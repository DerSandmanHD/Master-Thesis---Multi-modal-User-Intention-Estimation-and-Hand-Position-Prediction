#!/usr/bin/env python3
"""Data adapter and loss calibration helpers for terminal end-pose v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import Dataset

from data import DataBundle, INTENTION_TO_ID


DUAL_HORIZON_MODEL_TYPE = "hierarchical_dual_horizon_residual_pose_transformer_v3"
ADAPTER_VERSION = "terminal_endpose_v2_actual_capture_timing_and_balancing_v3"
AUXILIARY_TARGET_DEFINITION_VERSION = "future_offset_unique_hand_capture_v2"
AUXILIARY_CAPTURE_TIMESTAMP_BASIS = "hand_timestamp_ns"


def _nearest_future_capture_index(
    event_timestamps_ns: np.ndarray,
    capture_timestamps_ns: np.ndarray,
    capture_pose_valid: np.ndarray,
    endpoint: int,
    hand_id: int,
    horizon_ns: int,
    maximum_gap_ns: int,
) -> int | None:
    """Find the closest unique physical capture to the requested horizon."""

    event_timestamps = np.asarray(event_timestamps_ns, dtype=np.int64)
    capture_timestamps = np.asarray(capture_timestamps_ns, dtype=np.int64)
    valid = np.asarray(capture_pose_valid, dtype=bool)
    if (
        capture_timestamps.shape != event_timestamps.shape
        or valid.shape != (len(event_timestamps), 2)
        or endpoint < 0
        or endpoint >= len(event_timestamps)
        or hand_id not in (0, 1)
    ):
        raise ValueError("Inconsistent auxiliary capture-timing arrays")

    endpoint_timestamp = int(event_timestamps[endpoint])
    query_timestamp = endpoint_timestamp + int(horizon_ns)
    candidate_rows = np.flatnonzero(
        (capture_timestamps > endpoint_timestamp) & valid[:, hand_id]
    )
    if not len(candidate_rows):
        return None

    representatives = []
    for capture_timestamp in np.unique(capture_timestamps[candidate_rows]):
        matching = candidate_rows[
            capture_timestamps[candidate_rows] == capture_timestamp
        ]
        representatives.append(
            min(
                matching.tolist(),
                key=lambda row: (
                    abs(int(event_timestamps[row]) - int(capture_timestamp)),
                    int(event_timestamps[row]),
                    int(row),
                ),
            )
        )
    nearest = min(
        representatives,
        key=lambda row: (
            abs(int(capture_timestamps[row]) - query_timestamp),
            int(capture_timestamps[row]),
            int(row),
        ),
    )
    if abs(int(capture_timestamps[nearest]) - query_timestamp) > maximum_gap_ns:
        return None
    return int(nearest)


def _time_bin(seconds: float) -> str:
    if seconds <= 0.5:
        return "0-0.5s"
    if seconds <= 1.0:
        return "0.5-1s"
    if seconds <= 2.0:
        return "1-2s"
    if seconds <= 3.0:
        return "2-3s"
    return ">=3s"


class EndposeV2Dataset(Dataset):
    """Add t+1 auxiliary targets and deterministic balancing weights."""

    def __init__(self, base, auxiliary_config: dict) -> None:
        self.base = base
        self.records = base.records
        self.indices = base.indices
        self.auxiliary_config = dict(auxiliary_config)
        if self.auxiliary_config.get("mode") != "future_offset":
            raise ValueError("endpose-v2 auxiliary target must use mode='future_offset'")
        horizon_seconds = float(
            self.auxiliary_config.get("future_horizon_seconds", 1.0)
        )
        maximum_gap_seconds = float(
            self.auxiliary_config.get("maximum_target_gap_seconds", 0.1)
        )
        if horizon_seconds <= 0 or maximum_gap_seconds <= 0:
            raise ValueError("auxiliary horizon and maximum gap must be positive")
        configured_version = self.auxiliary_config.get(
            "target_definition_version", AUXILIARY_TARGET_DEFINITION_VERSION
        )
        if configured_version != AUXILIARY_TARGET_DEFINITION_VERSION:
            raise ValueError(
                "Unsupported auxiliary target_definition_version: "
                f"{configured_version!r}"
            )
        configured_basis = self.auxiliary_config.get(
            "capture_timestamp_basis", AUXILIARY_CAPTURE_TIMESTAMP_BASIS
        )
        if configured_basis != AUXILIARY_CAPTURE_TIMESTAMP_BASIS:
            raise ValueError(
                "Auxiliary targets require capture_timestamp_basis="
                f"{AUXILIARY_CAPTURE_TIMESTAMP_BASIS!r}"
            )
        self.auxiliary_config.update(
            {
                "future_horizon_seconds": horizon_seconds,
                "maximum_target_gap_seconds": maximum_gap_seconds,
                "target_definition_version": AUXILIARY_TARGET_DEFINITION_VERSION,
                "capture_timestamp_basis": AUXILIARY_CAPTURE_TIMESTAMP_BASIS,
            }
        )
        horizon_ns = int(round(horizon_seconds * 1e9))
        maximum_gap_ns = int(round(maximum_gap_seconds * 1e9))

        count = len(base)
        self.auxiliary_pose_targets = np.zeros((count, 7), dtype=np.float32)
        self.auxiliary_pose_targets[:, 6] = 1.0
        self.auxiliary_pose_valid = np.zeros(count, dtype=bool)
        self.auxiliary_target_timestamp_ns = np.full(count, -1, dtype=np.int64)
        self.auxiliary_query_timestamp_ns = np.full(count, -1, dtype=np.int64)
        self.auxiliary_target_time_error_ms = np.full(
            count, np.nan, dtype=np.float32
        )

        for dataset_index, (record_index, endpoint) in enumerate(self.indices):
            record = self.records[record_index]
            query_timestamp_ns = int(record.timestamps_ns[endpoint]) + horizon_ns
            self.auxiliary_query_timestamp_ns[dataset_index] = query_timestamp_ns
            if (
                int(record.intentions[endpoint]) != INTENTION_TO_ID["handover"]
                or record.receiving_hand_ids is None
                or record.hand_poses is None
                or record.hand_pose_valid is None
                or record.hand_timestamps_ns is None
            ):
                continue
            hand_id = int(record.receiving_hand_ids[endpoint])
            if hand_id not in (0, 1):
                continue
            target_index = _nearest_future_capture_index(
                record.timestamps_ns,
                record.hand_timestamps_ns,
                record.hand_pose_valid,
                endpoint,
                hand_id,
                horizon_ns,
                maximum_gap_ns,
            )
            if target_index is None:
                continue
            target = np.asarray(
                record.hand_poses[target_index, hand_id], dtype=np.float32
            )
            if not np.isfinite(target).all() or np.linalg.norm(target[3:7]) <= 1e-6:
                continue
            target = target.copy()
            target[3:7] /= np.linalg.norm(target[3:7])
            self.auxiliary_pose_targets[dataset_index] = target
            self.auxiliary_pose_valid[dataset_index] = True
            self.auxiliary_target_timestamp_ns[dataset_index] = int(
                record.hand_timestamps_ns[target_index]
            )
            self.auxiliary_target_time_error_ms[dataset_index] = float(
                (
                    int(record.hand_timestamps_ns[target_index])
                    - query_timestamp_ns
                )
                / 1e6
            )

        sequence_counts = Counter(record_index for record_index, _ in self.indices)
        self._sequence_sampling_weights = np.asarray(
            [1.0 / sequence_counts[record_index] for record_index, _ in self.indices],
            dtype=np.float64,
        )
        self._sequence_sampling_weights /= self._sequence_sampling_weights.mean()

        self._primary_pose_sample_weights = np.ones(count, dtype=np.float32)
        self._auxiliary_pose_sample_weights = np.ones(count, dtype=np.float32)
        pose_groups: list[tuple[int, str] | None] = [None] * count
        group_counts: Counter[tuple[int, str]] = Counter()
        for dataset_index, (record_index, endpoint) in enumerate(self.indices):
            record = self.records[record_index]
            # Match the actual residual-loss mask. A terminal target without a
            # valid receiving-hand reference is not executable and must not
            # dilute a sequence/time-bin group.
            if not bool(self.base[dataset_index]["residual_pose_valid"]):
                continue
            target_timestamps = record.pose_target_timestamp_ns
            if target_timestamps is None:
                continue
            seconds = (
                int(target_timestamps[endpoint]) - int(record.timestamps_ns[endpoint])
            ) / 1e9
            if not np.isfinite(seconds) or seconds < 0:
                continue
            group = (record_index, _time_bin(float(seconds)))
            pose_groups[dataset_index] = group
            group_counts[group] += 1
        valid_weights = []
        for dataset_index, group in enumerate(pose_groups):
            if group is None:
                continue
            record_index, _ = self.indices[dataset_index]
            # WeightedRandomSampler draws each window with probability
            # proportional to 1 / n_sequence. Multiplying by n_sequence here
            # cancels that proposal probability; division by the executable
            # group count then gives equal expected mass per sequence/time bin.
            weight = sequence_counts[record_index] / group_counts[group]
            self._primary_pose_sample_weights[dataset_index] = weight
            valid_weights.append(weight)
        if valid_weights:
            normalization = float(np.mean(valid_weights))
            for dataset_index, group in enumerate(pose_groups):
                if group is not None:
                    self._primary_pose_sample_weights[dataset_index] /= normalization

        auxiliary_groups: list[int | None] = [None] * count
        auxiliary_group_counts: Counter[int] = Counter()
        for dataset_index, (record_index, _) in enumerate(self.indices):
            item = self.base[dataset_index]
            receiving_hand = int(item["receiving_hand"])
            reference_valid = bool(
                receiving_hand in (0, 1)
                and item["hand_reference_valid"][receiving_hand]
            )
            if not (bool(self.auxiliary_pose_valid[dataset_index]) and reference_valid):
                continue
            auxiliary_groups[dataset_index] = record_index
            auxiliary_group_counts[record_index] += 1
        auxiliary_valid_weights = []
        for dataset_index, record_index in enumerate(auxiliary_groups):
            if record_index is None:
                continue
            # The auxiliary task has one fixed horizon, so it is balanced by
            # sequence only. This again corrects for the sequence-balanced
            # sampler and uses only executable auxiliary residual targets.
            weight = (
                sequence_counts[record_index]
                / auxiliary_group_counts[record_index]
            )
            self._auxiliary_pose_sample_weights[dataset_index] = weight
            auxiliary_valid_weights.append(weight)
        if auxiliary_valid_weights:
            normalization = float(np.mean(auxiliary_valid_weights))
            for dataset_index, record_index in enumerate(auxiliary_groups):
                if record_index is not None:
                    self._auxiliary_pose_sample_weights[dataset_index] /= normalization

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base[index])
        receiving_hand = int(item["receiving_hand"])
        reference_valid = bool(
            receiving_hand in (0, 1)
            and item["hand_reference_valid"][receiving_hand]
        )
        item.update(
            {
                "auxiliary_pose_target": torch.from_numpy(
                    self.auxiliary_pose_targets[index]
                ),
                "auxiliary_pose_valid": torch.tensor(
                    self.auxiliary_pose_valid[index], dtype=torch.bool
                ),
                "auxiliary_residual_pose_valid": torch.tensor(
                    bool(self.auxiliary_pose_valid[index]) and reference_valid,
                    dtype=torch.bool,
                ),
                "auxiliary_pose_target_timestamp_ns": torch.tensor(
                    self.auxiliary_target_timestamp_ns[index], dtype=torch.long
                ),
                "auxiliary_pose_query_timestamp_ns": torch.tensor(
                    self.auxiliary_query_timestamp_ns[index], dtype=torch.long
                ),
                "auxiliary_pose_target_time_error_ms": torch.tensor(
                    self.auxiliary_target_time_error_ms[index],
                    dtype=torch.float32,
                ),
                "primary_pose_sample_weight": torch.tensor(
                    self._primary_pose_sample_weights[index], dtype=torch.float32
                ),
                "auxiliary_pose_sample_weight": torch.tensor(
                    self._auxiliary_pose_sample_weights[index], dtype=torch.float32
                ),
                # Historical alias retained for old diagnostics. New training
                # code reads the task-specific keys above.
                "pose_sample_weight": torch.tensor(
                    self._primary_pose_sample_weights[index], dtype=torch.float32
                ),
            }
        )
        return item

    def sequence_sampling_weights(self) -> torch.Tensor:
        return torch.from_numpy(self._sequence_sampling_weights.copy()).double()

    def auxiliary_pose_count(self) -> int:
        return int(self.auxiliary_pose_valid.sum())

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def wrap_endpose_v2_bundle(bundle: DataBundle, data_config: dict) -> DataBundle:
    target = bundle.split_metadata.get("pose_target", {})
    if target.get("mode") != "terminal_endpose":
        raise ValueError("endpose-v2 requires terminal_endpose as its primary target")
    auxiliary = dict(data_config.get("auxiliary_pose_target", {}))
    if not auxiliary:
        raise ValueError("endpose-v2 requires data.auxiliary_pose_target")
    wrapped = {
        split: EndposeV2Dataset(getattr(bundle, split), auxiliary)
        for split in ("train", "validation", "test")
    }
    split_metadata = dict(bundle.split_metadata)
    split_metadata["endpose_v2_adapter"] = {
        "version": ADAPTER_VERSION,
        "primary_target": target,
        "auxiliary_target": wrapped["train"].auxiliary_config,
        "auxiliary_target_windows": {
            split: wrapped[split].auxiliary_pose_count() for split in wrapped
        },
        "loss_balancing": {
            "sampler": "sequence_balanced_weighted_random_sampler",
            "primary_terminal_pose": (
                "importance-corrected equal expected mass per executable "
                "sequence/time-to-terminal bin"
            ),
            "auxiliary_t_plus_1_pose": (
                "separate importance-corrected equal expected mass per "
                "executable sequence at the fixed horizon"
            ),
            "validity_masks": {
                "primary": "residual_pose_valid",
                "auxiliary": "auxiliary_residual_pose_valid",
            },
        },
    }
    return replace(
        bundle,
        train=wrapped["train"],
        validation=wrapped["validation"],
        test=wrapped["test"],
        split_metadata=split_metadata,
    )


def residual_position_scale_m(
    dataset: EndposeV2Dataset,
    *,
    target_key: str,
    valid_key: str,
    minimum_scale_m: float,
) -> list[float]:
    residuals = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not bool(item[valid_key]):
            continue
        hand_id = int(item["receiving_hand"])
        target = item[target_key][:3].numpy()
        reference = item["hand_reference_pose"][hand_id, :3].numpy()
        residuals.append(target - reference)
    if not residuals:
        raise ValueError(f"No valid samples available for {target_key} scaling")
    scale = np.std(np.asarray(residuals, dtype=np.float64), axis=0, ddof=0)
    scale = np.maximum(scale, float(minimum_scale_m))
    return [float(value) for value in scale]
