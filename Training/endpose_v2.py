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
ADAPTER_VERSION = "terminal_endpose_v2_dual_horizon_sequence_balanced_v1"


def _nearest_future_index(
    timestamps_ns: np.ndarray,
    endpoint: int,
    horizon_ns: int,
    maximum_gap_ns: int,
) -> int | None:
    target = int(timestamps_ns[endpoint]) + horizon_ns
    insertion = int(np.searchsorted(timestamps_ns, target, side="left"))
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if endpoint < index < len(timestamps_ns)
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda index: abs(int(timestamps_ns[index]) - target))
    if abs(int(timestamps_ns[nearest]) - target) > maximum_gap_ns:
        return None
    return nearest


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
        horizon_ns = int(round(horizon_seconds * 1e9))
        maximum_gap_ns = int(round(maximum_gap_seconds * 1e9))

        count = len(base)
        self.auxiliary_pose_targets = np.zeros((count, 7), dtype=np.float32)
        self.auxiliary_pose_targets[:, 6] = 1.0
        self.auxiliary_pose_valid = np.zeros(count, dtype=bool)
        self.auxiliary_target_timestamp_ns = np.full(count, -1, dtype=np.int64)

        for dataset_index, (record_index, endpoint) in enumerate(self.indices):
            record = self.records[record_index]
            if (
                int(record.intentions[endpoint]) != INTENTION_TO_ID["handover"]
                or not bool(record.pose_valid[endpoint])
                or record.receiving_hand_ids is None
                or record.hand_poses is None
                or record.hand_pose_valid is None
            ):
                continue
            hand_id = int(record.receiving_hand_ids[endpoint])
            if hand_id not in (0, 1):
                continue
            target_index = _nearest_future_index(
                record.timestamps_ns,
                endpoint,
                horizon_ns,
                maximum_gap_ns,
            )
            if target_index is None or not bool(
                record.hand_pose_valid[target_index, hand_id]
            ):
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
                record.timestamps_ns[target_index]
            )

        sequence_counts = Counter(record_index for record_index, _ in self.indices)
        self._sequence_sampling_weights = np.asarray(
            [1.0 / sequence_counts[record_index] for record_index, _ in self.indices],
            dtype=np.float64,
        )
        self._sequence_sampling_weights /= self._sequence_sampling_weights.mean()

        self._pose_sample_weights = np.ones(count, dtype=np.float32)
        pose_groups: list[tuple[int, str] | None] = [None] * count
        group_counts: Counter[tuple[int, str]] = Counter()
        for dataset_index, (record_index, endpoint) in enumerate(self.indices):
            record = self.records[record_index]
            if not bool(record.pose_valid[endpoint]):
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
            weight = 1.0 / group_counts[group]
            self._pose_sample_weights[dataset_index] = weight
            valid_weights.append(weight)
        if valid_weights:
            normalization = float(np.mean(valid_weights))
            for dataset_index, group in enumerate(pose_groups):
                if group is not None:
                    self._pose_sample_weights[dataset_index] /= normalization

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
                "pose_sample_weight": torch.tensor(
                    self._pose_sample_weights[index], dtype=torch.float32
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
        "auxiliary_target": auxiliary,
        "auxiliary_target_windows": {
            split: wrapped[split].auxiliary_pose_count() for split in wrapped
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
