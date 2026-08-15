from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from endpose_targets import (  # noqa: E402
    TARGET_DEFINITION_VERSION,
    estimate_terminal_endpose,
    future_terminal_target_mask,
    normalized_terminal_endpose_config,
)
from endpose_v2 import (  # noqa: E402
    AUXILIARY_TARGET_DEFINITION_VERSION,
    EndposeV2Dataset,
    _nearest_future_capture_index,
)


def stable_terminal_arrays() -> dict:
    rows = 120
    event = 10_000_000_000 + np.arange(rows, dtype=np.int64) * 33_333_333
    # A 10 Hz physical hand stream is nearest-merged to the 30 Hz master.
    capture = 10_000_000_000 + (np.arange(rows) // 3).astype(np.int64) * 100_000_000
    intention = np.zeros(rows, dtype=np.int64)
    intention[30:] = 2
    hand_ids = np.ones(rows, dtype=np.int64)
    poses = np.zeros((rows, 2, 7), dtype=np.float32)
    poses[:, :, 6] = 1.0
    poses[:, 1, :3] = np.asarray([0.4, -0.2, 0.8], dtype=np.float32)
    valid = np.ones((rows, 2), dtype=bool)
    return {
        "timestamps_ns": event,
        "hand_timestamps_ns": capture,
        "intention_ids": intention,
        "receiving_hand_ids": hand_ids,
        "hand_poses": poses,
        "hand_pose_valid": valid,
    }


def target_config() -> dict:
    return {
        "mode": "terminal_endpose",
        "target_definition_version": TARGET_DEFINITION_VERSION,
        "capture_timestamp_basis": "hand_timestamp_ns",
        "aggregation_window_seconds": 0.5,
        "minimum_valid_samples": 4,
        "minimum_valid_ratio": 0.7,
        "minimum_valid_span_seconds": 0.25,
        "maximum_position_p90_deviation_m": 0.05,
        "maximum_orientation_p90_deviation_deg": 25.0,
        "maximum_stable_window_lag_seconds": 1.0,
    }


def test_terminal_quality_uses_unique_physical_captures() -> None:
    arrays = stable_terminal_arrays()
    estimate = estimate_terminal_endpose(
        **arrays,
        handover_intent_id=2,
        receiving_hand_names=("left", "right"),
        config=target_config(),
    )
    assert estimate.eligible
    assert estimate.candidate_rows > estimate.candidate_unique_captures
    assert estimate.valid_samples == estimate.candidate_unique_captures
    assert estimate.target_capture_timestamp_ns == (
        estimate.aggregation_capture_end_timestamp_ns
    )
    assert estimate.target_capture_timestamp_ns != estimate.handover_event_end_timestamp_ns
    assert estimate.valid_span_seconds is not None
    assert estimate.valid_span_seconds >= 0.25


def test_repeated_master_rows_cannot_fake_capture_span_or_sample_count() -> None:
    arrays = stable_terminal_arrays()
    arrays["hand_timestamps_ns"][:] = arrays["hand_timestamps_ns"][-1]
    estimate = estimate_terminal_endpose(
        **arrays,
        handover_intent_id=2,
        receiving_hand_names=("left", "right"),
        config=target_config(),
    )
    assert not estimate.eligible
    assert estimate.candidate_unique_captures == 1
    assert estimate.valid_span_seconds == 0.0
    assert "insufficient_valid_samples" in estimate.reasons
    assert "insufficient_valid_span" in estimate.reasons


def test_terminal_target_is_strictly_future_at_endpoint() -> None:
    endpoint = np.asarray([99, 100, 101], dtype=np.int64)
    assert future_terminal_target_mask(endpoint, 100).tolist() == [True, False, False]


def test_legacy_terminal_definition_is_invalidated() -> None:
    config = target_config()
    config["target_definition_version"] = "terminal_endpose_master_rows_v1"
    try:
        normalized_terminal_endpose_config(config)
    except ValueError as exc:
        assert "target_definition_version" in str(exc)
    else:
        raise AssertionError("Legacy terminal target definition was accepted")


def test_auxiliary_target_uses_actual_capture_and_tolerance() -> None:
    event = np.arange(6, dtype=np.int64) * 100_000_000
    capture = np.asarray([0, 0, 210, 210, 405, 405], dtype=np.int64) * 1_000_000
    valid = np.ones((6, 2), dtype=bool)
    selected = _nearest_future_capture_index(
        event, capture, valid, endpoint=1, hand_id=1,
        horizon_ns=300_000_000, maximum_gap_ns=10_000_000,
    )
    assert selected is not None
    assert int(capture[selected]) == 405_000_000
    missing = _nearest_future_capture_index(
        event, capture, valid, endpoint=1, hand_id=1,
        horizon_ns=300_000_000, maximum_gap_ns=4_000_000,
    )
    assert missing is None


def test_auxiliary_availability_is_independent_of_terminal_target_validity() -> None:
    timestamps = np.arange(6, dtype=np.int64) * 100_000_000
    hand_poses = np.zeros((6, 2, 7), dtype=np.float32)
    hand_poses[:, :, 6] = 1.0
    hand_poses[:, 1, 0] = np.arange(6, dtype=np.float32) * 0.01
    record = SimpleNamespace(
        timestamps_ns=timestamps,
        hand_timestamps_ns=timestamps.copy(),
        hand_pose_valid=np.ones((6, 2), dtype=bool),
        hand_poses=hand_poses,
        intentions=np.full(6, 2, dtype=np.int64),
        receiving_hand_ids=np.ones(6, dtype=np.int64),
        pose_valid=np.zeros(6, dtype=bool),
        pose_target_timestamp_ns=np.full(6, -1, dtype=np.int64),
    )

    class FakeBase:
        records = [record]
        indices = [(0, 1)]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            assert index == 0
            return {
                "receiving_hand": torch.tensor(1),
                "hand_reference_valid": torch.tensor([True, True]),
                "hand_reference_pose": torch.zeros((2, 7)),
                "residual_pose_valid": torch.tensor(False),
            }

    dataset = EndposeV2Dataset(
        FakeBase(),
        {
            "mode": "future_offset",
            "target_definition_version": AUXILIARY_TARGET_DEFINITION_VERSION,
            "capture_timestamp_basis": "hand_timestamp_ns",
            "future_horizon_seconds": 0.2,
            "maximum_target_gap_seconds": 0.01,
        },
    )
    item = dataset[0]
    assert bool(item["auxiliary_pose_valid"])
    assert "primary_pose_sample_weight" in item
    assert "auxiliary_pose_sample_weight" in item
    assert int(item["auxiliary_pose_target_timestamp_ns"]) == 300_000_000
    assert not bool(record.pose_valid[1])


def test_task_specific_pose_weights_correct_sequence_sampler_proposal() -> None:
    def record(primary_offsets_s: list[float]) -> SimpleNamespace:
        count = 6
        timestamps = np.arange(count, dtype=np.int64) * 1_000_000_000
        poses = np.zeros((count, 2, 7), dtype=np.float32)
        poses[:, :, 6] = 1.0
        targets = timestamps.copy()
        for index, offset in enumerate(primary_offsets_s):
            targets[index] = timestamps[index] + int(offset * 1e9)
        return SimpleNamespace(
            timestamps_ns=timestamps,
            hand_timestamps_ns=timestamps.copy(),
            hand_pose_valid=np.ones((count, 2), dtype=bool),
            hand_poses=poses,
            intentions=np.full(count, 2, dtype=np.int64),
            receiving_hand_ids=np.ones(count, dtype=np.int64),
            pose_valid=np.ones(count, dtype=bool),
            pose_target_timestamp_ns=targets,
        )

    records = [
        record([0.25, 0.25, 1.5, 1.5]),
        record([0.25, 1.5]),
    ]
    indices = [(0, index) for index in range(4)] + [
        (1, index) for index in range(2)
    ]
    primary_valid = [True, True, False, True, True, True]

    class FakeBase:
        def __init__(self) -> None:
            self.records = records
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> dict:
            receiving_reference_valid = primary_valid[index]
            return {
                "receiving_hand": torch.tensor(1),
                "hand_reference_valid": torch.tensor(
                    [True, receiving_reference_valid]
                ),
                "hand_reference_pose": torch.zeros((2, 7)),
                "residual_pose_valid": torch.tensor(primary_valid[index]),
            }

    dataset = EndposeV2Dataset(
        FakeBase(),
        {
            "mode": "future_offset",
            "target_definition_version": AUXILIARY_TARGET_DEFINITION_VERSION,
            "capture_timestamp_basis": "hand_timestamp_ns",
            "future_horizon_seconds": 1.0,
            "maximum_target_gap_seconds": 0.01,
        },
    )
    proposal = dataset.sequence_sampling_weights().numpy()
    primary_weights = np.asarray(
        [float(dataset[index]["primary_pose_sample_weight"]) for index in range(6)]
    )
    primary_groups = {
        "r0_short": [0, 1],
        "r0_long": [3],
        "r1_short": [4],
        "r1_long": [5],
    }
    primary_masses = [
        float(np.sum(proposal[rows] * primary_weights[rows]))
        for rows in primary_groups.values()
    ]
    assert np.allclose(primary_masses, primary_masses[0])

    auxiliary_weights = np.asarray(
        [
            float(dataset[index]["auxiliary_pose_sample_weight"])
            for index in range(6)
        ]
    )
    # Terminal short/long bins do not affect fixed-horizon auxiliary weights.
    assert np.isclose(auxiliary_weights[0], auxiliary_weights[1])
    assert np.isclose(auxiliary_weights[1], auxiliary_weights[3])
    auxiliary_sequence_masses = [
        float(np.sum(proposal[[0, 1, 3]] * auxiliary_weights[[0, 1, 3]])),
        float(np.sum(proposal[[4, 5]] * auxiliary_weights[[4, 5]])),
    ]
    assert np.allclose(auxiliary_sequence_masses, auxiliary_sequence_masses[0])
    assert not bool(dataset[2]["auxiliary_residual_pose_valid"])
