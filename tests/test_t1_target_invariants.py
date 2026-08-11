from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Training"))
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from build_master_dataset import add_future_targets  # noqa: E402
from data import SequenceRecord, WindowDataset  # noqa: E402
from pose_baselines import resolve_target_timing  # noqa: E402


def _pose_master() -> pd.DataFrame:
    timestamps = np.asarray([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "timestamp_ns": timestamps,
            "hand_timestamp_ns": timestamps + 20_000_000,
        }
    )
    for side in ("left", "right"):
        for axis, values in zip("xyz", ([0.0, 1.0, 2.0], [0.0] * 3, [0.0] * 3)):
            frame[f"{side}_wrist_robot_{axis}_m"] = values
        for component in "xyz":
            frame[f"{side}_wrist_robot_q{component}"] = 0.0
        frame[f"{side}_wrist_robot_qw"] = 1.0
    return frame


def test_t1_target_keeps_master_and_physical_hand_timestamps() -> None:
    result = add_future_targets(
        _pose_master(), horizon_seconds=1.0, tolerance_ms=1.0
    )
    assert int(result.loc[0, "future_target_timestamp_ns"]) == 1_000_000_000
    assert int(result.loc[0, "future_target_hand_timestamp_ns"]) == 1_020_000_000
    assert float(result.loc[0, "future_1s_time_error_ms"]) == 0.0

    timing_frame = result.copy()
    timing = resolve_target_timing(timing_frame, 0, horizon_seconds=1.0)
    assert timing.nominal_timestamp_ns == 1_000_000_000
    assert timing.aligned_master_timestamp_ns == 1_000_000_000
    assert timing.actual_hand_capture_timestamp_ns == 1_020_000_000
    assert timing.prediction_horizon_timestamp_ns == 1_000_000_000
    assert timing.master_alignment_error_ms == 0.0
    assert timing.actual_capture_error_ms == 20.0


def _record() -> SequenceRecord:
    rows = 3
    hand_poses = np.zeros((rows, 2, 7), dtype=np.float32)
    hand_poses[:, :, 6] = 1.0
    hand_poses[:, 0, 0] = np.asarray([0.0, 99.0, 0.2], dtype=np.float32)
    return SequenceRecord(
        sequence_id="P1_0",
        participant="P1",
        timestamps_ns=np.asarray([0, 100_000_000, 200_000_000], dtype=np.int64),
        features=np.ones((rows, 2), dtype=np.float32),
        intentions=np.full(rows, 2, dtype=np.int64),
        pose_targets=np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (rows, 1),
        ),
        pose_valid=np.ones(rows, dtype=bool),
        receiving_hand_ids=np.zeros(rows, dtype=np.int64),
        hand_poses=hand_poses,
        hand_pose_valid=np.ones((rows, 2), dtype=bool),
        # Row 1 contains a nearest-merged capture from the future.  It must not
        # become the endpoint reference merely because it is on an earlier row.
        hand_timestamps_ns=np.asarray([0, 150_000_000, 200_000_000], dtype=np.int64),
        pose_target_timestamp_ns=np.asarray(
            [1_000_000_000, 1_100_000_000, 1_200_000_000], dtype=np.int64
        ),
        pose_target_hand_timestamp_ns=np.asarray(
            [1_000_000_000, 1_100_000_000, 1_200_000_000], dtype=np.int64
        ),
        pose_target_time_error_ms=np.zeros(rows, dtype=np.float64),
    )


def _window_dataset(maximum_age_seconds: float) -> WindowDataset:
    return WindowDataset(
        [_record()],
        window_size=2,
        stride=1,
        pose_intent_ids=[2],
        minimum_observed_fraction=0.0,
        max_timestamp_gap_seconds=1.0,
        include_hand_references=True,
        max_hand_reference_age_seconds=maximum_age_seconds,
    )


def test_hand_reference_is_causal_and_uses_capture_age() -> None:
    item = _window_dataset(0.25)[0]
    assert bool(item["hand_reference_valid"][0])
    assert np.isclose(float(item["hand_reference_age_seconds"][0]), 0.1)
    assert np.isclose(float(item["hand_reference_pose"][0, 0]), 0.0)


def test_stale_hand_reference_is_not_a_valid_pose_origin() -> None:
    item = _window_dataset(0.05)[0]
    assert not bool(item["hand_reference_valid"][0])
    assert not bool(item["residual_pose_valid"])
    assert np.isclose(float(item["hand_reference_age_seconds"][0]), 0.1)
