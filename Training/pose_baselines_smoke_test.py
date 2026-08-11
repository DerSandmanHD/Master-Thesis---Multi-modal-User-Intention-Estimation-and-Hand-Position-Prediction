#!/usr/bin/env python3
"""Scientific-invariant smoke tests for the primary t+1 pose baselines."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_pose_baselines import evaluate, mean_pose
from pose_baselines import (
    ObservationSeries,
    constant_velocity_pose,
    extract_hand_observations,
    observation_pose,
    persistence_pose,
    pose_matches,
    pose_metric_summary,
    resolve_target_timing,
)
from smoke_test import synthetic_sequence


IDENTITY_POSE = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
)


def observation_fixture(
    master_timestamps_ns: list[int],
    hand_timestamps_ns: list[int],
    positions_x: list[float],
) -> pd.DataFrame:
    rows = len(master_timestamps_ns)
    frame = pd.DataFrame(
        {
            "timestamp_ns": master_timestamps_ns,
            "hand_timestamp_ns": hand_timestamps_ns,
            "future_target_timestamp_ns": [np.nan] * rows,
            "receiving_hand": ["left"] * rows,
            "robot_frame_valid": np.ones(rows),
            "hand_left_valid": np.ones(rows),
            "hand_right_valid": np.ones(rows),
        }
    )
    for side in ("left", "right"):
        frame[f"{side}_wrist_robot_x_m"] = positions_x
        frame[f"{side}_wrist_robot_y_m"] = np.zeros(rows)
        frame[f"{side}_wrist_robot_z_m"] = np.zeros(rows)
        for component in "xyz":
            frame[f"{side}_wrist_robot_q{component}"] = np.zeros(rows)
        frame[f"{side}_wrist_robot_qw"] = np.ones(rows)
    return frame


def test_sign_invariant_pose_mean() -> None:
    average = mean_pose(
        np.asarray(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 4.0, 5.0, 0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
    )
    assert np.allclose(average[:3], [2.0, 3.0, 4.0])
    assert np.allclose(np.abs(average[3:7]), [0.0, 0.0, 0.0, 1.0])


def test_real_timestamp_deduplication_and_causality() -> None:
    frame = observation_fixture(
        [0, 100_000_000, 200_000_000, 300_000_000],
        [0, 100_000_000, 100_000_000, 310_000_000],
        [0.0, 1.0, 2.0, 3.0],
    )
    observations = extract_hand_observations(frame, 0, 3, "left")
    assert observations.capture_timestamps_ns.tolist() == [0, 100_000_000]
    assert observations.row_indices.tolist() == [0, 2]
    assert np.isclose(observations.poses[-1, 0], 2.0)
    assert observations.duplicate_captures_removed == 1
    assert observations.noncausal_captures_removed == 1


def test_persistence_missing_and_stale_handling() -> None:
    missing = ObservationSeries.empty()
    estimate = persistence_pose(
        missing, 1_000_000_000, maximum_age_seconds=0.25
    )
    assert not estimate.available and estimate.reason == "no_valid_observation"

    stale = ObservationSeries(
        row_indices=np.asarray([0], dtype=np.int64),
        capture_timestamps_ns=np.asarray([700_000_000], dtype=np.int64),
        poses=np.asarray([IDENTITY_POSE], dtype=np.float32),
    )
    estimate = persistence_pose(
        stale, 1_000_000_000, maximum_age_seconds=0.25
    )
    assert not estimate.available and estimate.reason == "stale_latest_observation"
    assert np.isclose(estimate.observation_age_seconds, 0.3)


def test_constant_velocity_uses_irregular_real_timestamps() -> None:
    timestamps = np.asarray(
        [0, 200_000_000, 700_000_000], dtype=np.int64
    )
    poses = np.tile(IDENTITY_POSE, (3, 1))
    poses[:, 0] = 2.0 * timestamps / 1e9
    # The last quaternion is intentionally -identity.  Holding it is a valid,
    # sign-equivalent zero-angular-velocity orientation baseline.
    poses[-1, 6] = -1.0
    observations = ObservationSeries(
        row_indices=np.arange(3, dtype=np.int64),
        capture_timestamps_ns=timestamps,
        poses=poses,
    )
    estimate = constant_velocity_pose(
        observations,
        endpoint_timestamp_ns=700_000_000,
        target_timestamp_ns=1_700_000_000,
        maximum_age_seconds=0.25,
        lookback_seconds=1.0,
        minimum_fit_span_seconds=0.1,
    )
    assert estimate.available
    assert estimate.fit_samples == 3
    assert np.isclose(estimate.fit_span_seconds, 0.7)
    assert np.isclose(estimate.estimated_speed_m_s, 2.0)
    assert estimate.pose is not None
    assert np.allclose(estimate.pose[:3], [3.4, 0.0, 0.0], atol=1e-6)
    assert np.allclose(estimate.pose[3:7], [0.0, 0.0, 0.0, -1.0])

    insufficient = ObservationSeries(
        row_indices=np.asarray([0, 1]),
        capture_timestamps_ns=np.asarray(
            [650_000_000, 700_000_000], dtype=np.int64
        ),
        poses=poses[-2:],
    )
    estimate = constant_velocity_pose(
        insufficient,
        endpoint_timestamp_ns=700_000_000,
        target_timestamp_ns=1_700_000_000,
        maximum_age_seconds=0.25,
        lookback_seconds=0.5,
        minimum_fit_span_seconds=0.1,
    )
    assert not estimate.available and estimate.reason == "insufficient_fit_span"


def test_target_timing_uses_actual_hand_capture() -> None:
    frame = observation_fixture(
        [0, 1_000_000_000],
        [0, 1_005_000_000],
        [0.0, 1.0],
    )
    frame.loc[0, "future_target_timestamp_ns"] = 1_000_000_000
    timing = resolve_target_timing(frame, 0, horizon_seconds=1.0)
    assert timing.nominal_timestamp_ns == 1_000_000_000
    assert timing.aligned_master_timestamp_ns == 1_000_000_000
    assert timing.actual_hand_capture_timestamp_ns == 1_005_000_000
    assert timing.prediction_horizon_timestamp_ns == 1_000_000_000
    assert np.isclose(timing.master_alignment_error_ms, 0.0)
    assert np.isclose(timing.actual_capture_error_ms, 5.0)
    target = observation_pose(frame, timing.target_row_index, "left")
    assert target is not None
    assert pose_matches(target, np.asarray([1.0, 0, 0, 0, 0, 0, 1]))


def test_constant_velocity_does_not_use_future_capture_jitter() -> None:
    observations = ObservationSeries(
        row_indices=np.asarray([0, 1], dtype=np.int64),
        capture_timestamps_ns=np.asarray([0, 500_000_000], dtype=np.int64),
        poses=np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    timing = resolve_target_timing(
        observation_fixture(
            [500_000_000, 1_500_000_000],
            [500_000_000, 1_600_000_000],
            [0.5, 1.6],
        ).assign(future_target_timestamp_ns=[1_500_000_000, np.nan]),
        0,
        horizon_seconds=1.0,
    )
    estimate = constant_velocity_pose(
        observations,
        endpoint_timestamp_ns=500_000_000,
        target_timestamp_ns=timing.prediction_horizon_timestamp_ns,
        maximum_age_seconds=0.25,
        lookback_seconds=1.0,
        minimum_fit_span_seconds=0.1,
    )
    assert estimate.available and estimate.pose is not None
    assert np.isclose(estimate.pose[0], 1.5)
    assert timing.actual_hand_capture_timestamp_ns == 1_600_000_000


def test_metric_mean_median_rms_orientation_and_coverage() -> None:
    targets = np.tile(IDENTITY_POSE, (3, 1))
    predictions = targets.copy()
    predictions[:, 0] = np.asarray([0.01, 0.03, 0.08])
    metrics = pose_metric_summary(
        predictions, targets, coverage_denominator=5
    )
    assert metrics["samples"] == 3
    assert np.isclose(metrics["position_mean_euclidean_error_cm"], 4.0)
    assert np.isclose(metrics["position_median_euclidean_error_cm"], 3.0)
    assert np.isclose(
        metrics["position_root_mean_square_euclidean_error_cm"],
        math.sqrt(74.0 / 3.0),
    )
    assert np.isclose(metrics["orientation_mean_deg"], 0.0)
    assert np.isclose(metrics["orientation_median_deg"], 0.0)
    assert metrics["coverage_numerator"] == 3
    assert metrics["coverage_denominator"] == 5
    assert np.isclose(metrics["coverage"], 0.6)


def make_consistent_t_plus_one_sequence(
    path: Path,
    participant: str,
    sequence_number: int,
) -> None:
    synthetic_sequence(path, participant, sequence_number)
    frame = pd.read_csv(path)
    rows = len(frame)
    timestamps = np.arange(rows, dtype=np.int64) * 50_000_000
    frame["timestamp_ns"] = timestamps
    frame["hand_timestamp_ns"] = timestamps
    time_seconds = timestamps / 1e9
    for side_index, side in enumerate(("left", "right")):
        frame[f"{side}_wrist_robot_x_m"] = (
            0.1 * side_index + 0.2 * time_seconds
        )
        frame[f"{side}_wrist_robot_y_m"] = 0.2 + 0.05 * side_index
        frame[f"{side}_wrist_robot_z_m"] = 0.3
        for component in "xyz":
            frame[f"{side}_wrist_robot_q{component}"] = 0.0
        frame[f"{side}_wrist_robot_qw"] = 1.0

    horizon_rows = 20
    aligned_targets = np.full(rows, np.nan)
    future_valid = np.zeros(rows, dtype=np.int8)
    future_values = {
        component: np.full(rows, np.nan)
        for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
    }
    side = str(frame["receiving_hand"].iloc[0])
    for endpoint in range(rows - horizon_rows):
        target_row = endpoint + horizon_rows
        aligned_targets[endpoint] = timestamps[target_row]
        future_valid[endpoint] = 1
        for component in ("x", "y", "z"):
            future_values[component][endpoint] = frame.iloc[target_row][
                f"{side}_wrist_robot_{component}_m"
            ]
        for component in ("qx", "qy", "qz", "qw"):
            future_values[component][endpoint] = frame.iloc[target_row][
                f"{side}_wrist_robot_{component}"
            ]
    frame["future_target_timestamp_ns"] = aligned_targets
    frame["future_target_hand_timestamp_ns"] = aligned_targets
    frame["future_1s_time_error_ms"] = np.where(future_valid, 0.0, np.nan)
    frame["future_1s_receiving_wrist_valid"] = future_valid
    for component in ("x", "y", "z"):
        frame[f"future_1s_receiving_wrist_robot_{component}_m"] = future_values[
            component
        ]
    for component in ("qx", "qy", "qz", "qw"):
        frame[f"future_1s_receiving_wrist_robot_{component}"] = future_values[
            component
        ]
    frame.to_csv(path, index=False)


def test_end_to_end_report_has_pure_and_fair_baselines() -> None:
    with tempfile.TemporaryDirectory(prefix="aria_pose_baselines_") as directory:
        root = Path(directory)
        master_dir = root / "master_datasets"
        master_dir.mkdir()
        for index in range(6):
            participant = f"P{index + 1}"
            make_consistent_t_plus_one_sequence(
                master_dir / f"{participant}_{index}_master.csv",
                participant,
                index,
            )

        config = {
            "data": {
                "master_dir": str(master_dir),
                "feature_profile": "multimodal_robot_frame_v1",
                "window_size": 20,
                "stride": 10,
                "future_horizon_seconds": 1.0,
                "pose_intent_ids": [2],
                "minimum_observed_fraction": 0.05,
                "max_timestamp_gap_seconds": 0.2,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
                "validation_participants": [],
                "test_participants": [],
            },
            "training": {"seed": 42},
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report_path = root / "report.json"
        details_path = root / "details.csv"
        report = evaluate(
            config,
            config_path=config_path,
            report_path=report_path,
            details_path=details_path,
            velocity_lookback_seconds=0.5,
            maximum_observation_age_seconds=0.25,
            minimum_velocity_fit_span_seconds=0.1,
        )

        assert report_path.exists() and details_path.exists()
        assert report["baseline_policy"]["fallbacks"] == "none"
        assert report["timestamp_policy"]["observation_time"].startswith(
            "real hand_timestamp_ns"
        )
        details = pd.read_csv(details_path)
        assert len(details) > 0
        assert details["fair_common"].all()
        assert (details["persistence_reason"] == "available").all()
        assert (details["constant_velocity_reason"] == "available").all()
        assert (details["constant_velocity_fit_span_seconds"] >= 0.1).all()
        assert (details["target_actual_capture_error_ms"] == 0.0).all()

        for split_name in ("train", "validation", "test"):
            split = report["splits"][split_name]
            assert split["valid_pose_targets"] > 0
            common_samples = split["fair_common"]["samples"]
            assert common_samples == split["valid_pose_targets"]
            persistence = split["baselines"]["persistence"]
            velocity = split["baselines"]["constant_velocity"]
            assert persistence["fair_common_metrics"]["samples"] == common_samples
            assert velocity["fair_common_metrics"]["samples"] == common_samples
            assert persistence["native_sample_key_fingerprint"] == velocity[
                "native_sample_key_fingerprint"
            ]
            assert persistence["fair_common_metrics"]["coverage"] == 1.0
            assert velocity["fair_common_metrics"]["coverage"] == 1.0
            assert velocity["fair_common_metrics"][
                "position_mean_euclidean_error_cm"
            ] < 1e-4
            assert persistence["fair_common_metrics"][
                "position_mean_euclidean_error_cm"
            ] > 1.0


def main() -> int:
    tests = (
        test_sign_invariant_pose_mean,
        test_real_timestamp_deduplication_and_causality,
        test_persistence_missing_and_stale_handling,
        test_constant_velocity_uses_irregular_real_timestamps,
        test_target_timing_uses_actual_hand_capture,
        test_constant_velocity_does_not_use_future_capture_jitter,
        test_metric_mean_median_rms_orientation_and_coverage,
        test_end_to_end_report_has_pure_and_fair_baselines,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"Pose baseline smoke test passed ({len(tests)} invariants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
