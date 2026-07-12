#!/usr/bin/env python3
"""Smoke-test the naive pose baselines on synthetic master CSVs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from evaluate_pose_baselines import constant_velocity_pose, evaluate, mean_pose
from smoke_test import synthetic_sequence


def main() -> int:
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

    velocity_prediction, samples, speed = constant_velocity_pose(
        np.asarray([0, 500_000_000], dtype=np.int64),
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        desired_timestamp_ns=1_500_000_000,
        lookback_seconds=0.5,
    )
    assert velocity_prediction is not None
    assert samples == 2 and np.isclose(speed, 2.0)
    assert np.allclose(velocity_prediction[:3], [3.0, 0.0, 0.0])

    with tempfile.TemporaryDirectory(prefix="aria_pose_baselines_") as directory:
        root = Path(directory)
        master_dir = root / "master_datasets"
        master_dir.mkdir()
        for index in range(6):
            participant = f"P{index + 1}"
            synthetic_sequence(
                master_dir / f"{participant}_{index}_master.csv", participant, index
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
        )

        assert report_path.exists() and details_path.exists()
        assert report["training_target_count_for_mean"] > 0
        for split_name in ("train", "validation", "test"):
            split = report["splits"][split_name]
            assert split["valid_pose_targets"] > 0
            expected_samples = split["valid_pose_targets"]
            for result in split["baselines"].values():
                assert result["metrics"]["samples"] == expected_samples
                assert result["metrics"]["position_mae_cm"] is not None
                assert sum(result["prediction_sources"].values()) == expected_samples
        print("Pose baseline smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
