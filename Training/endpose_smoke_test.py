#!/usr/bin/env python3
"""Synthetic checks for robust terminal targets and time-to-end grouping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data
from endpose_targets import (
    estimate_terminal_endpose,
    quaternion_average,
)
from model import HierarchicalResidualPoseTransformer
from smoke_test import synthetic_sequence
from train_residual import TIME_TO_END_GROUPS, run_epoch


TARGET_CONFIG = {
    "mode": "terminal_endpose",
    "aggregation_window_seconds": 0.5,
    "minimum_valid_samples": 8,
    "minimum_valid_ratio": 0.7,
    "minimum_valid_span_seconds": 0.35,
    "maximum_position_p90_deviation_m": 0.05,
    "maximum_orientation_p90_deviation_deg": 25.0,
    "maximum_stable_window_lag_seconds": 1.0,
}


def direct_aggregation_check() -> None:
    rows = 90
    timestamps = np.arange(rows, dtype=np.int64) * 33_333_333
    intentions = np.zeros(rows, dtype=np.int64)
    intentions[30:] = 2
    hand_ids = np.ones(rows, dtype=np.int64)
    poses = np.zeros((rows, 2, 7), dtype=np.float32)
    poses[:, :, 6] = 1.0
    valid = np.ones((rows, 2), dtype=bool)
    rng = np.random.default_rng(7)
    poses[:, 1, :3] = rng.normal(0.0, 0.2, size=(rows, 3))
    poses[-20:, 1, :3] = np.asarray([0.4, -0.2, 0.8]) + rng.normal(
        0.0, 0.002, size=(20, 3)
    )
    poses[::2, 1, 3:7] *= -1.0
    valid[-5, 1] = False
    estimate = estimate_terminal_endpose(
        timestamps_ns=timestamps,
        intention_ids=intentions,
        receiving_hand_ids=hand_ids,
        hand_poses=poses,
        hand_pose_valid=valid,
        hand_timestamps_ns=timestamps,
        handover_intent_id=2,
        receiving_hand_names=("left", "right"),
        config=TARGET_CONFIG,
    )
    assert estimate.eligible
    assert estimate.receiving_hand == "right"
    assert np.allclose(np.asarray(estimate.pose)[:3], [0.4, -0.2, 0.8], atol=0.01)
    assert np.allclose(
        np.abs(quaternion_average(poses[-20:, 1, 3:7])),
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )
    moving = poses.copy()
    moving[30:, 1, 0] = np.linspace(0.0, 2.0, rows - 30)
    rejected = estimate_terminal_endpose(
        timestamps_ns=timestamps,
        intention_ids=intentions,
        receiving_hand_ids=hand_ids,
        hand_poses=moving,
        hand_pose_valid=valid,
        hand_timestamps_ns=timestamps,
        handover_intent_id=2,
        receiving_hand_names=("left", "right"),
        config=TARGET_CONFIG,
    )
    assert not rejected.eligible
    assert "unstable_position" in rejected.reasons


def make_stable(path: Path, sequence_number: int) -> None:
    frame = pd.read_csv(path)
    hand = str(frame["receiving_hand"].iloc[0])
    rng = np.random.default_rng(100 + sequence_number)
    handover = frame["intent_label"] == "handover"
    base = np.asarray([0.3, -0.1, 0.7]) + sequence_number * 0.01
    frame.loc[handover, [f"{hand}_wrist_robot_{axis}_m" for axis in "xyz"]] = (
        base + rng.normal(0.0, 0.002, size=(int(handover.sum()), 3))
    )
    frame.loc[handover, [f"{hand}_wrist_robot_q{c}" for c in "xyzw"]] = (
        np.tile([0.0, 0.0, 0.0, 1.0], (int(handover.sum()), 1))
    )
    frame.to_csv(path, index=False)


def pipeline_check() -> None:
    with tempfile.TemporaryDirectory(prefix="aria_endpose_smoke_") as directory:
        master_dir = Path(directory)
        for index in range(6):
            participant = f"P{index + 1}"
            path = master_dir / f"{participant}_{index}_master.csv"
            synthetic_sequence(path, participant, index)
            make_stable(path, index)
        config = {
            "master_dir": str(master_dir),
            "feature_profile": "multimodal_robot_frame_v1",
            "window_size": 20,
            "stride": 10,
            "future_horizon_seconds": 1.0,
            "pose_target": TARGET_CONFIG,
            "pose_intent_ids": [2],
            "include_hand_references": True,
            "minimum_observed_fraction": 0.05,
            "max_timestamp_gap_seconds": 0.2,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "validation_participants": [],
            "test_participants": [],
        }
        bundle = prepare_data(config, seed=42)
        assert bundle.split_metadata["pose_target"]["mode"] == "terminal_endpose"
        assert bundle.provenance["builder_version"].endswith("terminal_endpose")
        for split_name in ("train", "validation", "test"):
            dataset = getattr(bundle, split_name)
            audit = dataset.pose_target_sequence_audit()
            assert audit["accepted_handover_sequences"] == len(dataset.records)
            assert dataset.residual_pose_count() > 0
            for record in dataset.records:
                valid = record.pose_valid
                assert len(np.unique(record.pose_targets[valid], axis=0)) == 1

        loader = DataLoader(bundle.train, batch_size=8, shuffle=False)
        model = HierarchicalResidualPoseTransformer(
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=20,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        metrics = run_epoch(
            model,
            loader,
            torch.device("cpu"),
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            {
                "assistance_loss_weight": 1.0,
                "assistance_type_loss_weight": 1.0,
                "receiving_hand_loss_weight": 1.0,
                "pose_loss_weight": 1.0,
                "orientation_loss_weight": 0.5,
                "gradient_clip_norm": 1.0,
            },
        )
        assert set(metrics["pose_by_time_to_sequence_end"]) == set(
            TIME_TO_END_GROUPS
        )
        assert sum(
            group["pose_oracle"]["samples"]
            for group in metrics["pose_by_time_to_sequence_end"].values()
        ) == metrics["pose_oracle"]["samples"]


def main() -> int:
    direct_aggregation_check()
    pipeline_check()
    print("Terminal end-pose smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
