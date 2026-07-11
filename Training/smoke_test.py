#!/usr/bin/env python3
"""Exercise CSV loading, participant split, model forward pass and all losses."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data
from model import HierarchicalGatedMultimodalTransformer
from train import multitask_loss, run_epoch


def synthetic_sequence(path: Path, participant: str, sequence_number: int) -> None:
    rng = np.random.default_rng(sequence_number)
    rows = 90
    timestamps = np.arange(rows, dtype=np.int64) * 33_333_333
    timestamps[5:] += 1_000_000_000
    intent_labels = ["continue"] * 30
    intent_labels += ["fetch"] * 15
    intent_labels += ["transition"] * 15
    intent_labels += ["handover"] * 30
    frame = pd.DataFrame(
        {
            "sequence_id": [f"{participant}_{sequence_number}"] * rows,
            "participant": [participant] * rows,
            "timestamp_ns": timestamps,
            "intent_label": intent_labels,
            "gaze_valid": np.ones(rows),
            "gaze_yaw_rad": rng.normal(size=rows),
            "gaze_pitch_rad": rng.normal(size=rows),
            "gaze_depth_m": rng.uniform(0.2, 2.0, size=rows),
            "hand_left_tracking_confidence": np.ones(rows),
            "hand_right_tracking_confidence": np.ones(rows),
            "hand_left_valid": np.ones(rows),
            "hand_right_valid": np.ones(rows),
            "slam_device_linear_velocity_x_device": rng.normal(size=rows),
            "slam_device_linear_velocity_y_device": rng.normal(size=rows),
            "slam_device_linear_velocity_z_device": rng.normal(size=rows),
            "slam_angular_velocity_x_device": rng.normal(size=rows),
            "slam_angular_velocity_y_device": rng.normal(size=rows),
            "slam_angular_velocity_z_device": rng.normal(size=rows),
            "slam_quality_score": np.ones(rows),
            "apriltag_0_valid": np.ones(rows),
            "future_1s_receiving_wrist_valid": np.ones(rows),
        }
    )
    for side in ("left", "right"):
        for axis in "xyz":
            frame[f"{side}_wrist_robot_{axis}_m"] = rng.normal(size=rows)
        quaternion = rng.normal(size=(rows, 4))
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        for index, component in enumerate("xyzw"):
            frame[f"{side}_wrist_robot_q{component}"] = quaternion[:, index]
    for marker_id in range(6, 15):
        for axis in "xyz":
            frame[f"aruco_{marker_id}_robot_{axis}_m"] = rng.normal(size=rows)
        frame[f"aruco_{marker_id}_gaze_angle_rad"] = rng.uniform(0, 2, size=rows)
        frame[f"aruco_{marker_id}_gaze_distance_m"] = rng.uniform(0.2, 2, size=rows)
        frame[f"aruco_{marker_id}_valid"] = np.ones(rows)
    target_quaternion = rng.normal(size=(rows, 4))
    target_quaternion /= np.linalg.norm(target_quaternion, axis=1, keepdims=True)
    for axis in "xyz":
        frame[f"future_1s_receiving_wrist_robot_{axis}_m"] = rng.normal(size=rows)
    for index, component in enumerate("xyzw"):
        frame[f"future_1s_receiving_wrist_robot_q{component}"] = target_quaternion[:, index]
    frame.to_csv(path, index=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aria_training_smoke_") as directory:
        master_dir = Path(directory)
        for index in range(6):
            participant = f"P{index + 1}"
            synthetic_sequence(master_dir / f"{participant}_1_master.csv", participant, index)
        data_config = {
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
        }
        bundle = prepare_data(data_config, seed=42)
        model = HierarchicalGatedMultimodalTransformer(
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=20,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        batch = next(iter(DataLoader(bundle.train, batch_size=4, shuffle=False)))
        outputs = model(batch["features"])
        assistance_criterion = nn.CrossEntropyLoss()
        assistance_type_criterion = nn.CrossEntropyLoss()
        loss, components = multitask_loss(
            outputs,
            batch,
            assistance_criterion,
            assistance_type_criterion,
            {
                "assistance_loss_weight": 1.0,
                "assistance_type_loss_weight": 1.0,
                "pose_loss_weight": 1.0,
                "orientation_loss_weight": 0.25,
            },
        )
        loss.backward()
        assert outputs["assistance_logits"].shape == (4, 2)
        assert outputs["assistance_type_logits"].shape == (4, 2)
        assert outputs["pose"].shape == (4, 7)
        assert torch.isfinite(loss)
        assert all(np.isfinite(value) for value in components.values())
        assert sum(
            dataset.discarded_gap_windows
            for dataset in (bundle.train, bundle.validation, bundle.test)
        ) > 0
        assert sum(
            dataset.discarded_unlabeled_windows
            for dataset in (bundle.train, bundle.validation, bundle.test)
        ) > 0
        training_config = {
            "assistance_loss_weight": 1.0,
            "assistance_type_loss_weight": 1.0,
            "pose_loss_weight": 1.0,
            "orientation_loss_weight": 0.25,
            "gradient_clip_norm": 1.0,
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        epoch_metrics = run_epoch(
            model,
            DataLoader(bundle.train, batch_size=8, shuffle=False),
            torch.device("cpu"),
            assistance_criterion,
            assistance_type_criterion,
            training_config,
            optimizer,
        )
        assert epoch_metrics["intention"]["samples"] == len(bundle.train)
        assert epoch_metrics["assistance"]["samples"] == len(bundle.train)
        assert epoch_metrics["assistance_type"]["samples"] > 0
        assert epoch_metrics["pose"]["samples"] > 0
        print("Training smoke test passed")
        print(f"Input features including masks: {len(bundle.normalizer.output_feature_names)}")
        print(f"Windows: {len(bundle.train)}/{len(bundle.validation)}/{len(bundle.test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
