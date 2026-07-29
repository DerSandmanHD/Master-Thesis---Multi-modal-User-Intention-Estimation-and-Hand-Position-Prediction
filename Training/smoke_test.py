#!/usr/bin/env python3
"""Exercise data loading, all absolute-pose backbones, losses and metrics."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data, save_data_metadata, sha256_file
from train import build_model, multitask_loss, run_epoch


MODEL_CONFIGS = (
    {
        "model_type": "hierarchical_gated_multimodal_transformer",
        "model": {
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
    },
    {
        "model_type": "hierarchical_window_mlp",
        "model": {"hidden_dims": [32, 16], "dropout": 0.0},
    },
    {
        "model_type": "hierarchical_gru",
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
    },
)


def synthetic_sequence(path: Path, participant: str, sequence_number: int) -> None:
    rng = np.random.default_rng(sequence_number)
    rows = 90
    timestamps = np.arange(rows, dtype=np.int64) * 33_333_333
    timestamps[5:] += 1_000_000_000
    intent_labels = ["continue"] * 30
    intent_labels += ["fetch"] * 15
    intent_labels += ["transition"] * 15
    intent_labels += ["handover"] * 30
    receiving_hand = "left" if sequence_number % 2 else "right"
    frame = pd.DataFrame(
        {
            "sequence_id": [f"{participant}_{sequence_number}"] * rows,
            "participant": [participant] * rows,
            "timestamp_ns": timestamps,
            "intent_label": intent_labels,
            "receiving_hand": [receiving_hand] * rows,
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
            "robot_frame_valid": np.ones(rows),
            "robot_anchor_interpolated": np.zeros(rows),
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
        frame[f"future_1s_receiving_wrist_robot_q{component}"] = target_quaternion[
            :, index
        ]
    frame.to_csv(path, index=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aria_training_smoke_") as directory:
        data_root = Path(directory)
        master_dir = data_root / "master_datasets"
        master_dir.mkdir()
        participants = ("David", "david", "Test", "P4", "P5", "P6", "Warn")
        sequence_ids = []
        for index, participant in enumerate(participants):
            sequence_id = f"{participant}_{index}"
            synthetic_sequence(
                master_dir / f"{sequence_id}_master.csv", participant, index
            )
            sequence_ids.append(sequence_id)
        manifest_path = data_root / "dataset_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sequence_id",
                    "include_in_training",
                    "status",
                    "next_action",
                    "master_csv_exists",
                ),
            )
            writer.writeheader()
            for sequence_id, participant in zip(sequence_ids, participants):
                writer.writerow(
                    {
                        "sequence_id": sequence_id,
                        "include_in_training": True,
                        "status": (
                            "valid_with_warnings" if participant == "Warn" else "valid"
                        ),
                        "next_action": (
                            "manual_review"
                            if participant == "Warn"
                            else "ready_for_master_merge"
                        ),
                        "master_csv_exists": True,
                    }
                )
        data_config = {
            "master_dir": str(master_dir),
            "manifest_filter": {
                "path": manifest_path.name,
                "allowed_statuses": ["valid"],
                "allowed_next_actions": ["ready_for_master_merge"],
                "strict": True,
            },
            "feature_profile": "multimodal_robot_frame_v1",
            "window_size": 20,
            "stride": 10,
            "future_horizon_seconds": 1.0,
            "pose_intent_ids": [2],
            "minimum_observed_fraction": 0.05,
            "max_timestamp_gap_seconds": 0.2,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "validation_participants": ["p4"],
            "test_participants": ["P6"],
        }
        bundle = prepare_data(data_config, seed=42)
        provenance = bundle.provenance
        assert len(provenance["master_files"]) == 6
        assert len(provenance["dataset_content_fingerprint"]) == 64
        assert bundle.split_metadata["dataset_filter"][
            "dataset_content_fingerprint"
        ] == provenance["dataset_content_fingerprint"]
        assert len(provenance["schema"]["fingerprint"]) == 64
        assert all(
            len(item["sha256"]) == 64
            for item in provenance["master_files"]
        )
        first_master = provenance["master_files"][0]
        assert first_master["sha256"] == sha256_file(
            master_dir / first_master["file_name"]
        )
        assert provenance["manifest"]["snapshot_file"] == (
            "dataset_manifest_snapshot.csv"
        )
        metadata_dir = data_root / "metadata_artifacts"
        save_data_metadata(
            bundle,
            metadata_dir / "data_metadata.json",
        )
        assert (metadata_dir / "dataset_provenance.json").is_file()
        assert (
            metadata_dir / "dataset_manifest_snapshot.csv"
        ).read_text(encoding="utf-8") == bundle.manifest_snapshot
        assert sha256_file(
            metadata_dir / "dataset_manifest_snapshot.csv"
        ) == provenance["manifest"]["sha256"]
        assert bundle.split_metadata["dataset_filter"]["selected_sequences"] == 6
        assert bundle.split_metadata["dataset_filter"]["excluded_master_files"] == 1
        assert bundle.split_metadata["participants"]["validation"] == ["P4"]
        assert bundle.split_metadata["participants"]["test"] == ["P6"]
        assert "David" in bundle.split_metadata["participants"]["train"]
        assert "david" not in bundle.split_metadata["participants"]["train"]
        assert "Test" in bundle.split_metadata["participants"]["train"]
        assert {
            sequence_id
            for sequence_id in bundle.split_metadata["sequences"]["train"]
            if sequence_id.casefold().startswith("david_")
        } == {"David_0", "david_1"}
        batch = next(iter(DataLoader(bundle.train, batch_size=4, shuffle=False)))
        assistance_criterion = nn.CrossEntropyLoss()
        assistance_type_criterion = nn.CrossEntropyLoss()
        assert (
            sum(
                dataset.discarded_gap_windows
                for dataset in (bundle.train, bundle.validation, bundle.test)
            )
            > 0
        )
        assert (
            sum(
                dataset.discarded_unlabeled_windows
                for dataset in (bundle.train, bundle.validation, bundle.test)
            )
            > 0
        )
        training_config = {
            "assistance_loss_weight": 1.0,
            "assistance_type_loss_weight": 1.0,
            "pose_loss_weight": 1.0,
            "orientation_loss_weight": 0.25,
            "gradient_clip_norm": 1.0,
        }
        for model_config in MODEL_CONFIGS:
            model, model_type = build_model(
                model_config,
                input_dim=len(bundle.normalizer.output_feature_names),
                window_size=20,
            )
            outputs = model(batch["features"])
            loss, components = multitask_loss(
                outputs,
                batch,
                assistance_criterion,
                assistance_type_criterion,
                training_config,
            )
            loss.backward()
            assert outputs["assistance_logits"].shape == (4, 2)
            assert outputs["assistance_type_logits"].shape == (4, 2)
            assert outputs["pose"].shape == (4, 7)
            assert torch.isfinite(loss)
            assert all(np.isfinite(value) for value in components.values())

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
            assert (
                epoch_metrics["pose"][
                    "position_mean_euclidean_error_cm"
                ]
                == epoch_metrics["pose"]["position_mae_cm"]
            )
            assert (
                epoch_metrics["pose"][
                    "position_root_mean_square_euclidean_error_cm"
                ]
                == epoch_metrics["pose"]["position_rmse_cm"]
            )
            assert "mean Euclidean norm" in epoch_metrics["pose"][
                "position_error_definition"
            ]
            if model_type == "hierarchical_gated_multimodal_transformer":
                assert epoch_metrics["mean_gate"] is not None
            else:
                assert epoch_metrics["mean_gate"] is None
            print(f"{model_type} smoke test passed")
        print(
            f"Input features including masks: {len(bundle.normalizer.output_feature_names)}"
        )
        print(
            f"Windows: {len(bundle.train)}/{len(bundle.validation)}/{len(bundle.test)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
