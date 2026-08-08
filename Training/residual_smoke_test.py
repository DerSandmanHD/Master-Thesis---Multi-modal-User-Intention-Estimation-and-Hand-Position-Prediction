#!/usr/bin/env python3
"""Exercise residual references, model outputs, losses and evaluation metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data
from model import HierarchicalResidualPoseTransformer, quaternion_multiply
from smoke_test import synthetic_sequence
from train_residual import residual_multitask_loss, run_epoch, train


def main() -> int:
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    quarter_turn_z = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])
    assert torch.allclose(
        quaternion_multiply(identity, quarter_turn_z), quarter_turn_z, atol=1e-6
    )

    with tempfile.TemporaryDirectory(prefix="aria_residual_smoke_") as directory:
        master_dir = Path(directory)
        for index in range(6):
            participant = f"P{index + 1}"
            synthetic_sequence(
                master_dir / f"{participant}_{index}_master.csv", participant, index
            )
        data_config = {
            "master_dir": str(master_dir),
            "feature_profile": "multimodal_robot_frame_v1",
            "window_size": 20,
            "stride": 10,
            "future_horizon_seconds": 1.0,
            "pose_intent_ids": [2],
            "include_hand_references": True,
            "minimum_observed_fraction": 0.05,
            "max_timestamp_gap_seconds": 0.2,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "validation_participants": [],
            "test_participants": [],
        }
        bundle = prepare_data(data_config, seed=42)
        assert sum(bundle.train.receiving_hand_counts()) > 0
        assert bundle.train.residual_pose_count() > 0
        batch = next(iter(DataLoader(bundle.train, batch_size=8, shuffle=False)))
        assert batch["hand_reference_pose"].shape == (8, 2, 7)
        assert batch["hand_reference_valid"].shape == (8, 2)

        model = HierarchicalResidualPoseTransformer(
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=20,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        outputs = model(batch["features"], batch["hand_reference_pose"])
        assert outputs["receiving_hand_logits"].shape == (8, 2)
        assert outputs["pose_candidates"].shape == (8, 2, 7)
        assert torch.allclose(outputs["position_delta"], torch.zeros((8, 3)))
        assert torch.allclose(
            outputs["quaternion_delta"],
            torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(8, -1),
        )
        assert torch.allclose(
            outputs["pose_candidates"], batch["hand_reference_pose"], atol=1e-6
        )

        assistance_criterion = nn.CrossEntropyLoss()
        assistance_type_criterion = nn.CrossEntropyLoss()
        hand_criterion = nn.CrossEntropyLoss()
        training_config = {
            "assistance_loss_weight": 1.0,
            "assistance_type_loss_weight": 1.0,
            "receiving_hand_loss_weight": 1.0,
            "pose_loss_weight": 1.0,
            "orientation_loss_weight": 0.25,
            "gradient_clip_norm": 1.0,
        }
        loss, components = residual_multitask_loss(
            outputs,
            batch,
            assistance_criterion,
            assistance_type_criterion,
            hand_criterion,
            training_config,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(np.isfinite(value) for value in components.values())

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = run_epoch(
            model,
            DataLoader(bundle.train, batch_size=8, shuffle=False),
            torch.device("cpu"),
            assistance_criterion,
            assistance_type_criterion,
            hand_criterion,
            training_config,
            optimizer,
        )
        assert metrics["intention"]["samples"] == len(bundle.train)
        assert metrics["receiving_hand"]["samples"] > 0
        assert metrics["pose_oracle"]["samples"] > 0
        assert metrics["pose_end_to_end"]["samples"] > 0
        assert metrics["last_observation_oracle"]["samples"] > 0
        assert set(metrics["pose_by_handover_progress"]) == {
            "0-25%",
            "25-50%",
            "50-75%",
            "75-100%",
        }
        assert set(metrics["pose_by_receiving_hand"]) == {"left", "right"}

        config = {
            "run_name": "residual_smoke",
            "data": data_config,
            "model": {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            "training": {
                "seed": 42,
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "assistance_loss_weight": 1.0,
                "assistance_type_loss_weight": 1.0,
                "receiving_hand_loss_weight": 1.0,
                "pose_loss_weight": 1.0,
                "orientation_loss_weight": 0.25,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 2,
                "num_workers": 0,
            },
        }
        config_path = Path(directory) / "residual_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_dir = Path(directory) / "full_run"
        completed_run = train(
            SimpleNamespace(
                config=config_path,
                run_dir=run_dir,
                epochs=None,
                device="cpu",
                limit_sequences=None,
                skip_test_evaluation=False,
            )
        )
        full_metrics = json.loads(
            (completed_run / "metrics.json").read_text(encoding="utf-8")
        )
        data_metadata = json.loads(
            (completed_run / "data_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert (completed_run / "best_intention_model.pt").exists()
        assert (completed_run / "best_pose_model.pt").exists()
        assert (completed_run / "dataset_provenance.json").exists()
        checkpoint = torch.load(
            completed_run / "best_intention_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["dataset_provenance"][
            "dataset_content_fingerprint"
        ] == data_metadata["provenance"][
            "dataset_content_fingerprint"
        ]
        assert set(full_metrics["test"]) == {"best_intention", "best_pose"}
        assert set(full_metrics["validation_by_checkpoint"]) == {
            "best_intention",
            "best_pose",
        }
        assert full_metrics["test_evaluation_skipped"] is False
        assert full_metrics["runtime"]["wall_seconds"] > 0
        assert len(full_metrics["history"]) == 2
        assert full_metrics["legacy_pose_metric_alias"][
            "position_mae_cm"
        ]

        validation_only_run = train(
            SimpleNamespace(
                config=config_path,
                run_dir=Path(directory) / "validation_only_run",
                epochs=1,
                device="cpu",
                limit_sequences=None,
                skip_test_evaluation=True,
            )
        )
        validation_only_metrics = json.loads(
            (validation_only_run / "metrics.json").read_text(encoding="utf-8")
        )
        assert validation_only_metrics["test_evaluation_skipped"] is True
        assert "test" not in validation_only_metrics
        assert set(validation_only_metrics["validation_by_checkpoint"]) == {
            "best_intention",
            "best_pose",
        }
        print("Residual v2 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
