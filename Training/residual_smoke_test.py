#!/usr/bin/env python3
"""Exercise residual references, model outputs, losses and evaluation metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data
from model import HierarchicalResidualPoseTransformer, quaternion_multiply
from pose_baselines_smoke_test import make_consistent_t_plus_one_sequence
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
            make_consistent_t_plus_one_sequence(
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
            "max_hand_reference_age_seconds": 0.25,
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
        fair = metrics["pose_fair_common"]
        assert fair["shared_samples"] > 0
        assert fair["methods"]["learned_end_to_end"]["samples"] == fair[
            "methods"
        ]["persistence"]["samples"]
        assert fair["methods"]["learned_oracle_hand"]["samples"] == fair[
            "shared_samples"
        ]
        assert set(metrics["pose_by_handover_progress"]) == {
            "0-25%",
            "25-50%",
            "50-75%",
            "75-100%",
        }
        assert set(metrics["pose_by_receiving_hand"]) == {"left", "right"}

        architecture_variants = (
            ("temporal_channel_simple", "hierarchical"),
            ("temporal_only", "hierarchical"),
            ("modality_gated", "hierarchical"),
            ("temporal_channel_gated", "flat"),
        )
        for fusion_mode, head_mode in architecture_variants:
            variant_kwargs = {
                "fusion_mode": fusion_mode,
                "intention_head_mode": head_mode,
            }
            if fusion_mode == "modality_gated":
                variant_kwargs["modality_schema"] = bundle.split_metadata[
                    "modality_schema"
                ]
            variant = HierarchicalResidualPoseTransformer(
                input_dim=len(bundle.normalizer.output_feature_names),
                window_size=20,
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                **variant_kwargs,
            )
            variant_training = dict(training_config)
            if head_mode == "flat":
                variant_training.update(
                    {
                        "intention_loss_weight": 1.0,
                        "resolved_flat_intention_class_weights": [
                            1.0,
                            1.0,
                            1.0,
                        ],
                    }
                )
            variant_metrics = run_epoch(
                variant,
                DataLoader(bundle.train, batch_size=8, shuffle=False),
                torch.device("cpu"),
                assistance_criterion,
                assistance_type_criterion,
                hand_criterion,
                variant_training,
            )
            assert variant_metrics["intention"]["samples"] == len(bundle.train)
            assert variant_metrics["fusion"]["mode"] == fusion_mode
            if fusion_mode == "modality_gated":
                assert set(
                    variant_metrics["fusion"][
                        "modality_mean_weights_when_available"
                    ]
                ) == set(bundle.split_metadata["modality_schema"]["active_modalities"])

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
        validation_manifest = json.loads(
            (validation_only_run / "artifact_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        validation_selection = Path(directory) / "validation_selection.json"
        validation_selection.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "complete": True,
                    "selection_split": "validation",
                    "test_metrics_read": False,
                    "matrix_id": "residual_smoke",
                    "final_test_runs": [
                        {
                            "experiment_id": "residual_smoke",
                            "seed": int(config["training"]["seed"]),
                            "run_dir": str(validation_only_run),
                            "checkpoint_name": "best_intention",
                            "checkpoint_sha256": validation_manifest[
                                "output_artifacts"
                            ]["checkpoints"]["best_intention"]["sha256"],
                            "checkpoint_epoch": validation_only_metrics[
                                "checkpoints"
                            ]["best_intention"]["epoch"],
                            "checkpoint_selection_metric": validation_only_metrics[
                                "checkpoints"
                            ]["best_intention"]["selection_metric"],
                            "checkpoint_selection_value": validation_only_metrics[
                                "checkpoints"
                            ]["best_intention"]["selection_value"],
                            "artifact_manifest_fingerprint": validation_manifest[
                                "manifest_fingerprint"
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        frozen_test_output = Path(directory) / "frozen_final_test.json"
        frozen_evaluation = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate_frozen_run.py")),
                "--run-dir",
                str(validation_only_run),
                "--master-dir",
                str(master_dir),
                "--selection-file",
                str(validation_selection),
                "--experiment-id",
                "residual_smoke",
                "--output",
                str(frozen_test_output),
                "--device",
                "cpu",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert frozen_evaluation.returncode == 0, (
            frozen_evaluation.stdout + frozen_evaluation.stderr
        )
        frozen_test = json.loads(frozen_test_output.read_text(encoding="utf-8"))
        assert frozen_test["split"] == "test"
        assert frozen_test["checkpoint"]["name"] == "best_intention"
        assert frozen_test["test_used_for_model_or_checkpoint_selection"] is False
        assert frozen_test["test_metrics"]["intention"]["samples"] > 0

        prediction_csv = Path(directory) / "t1_predictions.csv"
        prediction_report = Path(directory) / "t1_predictions.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("export_residual_predictions.py")),
                "--run-dir",
                str(validation_only_run),
                "--master-dir",
                str(master_dir),
                "--split",
                "test",
                "--final-test-report",
                str(frozen_test_output),
                "--output-csv",
                str(prediction_csv),
                "--report-out",
                str(prediction_report),
                "--device",
                "cpu",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        exported = json.loads(prediction_report.read_text(encoding="utf-8"))
        assert exported["schema_version"] == 3
        assert exported["result_role"] == "primary_validation_selected_checkpoint"
        assert exported["full_split_export"] is True
        assert exported["sequence_filter"] == []
        assert exported["exported_endpoint_count"] == exported[
            "frozen_split_endpoint_count"
        ]
        assert exported["exported_endpoint_fingerprint"] == exported[
            "frozen_split_endpoint_fingerprint"
        ]
        assert exported["artifact_freeze"]["manifest_fingerprint"]
        assert exported["final_test_authorization"]["report_fingerprint"] == (
            frozen_test["report_fingerprint"]
        )
        comparison = exported["pose_comparison"]
        assert comparison["fair_common_samples"] > 0
        assert set(comparison["methods"]) == {
            "persistence",
            "constant_velocity",
            "learned_model_oracle_hand",
        }
        common_counts = {
            values["fair_common_metrics"]["samples"]
            for values in comparison["methods"].values()
        }
        assert common_counts == {comparison["fair_common_samples"]}
        exported_rows = pd.read_csv(prediction_csv)
        assert exported_rows["sequence_receiving_hand"].isin(
            ["left", "right"]
        ).all()
        assert exported_rows["target_object_id"].notna().all()

        modality_config = json.loads(json.dumps(config))
        modality_config["run_name"] = "residual_modality_smoke"
        modality_config["model"].update(
            {
                "fusion_mode": "modality_gated",
                "intention_head_mode": "hierarchical",
            }
        )
        modality_config_path = Path(directory) / "modality_config.json"
        modality_config_path.write_text(
            json.dumps(modality_config), encoding="utf-8"
        )
        modality_run = train(
            SimpleNamespace(
                config=modality_config_path,
                run_dir=Path(directory) / "modality_run",
                epochs=1,
                device="cpu",
                limit_sequences=None,
                skip_test_evaluation=True,
            )
        )
        modality_csv = Path(directory) / "modality_predictions.csv"
        modality_report_path = Path(directory) / "modality_predictions.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("export_residual_predictions.py")),
                "--run-dir",
                str(modality_run),
                "--master-dir",
                str(master_dir),
                "--split",
                "validation",
                "--output-csv",
                str(modality_csv),
                "--report-out",
                str(modality_report_path),
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        modality_report = json.loads(
            modality_report_path.read_text(encoding="utf-8")
        )
        assert modality_report["architecture"]["fusion_mode"] == "modality_gated"
        modality_columns = modality_csv.read_text(encoding="utf-8").splitlines()[0]
        for modality_name in bundle.split_metadata["modality_schema"][
            "active_modalities"
        ]:
            assert f"modality_{modality_name}_weight" in modality_columns
            assert f"modality_{modality_name}_available" in modality_columns
        print("Residual v2 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
