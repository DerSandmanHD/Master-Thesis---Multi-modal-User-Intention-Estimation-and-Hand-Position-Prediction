#!/usr/bin/env python3
"""One-epoch validation-freeze/final-test smoke test for standard baselines."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from artifact_freeze import validate_artifact_freeze
from pose_baselines_smoke_test import make_consistent_t_plus_one_sequence
from train import train


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aria_standard_train_smoke_") as value:
        root = Path(value)
        masters = root / "masters"
        masters.mkdir()
        for index in range(6):
            participant = f"P{index + 1}"
            make_consistent_t_plus_one_sequence(
                masters / f"{participant}_{index}_master.csv",
                participant,
                index,
            )
        config = {
            "run_name": "mlp_smoke",
            "model_type": "hierarchical_window_mlp",
            "data": {
                "master_dir": str(masters),
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
            },
            "model": {"hidden_dims": [16], "dropout": 0.0},
            "training": {
                "seed": 42,
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
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
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_dir = train(
            SimpleNamespace(
                config=config_path,
                run_dir=root / "run",
                dataset_tag="development",
                experiment_tag="standard_smoke",
                epochs=None,
                seed=None,
                device="cpu",
                limit_sequences=None,
                skip_test_evaluation=True,
            )
        )
        manifest = validate_artifact_freeze(run_dir / "artifact_manifest.json")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["test_evaluation_skipped"] is True
        assert "test" not in metrics
        checkpoint_hash = manifest["output_artifacts"]["checkpoints"][
            "best_intention"
        ]["sha256"]
        selection = root / "validation_selection.json"
        selection.write_text(
            json.dumps(
                {
                    "complete": True,
                    "selection_split": "validation",
                    "test_metrics_read": False,
                    "matrix_id": "standard_smoke",
                    "final_test_runs": [
                        {
                            "experiment_id": "baseline_mlp",
                            "seed": 42,
                            "run_dir": str(run_dir),
                            "checkpoint_name": "best_intention",
                            "checkpoint_sha256": checkpoint_hash,
                            "artifact_manifest_fingerprint": manifest[
                                "manifest_fingerprint"
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output = root / "final_test.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate_frozen_run.py")),
                "--run-dir",
                str(run_dir),
                "--master-dir",
                str(masters),
                "--selection-file",
                str(selection),
                "--experiment-id",
                "baseline_mlp",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["model_type"] == "hierarchical_window_mlp"
        assert report["checkpoint"]["selection_split"] == "validation"
        assert report["test_metrics"]["receiving_hand"]["samples"] > 0
        assert report["test_metrics"]["pose_coverage"]["pose_targets"] == (
            report["test_metrics"]["pose"]["samples"]
        )
        assert report["test_metrics"]["pose_coverage"]["coverage"] == 1.0
        assert report["matrix_authorization"]["experiment_id"] == "baseline_mlp"
        assert report["matrix_authorization"]["authorized_checkpoint_sha256"] == (
            report["checkpoint"]["sha256"]
        )
        assert report["test_metrics"]["intention"]["samples"] > 0
    print("Standard validation-freeze/final-test smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
