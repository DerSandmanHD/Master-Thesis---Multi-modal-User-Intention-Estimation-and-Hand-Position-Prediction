#!/usr/bin/env python3
"""Smoke-test checkpoint prediction export on synthetic master datasets."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import torch

from data import prepare_data
from export_checkpoint_predictions import export_predictions
from metrics import classification_metrics
from model import HierarchicalGatedMultimodalTransformer
from smoke_test import synthetic_sequence


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aria_prediction_export_") as directory:
        root = Path(directory)
        master_dir = root / "master_datasets"
        run_dir = root / "run"
        master_dir.mkdir()
        run_dir.mkdir()
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
            "model": {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            "training": {"seed": 42, "batch_size": 4},
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        bundle = prepare_data(config["data"], seed=42)
        model = HierarchicalGatedMultimodalTransformer(
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=20,
            **config["model"],
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": config["model"],
                "model_type": "hierarchical_gated_multimodal_transformer",
                "input_dim": len(bundle.normalizer.output_feature_names),
                "window_size": 20,
                "epoch": 1,
                "validation_intention_macro_f1": 0.0,
            },
            run_dir / "best_model.pt",
        )
        output_csv = run_dir / "predictions.csv"
        report_path = run_dir / "analysis.json"
        summary = export_predictions(
            run_dir,
            output_csv=output_csv,
            report_path=report_path,
            device=torch.device("cpu"),
            batch_size=4,
            verify_metrics=False,
        )

        assert output_csv.exists() and report_path.exists()
        assert summary["overall"]["windows"] == len(bundle.test)
        assert summary["overall"]["valid_pose_targets"] > 0
        assert summary["reference_metrics"]["status"] == "not_checked"
        assert set(summary["participants"]) == set(
            bundle.split_metadata["participants"]["test"]
        )

        prediction_rows = list(csv.DictReader(output_csv.open(newline="")))
        intention = classification_metrics(
            torch.tensor([int(row["predicted_intention_id"]) for row in prediction_rows]),
            torch.tensor([int(row["target_intention_id"]) for row in prediction_rows]),
            3,
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "test": {
                        "intention": intention,
                        "pose": summary["overall"]["transformer_pose"],
                    }
                }
            ),
            encoding="utf-8",
        )
        verified = export_predictions(
            run_dir,
            output_csv=run_dir / "verified_predictions.csv",
            report_path=run_dir / "verified_analysis.json",
            device=torch.device("cpu"),
            batch_size=4,
            verify_metrics=True,
        )
        assert verified["reference_metrics"]["status"] == "matched"
        print("Checkpoint prediction export smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
