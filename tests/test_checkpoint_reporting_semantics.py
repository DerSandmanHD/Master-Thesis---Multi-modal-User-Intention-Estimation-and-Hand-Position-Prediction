from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Training" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from build_final_experiment_summary import metric_row  # noqa: E402
from checkpoint_semantics import (  # noqa: E402
    CheckpointSemanticsError,
    assert_single_checkpoint_row,
    pose_selected_diagnostic,
    primary_result_row,
    select_pose_primary_checkpoint_row,
    select_primary_checkpoint_row,
)


def classification(value: float, samples: int = 10) -> dict:
    return {
        "macro_f1": value,
        "macro_f1_supported": value,
        "accuracy": value,
        "samples": samples,
        "per_class_f1": [value, value],
        "confusion_matrix": [[4, 1], [1, 4]],
    }


def evaluation(value: float, pose_cm: float) -> dict:
    intention = classification(value, 100)
    intention["per_class_f1"] = [value, value, value]
    intention["confusion_matrix"] = [[8, 1, 1], [1, 8, 1], [1, 1, 8]]
    intention["class_names"] = ["continue", "fetch", "handover"]
    return {
        "intention": intention,
        "assistance": classification(value - 0.01, 100),
        "assistance_type": classification(value - 0.02, 60),
        "receiving_hand": classification(value - 0.03, 20),
        "pose_oracle": {
            "position_mae_cm": pose_cm,
            "orientation_mean_deg": pose_cm + 10.0,
            "samples": 18,
        },
        "pose_end_to_end": {
            "position_mae_cm": pose_cm + 2.0,
            "orientation_mean_deg": pose_cm + 12.0,
            "samples": 17,
        },
        "last_observation_oracle": {
            "position_mae_cm": pose_cm + 1.0,
            "orientation_mean_deg": pose_cm + 11.0,
            "samples": 18,
        },
        "pose_coverage": {
            "pose_targets": 18,
            "oracle_reference_valid": 18,
            "predicted_reference_valid": 17,
        },
        "auxiliary_t_plus_1": {"pose_oracle": {}},
    }


def report() -> dict:
    return {
        "trainable_parameters": 123,
        "checkpoints": {
            "best_intention": {
                "path": "run/best_intention_model.pt",
                "epoch": 3,
                "selection_metric": "validation_intention_macro_f1",
                "selection_value": 0.9,
            },
            "best_pose": {
                "path": "run/best_pose_model.pt",
                "epoch": 5,
                "selection_metric": "validation_pose_oracle_position_mae_cm",
                "selection_value": 5.0,
            },
        },
        "validation_by_checkpoint": {
            "best_intention": evaluation(0.90, 7.0),
            "best_pose": evaluation(0.70, 5.0),
        },
        "test": {
            "best_intention": evaluation(0.80, 8.0),
            "best_pose": evaluation(0.60, 4.0),
        },
    }


class CheckpointReportingTests(unittest.TestCase):
    def test_primary_metrics_all_come_from_one_checkpoint(self) -> None:
        row = primary_result_row(
            report(), checkpoint="best_intention", split="test"
        )
        self.assertEqual(row["test_intention_macro_f1"], 0.8)
        self.assertEqual(row["test_pose_mae_cm"], 10.0)
        self.assertEqual(row["test_pose_orientation_error_deg"], 20.0)
        self.assertEqual(row["test_pose_samples"], 17)
        self.assertEqual(row["test_pose_coverage"], 17 / 18)
        self.assertEqual(
            row["diagnostic_pose_oracle_test_position_mean_cm"], 8.0
        )
        self.assertEqual(row["test_pose_target_coverage"], 0.9)
        self.assertEqual(
            [item["class_name"] for item in row["test_intention_per_class"]],
            ["continue", "fetch", "handover"],
        )
        self.assertEqual(row["test_intention_per_class"][0]["precision"], 0.8)
        self.assertEqual(row["test_intention_confusion_matrix"][0], [8, 1, 1])
        self.assertEqual(
            row["test_pose_coverage_denominator_receiving_hand_samples"], 20
        )
        self.assertEqual(row["metric_source_checkpoint"], "best_intention")
        assert_single_checkpoint_row(row)

    def test_pose_selected_values_are_diagnostic_only(self) -> None:
        values = pose_selected_diagnostic(report(), split="test")
        self.assertEqual(
            values["diagnostic_pose_selected_test_pose_mae_cm"], 6.0
        )
        self.assertFalse(any(key == "test_pose_mae_cm" for key in values))

    def test_mixed_checkpoint_row_is_rejected(self) -> None:
        row = primary_result_row(
            report(), checkpoint="best_intention", split="test"
        )
        row["pose_source_checkpoint"] = "best_pose"
        with self.assertRaises(CheckpointSemanticsError):
            assert_single_checkpoint_row(row)

    def test_non_validation_checkpoint_is_rejected(self) -> None:
        broken = report()
        broken["checkpoints"]["best_intention"]["selection_metric"] = (
            "test_intention_macro_f1"
        )
        with self.assertRaises(CheckpointSemanticsError):
            primary_result_row(
                broken, checkpoint="best_intention", split="test"
            )

    def test_intention_seed_selection_uses_validation_only(self) -> None:
        rows = []
        for seed, f1, pose, test_f1 in (
            (42, 0.900, 7.0, 0.1),
            (43, 0.898, 5.0, 0.99),
        ):
            row = primary_result_row(
                report(), checkpoint="best_intention", split="test"
            )
            row.update(
                {
                    "seed": seed,
                    "validation_intention_macro_f1": f1,
                    "validation_pose_mae_cm": pose,
                    "validation_receiving_hand_macro_f1": 0.8,
                    "test_intention_macro_f1": test_f1,
                }
            )
            rows.append(row)
        self.assertEqual(select_primary_checkpoint_row(rows)["seed"], 43)

    def test_pose_seed_selection_uses_validation_pose(self) -> None:
        rows = []
        for seed, pose, test_pose in ((42, 5.0, 99.0), (43, 6.0, 1.0)):
            row = primary_result_row(
                report(), checkpoint="best_pose", split="test"
            )
            row.update(
                {
                    "seed": seed,
                    "validation_pose_mae_cm": pose,
                    "validation_pose_orientation_error_deg": 20.0,
                    "validation_intention_macro_f1": 0.7,
                    "test_pose_mae_cm": test_pose,
                }
            )
            rows.append(row)
        self.assertEqual(select_pose_primary_checkpoint_row(rows)["seed"], 42)

    def test_seed_selection_allows_backbones_without_hand_head(self) -> None:
        rows = []
        for seed, pose in ((42, 7.0), (43, 5.0)):
            row = primary_result_row(
                report(), checkpoint="best_intention", split="test"
            )
            row.update(
                {
                    "seed": seed,
                    "validation_intention_macro_f1": 0.9,
                    "validation_pose_mae_cm": pose,
                    "validation_receiving_hand_macro_f1": None,
                }
            )
            rows.append(row)
        self.assertEqual(select_primary_checkpoint_row(rows)["seed"], 43)

    def test_final_table_requires_checkpoint_coherent_source(self) -> None:
        source = primary_result_row(
            report(), checkpoint="best_intention", split="test"
        )
        source.update({"seed": 42, "trainable_parameters": 123})
        final = metric_row("model", "main", source)
        self.assertEqual(final["test_pose_mae_cm"], 10.0)
        broken = copy.deepcopy(source)
        broken["metric_source_checkpoint"] = "best_pose"
        with self.assertRaises(CheckpointSemanticsError):
            metric_row("model", "main", broken)


if __name__ == "__main__":
    unittest.main()
