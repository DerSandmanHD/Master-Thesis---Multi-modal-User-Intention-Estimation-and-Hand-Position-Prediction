from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Training" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from grouped_metrics import (  # noqa: E402
    SEQUENCE_AGGREGATION_DEFINITION,
    build_grouped_evaluation,
    discover_pose_methods,
    prepare_prediction_frame,
    report_table_rows,
    system_success_metrics,
)
from evaluate_grouped_predictions import checkpoint_binding  # noqa: E402


def prediction_frame() -> pd.DataFrame:
    specifications = (
        # participant, sequence, predictions; all targets cycle 0/1/2.
        ("P1", "S1", [0, 1, 2, 0, 0], "left"),
        ("P1", "S2", [0, 2, 2], "right"),
        ("P2", "S3", [1, 1, 2], "left"),
        ("P3", "S4", [0, 1, 0], "right"),
    )
    pose_error = iter((1.0, 2.0, 3.0, 4.0))
    oracle_error = iter((0.8, 1.8, 2.8, 3.8))
    persistence_error = iter((2.0, np.nan, 4.0, 5.0))
    velocity_error = iter((1.5, 2.5, 2.5, 3.5))
    rows = []
    for participant, sequence, predictions, hand in specifications:
        if len(predictions) == 5:
            targets = [0, 1, 2, 0, 0]
        else:
            targets = [0, 1, 2]
        for index, (target, prediction) in enumerate(zip(targets, predictions)):
            is_pose = target == 2
            row = {
                "sample_key": f"test|{sequence}|{index}",
                "participant": participant,
                "sequence_id": sequence,
                "target_intention_id": target,
                "predicted_intention_id": prediction,
                # Current residual export only fills this on handover rows.
                "target_receiving_hand": hand if is_pose else "",
                "sequence_receiving_hand": hand,
                "predicted_receiving_hand": (
                    hand if sequence != "S4" else "left"
                ),
                "pose_valid": is_pose,
                "fair_common": False,
                "predicted_position_error_cm": np.nan,
                "predicted_orientation_error_deg": np.nan,
                "oracle_position_error_cm": np.nan,
                "oracle_orientation_error_deg": np.nan,
                "persistence_position_error_cm": np.nan,
                "persistence_orientation_error_deg": np.nan,
                "constant_velocity_position_error_cm": np.nan,
                "constant_velocity_orientation_error_deg": np.nan,
            }
            if is_pose:
                learned = next(pose_error)
                oracle = next(oracle_error)
                persistence = next(persistence_error)
                velocity = next(velocity_error)
                row.update(
                    {
                        "predicted_position_error_cm": learned,
                        "predicted_orientation_error_deg": learned * 10.0,
                        "oracle_position_error_cm": oracle,
                        "oracle_orientation_error_deg": oracle * 10.0,
                        "persistence_position_error_cm": persistence,
                        "persistence_orientation_error_deg": (
                            persistence * 10.0
                            if np.isfinite(persistence)
                            else np.nan
                        ),
                        "constant_velocity_position_error_cm": velocity,
                        "constant_velocity_orientation_error_deg": velocity * 10.0,
                        "fair_common": bool(np.isfinite(persistence)),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


class GroupedEvaluationTests(unittest.TestCase):
    def test_window_sequence_participant_and_hand_metrics(self) -> None:
        report = build_grouped_evaluation(
            prediction_frame(),
            bootstrap_iterations=30,
            bootstrap_seed=17,
        )
        window = report["window_level"]
        self.assertEqual(window["windows"], 14)
        self.assertAlmostEqual(
            window["classification"]["intention"]["accuracy"], 11 / 14
        )
        self.assertEqual(window["pose_target_denominator"], 4)
        self.assertEqual(window["pose"]["learned_end_to_end"]["position_samples"], 4)
        self.assertEqual(window["pose"]["learned_oracle_hand"]["position_samples"], 4)
        self.assertEqual(window["pose"]["persistence"]["position_samples"], 3)
        self.assertAlmostEqual(window["pose"]["persistence"]["coverage"], 0.75)
        self.assertEqual(window["pose_fair_common"]["shared_samples"], 3)
        self.assertEqual(
            set(window["pose_fair_common"]["methods"]),
            {"learned_oracle_hand", "persistence", "constant_velocity"},
        )
        self.assertNotIn("learned_end_to_end", window["pose_fair_common"]["methods"])
        paired = window["paired_pose_comparisons"][
            "learned_oracle_hand_minus_persistence"
        ]
        self.assertEqual(paired["shared_samples"], 3)
        self.assertAlmostEqual(paired["position_mean_difference_cm"], -1.2)

        sequence = report["sequence_level"]
        self.assertEqual(sequence["group_count"], 4)
        self.assertEqual(
            sequence["aggregation_definition"], SEQUENCE_AGGREGATION_DEFINITION
        )
        sequence_accuracy = sequence["equal_weighted_group_summary"][
            "classification"
        ]["intention"]["accuracy"]["unweighted_mean"]
        self.assertAlmostEqual(sequence_accuracy, 0.75)
        self.assertNotAlmostEqual(
            sequence_accuracy,
            window["classification"]["intention"]["accuracy"],
        )

        participant = report["participant_level"]
        self.assertEqual(participant["group_count"], 3)
        self.assertEqual(set(participant["groups"]), {"P1", "P2", "P3"})
        self.assertAlmostEqual(
            participant["groups"]["P1"]["metrics"]["classification"][
                "intention"
            ]["accuracy"],
            (1.0 + 2 / 3) / 2,
        )
        self.assertAlmostEqual(
            participant["point_estimate"]["classification"]["intention"][
                "accuracy"
            ],
            ((1.0 + 2 / 3) / 2 + 2 / 3 + 2 / 3) / 3,
        )
        self.assertEqual(
            report["per_receiving_hand"]["groups"]["left"]["windows"], 8
        )
        self.assertEqual(
            report["per_receiving_hand"]["groups"]["right"]["windows"], 6
        )
        self.assertIn(
            "participant_balanced",
            report["per_receiving_hand"]["groups"]["left"],
        )
        self.assertIsNotNone(
            report["per_receiving_hand"]["groups"]["left"][
                "participant_cluster_bootstrap"
            ]
        )
        self.assertIn("pose_fair_common", sequence["point_estimate"])
        self.assertIn("pose_fair_common", participant["point_estimate"])
        hand_metrics = window["classification"][
            "receiving_hand_given_ground_truth_handover"
        ]
        self.assertEqual(hand_metrics["samples"], 4)
        self.assertEqual(hand_metrics["denominator"], 4)
        self.assertAlmostEqual(hand_metrics["accuracy"], 0.75)
        system = window["system_success"]
        self.assertEqual(system["valid_t1_pose_target_windows"], 4)
        self.assertEqual(system["stages"]["handover_correct"]["successes"], 3)
        self.assertEqual(
            system["stages"]["handover_and_receiving_hand_correct"]["successes"],
            3,
        )
        self.assertEqual(system["stages"]["success_at_5_cm"]["successes"], 3)

    def test_system_success_is_a_strict_full_cascade(self) -> None:
        frame = prediction_frame()
        handover_indices = frame.index[frame["target_intention_id"].eq(2)].tolist()
        frame.loc[handover_indices[1], "predicted_receiving_hand"] = "left"
        frame.loc[handover_indices[2], "predicted_position_error_cm"] = 12.0
        normalized, _ = prepare_prediction_frame(frame)
        result = system_success_metrics(
            normalized, discover_pose_methods(normalized)
        )
        stages = result["stages"]
        self.assertEqual(result["valid_t1_pose_target_windows"], 4)
        self.assertEqual(stages["handover_correct"]["successes"], 3)
        self.assertEqual(
            stages["handover_and_receiving_hand_correct"]["successes"], 2
        )
        self.assertEqual(stages["success_at_5_cm"]["successes"], 1)
        self.assertEqual(stages["success_at_10_cm"]["successes"], 1)
        self.assertEqual(stages["success_at_15_cm"]["successes"], 2)
        self.assertEqual(stages["success_at_20_cm"]["successes"], 2)

    def test_participant_and_sequence_bootstraps_are_deterministic(self) -> None:
        first = build_grouped_evaluation(
            prediction_frame(),
            bootstrap_iterations=40,
            bootstrap_seed=123,
            include_sequence_bootstrap=True,
        )
        second = build_grouped_evaluation(
            prediction_frame(),
            bootstrap_iterations=40,
            bootstrap_seed=123,
            include_sequence_bootstrap=True,
        )
        self.assertEqual(first["cluster_bootstrap"], second["cluster_bootstrap"])
        participant = first["cluster_bootstrap"]["participant_cluster"]
        sequence = first["cluster_bootstrap"]["sequence_cluster"]
        self.assertEqual(participant["cluster_count"], 3)
        self.assertEqual(sequence["cluster_count"], 4)
        metric = participant["metrics"]["classification.intention.accuracy"]
        self.assertEqual(metric["valid_replicates"], 40)
        self.assertLessEqual(metric["lower"], metric["upper"])
        self.assertAlmostEqual(
            metric["estimate"],
            first["participant_level"]["point_estimate"]["classification"][
                "intention"
            ]["accuracy"],
        )
        paired_metric = participant["metrics"][
            "paired_pose.learned_oracle_hand_minus_persistence.position_mean_difference_cm"
        ]
        self.assertEqual(paired_metric["valid_replicates"], 40)
        serialized = json.dumps(first).casefold()
        self.assertIn("standard deviation across training seeds", serialized)
        self.assertNotIn("population uncertainty", serialized)

    def test_schema_is_head_neutral_and_discovers_pose_baselines(self) -> None:
        normalized, columns = prepare_prediction_frame(prediction_frame())
        self.assertTrue(columns["head_neutral"])
        self.assertEqual(normalized["_receiving_hand_context"].eq("").sum(), 0)
        report = build_grouped_evaluation(
            prediction_frame(), bootstrap_iterations=5
        )
        methods = {item["name"] for item in report["pose_methods"]}
        self.assertEqual(
            methods,
            {
                "learned_end_to_end",
                "learned_oracle_hand",
                "persistence",
                "constant_velocity",
            },
        )
        table = pd.DataFrame(report_table_rows(report))
        self.assertTrue(
            {"window", "sequence", "participant", "receiving_hand"}.issubset(
                set(table["level"])
            )
        )
        self.assertIn("pose_fair_common", set(table["domain"]))

    def test_fair_common_is_recomputed_and_rejects_cherry_picked_flags(self) -> None:
        frame = prediction_frame()
        eligible = frame.index[frame["fair_common"]].tolist()[0]
        frame.loc[eligible, "fair_common"] = False
        with self.assertRaisesRegex(ValueError, "recomputed intersection"):
            build_grouped_evaluation(frame, bootstrap_iterations=5)

        frame = prediction_frame()
        eligible = frame.index[frame["fair_common"]].tolist()[0]
        frame.loc[eligible, "constant_velocity_position_error_cm"] = np.inf
        with self.assertRaisesRegex(ValueError, "recomputed intersection"):
            build_grouped_evaluation(frame, bootstrap_iterations=5)

    def test_cluster_ci_requires_two_metric_contributing_participants(self) -> None:
        frame = prediction_frame()
        other = frame["participant"].ne("P1") & frame["pose_valid"]
        for column in (
            "oracle_position_error_cm",
            "persistence_position_error_cm",
            "constant_velocity_position_error_cm",
        ):
            frame.loc[other, column] = np.nan
        frame.loc[other, "fair_common"] = False
        report = build_grouped_evaluation(
            frame,
            bootstrap_iterations=30,
            bootstrap_seed=7,
        )
        metric = report["cluster_bootstrap"]["participant_cluster"]["metrics"][
            "pose.learned_oracle_hand.fair_common.position_mean_cm"
        ]
        self.assertEqual(metric["metric_contributing_clusters"], 1)
        self.assertEqual(
            metric["status"], "insufficient_metric_contributing_clusters"
        )
        self.assertIsNone(metric["lower"])
        self.assertIsNone(metric["upper"])

    def test_explicit_assistance_type_is_preferred_over_final_decision(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "participant": "P1",
                    "sequence_id": "S1",
                    "target_intention_id": 2,
                    "predicted_intention_id": 0,
                    "target_assistance_id": 1,
                    "predicted_assistance_id": 0,
                    "target_assistance_type_id": 1,
                    "predicted_assistance_type_id": 1,
                    "sequence_receiving_hand": "left",
                    "target_receiving_hand": "left",
                    "predicted_receiving_hand": "left",
                    "pose_valid": False,
                }
            ]
        )
        report = build_grouped_evaluation(frame, bootstrap_iterations=5)
        assistance_type = report["window_level"]["classification"][
            "assistance_type_given_ground_truth_assistance"
        ]
        self.assertEqual(assistance_type["samples"], 1)
        self.assertEqual(assistance_type["denominator"], 1)
        self.assertEqual(assistance_type["accuracy"], 1.0)
        self.assertEqual(
            report["input_columns"]["assistance_type_metric_source"],
            "explicit_exported_head_decisions",
        )
        bootstrap = report["cluster_bootstrap"]["participant_cluster"]
        self.assertEqual(bootstrap["status"], "insufficient_independent_clusters")
        self.assertIsNone(
            bootstrap["metrics"]["classification.intention.accuracy"]["lower"]
        )

    def test_inconsistent_participant_or_duplicate_sample_is_rejected(self) -> None:
        inconsistent = prediction_frame()
        inconsistent.loc[inconsistent.index[0], "participant"] = "P9"
        with self.assertRaises(ValueError):
            prepare_prediction_frame(inconsistent)
        duplicate = prediction_frame()
        duplicate.loc[duplicate.index[1], "sample_key"] = duplicate.loc[
            duplicate.index[0], "sample_key"
        ]
        with self.assertRaises(ValueError):
            prepare_prediction_frame(duplicate)
        conflicting_hand = prediction_frame()
        conflicting_hand.loc[
            conflicting_hand["sequence_id"].eq("S1"),
            "sequence_receiving_hand",
        ] = ["left", "right", "left", "left", "left"]
        with self.assertRaisesRegex(ValueError, "Conflicting receiving-hand"):
            prepare_prediction_frame(conflicting_hand)

    def test_cli_writes_json_and_long_form_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="grouped_eval_") as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            report_path = root / "report.json"
            table_path = root / "metrics.csv"
            source_report = root / "prediction_report.json"
            prediction_frame().to_csv(predictions, index=False)
            predictions_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
            source_report.write_text(
                json.dumps(
                    {
                        "result_role": "primary_validation_selected_checkpoint",
                        "checkpoint": "/cluster/run/best_intention_model.pt",
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_epoch": 7,
                        "predictions_csv_sha256": predictions_hash,
                        "checkpoint_selection_split": "validation",
                        "checkpoint_selection_metric": (
                            "validation_intention_macro_f1"
                        ),
                        "checkpoint_selection_value": 0.7,
                        "dataset_content_fingerprint": "dataset123",
                        "split": "test",
                        "rows": 14,
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATION / "evaluate_grouped_predictions.py"),
                    "--predictions",
                    str(predictions),
                    "--report-out",
                    str(report_path),
                    "--table-out",
                    str(table_path),
                    "--prediction-report",
                    str(source_report),
                    "--bootstrap-iterations",
                    "10",
                    "--bootstrap-seed",
                    "9",
                    "--sequence-bootstrap",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            table = pd.read_csv(table_path)
            self.assertEqual(report["prediction_rows"], 14)
            self.assertIn("participant_cluster", report["cluster_bootstrap"])
            self.assertIn("sequence_cluster", report["cluster_bootstrap"])
            self.assertEqual(
                report["checkpoint_binding"]["status"],
                "bound_single_checkpoint",
            )
            self.assertEqual(
                report["checkpoint_binding"]["checkpoint_sha256"], "a" * 64
            )
            self.assertGreater(len(table), 0)

            changed = pd.read_csv(predictions)
            changed.loc[0, "predicted_intention"] = 2
            changed.to_csv(predictions, index=False)
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATION / "evaluate_grouped_predictions.py"),
                    "--predictions",
                    str(predictions),
                    "--prediction-report",
                    str(source_report),
                    "--report-out",
                    str(root / "changed.json"),
                    "--table-out",
                    str(root / "changed.csv"),
                    "--bootstrap-iterations",
                    "10",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("SHA-256 differs", rejected.stderr + rejected.stdout)

    def test_oracle_sidecar_requires_explicit_diagnostic_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="grouped_binding_") as directory:
            predictions = Path(directory) / "diagnostic.csv"
            prediction_frame().to_csv(predictions, index=False)
            path = Path(directory) / "diagnostic.json"
            path.write_text(
                json.dumps(
                    {
                        "result_role": "oracle_pose_selected_diagnostic",
                        "checkpoint": "best_pose_model.pt",
                        "checkpoint_sha256": "b" * 64,
                        "checkpoint_epoch": 4,
                        "predictions_csv_sha256": hashlib.sha256(
                            predictions.read_bytes()
                        ).hexdigest(),
                        "checkpoint_selection_split": "validation",
                        "checkpoint_selection_metric": (
                            "validation_pose_oracle_position_mae_cm"
                        ),
                        "rows": 14,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                checkpoint_binding(
                    path,
                    predictions_path=predictions,
                    prediction_rows=14,
                    allow_diagnostic=False,
                )
            bound = checkpoint_binding(
                path,
                predictions_path=predictions,
                prediction_rows=14,
                allow_diagnostic=True,
            )
            self.assertEqual(
                bound["result_role"], "checkpoint_bound_grouped_diagnostic"
            )


if __name__ == "__main__":
    unittest.main()
