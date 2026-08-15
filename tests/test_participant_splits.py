from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Training"))

from participant_splits import (  # noqa: E402
    SequenceSummary,
    generate_balanced_participant_split,
    generate_participant_group_cv,
    load_static_sequence_summaries,
    summarize_master_csv,
)
from audit_participant_splits import build_report  # noqa: E402
from prepare_group_cv_runs import build_group_cv_plan  # noqa: E402


def summary(
    participant: str,
    hand: str,
    index: int,
    *,
    object_id: int = 6,
) -> SequenceSummary:
    return SequenceSummary(
        sequence_id=f"{participant}_{index}",
        participant=participant,
        receiving_hand=hand,
        target_object_id=object_id,
        phase_distribution={
            "continue": float(10 + index),
            "fetch": 4.0,
            "transition": 2.0,
            "handover": 6.0,
        },
        phase_unit="rows",
        phase_scope="synthetic_pre_window_rows",
        row_count=22 + index,
        source="synthetic",
    )


def balanced_summaries() -> list[SequenceSummary]:
    values = []
    for index in range(6):
        values.append(summary(f"Left{index}", "left", index, object_id=6 + index % 3))
        values.append(
            summary(f"Right{index}", "right", index, object_id=6 + index % 3)
        )
    return values


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ParticipantSplitTests(unittest.TestCase):
    def test_master_summary_extracts_sequence_labels_and_phase_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P1_1_master.csv"
            fields = [
                "sequence_id",
                "participant",
                "receiving_hand",
                "target_object_id",
                "intent_label",
            ]
            write_csv(
                path,
                fields,
                [
                    {
                        "sequence_id": "P1_1",
                        "participant": "p1",
                        "receiving_hand": "left",
                        "target_object_id": "7",
                        "intent_label": label,
                    }
                    for label in ("continue", "continue", "fetch", "handover")
                ],
            )
            result = summarize_master_csv(path)

        self.assertEqual(result.sequence_id, "P1_1")
        self.assertEqual(result.participant, "P1")
        self.assertEqual(result.receiving_hand, "left")
        self.assertEqual(result.target_object_id, 7)
        self.assertEqual(
            result.phase_distribution,
            {"continue": 2.0, "fetch": 1.0, "handover": 1.0},
        )
        self.assertEqual(result.phase_unit, "rows")
        self.assertEqual(result.phase_scope, "master_rows_before_windowing")

    def test_static_audit_and_annotations_preserve_split_and_supply_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.csv"
            annotations = root / "annotations.csv"
            write_csv(
                audit,
                [
                    "split",
                    "participant",
                    "sequence_id",
                    "receiving_hand",
                    "handover_duration_seconds",
                ],
                [
                    {
                        "split": "validation",
                        "participant": "P1",
                        "sequence_id": "P1_1",
                        "receiving_hand": "right",
                        "handover_duration_seconds": "3",
                    },
                    {
                        "split": "test",
                        "participant": "P2",
                        "sequence_id": "P2_1",
                        "receiving_hand": "left",
                        "handover_duration_seconds": "4",
                    },
                    {
                        "split": "train",
                        "participant": "P3",
                        "sequence_id": "P3_1",
                        "receiving_hand": "right",
                        "handover_duration_seconds": "5",
                    },
                ],
            )
            write_csv(
                annotations,
                [
                    "sequence_id",
                    "target_object_id",
                    "manual_start_s",
                    "manual_second_s",
                    "manual_done_s",
                    "manual_third_s",
                ],
                [
                    {
                        "sequence_id": f"P{index}_1",
                        "target_object_id": str(5 + index),
                        "manual_start_s": "0",
                        "manual_second_s": "10",
                        "manual_done_s": "14",
                        "manual_third_s": "16",
                    }
                    for index in (1, 2, 3)
                ],
            )
            summaries, historical, metadata = load_static_sequence_summaries(
                audit, annotation_csv=annotations
            )

        self.assertEqual(historical["validation"], ["P1"])
        self.assertEqual(historical["test"], ["P2"])
        self.assertEqual(historical["train"], ["P3"])
        self.assertEqual(summaries[0].target_object_id, 6)
        self.assertEqual(summaries[0].phase_unit, "seconds")
        self.assertEqual(summaries[0].phase_scope, "annotation_phase_durations")
        self.assertEqual(
            summaries[0].phase_distribution,
            {"continue": 10.0, "fetch": 4.0, "transition": 2.0, "handover": 3.0},
        )
        self.assertEqual(metadata["annotations_joined"], 3)

    def test_explicit_historical_split_is_preserved_exactly(self) -> None:
        summaries = balanced_summaries()
        historical = {
            "validation": ["Left0", "Right0"],
            "test": ["Left1", "Right1"],
        }
        plan = generate_balanced_participant_split(
            summaries,
            seed=42,
            validation_fraction=0.2,
            test_fraction=0.2,
            historical_split=historical,
            restarts=32,
        )
        self.assertTrue(plan["historical_split_preserved"])
        self.assertFalse(plan["selection_optimized"])
        self.assertEqual(plan["participants"]["validation"], ["Left0", "Right0"])
        self.assertEqual(plan["participants"]["test"], ["Left1", "Right1"])
        self.assertEqual(len(plan["participants"]["train"]), 8)
        self.assertEqual(len(plan["participant_table"]), 12)
        p1_row = next(
            row
            for row in plan["participant_table"]
            if row["split"] == "validation" and row["participant"] == "Left0"
        )
        self.assertEqual(p1_row["sequence_count"], 1)
        self.assertEqual(p1_row["receiving_hand_sequence_counts"], {"left": 1})
        self.assertEqual(len(plan["split_fingerprint_sha256"]), 64)
        test_coupling = plan["participant_hand_diagnostics_by_split"]["test"]
        self.assertEqual(test_coupling["participant_majority_hand_accuracy"], 1.0)
        self.assertTrue(test_coupling["all_known_participants_are_hand_pure"])
        self.assertTrue(
            any("perfectly determined" in value for value in test_coupling["warnings"])
        )

    def test_balanced_split_is_disjoint_deterministic_and_covers_both_hands(
        self,
    ) -> None:
        summaries = balanced_summaries()
        first = generate_balanced_participant_split(
            summaries,
            seed=17,
            validation_fraction=1 / 6,
            test_fraction=1 / 6,
            restarts=96,
        )
        second = generate_balanced_participant_split(
            summaries,
            seed=17,
            validation_fraction=1 / 6,
            test_fraction=1 / 6,
            restarts=96,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["split_fingerprint_sha256"], second["split_fingerprint_sha256"]
        )
        groups = {
            name: set(first["participants"][name])
            for name in ("train", "validation", "test")
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(len(set().union(*groups.values())), 12)
        for row in first["table"]:
            self.assertGreater(row["receiving_hand_sequence_counts"].get("left", 0), 0)
            self.assertGreater(
                row["receiving_hand_sequence_counts"].get("right", 0), 0
            )
        self.assertTrue(first["objective"]["uses_only_dataset_labels_and_counts"])
        self.assertFalse(first["objective"]["model_performance_metrics_used"])

    def test_balancing_prefers_mixed_hand_participants_for_eval_splits(self) -> None:
        summaries = []
        for index in range(3):
            summaries.extend(
                [
                    summary(f"Mixed{index}", "left", index * 2),
                    summary(f"Mixed{index}", "right", index * 2 + 1),
                    summary(f"Leftonly{index}", "left", 20 + index),
                    summary(f"Rightonly{index}", "right", 30 + index),
                ]
            )
        plan = generate_balanced_participant_split(
            summaries,
            seed=11,
            validation_fraction=2 / 9,
            test_fraction=2 / 9,
            restarts=256,
        )
        mixed = {f"Mixed{index}" for index in range(3)}
        for split_name in ("validation", "test"):
            self.assertTrue(
                set(plan["participants"][split_name]) & mixed,
                f"{split_name} should contain a mixed-hand participant",
            )

    def test_impossible_hand_balance_is_reported_not_hidden(self) -> None:
        summaries = [summary(f"P{index}", "right", index) for index in range(6)]
        plan = generate_balanced_participant_split(
            summaries,
            seed=5,
            validation_fraction=1 / 3,
            test_fraction=1 / 3,
            restarts=32,
        )
        diagnostics = plan["participant_hand_diagnostics"]
        self.assertFalse(
            diagnostics["both_hands_each_partition_necessary_condition"]
        )
        self.assertFalse(diagnostics["both_hands_each_partition_exactly_feasible"])
        self.assertTrue(diagnostics["all_known_participants_are_hand_pure"])
        self.assertFalse(
            diagnostics[
                "reassignment_alone_can_remove_participant_hand_coupling"
            ]
        )
        self.assertTrue(any("No participant has a known left" in item for item in plan["warnings"]))

    def test_group_cv_is_participant_disjoint_complete_and_deterministic(self) -> None:
        summaries = balanced_summaries()
        first = generate_participant_group_cv(
            summaries, folds=4, seed=23, restarts=64
        )
        second = generate_participant_group_cv(
            summaries, folds=4, seed=23, restarts=64
        )
        self.assertEqual(first, second)
        validation_occurrences = []
        test_occurrences = []
        for fold in first["folds"]:
            train = set(fold["train_participants"])
            validation = set(fold["validation_participants"])
            test = set(fold["test_participants"])
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(len(train | validation | test), 12)
            validation_occurrences.extend(validation)
            test_occurrences.extend(test)
        self.assertEqual(len(validation_occurrences), 12)
        self.assertEqual(len(set(validation_occurrences)), 12)
        self.assertEqual(len(test_occurrences), 12)
        self.assertEqual(len(set(test_occurrences)), 12)

    def test_group_cv_plan_materializes_executable_nested_configs(self) -> None:
        cv = generate_participant_group_cv(
            balanced_summaries(), folds=12, seed=23, restarts=32
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            base = {
                "run_name": "residual",
                "model_type": "hierarchical_residual_pose_transformer_v2",
                "data": {
                    "validation_participants": ["old_validation"],
                    "test_participants": ["old_test"],
                },
                "training": {"seed": 42},
            }
            base_path.write_text(json.dumps(base), encoding="utf-8")
            plan = build_group_cv_plan(
                audit={"participant_group_cv": cv},
                base_config=base,
                base_config_path=base_path,
                output_dir=root / "generated",
                dataset_tag="dataset",
                experiment_tag="group_cv",
                seeds=[42, 43],
            )
            self.assertEqual(len(plan["runs"]), 24)
            self.assertFalse(plan["outer_evaluation_used_for_selection"])
            self.assertIsInstance(plan["plan_fingerprint"], str)
            self.assertEqual(len(plan["plan_fingerprint"]), 64)
            for row in plan["runs"]:
                config = json.loads(Path(row["config"]).read_text(encoding="utf-8"))
                self.assertEqual(
                    config["data"]["train_participants"],
                    row["train_participants"],
                )
                self.assertEqual(
                    config["data"]["validation_participants"],
                    row["validation_participants"],
                )
                self.assertEqual(
                    config["data"]["test_participants"],
                    row["outer_evaluation_participants"],
                )
                self.assertIn("--skip-test-evaluation", row["validation_command"])
                self.assertIn(
                    "evaluate_frozen_run.py", row["outer_evaluation_command"]
                )

    def test_leave_one_participant_out_is_singleton_and_aggregatable(self) -> None:
        summaries = balanced_summaries()
        args = SimpleNamespace(
            summary_csv=None,
            annotations=None,
            master_dir=None,
            manifest=None,
            config=None,
            strict_manifest=True,
            balanced_candidate=False,
            seed=42,
            validation_fraction=0.2,
            test_fraction=0.2,
            group_cv_folds=5,
            leave_one_participant_out=True,
            restarts=32,
        )
        with patch(
            "audit_participant_splits.load_inputs",
            return_value=(summaries, None, {"source": "synthetic"}),
        ):
            report = build_report(args)
        cv = report["participant_group_cv"]
        self.assertEqual(cv["fold_count"], 12)
        self.assertEqual(cv["outer_evaluation_unit"], "single_participant")
        self.assertTrue(cv["participant_balanced_aggregation_identifiable"])
        self.assertEqual(
            {fold["outer_evaluation_participants"][0] for fold in cv["folds"]},
            {item.participant for item in summaries},
        )
        self.assertTrue(
            all(len(fold["outer_evaluation_participants"]) == 1 for fold in cv["folds"])
        )

    def test_cli_report_exposes_static_limitations_and_test_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.csv"
            annotations = root / "annotations.csv"
            write_csv(
                audit,
                ["split", "participant", "sequence_id", "receiving_hand"],
                [
                    {
                        "split": "train",
                        "participant": "Test",
                        "sequence_id": "Test_1",
                        "receiving_hand": "right",
                    },
                    {
                        "split": "validation",
                        "participant": "P2",
                        "sequence_id": "P2_1",
                        "receiving_hand": "right",
                    },
                    {
                        "split": "test",
                        "participant": "P3",
                        "sequence_id": "P3_1",
                        "receiving_hand": "left",
                    },
                ],
            )
            write_csv(
                annotations,
                ["sequence_id", "target_object_id"],
                [
                    {"sequence_id": "Test_1", "target_object_id": "6"},
                    {"sequence_id": "P2_1", "target_object_id": "7"},
                    {"sequence_id": "P3_1", "target_object_id": "8"},
                ],
            )
            report = build_report(
                SimpleNamespace(
                    summary_csv=audit,
                    annotations=annotations,
                    master_dir=None,
                    manifest=None,
                    config=None,
                    strict_manifest=True,
                    balanced_candidate=False,
                    seed=42,
                    validation_fraction=0.2,
                    test_fraction=0.2,
                    group_cv_folds=0,
                    restarts=16,
                )
            )

        self.assertFalse(
            report["distribution_semantics"][
                "is_window_endpoint_target_distribution"
            ]
        )
        self.assertTrue(
            report["distribution_semantics"]["transition_is_context_only"]
        )
        self.assertEqual(
            report["data_completeness"]["known_target_object_sequences"], 3
        )
        self.assertEqual(report["identity_provenance_flags"][0]["participant"], "Test")
        self.assertEqual(
            report["identity_provenance_flags"][0]["historical_splits"],
            ["train"],
        )
        self.assertTrue(any("Static audit" in value for value in report["limitations"]))


if __name__ == "__main__":
    unittest.main()
