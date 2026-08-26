from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
EVALUATION = TRAINING / "evaluation"
for directory in (TRAINING, EVALUATION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from artifact_freeze import canonical_json_hash, sha256_file  # noqa: E402
from prepare_group_cv_runs import build_group_cv_plan  # noqa: E402
from summarize_group_cv import (  # noqa: E402
    GroupCVSummaryError,
    build_summary,
    validate_plan,
)


PARTICIPANTS = ("P1", "P2", "P3")
CHECKPOINT_SHA = "c" * 64


def classification(matrix: list[list[int]], names: list[str]) -> dict:
    size = len(matrix)
    support = [sum(row) for row in matrix]
    f1 = []
    precision = []
    recall = []
    for index in range(size):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(size)) - tp
        fn = support[index] - tp
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    samples = sum(support)
    return {
        "class_names": names,
        "confusion_matrix": matrix,
        "samples": samples,
        "accuracy": sum(matrix[index][index] for index in range(size)) / samples,
        "macro_f1": sum(f1) / size,
        "macro_f1_supported": (
            sum(value for value, count in zip(f1, support) if count > 0)
            / sum(count > 0 for count in support)
        ),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    base_path = tmp_path / "base.json"
    base = {
        "run_name": "residual",
        "model_type": "hierarchical_residual_pose_transformer_v2",
        "data": {},
        "training": {"seed": 42},
    }
    write_json(base_path, base)
    folds = []
    for fold, outer in enumerate(PARTICIPANTS):
        validation = PARTICIPANTS[(fold + 1) % len(PARTICIPANTS)]
        train = PARTICIPANTS[(fold + 2) % len(PARTICIPANTS)]
        folds.append(
            {
                "fold": fold,
                "train_participants": [train],
                "validation_participants": [validation],
                "test_participants": [outer],
            }
        )
    plan = build_group_cv_plan(
        audit={
            "participant_group_cv": {
                "folds": folds,
                "execution_protocol": "nested participant Group-CV",
                "split_fingerprint_sha256": "s" * 64,
                "fold_count": 3,
                "participant_balanced_aggregation_identifiable": True,
            }
        },
        base_config=base,
        base_config_path=base_path,
        output_dir=tmp_path / "generated",
        dataset_tag="dataset_test",
        experiment_tag="group_cv_test",
        seeds=[42, 43],
    )
    for row in plan["runs"]:
        run_dir = tmp_path / "runs" / f"fold{row['fold']}_seed{row['seed']}"
        report_path = tmp_path / "outer" / f"fold{row['fold']}_seed{row['seed']}.json"
        row["run_dir"] = str(run_dir)
        row["outer_evaluation_output"] = str(report_path)
        planned_config = json.loads(Path(row["config"]).read_text(encoding="utf-8"))
        write_json(run_dir / "config.json", planned_config)
        write_json(
            run_dir / "metrics.json",
            {
                "test_evaluation_skipped": True,
                "checkpoints": {
                    "best_intention": {
                        "sha256": CHECKPOINT_SHA,
                        "selection_metric": "validation_intention_macro_f1",
                    }
                },
            },
        )
        write_json(run_dir / "artifact_manifest.json", {"placeholder": True})
        diagonal = 6 if int(row["seed"]) == 42 else 4
        report = {
            "schema_version": 2,
            "evaluation_protocol": "validation_frozen_checkpoint_single_test_v2",
            "report_fingerprint": None,
            "split": "test",
            "source_run": str(run_dir),
            "source_artifact_manifest": str(run_dir / "artifact_manifest.json"),
            "source_artifact_manifest_fingerprint": "f" * 64,
            "dataset_identifier": "dataset_test",
            "dataset_content_fingerprint": "d" * 64,
            "source_content_fingerprint": "s" * 64,
            "matrix_authorization": None,
            "checkpoint": {
                "name": "best_intention",
                "sha256": CHECKPOINT_SHA,
                "selection_split": "validation",
                "selection_metric": "validation_intention_macro_f1",
            },
            "test_metrics": {
                "intention": classification(
                    [[diagonal, 1, 0], [1, diagonal, 0], [0, 1, diagonal]],
                    ["continue", "fetch", "handover"],
                ),
                "receiving_hand": classification(
                    (
                        [[diagonal, 1], [0, 0]]
                        if int(row["fold"]) == 0
                        else (
                            [[0, 0], [1, diagonal]]
                            if int(row["fold"]) == 1
                            else [[diagonal, 1], [1, diagonal]]
                        )
                    ),
                    ["left", "right"],
                ),
            },
            "test_used_for_model_or_checkpoint_selection": False,
        }
        report["report_fingerprint"] = canonical_json_hash(report)
        write_json(report_path, report)
    plan["plan_fingerprint"] = canonical_json_hash(
        {**plan, "plan_fingerprint": None}
    )
    plan_path = tmp_path / "group_cv_plan.json"
    write_json(plan_path, plan)
    freeze = {
        "manifest_fingerprint": "f" * 64,
        "dataset": {"identifier": "dataset_test"},
        "selection_policy": {
            "primary_checkpoint": "best_intention",
            "selection_split": "validation",
        },
        "output_artifacts": {
            "checkpoints": {"best_intention": {"sha256": CHECKPOINT_SHA}}
        },
    }
    freeze["dataset"].update(
        {
            "dataset_content_fingerprint": "d" * 64,
            "source_content_fingerprint": "s" * 64,
        }
    )
    return plan_path, freeze


def test_complete_group_cv_is_participant_balanced_and_seed_sd_is_separate(
    tmp_path: Path,
) -> None:
    plan_path, freeze = fixture(tmp_path)
    summary, participant_rows, seed_rows = build_summary(
        plan_path, freeze_validator=lambda _: freeze
    )
    assert summary["completeness"]["status"] == "complete"
    assert summary["completeness"]["validated_outer_evaluations"] == 6
    assert len(participant_rows) == 6
    assert len(seed_rows) == 2
    assert all(row["participant_count"] == 3 for row in seed_rows)
    metric = summary["seed_metric_summary"]["intention_macro_f1"]
    assert metric["seed_count"] == 2
    assert metric["sample_sd_across_seeds"] is not None
    assert summary["aggregation"]["seed_sd_is_population_uncertainty"] is False
    assert summary["schema_version"] == 2
    assert summary["protocol"] == "complete_participant_balanced_nested_group_cv_v2"
    assert summary["receiving_hand_reporting"][
        "fixed_two_class_mixed_hand_participants"
    ]["participants"] == ["P3"]
    assert all(
        row["receiving_hand_mixed_hand_participants"] == 1 for row in seed_rows
    )
    assert all(
        row["receiving_hand_macro_f1_supported"]
        > row["receiving_hand_macro_f1"]
        for row in seed_rows
    )
    assert all(
        row["receiving_hand_mixed_hand_macro_f1"]
        == next(
            participant["receiving_hand_macro_f1"]
            for participant in participant_rows
            if participant["seed"] == row["seed"]
            and participant["participant"] == "P3"
        )
        for row in seed_rows
    )
    assert summary["report_fingerprint"] == canonical_json_hash(
        {**summary, "report_fingerprint": None}
    )
    assert summary["plan"]["sha256"] == sha256_file(plan_path)


def test_missing_outer_evaluation_fails_closed(tmp_path: Path) -> None:
    plan_path, freeze = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    Path(plan["runs"][0]["outer_evaluation_output"]).unlink()
    with pytest.raises(GroupCVSummaryError, match="Missing outer evaluation"):
        build_summary(plan_path, freeze_validator=lambda _: freeze)


def test_outer_report_fingerprint_and_validation_only_source_are_enforced(
    tmp_path: Path,
) -> None:
    plan_path, freeze = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    row = plan["runs"][0]
    report_path = Path(row["outer_evaluation_output"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["test_metrics"]["intention"]["macro_f1"] = 0.999
    write_json(report_path, report)
    with pytest.raises(GroupCVSummaryError, match="fingerprint mismatch"):
        build_summary(plan_path, freeze_validator=lambda _: freeze)

    report["report_fingerprint"] = canonical_json_hash(
        {**report, "report_fingerprint": None}
    )
    write_json(report_path, report)
    metrics_path = Path(row["run_dir"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test"] = {"leak": True}
    write_json(metrics_path, metrics)
    with pytest.raises(GroupCVSummaryError, match="contains test metrics"):
        build_summary(plan_path, freeze_validator=lambda _: freeze)


def test_plan_requires_singleton_unique_outer_participants(tmp_path: Path) -> None:
    plan_path, _ = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for row in plan["runs"]:
        if row["fold"] == 1:
            row["outer_evaluation_participants"] = ["P1"]
            row["train_participants"] = ["P2"]
    plan["plan_fingerprint"] = canonical_json_hash(
        {**plan, "plan_fingerprint": None}
    )
    with pytest.raises(GroupCVSummaryError, match="exactly one outer fold"):
        validate_plan(plan)


def test_plan_fingerprint_is_mandatory(tmp_path: Path) -> None:
    plan_path, _ = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["dataset_tag"] = "tampered"
    with pytest.raises(GroupCVSummaryError, match="plan fingerprint mismatch"):
        validate_plan(plan)


def test_outer_runs_must_share_the_same_master_source(tmp_path: Path) -> None:
    plan_path, freeze = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report_path = Path(plan["runs"][0]["outer_evaluation_output"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_content_fingerprint"] = "x" * 64
    report["report_fingerprint"] = canonical_json_hash(
        {**report, "report_fingerprint": None}
    )
    write_json(report_path, report)

    def freeze_validator(path: Path) -> dict:
        value = json.loads(json.dumps(freeze))
        if path.resolve() == (Path(plan["runs"][0]["run_dir"]) / "artifact_manifest.json").resolve():
            value["dataset"]["source_content_fingerprint"] = "x" * 64
        return value

    with pytest.raises(GroupCVSummaryError, match="one immutable master/manifest"):
        build_summary(plan_path, freeze_validator=freeze_validator)


def test_resolved_scientific_config_must_match_the_plan(tmp_path: Path) -> None:
    plan_path, freeze = fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config_path = Path(plan["runs"][0]["run_dir"]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("model", {})["d_model"] = 999
    write_json(config_path, config)
    with pytest.raises(GroupCVSummaryError, match="Resolved run config differs"):
        build_summary(plan_path, freeze_validator=lambda _: freeze)
