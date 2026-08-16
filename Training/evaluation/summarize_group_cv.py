#!/usr/bin/env python3
"""Validate and summarize complete nested participant Group-CV results.

The input plan is immutable and hash-bound.  Every participant/seed outer
evaluation must exist and must come from that run's validation-selected,
artifact-frozen ``best_intention`` checkpoint.  Because frozen outer reports
contain aggregate window metrics, participant-balanced reporting is only
scientifically identifiable when each outer fold contains one participant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from artifact_freeze import (  # noqa: E402
    MANIFEST_NAME,
    canonical_json_hash,
    sha256_file,
    validate_artifact_freeze,
)


PLAN_PROTOCOL = "nested_participant_group_cv_executable_v1"
FINAL_REPORT_PROTOCOL = "validation_frozen_checkpoint_single_test_v2"
SUMMARY_PROTOCOL = "complete_participant_balanced_nested_group_cv_v1"
INTENTION_NAMES = ("continue", "fetch", "handover")
HAND_NAMES = ("left", "right")


class GroupCVSummaryError(ValueError):
    """Raised when Group-CV completeness or provenance cannot be proved."""


def validate_historical_artifact_freeze(path: Path) -> Mapping[str, Any]:
    """Validate an immutable run without requiring its historical checkout."""

    return validate_artifact_freeze(path, require_current_git_state=False)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GroupCVSummaryError(f"Expected JSON object: {path}")
    return value


def _resolve(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GroupCVSummaryError(message)


def validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(plan.get("schema_version") == 1, "Unsupported Group-CV plan schema")
    _require(plan.get("protocol") == PLAN_PROTOCOL, "Unsupported Group-CV protocol")
    expected_fingerprint = canonical_json_hash(
        {**plan, "plan_fingerprint": None}
    )
    _require(
        plan.get("plan_fingerprint") == expected_fingerprint,
        "Group-CV plan fingerprint mismatch",
    )
    _require(
        plan.get("checkpoint_selection_split") == "inner_validation",
        "Group-CV checkpoints were not selected on inner validation",
    )
    _require(
        plan.get("outer_evaluation_used_for_selection") is False,
        "Outer evaluation was used for model/checkpoint selection",
    )
    base_config = _resolve(str(plan.get("base_config", "")))
    _require(base_config.is_file(), f"Missing Group-CV base config: {base_config}")
    _require(
        sha256_file(base_config) == plan.get("base_config_sha256"),
        "Group-CV base config hash mismatch",
    )
    seeds = [int(value) for value in plan.get("seeds", [])]
    _require(bool(seeds) and len(seeds) == len(set(seeds)), "Invalid plan seeds")
    fold_count = int(plan.get("fold_count", 0))
    _require(fold_count > 0, "Invalid fold count")
    runs = plan.get("runs")
    _require(isinstance(runs, list), "Plan runs must be a list")
    _require(
        len(runs) == fold_count * len(seeds),
        "Plan is not the complete fold-by-seed Cartesian product",
    )
    expected_pairs = {(fold, seed) for fold in range(fold_count) for seed in seeds}
    actual_pairs = {(int(row["fold"]), int(row["seed"])) for row in runs}
    _require(
        len(actual_pairs) == len(runs) and actual_pairs == expected_pairs,
        "Plan has duplicate or missing fold/seed runs",
    )

    fold_partitions: dict[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
    outer_by_fold: dict[int, str] = {}
    participant_universes: set[tuple[str, ...]] = set()
    for row in runs:
        fold = int(row["fold"])
        train = tuple(sorted(str(value) for value in row["train_participants"]))
        validation = tuple(
            sorted(str(value) for value in row["validation_participants"])
        )
        outer = tuple(
            sorted(str(value) for value in row["outer_evaluation_participants"])
        )
        _require(bool(train) and bool(validation) and bool(outer), "Empty CV partition")
        _require(
            not (set(train) & set(validation) or set(train) & set(outer) or set(validation) & set(outer)),
            f"Fold {fold} is not participant-disjoint",
        )
        _require(
            len(outer) == 1,
            "Participant-balanced aggregation requires exactly one outer participant "
            f"per fold; fold {fold} has {len(outer)}",
        )
        partition = (train, validation, outer)
        participant_universes.add(tuple(sorted(set(train) | set(validation) | set(outer))))
        previous = fold_partitions.setdefault(fold, partition)
        _require(previous == partition, f"Fold {fold} partition changes across seeds")
        outer_by_fold[fold] = outer[0]
    _require(
        len(set(outer_by_fold.values())) == fold_count,
        "Each participant must occur in exactly one outer fold",
    )
    _require(len(participant_universes) == 1, "Participant universe changes across folds")
    universe = set(next(iter(participant_universes)))
    _require(
        set(outer_by_fold.values()) == universe,
        "Outer folds do not cover every participant exactly once",
    )
    return [dict(row) for row in runs]


def _classification_from_confusion(
    metric: Mapping[str, Any], *, expected_names: Sequence[str], label: str
) -> dict[str, Any]:
    names = tuple(str(value) for value in metric.get("class_names", []))
    _require(names == tuple(expected_names), f"{label} class names mismatch: {names}")
    matrix = metric.get("confusion_matrix")
    size = len(expected_names)
    _require(
        isinstance(matrix, list)
        and len(matrix) == size
        and all(isinstance(row, list) and len(row) == size for row in matrix),
        f"{label} confusion matrix has invalid dimensions",
    )
    values: list[list[int]] = []
    for row in matrix:
        parsed = []
        for value in row:
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{label} confusion matrix must contain non-negative integers",
            )
            parsed.append(value)
        values.append(parsed)
    support = [sum(values[index]) for index in range(size)]
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index in range(size):
        true_positive = values[index][index]
        false_positive = sum(values[row][index] for row in range(size)) - true_positive
        false_negative = support[index] - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative
        precision.append(true_positive / precision_denominator if precision_denominator else 0.0)
        recall.append(true_positive / recall_denominator if recall_denominator else 0.0)
        f1.append(2 * true_positive / f1_denominator if f1_denominator else 0.0)
    total = sum(support)
    result = {
        "samples": total,
        "macro_f1": sum(f1) / size,
        "macro_f1_supported": (
            sum(value for value, count in zip(f1, support) if count > 0)
            / max(1, sum(count > 0 for count in support))
        ),
        "accuracy": sum(values[index][index] for index in range(size)) / total if total else 0.0,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
    }
    for key in ("samples", "macro_f1", "macro_f1_supported", "accuracy"):
        reported = metric.get(key)
        if reported is not None:
            matches = (
                int(reported) == int(result[key])
                if key == "samples"
                else math.isclose(
                    float(reported), float(result[key]), rel_tol=1e-6, abs_tol=1e-6
                )
            )
            _require(matches, f"{label} {key} disagrees with its confusion matrix")
    return result


def _checkpoint_identity(freeze: Mapping[str, Any], checkpoint: str) -> Mapping[str, Any]:
    identity = freeze.get("output_artifacts", {}).get("checkpoints", {}).get(checkpoint)
    _require(isinstance(identity, Mapping), f"Freeze has no {checkpoint} checkpoint")
    return identity


def _scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only fields that may affect the trained/evaluated system."""

    data = dict(value.get("data", {}))
    if data.get("master_dir"):
        data["master_dir"] = str(_resolve(str(data["master_dir"])))
    return {
        "run_name": value.get("run_name"),
        "model_type": value.get("model_type"),
        "data": data,
        "model": value.get("model"),
        "training": value.get("training"),
        "group_cv": value.get("group_cv"),
    }


def validate_outer_report(
    row: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    freeze_validator: Callable[[Path], Mapping[str, Any]] = validate_historical_artifact_freeze,
) -> dict[str, Any]:
    config_path = _resolve(str(row["config"]))
    run_dir = _resolve(str(row["run_dir"]))
    report_path = _resolve(str(row["outer_evaluation_output"]))
    _require(config_path.is_file(), f"Missing Group-CV config: {config_path}")
    _require(
        sha256_file(config_path) == row.get("config_sha256"),
        f"Group-CV config hash mismatch: {config_path}",
    )
    _require(report_path.is_file(), f"Missing outer evaluation: {report_path}")
    report = _read_json(report_path)
    _require(report.get("schema_version") == 2, "Unsupported outer report schema")
    _require(report.get("evaluation_protocol") == FINAL_REPORT_PROTOCOL, "Wrong outer protocol")
    _require(
        report.get("report_fingerprint")
        == canonical_json_hash({**report, "report_fingerprint": None}),
        f"Outer report fingerprint mismatch: {report_path}",
    )
    _require(report.get("split") == "test", "Outer report is not the held-out split")
    _require(report.get("matrix_authorization") is None, "Group-CV report has matrix authorization")
    _require(
        report.get("test_used_for_model_or_checkpoint_selection") is False,
        "Outer metrics were used for selection",
    )
    _require(_resolve(str(report.get("source_run"))) == run_dir, "Outer report source run mismatch")

    config = _read_json(config_path)
    run_config = _read_json(run_dir / "config.json")
    _require(
        _scientific_config(config) == _scientific_config(run_config),
        "Resolved run config differs from the hash-bound Group-CV plan",
    )
    for value, label in ((config, "planned"), (run_config, "resolved")):
        group_cv = value.get("group_cv", {})
        _require(group_cv.get("checkpoint_selection_split") == "validation", f"{label} config is not validation-selected")
        _require(group_cv.get("outer_evaluation_used_for_selection") is False, f"{label} config permits outer selection")
        _require(int(group_cv.get("fold", -1)) == int(row["fold"]), f"{label} config fold mismatch")
        for config_key, row_key in (
            ("train_participants", "train_participants"),
            ("validation_participants", "validation_participants"),
            ("outer_evaluation_participants", "outer_evaluation_participants"),
        ):
            _require(
                sorted(group_cv.get(config_key, [])) == sorted(row[row_key]),
                f"{label} config {config_key} mismatch",
            )
        data = value.get("data", {})
        for data_key, row_key in (
            ("train_participants", "train_participants"),
            ("validation_participants", "validation_participants"),
            ("test_participants", "outer_evaluation_participants"),
        ):
            _require(
                sorted(data.get(data_key, [])) == sorted(row[row_key]),
                f"{label} data split {data_key} mismatch",
            )
        _require(
            group_cv.get("split_fingerprint_sha256")
            == plan.get("split_fingerprint_sha256"),
            f"{label} config split fingerprint mismatch",
        )
        _require(int(value["training"]["seed"]) == int(row["seed"]), f"{label} config seed mismatch")

    metrics = _read_json(run_dir / "metrics.json")
    _require(metrics.get("test_evaluation_skipped") is True, "Source run was not validation-only")
    _require("test" not in metrics and "test_by_checkpoint" not in metrics, "Source run contains test metrics")
    checkpoint_meta = metrics.get("checkpoints", {}).get("best_intention")
    _require(isinstance(checkpoint_meta, Mapping), "Source run lacks best_intention metadata")
    _require(
        checkpoint_meta.get("selection_metric") == "validation_intention_macro_f1",
        "Source checkpoint was not selected by validation intention macro-F1",
    )

    freeze = freeze_validator(run_dir / MANIFEST_NAME)
    _require(
        _resolve(str(report.get("source_artifact_manifest")))
        == (run_dir / MANIFEST_NAME).resolve(),
        "Outer report artifact-manifest path mismatch",
    )
    _require(
        report.get("source_artifact_manifest_fingerprint") == freeze.get("manifest_fingerprint"),
        "Outer report artifact-freeze fingerprint mismatch",
    )
    _require(
        freeze.get("selection_policy", {}).get("primary_checkpoint")
        == "best_intention"
        and freeze.get("selection_policy", {}).get("selection_split")
        == "validation",
        "Artifact freeze does not bind the validation-selected primary checkpoint",
    )
    _require(freeze.get("dataset", {}).get("identifier") == plan.get("dataset_tag"), "Frozen dataset tag mismatch")
    _require(report.get("dataset_identifier") == plan.get("dataset_tag"), "Report dataset tag mismatch")
    for key in ("dataset_content_fingerprint", "source_content_fingerprint"):
        _require(
            report.get(key) == freeze.get("dataset", {}).get(key),
            f"Outer report {key} mismatch",
        )
    checkpoint = report.get("checkpoint", {})
    _require(checkpoint.get("name") == "best_intention", "Outer report is not best_intention")
    _require(checkpoint.get("selection_split") == "validation", "Outer checkpoint selection split mismatch")
    _require(
        checkpoint.get("selection_metric") == "validation_intention_macro_f1",
        "Outer checkpoint metric is not validation intention macro-F1",
    )
    frozen_checkpoint = _checkpoint_identity(freeze, "best_intention")
    hashes = {
        str(checkpoint.get("sha256")),
        str(checkpoint_meta.get("sha256")),
        str(frozen_checkpoint.get("sha256")),
    }
    _require(len(hashes) == 1 and "None" not in hashes and "" not in hashes, "Checkpoint hashes disagree")

    metrics_block = report.get("test_metrics")
    _require(isinstance(metrics_block, Mapping), "Outer report lacks test metrics")
    intention = _classification_from_confusion(
        metrics_block.get("intention", {}),
        expected_names=INTENTION_NAMES,
        label="intention",
    )
    hand_metric = metrics_block.get("receiving_hand")
    hand = None
    if isinstance(hand_metric, Mapping):
        hand = _classification_from_confusion(
            hand_metric, expected_names=HAND_NAMES, label="receiving_hand"
        )
    return {
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "participant": str(row["outer_evaluation_participants"][0]),
        "run_dir": str(run_dir),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "report_fingerprint": report["report_fingerprint"],
        "artifact_manifest_fingerprint": freeze["manifest_fingerprint"],
        "checkpoint_sha256": checkpoint["sha256"],
        "source_content_fingerprint": report["source_content_fingerprint"],
        "intention": intention,
        "receiving_hand": hand,
    }


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "Cannot aggregate an empty metric")
    return statistics.fmean(values)


def _participant_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        intention = row["intention"]
        hand = row["receiving_hand"]
        output: dict[str, Any] = {
            "fold": row["fold"],
            "seed": row["seed"],
            "participant": row["participant"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "report_fingerprint": row["report_fingerprint"],
            "intention_samples": intention["samples"],
            "intention_accuracy": intention["accuracy"],
            "intention_macro_f1": intention["macro_f1"],
        }
        for index, name in enumerate(INTENTION_NAMES):
            output[f"{name}_precision"] = intention["per_class_precision"][index]
            output[f"{name}_recall"] = intention["per_class_recall"][index]
            output[f"{name}_f1"] = intention["per_class_f1"][index]
            output[f"{name}_support"] = intention["support"][index]
        output["receiving_hand_samples"] = hand["samples"] if hand else None
        output["receiving_hand_macro_f1"] = (
            hand["macro_f1"] if hand and hand["samples"] > 0 else None
        )
        result.append(output)
    return result


def build_summary(plan_path: Path, *, freeze_validator: Callable[[Path], Mapping[str, Any]] = validate_historical_artifact_freeze) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan_path = plan_path.expanduser().resolve()
    plan = _read_json(plan_path)
    runs = validate_plan(plan)
    validated = [
        validate_outer_report(row, plan=plan, freeze_validator=freeze_validator)
        for row in runs
    ]
    source_fingerprints = {
        str(row["source_content_fingerprint"]) for row in validated
    }
    _require(
        len(source_fingerprints) == 1,
        "Group-CV runs do not share one immutable master/manifest source",
    )
    validated.sort(key=lambda row: (row["seed"], row["participant"], row["fold"]))
    participant_rows = _participant_csv_rows(validated)

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        by_seed[int(row["seed"])].append(row)
    expected_participants = {row["participant"] for row in participant_rows}
    seed_rows: list[dict[str, Any]] = []
    numeric_keys = [
        "intention_accuracy",
        "intention_macro_f1",
        *(f"{name}_{metric}" for name in INTENTION_NAMES for metric in ("precision", "recall", "f1")),
        "receiving_hand_macro_f1",
    ]
    for seed in sorted(by_seed):
        values = by_seed[seed]
        _require({row["participant"] for row in values} == expected_participants, f"Seed {seed} participant coverage mismatch")
        summary_row: dict[str, Any] = {
            "seed": seed,
            "participant_count": len(values),
            "aggregation": "equal_weight_per_outer_participant",
            "receiving_hand_contributing_participants": sum(
                row.get("receiving_hand_macro_f1") is not None for row in values
            ),
        }
        for key in numeric_keys:
            available = [float(row[key]) for row in values if row.get(key) is not None]
            summary_row[key] = _mean(available) if available else None
        seed_rows.append(summary_row)

    seed_metric_summary = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in seed_rows if row.get(key) is not None]
        seed_metric_summary[key] = {
            "mean_across_seed_participant_balanced_estimates": _mean(values) if values else None,
            "sample_sd_across_seeds": statistics.stdev(values) if len(values) >= 2 else None,
            "seed_count": len(values),
        }
    summary = {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_fingerprint": None,
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "plan_fingerprint": plan["plan_fingerprint"],
            "split_fingerprint_sha256": plan["split_fingerprint_sha256"],
            "dataset_tag": plan["dataset_tag"],
            "experiment_tag": plan["experiment_tag"],
            "common_source_content_fingerprint": next(
                iter(source_fingerprints)
            ),
        },
        "completeness": {
            "status": "complete",
            "expected_outer_evaluations": len(runs),
            "validated_outer_evaluations": len(validated),
            "participants": sorted(expected_participants),
            "seeds": sorted(by_seed),
            "one_outer_participant_per_fold": True,
        },
        "selection_discipline": {
            "checkpoint": "best_intention",
            "checkpoint_selection_split": "inner_validation",
            "outer_evaluation_used_for_selection": False,
            "all_reports_artifact_frozen": True,
            "all_report_fingerprints_verified": True,
        },
        "aggregation": {
            "participant_balance": "equal weight per held-out participant within each seed",
            "seed_summary": "arithmetic mean and sample SD across seed-level participant-balanced estimates",
            "seed_sd_is_population_uncertainty": False,
            "window_counts_are_not_used_as_participant_weights": True,
            "metrics_recomputed_from_confusion_matrices": True,
        },
        "seed_metric_summary": seed_metric_summary,
        "validated_reports": [
            {key: row[key] for key in (
                "fold", "seed", "participant", "run_dir", "report", "report_sha256",
                "report_fingerprint", "artifact_manifest_fingerprint", "checkpoint_sha256",
            )}
            for row in validated
        ],
    }
    summary["report_fingerprint"] = canonical_json_hash(summary)
    return summary, participant_rows, seed_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"Refusing to write empty table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_dir = args.output_dir.expanduser().resolve()
        outputs = {
            "json": output_dir / "group_cv_summary.json",
            "participants": output_dir / "group_cv_participant_metrics.csv",
            "seeds": output_dir / "group_cv_seed_metrics.csv",
        }
        existing = [path for path in outputs.values() if path.exists()]
        _require(not existing, "Refusing to overwrite Group-CV summary artifacts: " + ", ".join(map(str, existing)))
        summary, participant_rows, seed_rows = build_summary(args.plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs["json"].write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(outputs["participants"], participant_rows)
        _write_csv(outputs["seeds"], seed_rows)
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Complete participant-balanced Group-CV summary: {outputs['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
