#!/usr/bin/env python3
"""Shared invariants for checkpoint-coherent experiment reporting.

The training reports intentionally retain multiple validation-selected checkpoints
(for example ``best_intention`` and ``best_pose``).  A primary result must never
silently combine metrics from those checkpoints.  This module is the single place
where report generators resolve one checkpoint and flatten all of its metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


PRIMARY_RESULT_SEMANTICS = "single_validation_selected_checkpoint"
SEED_AGGREGATE_SEMANTICS = "multi_seed_aggregate_diagnostic"
DEFAULT_VALIDATION_SELECTION_RULE = (
    "retain checkpoints within 0.005 of the best validation intention "
    "macro-F1; then minimize validation pose error, maximize validation "
    "receiving-hand macro-F1 when that head exists, and use the lower seed "
    "as deterministic tie-break"
)
POSE_VALIDATION_SELECTION_RULE = (
    "minimize validation end-to-end pose error; then minimize validation orientation "
    "error, maximize validation intention macro-F1, and use the lower seed "
    "as deterministic tie-break"
)


class CheckpointSemanticsError(ValueError):
    """Raised when a report cannot prove single-checkpoint metric provenance."""


def _checkpoint_epoch(metadata: Mapping[str, Any]) -> int:
    value = metadata.get("epoch", metadata.get("source_epoch"))
    if value is None:
        raise CheckpointSemanticsError("checkpoint epoch/source_epoch is missing")
    return int(value)


def _selection_metric(metadata: Mapping[str, Any]) -> str:
    value = metadata.get(
        "selection_metric", metadata.get("source_selection_metric")
    )
    if not isinstance(value, str) or not value:
        raise CheckpointSemanticsError("checkpoint selection metric is missing")
    if not value.startswith("validation_"):
        raise CheckpointSemanticsError(
            f"checkpoint was not selected on validation: {value!r}"
        )
    return value


def checkpoint_metadata(report: Mapping[str, Any], checkpoint: str) -> dict:
    """Return validated, normalized metadata for one executable checkpoint."""

    try:
        raw = report["checkpoints"][checkpoint]
    except (KeyError, TypeError) as exc:
        raise CheckpointSemanticsError(
            f"checkpoint metadata is missing for {checkpoint!r}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CheckpointSemanticsError(
            f"checkpoint metadata for {checkpoint!r} is not an object"
        )
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise CheckpointSemanticsError(
            f"checkpoint path is missing for {checkpoint!r}"
        )
    selection_value = raw.get(
        "selection_value", raw.get("source_selection_value")
    )
    return {
        "name": checkpoint,
        "path": path,
        "epoch": _checkpoint_epoch(raw),
        "selection_split": "validation",
        "selection_metric": _selection_metric(raw),
        "selection_value": (
            None if selection_value is None else float(selection_value)
        ),
        "sha256": raw.get("sha256"),
    }


def checkpoint_metrics(
    report: Mapping[str, Any],
    *,
    split: str,
    checkpoint: str,
) -> Mapping[str, Any]:
    """Resolve metrics for exactly ``checkpoint`` across current/legacy schemas."""

    checkpoint_metadata(report, checkpoint)
    if split == "validation":
        by_checkpoint = report.get("validation_by_checkpoint")
        if isinstance(by_checkpoint, Mapping):
            values = by_checkpoint.get(checkpoint)
            if isinstance(values, Mapping):
                return values
        epoch = checkpoint_metadata(report, checkpoint)["epoch"]
        for record in report.get("history", []):
            if int(record.get("epoch", -1)) == epoch:
                values = record.get("validation")
                if isinstance(values, Mapping):
                    return values
        raise CheckpointSemanticsError(
            f"validation metrics are missing for checkpoint {checkpoint!r}"
        )

    if split != "test":
        raise CheckpointSemanticsError(f"unsupported metric split: {split!r}")

    test_by_checkpoint = report.get("test_by_checkpoint")
    if isinstance(test_by_checkpoint, Mapping):
        values = test_by_checkpoint.get(checkpoint)
        if isinstance(values, Mapping):
            return values

    test = report.get("test")
    if isinstance(test, Mapping):
        values = test.get(checkpoint)
        if isinstance(values, Mapping):
            return values
        # Legacy standard-model reports store best-intention metrics directly.
        if checkpoint == "best_intention" and isinstance(
            test.get("intention"), Mapping
        ):
            return test
    raise CheckpointSemanticsError(
        f"test metrics are missing for checkpoint {checkpoint!r}"
    )


def _number(
    values: Mapping[str, Any], *keys: str, default: Any = None
) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


def _classification_details(
    values: Mapping[str, Any],
    *,
    default_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[list[int]] | None]:
    """Normalize per-class metrics, including legacy confusion-only reports."""

    confusion_value = values.get("confusion_matrix")
    if not isinstance(confusion_value, list) or len(confusion_value) != len(
        default_names
    ):
        return [], None
    try:
        confusion = [
            [int(entry) for entry in row]
            for row in confusion_value
        ]
    except (TypeError, ValueError):
        return [], None
    if any(len(row) != len(default_names) for row in confusion):
        return [], None
    names_value = values.get("class_names")
    names = (
        [str(name) for name in names_value]
        if isinstance(names_value, list) and len(names_value) == len(default_names)
        else list(default_names)
    )
    f1_value = values.get("per_class_f1")
    f1_values = (
        [float(entry) for entry in f1_value]
        if isinstance(f1_value, list) and len(f1_value) == len(default_names)
        else [None] * len(default_names)
    )
    details = []
    for index, name in enumerate(names):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        computed_f1 = (
            2.0 * precision * recall / denominator if denominator else 0.0
        )
        details.append(
            {
                "class_id": index,
                "class_name": name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": (
                    float(f1_values[index])
                    if f1_values[index] is not None
                    else float(computed_f1)
                ),
                "support": int(support),
                "predicted": int(predicted),
            }
        )
    return details, confusion


def primary_result_row(
    report: Mapping[str, Any],
    *,
    checkpoint: str = "best_intention",
    split: str = "test",
) -> dict[str, Any]:
    """Flatten all primary metrics from one proven checkpoint.

    Generic pose fields describe the executable end-to-end system: receiving
    hand and reference pose are predicted by the same checkpoint.  Ground-truth
    receiving-hand (``pose_oracle``) values are retained only under the explicit
    ``diagnostic_pose_oracle_*`` namespace.  Neither may be substituted from
    another checkpoint.
    """

    metadata = checkpoint_metadata(report, checkpoint)
    values = checkpoint_metrics(report, split=split, checkpoint=checkpoint)
    prefix = split
    oracle_pose_key = (
        "pose_oracle" if isinstance(values.get("pose_oracle"), Mapping) else "pose"
    )
    oracle_pose = values.get(oracle_pose_key, {})
    end_to_end = values.get("pose_end_to_end", {})
    has_end_to_end_pose = isinstance(end_to_end, Mapping) and (
        "position_mae_cm" in end_to_end or "samples" in end_to_end
    )
    # Legacy single-head models expose only ``pose``.  Current residual models
    # must use the predicted-hand end-to-end branch for every generic metric.
    primary_pose = end_to_end if has_end_to_end_pose else oracle_pose
    coverage = values.get("pose_coverage", {})
    if not isinstance(oracle_pose, Mapping):
        oracle_pose = {}
    if not isinstance(primary_pose, Mapping):
        primary_pose = {}
    if not isinstance(end_to_end, Mapping):
        end_to_end = {}
    if not isinstance(coverage, Mapping):
        coverage = {}
    target_count = coverage.get("pose_targets", coverage.get("future_targets"))
    intention_values = values.get("intention", {})
    assistance_values = values.get("assistance", {})
    assistance_type_values = values.get("assistance_type", {})
    receiving_hand_values = values.get("receiving_hand", {})
    intention_per_class, intention_confusion = _classification_details(
        intention_values if isinstance(intention_values, Mapping) else {},
        default_names=("continue", "fetch", "handover"),
    )
    assistance_per_class, assistance_confusion = _classification_details(
        assistance_values if isinstance(assistance_values, Mapping) else {},
        default_names=("continue", "assistance"),
    )
    assistance_type_per_class, assistance_type_confusion = (
        _classification_details(
            assistance_type_values
            if isinstance(assistance_type_values, Mapping)
            else {},
            default_names=("fetch", "handover"),
        )
    )
    receiving_hand_per_class, receiving_hand_confusion = (
        _classification_details(
            receiving_hand_values
            if isinstance(receiving_hand_values, Mapping)
            else {},
            default_names=("left", "right"),
        )
    )

    row: dict[str, Any] = {
        "result_semantics": PRIMARY_RESULT_SEMANTICS,
        "metric_source_checkpoint": checkpoint,
        "primary_checkpoint_name": checkpoint,
        "primary_checkpoint_path": metadata["path"],
        "primary_checkpoint_epoch": metadata["epoch"],
        "primary_checkpoint_selection_split": metadata["selection_split"],
        "primary_checkpoint_selection_metric": metadata["selection_metric"],
        "primary_checkpoint_selection_value": metadata["selection_value"],
        "primary_checkpoint_sha256": metadata["sha256"],
        "primary_pose_metric_semantics": (
            "learned_end_to_end_predicted_receiving_hand_and_reference"
            if has_end_to_end_pose
            else "legacy_single_pose_output"
        ),
        f"{prefix}_intention_macro_f1": _float(
            _number(values, "intention", "macro_f1")
        ),
        f"{prefix}_intention_accuracy": _float(
            _number(values, "intention", "accuracy")
        ),
        f"{prefix}_intention_samples": _int(
            _number(values, "intention", "samples")
        ),
        f"{prefix}_intention_per_class": intention_per_class,
        f"{prefix}_intention_confusion_matrix": intention_confusion,
        f"{prefix}_assistance_macro_f1": _float(
            _number(values, "assistance", "macro_f1")
        ),
        f"{prefix}_assistance_accuracy": _float(
            _number(values, "assistance", "accuracy")
        ),
        f"{prefix}_assistance_samples": _int(
            _number(values, "assistance", "samples")
        ),
        f"{prefix}_assistance_per_class": assistance_per_class,
        f"{prefix}_assistance_confusion_matrix": assistance_confusion,
        f"{prefix}_assistance_type_macro_f1": _float(
            _number(values, "assistance_type", "macro_f1")
        ),
        f"{prefix}_assistance_type_accuracy": _float(
            _number(values, "assistance_type", "accuracy")
        ),
        f"{prefix}_assistance_type_samples": _int(
            _number(values, "assistance_type", "samples")
        ),
        f"{prefix}_assistance_type_per_class": assistance_type_per_class,
        f"{prefix}_assistance_type_confusion_matrix": assistance_type_confusion,
        f"{prefix}_receiving_hand_macro_f1": _float(
            _number(
                values,
                "receiving_hand",
                "macro_f1_supported",
                default=_number(values, "receiving_hand", "macro_f1"),
            )
        ),
        f"{prefix}_receiving_hand_accuracy": _float(
            _number(values, "receiving_hand", "accuracy")
        ),
        f"{prefix}_receiving_hand_samples": _int(
            _number(values, "receiving_hand", "samples")
        ),
        f"{prefix}_receiving_hand_per_class": receiving_hand_per_class,
        f"{prefix}_receiving_hand_confusion_matrix": receiving_hand_confusion,
        f"{prefix}_pose_mae_cm": _float(primary_pose.get("position_mae_cm")),
        f"{prefix}_pose_median_cm": _float(
            primary_pose.get("position_median_cm")
        ),
        f"{prefix}_pose_orientation_error_deg": _float(
            primary_pose.get("orientation_mean_deg")
        ),
        f"{prefix}_pose_orientation_median_error_deg": _float(
            primary_pose.get("orientation_median_deg")
        ),
        f"{prefix}_pose_samples": _int(primary_pose.get("samples")),
        f"{prefix}_pose_end_to_end_mae_cm": _float(
            end_to_end.get("position_mae_cm")
        ),
        f"{prefix}_pose_end_to_end_median_cm": _float(
            end_to_end.get("position_median_cm")
        ),
        f"{prefix}_pose_end_to_end_orientation_error_deg": _float(
            end_to_end.get("orientation_mean_deg")
        ),
        f"{prefix}_pose_end_to_end_orientation_median_error_deg": _float(
            end_to_end.get("orientation_median_deg")
        ),
        f"{prefix}_pose_end_to_end_samples": _int(end_to_end.get("samples")),
        f"{prefix}_pose_target_samples": _int(target_count),
        f"{prefix}_pose_oracle_reference_valid": _int(
            coverage.get("oracle_reference_valid")
        ),
        f"{prefix}_pose_predicted_reference_valid": _int(
            coverage.get("predicted_reference_valid")
        ),
        f"{prefix}_pose_coverage_denominator_receiving_hand_samples": _int(
            _number(values, "receiving_hand", "samples")
        ),
        f"diagnostic_pose_oracle_{prefix}_position_mean_cm": _float(
            oracle_pose.get("position_mae_cm")
        ),
        f"diagnostic_pose_oracle_{prefix}_position_median_cm": _float(
            oracle_pose.get("position_median_cm")
        ),
        f"diagnostic_pose_oracle_{prefix}_orientation_mean_deg": _float(
            oracle_pose.get("orientation_mean_deg")
        ),
        f"diagnostic_pose_oracle_{prefix}_orientation_median_deg": _float(
            oracle_pose.get("orientation_median_deg")
        ),
        f"diagnostic_pose_oracle_{prefix}_samples": _int(
            oracle_pose.get("samples")
        ),
    }
    denominator = row[f"{prefix}_receiving_hand_samples"]
    numerator = row[f"{prefix}_pose_target_samples"]
    row[f"{prefix}_pose_target_coverage"] = (
        None
        if denominator in (None, 0) or numerator is None
        else float(numerator) / float(denominator)
    )
    end_to_end_samples = row[f"{prefix}_pose_samples"]
    row[f"{prefix}_pose_coverage"] = (
        None
        if numerator in (None, 0) or end_to_end_samples is None
        else float(end_to_end_samples) / float(numerator)
    )
    assert_single_checkpoint_row(row)
    return row


def persistence_diagnostic(
    report: Mapping[str, Any],
    *,
    checkpoint: str = "best_intention",
    split: str = "test",
) -> dict[str, Any]:
    """Return a clearly namespaced last-observation pose baseline.

    Persistence uses the latest valid wrist observation of the ground-truth
    receiving hand.  It is therefore a conditional baseline, not an executable
    checkpoint.  The checkpoint argument identifies only the evaluation/sample
    context and is validated to prevent accidental cross-checkpoint mixing.
    """

    metadata = checkpoint_metadata(report, checkpoint)
    values = checkpoint_metrics(report, split=split, checkpoint=checkpoint)
    fair_common = values.get("pose_fair_common", {})
    fair_methods = (
        fair_common.get("methods", {})
        if isinstance(fair_common, Mapping)
        else {}
    )
    baseline = fair_methods.get(
        "persistence", values.get("last_observation_oracle", {})
    )
    oracle = fair_methods.get(
        "learned_end_to_end",
        values.get("pose_end_to_end", values.get("pose_oracle", values.get("pose", {}))),
    )
    coverage = values.get("pose_coverage", {})
    if not isinstance(baseline, Mapping):
        baseline = {}
    if not isinstance(oracle, Mapping):
        oracle = {}
    if not isinstance(coverage, Mapping):
        coverage = {}
    samples = _int(baseline.get("samples"))
    oracle_samples = _int(oracle.get("samples"))
    if (
        samples is not None
        and oracle_samples is not None
        and samples != oracle_samples
    ):
        raise CheckpointSemanticsError(
            "persistence and learned oracle pose do not use the same valid samples"
        )
    target_samples = _int(
        fair_common.get("coverage_denominator_pose_targets")
        if isinstance(fair_common, Mapping)
        else None
    )
    if target_samples is None:
        target_samples = _int(
            coverage.get("pose_targets", coverage.get("future_targets"))
        )
    valid_coverage = (
        None
        if samples is None or target_samples in (None, 0)
        else float(samples) / float(target_samples)
    )
    prefix = f"diagnostic_persistence_{split}"
    return {
        "diagnostic_result_semantics": (
            "paired_persistence_last_observation_ground_truth_receiving_hand_"
            "versus_primary_learned_end_to_end"
        ),
        "diagnostic_persistence_metric_source_checkpoint": checkpoint,
        "diagnostic_persistence_checkpoint_path": metadata["path"],
        "diagnostic_persistence_checkpoint_epoch": metadata["epoch"],
        "diagnostic_persistence_checkpoint_selection_metric": metadata[
            "selection_metric"
        ],
        f"{prefix}_position_mean_cm": _float(baseline.get("position_mae_cm")),
        f"{prefix}_position_median_cm": _float(
            baseline.get("position_median_cm")
        ),
        f"{prefix}_orientation_mean_deg": _float(
            baseline.get("orientation_mean_deg")
        ),
        f"{prefix}_orientation_median_deg": _float(
            baseline.get("orientation_median_deg")
        ),
        f"{prefix}_samples": samples,
        f"{prefix}_target_samples": target_samples,
        f"{prefix}_coverage": valid_coverage,
    }


def pose_selected_diagnostic(
    report: Mapping[str, Any], *, split: str = "test"
) -> dict[str, Any]:
    """Return explicitly namespaced best-pose diagnostics.

    This intentionally cannot populate generic ``test_pose_*`` fields.
    """

    row = primary_result_row(report, checkpoint="best_pose", split=split)
    prefix = split
    return {
        "diagnostic_result_semantics": "pose_selected_checkpoint_diagnostic",
        "diagnostic_pose_selected_checkpoint_name": row[
            "primary_checkpoint_name"
        ],
        "diagnostic_pose_selected_checkpoint_path": row[
            "primary_checkpoint_path"
        ],
        "diagnostic_pose_selected_checkpoint_epoch": row[
            "primary_checkpoint_epoch"
        ],
        "diagnostic_pose_selected_checkpoint_selection_metric": row[
            "primary_checkpoint_selection_metric"
        ],
        f"diagnostic_pose_selected_{prefix}_intention_macro_f1": row[
            f"{prefix}_intention_macro_f1"
        ],
        f"diagnostic_pose_selected_{prefix}_pose_mae_cm": row[
            f"{prefix}_pose_mae_cm"
        ],
        f"diagnostic_pose_selected_{prefix}_pose_orientation_error_deg": row[
            f"{prefix}_pose_orientation_error_deg"
        ],
        f"diagnostic_pose_selected_{prefix}_pose_samples": row[
            f"{prefix}_pose_samples"
        ],
        f"diagnostic_pose_selected_{prefix}_pose_end_to_end_mae_cm": row[
            f"{prefix}_pose_end_to_end_mae_cm"
        ],
        f"diagnostic_pose_selected_{prefix}_pose_end_to_end_samples": row[
            f"{prefix}_pose_end_to_end_samples"
        ],
        f"diagnostic_pose_selected_{prefix}_oracle_position_mean_cm": row[
            f"diagnostic_pose_oracle_{prefix}_position_mean_cm"
        ],
        f"diagnostic_pose_selected_{prefix}_oracle_orientation_mean_deg": row[
            f"diagnostic_pose_oracle_{prefix}_orientation_mean_deg"
        ],
        f"diagnostic_pose_selected_{prefix}_oracle_samples": row[
            f"diagnostic_pose_oracle_{prefix}_samples"
        ],
    }


def assert_single_checkpoint_row(row: Mapping[str, Any]) -> None:
    """Reject ambiguous/mixed primary result-row provenance."""

    if row.get("result_semantics") != PRIMARY_RESULT_SEMANTICS:
        raise CheckpointSemanticsError(
            "primary row must declare single-checkpoint result semantics"
        )
    checkpoint = row.get("primary_checkpoint_name")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise CheckpointSemanticsError("primary checkpoint name is missing")
    if row.get("metric_source_checkpoint") != checkpoint:
        raise CheckpointSemanticsError(
            "primary metrics do not share the declared checkpoint"
        )
    if row.get("primary_checkpoint_selection_split") != "validation":
        raise CheckpointSemanticsError(
            "primary checkpoint was not selected on validation"
        )
    selection_metric = row.get("primary_checkpoint_selection_metric")
    if not isinstance(selection_metric, str) or not selection_metric.startswith(
        "validation_"
    ):
        raise CheckpointSemanticsError(
            "primary checkpoint selection metric is not validation-based"
        )
    if not row.get("primary_checkpoint_path"):
        raise CheckpointSemanticsError("primary checkpoint path is missing")
    pose_semantics = row.get("primary_pose_metric_semantics")
    if pose_semantics not in {
        "learned_end_to_end_predicted_receiving_hand_and_reference",
        "legacy_single_pose_output",
    }:
        raise CheckpointSemanticsError(
            "generic pose metrics are not declared as executable end-to-end output"
        )
    for key, value in row.items():
        if key.startswith("diagnostic_"):
            continue
        if key.endswith("_source_checkpoint") and value != checkpoint:
            raise CheckpointSemanticsError(
                f"mixed checkpoint source in {key}: {value!r} != {checkpoint!r}"
            )


def assert_primary_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        assert_single_checkpoint_row(row)


def mark_seed_aggregate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Label mean/std rows as diagnostics rather than executable checkpoints."""

    result = dict(row)
    result["result_semantics"] = SEED_AGGREGATE_SEMANTICS
    return result


def select_primary_checkpoint_row(
    rows: Iterable[Mapping[str, Any]], *, f1_tolerance: float = 0.005
) -> dict[str, Any]:
    """Select one executable checkpoint using validation fields only."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise CheckpointSemanticsError("cannot select from an empty candidate set")
    assert_primary_rows(candidates)
    required = (
        "validation_intention_macro_f1",
        "validation_pose_mae_cm",
        "seed",
    )
    for row in candidates:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise CheckpointSemanticsError(
                f"validation selection fields are missing: {missing}"
            )
    best_f1 = max(float(row[required[0]]) for row in candidates)
    eligible = [
        row
        for row in candidates
        if float(row[required[0]]) >= best_f1 - f1_tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: (
            float(row[required[1]]),
            (
                -float(row["validation_receiving_hand_macro_f1"])
                if row.get("validation_receiving_hand_macro_f1") is not None
                else float("inf")
            ),
            int(row[required[2]]),
        ),
    )
    selected["validation_selection_rule"] = DEFAULT_VALIDATION_SELECTION_RULE
    selected["validation_selection_f1_tolerance"] = float(f1_tolerance)
    return selected


def select_pose_primary_checkpoint_row(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one pose-task checkpoint using validation metrics only."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise CheckpointSemanticsError("cannot select from an empty candidate set")
    assert_primary_rows(candidates)
    required = (
        "validation_pose_mae_cm",
        "validation_pose_orientation_error_deg",
        "validation_intention_macro_f1",
        "seed",
    )
    for row in candidates:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise CheckpointSemanticsError(
                f"validation pose-selection fields are missing: {missing}"
            )
    selected = min(
        candidates,
        key=lambda row: (
            float(row[required[0]]),
            float(row[required[1]]),
            -float(row[required[2]]),
            int(row[required[3]]),
        ),
    )
    selected["validation_selection_rule"] = POSE_VALIDATION_SELECTION_RULE
    return selected
