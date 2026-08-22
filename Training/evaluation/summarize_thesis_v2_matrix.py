#!/usr/bin/env python3
"""Build the authoritative thesis-v2 table from frozen final-test reports.

One seed-level row is one executable ``best_intention`` checkpoint.  The
summarizer verifies the complete validation authorization before reading any
test metric and never substitutes metrics from a second checkpoint.  Across-
seed means and standard deviations are emitted in separate aggregate files;
an aggregate row is deliberately not presented as an executable model.

Optional grouped prediction reports add the paired t+1 persistence, constant-
velocity and learned-oracle-hand comparison.  Missing grouped reports remain
explicitly unavailable: baseline values are never inferred from the main
checkpoint report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from artifact_freeze import (  # noqa: E402
    ArtifactFreezeError,
    canonical_json_hash,
    sha256_file,
    validate_artifact_freeze,
)
from experiment_matrix import (  # noqa: E402
    DEFAULT_MATRIX,
    run_directory,
    validate_matrix,
)
from select_matrix_checkpoints import (  # noqa: E402
    validate_final_test_authorization,
    validate_embedded_final_test_authorization,
)
from grouped_metrics import (  # noqa: E402
    discover_pose_methods,
    prepare_prediction_frame,
    summarize_windows,
)
from pose_baselines import sample_key_fingerprint  # noqa: E402


FINAL_TEST_PROTOCOL = "validation_frozen_checkpoint_single_test_v2"
SEED_RESULT_SEMANTICS = "single_validation_selected_executable_checkpoint"
AGGREGATE_RESULT_SEMANTICS = "across_seed_summary_not_an_executable_checkpoint"
INTENTION_NAMES = ("continue", "fetch", "handover")
ASSISTANCE_NAMES = ("continue", "assistance")
ASSISTANCE_TYPE_NAMES = ("fetch", "handover")
HAND_NAMES = ("left", "right")
GROUPED_REPORT_NAME = "grouped_metrics.json"
GROUPED_PROTOCOL = "grouped_prediction_evaluation_v1"


class MatrixSummaryError(ValueError):
    """Raised when a final result cannot prove its immutable provenance."""


def validate_historical_artifact_freeze(manifest_path: Path) -> dict[str, Any]:
    """Validate immutable run artifacts from a later reporting checkout.

    The training Git identity is already fingerprint-bound in the manifest.
    Reporting must therefore validate the recorded run inputs and outputs, not
    require the report generator to be checked out at the old training commit.
    """

    return validate_artifact_freeze(
        manifest_path, require_current_git_state=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--final-test-dir", type=Path, default=None)
    parser.add_argument(
        "--postprocess-root",
        type=Path,
        default=None,
        help=(
            "Root containing <experiment>_seed<seed>/grouped_metrics.json or "
            "<experiment>_seed<seed>_grouped_metrics.json. It is mandatory when "
            "the matrix declares required t+1 postprocessing."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing derived summary (default: refuse).",
    )
    return parser.parse_args()


def resolve(path: Path, *, root: Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixSummaryError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixSummaryError(f"Expected a JSON object in {path}")
    return value


def _normalized_run_path(value: str | Path, *, project_root: Path) -> Path:
    text = str(value).replace("\\", "/")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_source_manifest_binding(
    report: Mapping[str, Any],
    *,
    expected_run: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    dataset_identifier: str,
    dataset_content_fingerprint: str,
    artifact_validator: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = (expected_run / "artifact_manifest.json").resolve()
    declared_value = str(report.get("source_artifact_manifest", ""))
    declared_manifest = Path(declared_value).expanduser()
    if not declared_manifest.is_absolute():
        declared_manifest = (expected_run / declared_manifest).resolve()
    else:
        declared_manifest = declared_manifest.resolve()
    if declared_manifest != manifest_path or not manifest_path.is_file():
        raise MatrixSummaryError(
            "Final report is not bound to the expected artifact_manifest.json"
        )
    try:
        manifest = artifact_validator(manifest_path)
    except (ArtifactFreezeError, FileNotFoundError, OSError, ValueError) as exc:
        raise MatrixSummaryError(
            f"Source artifact freeze is invalid: {exc}"
        ) from exc
    if manifest.get("status") != "complete":
        raise MatrixSummaryError("Source artifact manifest is not complete")
    stored_fingerprint = str(manifest.get("manifest_fingerprint", ""))
    calculated_fingerprint = canonical_json_hash(
        {**manifest, "manifest_fingerprint": None}
    )
    if stored_fingerprint != calculated_fingerprint:
        raise MatrixSummaryError("Source artifact manifest fingerprint is invalid")
    if stored_fingerprint != str(
        report.get("source_artifact_manifest_fingerprint", "")
    ):
        raise MatrixSummaryError(
            "Final report artifact-manifest fingerprint differs from source"
        )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise MatrixSummaryError("Source artifact manifest has no dataset identity")
    if dataset.get("identifier") != dataset_identifier:
        raise MatrixSummaryError("Source manifest dataset identifier differs")
    if dataset.get("dataset_content_fingerprint") != dataset_content_fingerprint:
        raise MatrixSummaryError(
            "Final report dataset fingerprint differs from source manifest"
        )
    source_fingerprint = dataset.get("source_content_fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise MatrixSummaryError(
            "Source artifact manifest has no source_content_fingerprint"
        )
    reported_source = report.get("source_content_fingerprint")
    if reported_source is not None and reported_source != source_fingerprint:
        raise MatrixSummaryError(
            "Final report source fingerprint differs from source manifest"
        )
    checkpoints = manifest.get("output_artifacts", {}).get("checkpoints", {})
    best_intention = checkpoints.get("best_intention")
    if not isinstance(best_intention, Mapping) or str(
        best_intention.get("sha256", "")
    ).lower() != checkpoint_sha256:
        raise MatrixSummaryError(
            "Source artifact manifest does not bind the reported checkpoint"
        )
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha256:
        raise MatrixSummaryError("Reported checkpoint file hash is no longer valid")
    eligibility = dataset.get("window_eligibility")
    if not isinstance(eligibility, Mapping):
        raise MatrixSummaryError("Source manifest lacks common window eligibility")
    endpoint_fingerprints = eligibility.get("endpoint_fingerprints")
    endpoint_counts = eligibility.get("endpoint_counts")
    if not isinstance(endpoint_fingerprints, Mapping) or not isinstance(
        endpoint_counts, Mapping
    ):
        raise MatrixSummaryError("Source manifest lacks endpoint identities")
    test_endpoint_fingerprint = endpoint_fingerprints.get("test")
    if not isinstance(test_endpoint_fingerprint, str) or len(
        test_endpoint_fingerprint
    ) != 64:
        raise MatrixSummaryError("Source manifest has no test endpoint fingerprint")
    test_endpoint_count = _nonnegative_count(
        endpoint_counts.get("test"), field="window_eligibility.endpoint_counts.test"
    )
    return {
        "source_content_fingerprint": source_fingerprint,
        "test_window_endpoint_fingerprint": test_endpoint_fingerprint,
        "test_window_endpoint_count": test_endpoint_count,
    }


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    """Compare persisted metrics while accepting expected Float32 rounding.

    Final-test metrics are emitted by PyTorch as Float32 values, whereas the
    summary recomputes classification rates from integer confusion matrices in
    Python Float64.  A 1e-6 tolerance is far below any reportable metric
    precision, but comfortably exceeds one Float32 unit in the last place for
    probabilities near one.
    """
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _nonnegative_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixSummaryError(f"{field} must be a nonnegative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise MatrixSummaryError(f"{field} must be a nonnegative integer")
    return int(numeric)


def _validate_selection(
    matrix: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    matrix_path: Path,
    project_root: Path,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    if int(selection.get("schema_version", -1)) != 2:
        raise MatrixSummaryError("Validation selection must use schema_version=2")
    if selection.get("complete") is not True:
        raise MatrixSummaryError("Validation selection is incomplete")
    if selection.get("selection_split") != "validation":
        raise MatrixSummaryError("Selection is not validation-only")
    if selection.get("test_metrics_read") is not False:
        raise MatrixSummaryError("Selection manifest read test metrics")
    if selection.get("matrix_id") != matrix.get("matrix_id"):
        raise MatrixSummaryError("Selection matrix_id differs from the matrix")
    if selection.get("dataset_tag") != matrix.get("dataset_tag"):
        raise MatrixSummaryError("Selection dataset_tag differs from the matrix")
    if str(selection.get("matrix_sha256", "")).lower() != sha256_file(matrix_path):
        raise MatrixSummaryError("Selection is bound to another matrix file hash")
    selected_matrix_path = _normalized_run_path(
        str(selection.get("matrix_file", "")), project_root=project_root
    )
    if selected_matrix_path != matrix_path.resolve():
        raise MatrixSummaryError("Selection matrix_file differs from the matrix path")

    expected = {
        (str(entry["id"]), int(seed))
        for entry in matrix["training_experiments"]
        for seed in matrix["seeds"]
    }
    rows = selection.get("final_test_runs")
    if not isinstance(rows, list):
        raise MatrixSummaryError("Selection final_test_runs is not a list")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MatrixSummaryError("Selection contains a non-object authorization")
        try:
            key = (str(row["experiment_id"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MatrixSummaryError("Selection row has no experiment/seed identity") from exc
        if key in indexed:
            raise MatrixSummaryError(f"Duplicate selection authorization for {key}")
        indexed[key] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise MatrixSummaryError(
            f"Selection rows do not exactly match the matrix; missing={missing}, extra={extra}"
        )
    return indexed


def _classification_fields(
    values: Mapping[str, Any],
    *,
    prefix: str,
    expected_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise MatrixSummaryError(f"Missing classification metrics: {prefix}")
    samples = _nonnegative_count(values.get("samples", 0), field=f"{prefix}.samples")
    confusion_value = values.get("confusion_matrix")
    if samples == 0 and confusion_value in (None, []):
        confusion = [[0] * len(expected_names) for _ in expected_names]
    else:
        if not isinstance(confusion_value, list) or len(confusion_value) != len(
            expected_names
        ):
            raise MatrixSummaryError(f"Invalid confusion matrix: {prefix}")
        if any(
            not isinstance(row, list) or len(row) != len(expected_names)
            for row in confusion_value
        ):
            raise MatrixSummaryError(f"Invalid confusion matrix dimensions: {prefix}")
        confusion = [
            [
                _nonnegative_count(
                    item,
                    field=f"{prefix}.confusion_matrix[{row_index}][{column_index}]",
                )
                for column_index, item in enumerate(row)
            ]
            for row_index, row in enumerate(confusion_value)
        ]
    if sum(sum(row) for row in confusion) != samples:
        raise MatrixSummaryError(f"Confusion-matrix denominator differs from samples: {prefix}")
    names = values.get("class_names", list(expected_names))
    if list(names) != list(expected_names):
        raise MatrixSummaryError(
            f"Class order differs for {prefix}: {names!r} != {list(expected_names)!r}"
        )

    result: dict[str, Any] = {
        f"{prefix}_samples": samples,
        f"{prefix}_confusion_matrix": confusion,
    }
    provided = {
        key: (
            None
            if samples == 0 and values.get(key) == []
            else values.get(key)
        )
        for key in ("per_class_precision", "per_class_recall", "per_class_f1")
    }
    support_value = values.get("support")
    if samples == 0 and support_value == []:
        support_value = None
    per_class_values: list[tuple[float, float, float, int, int]] = []
    for index, name in enumerate(expected_names):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class_values.append((precision, recall, f1, support, predicted))
        calculated = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        source_names = {
            "precision": "per_class_precision",
            "recall": "per_class_recall",
            "f1": "per_class_f1",
        }
        for metric, calculated_value in calculated.items():
            sequence = provided[source_names[metric]]
            if sequence is not None:
                if not isinstance(sequence, list) or len(sequence) != len(expected_names):
                    raise MatrixSummaryError(
                        f"Invalid {source_names[metric]} dimensions: {prefix}"
                    )
                if not _same_number(sequence[index], calculated_value):
                    raise MatrixSummaryError(
                        f"Stored {metric} conflicts with confusion matrix: {prefix}/{name}"
                    )
            result[f"{prefix}_{name}_{metric}"] = (
                None if samples == 0 else calculated_value
            )
        if support_value is not None:
            if not isinstance(support_value, list) or len(support_value) != len(
                expected_names
            ):
                raise MatrixSummaryError(f"Invalid support dimensions: {prefix}")
            if _nonnegative_count(
                support_value[index], field=f"{prefix}.support[{index}]"
            ) != support:
                raise MatrixSummaryError(f"Stored support conflicts with confusion: {prefix}")
        result[f"{prefix}_{name}_support"] = support
        result[f"{prefix}_{name}_predicted"] = predicted

    accuracy = (
        sum(confusion[index][index] for index in range(len(expected_names))) / samples
        if samples
        else None
    )
    macro_f1 = (
        sum(item[2] for item in per_class_values) / len(expected_names)
        if samples
        else None
    )
    supported_f1_values = [
        item[2] for item in per_class_values if item[3] > 0
    ]
    macro_f1_supported = (
        sum(supported_f1_values) / len(supported_f1_values)
        if supported_f1_values
        else None
    )
    calculated_summaries = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_f1_supported": macro_f1_supported,
    }
    for metric, calculated_value in calculated_summaries.items():
        stored = values.get(metric)
        if samples == 0:
            if stored is not None and not _same_number(stored, 0.0):
                raise MatrixSummaryError(
                    f"Stored {metric} is nonzero for empty metrics: {prefix}"
                )
        elif stored is None or not _same_number(stored, calculated_value):
            raise MatrixSummaryError(
                f"Stored {metric} conflicts with confusion matrix: {prefix}"
            )
        result[f"{prefix}_{metric}"] = calculated_value
    return result


def _pose_value(values: Mapping[str, Any], canonical: str, legacy: str) -> Any:
    canonical_value = values.get(canonical)
    legacy_value = values.get(legacy)
    if canonical_value is not None and legacy_value is not None and not _same_number(
        canonical_value, legacy_value
    ):
        raise MatrixSummaryError(
            f"Conflicting canonical/legacy pose metric: {canonical} vs {legacy}"
        )
    return canonical_value if canonical_value is not None else legacy_value


def _pose_fields(metrics: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        f"{prefix}_position_mean_cm": (
            metrics.get("position_mean_cm")
            if metrics.get("position_mean_cm") is not None
            else _pose_value(
                metrics, "position_mean_euclidean_error_cm", "position_mae_cm"
            )
        ),
        f"{prefix}_position_median_cm": metrics.get(
            "position_median_cm", metrics.get("position_median_euclidean_error_cm")
        ),
        f"{prefix}_position_rmse_cm": _pose_value(
            metrics,
            "position_root_mean_square_euclidean_error_cm",
            "position_rmse_cm",
        ),
        f"{prefix}_orientation_mean_deg": metrics.get("orientation_mean_deg"),
        f"{prefix}_orientation_median_deg": metrics.get("orientation_median_deg"),
        f"{prefix}_samples": (
            _nonnegative_count(
                metrics["samples"], field=f"{prefix}.samples"
            )
            if metrics.get("samples") is not None
            else (
                _nonnegative_count(
                    metrics["position_samples"],
                    field=f"{prefix}.position_samples",
                )
                if metrics.get("position_samples") is not None
                else None
            )
        ),
    }


def _terminal_fair_fields(
    values: Mapping[str, Any], *, expected_pose_target_denominator: int | None
) -> dict[str, Any]:
    fair = values.get("pose_fair_common")
    if not isinstance(fair, Mapping):
        raise MatrixSummaryError("Terminal result lacks pose_fair_common")
    methods = fair.get("methods")
    if not isinstance(methods, Mapping):
        raise MatrixSummaryError("Terminal pose_fair_common has no methods")
    learned_oracle = methods.get("learned_oracle_hand")
    learned = methods.get("learned_end_to_end")
    persistence = methods.get("persistence")
    if (
        not isinstance(learned_oracle, Mapping)
        or not isinstance(learned, Mapping)
        or not isinstance(persistence, Mapping)
    ):
        raise MatrixSummaryError(
            "Terminal fair comparison requires learned_oracle_hand, "
            "learned_end_to_end and persistence"
        )
    result = {
        "terminal_fair_comparison_role": fair.get("comparison_role"),
        "terminal_fair_receiving_hand_context": fair.get("receiving_hand_context"),
        "terminal_fair_shared_samples": int(fair.get("shared_samples", 0)),
        "terminal_fair_pose_target_denominator": int(
            fair.get("coverage_denominator_pose_targets", 0)
        ),
    }
    result.update(
        _pose_fields(
            learned_oracle, prefix="terminal_fair_learned_oracle_hand"
        )
    )
    result.update(_pose_fields(learned, prefix="terminal_fair_learned_end_to_end"))
    result.update(_pose_fields(persistence, prefix="terminal_fair_persistence"))
    shared = result["terminal_fair_shared_samples"]
    if any(
        result[key] != shared
        for key in (
            "terminal_fair_learned_oracle_hand_samples",
            "terminal_fair_learned_end_to_end_samples",
            "terminal_fair_persistence_samples",
        )
    ):
        raise MatrixSummaryError(
            "Terminal learned/persistence metrics do not use the declared shared samples"
        )
    denominator = result["terminal_fair_pose_target_denominator"]
    if expected_pose_target_denominator is None:
        raise MatrixSummaryError("Terminal result has no pose-target denominator")
    if denominator != expected_pose_target_denominator:
        raise MatrixSummaryError(
            "Terminal fair-common denominator differs from pose_coverage.pose_targets"
        )
    if shared < 0 or shared > denominator:
        raise MatrixSummaryError(
            "Terminal fair-common shared samples are outside the target denominator"
        )
    result["terminal_fair_coverage"] = shared / denominator if denominator else None
    return result


def _terminal_regime_fields(
    values: Mapping[str, Any], *, expected_pose_target_denominator: int
) -> dict[str, Any]:
    regimes = values.get("pose_by_terminal_target_regime")
    if not isinstance(regimes, Mapping):
        raise MatrixSummaryError(
            "Terminal result lacks pose_by_terminal_target_regime"
        )
    expected = (
        "strictly_before_aggregation",
        "partially_overlapping_aggregation",
    )
    result: dict[str, Any] = {
        "terminal_main_pose_reporting_regime": "strictly_before_aggregation",
        "terminal_pooled_pose_is_diagnostic": True,
    }
    denominator_sum = 0
    for name in expected:
        regime = regimes.get(name)
        if not isinstance(regime, Mapping):
            raise MatrixSummaryError(f"Terminal result lacks target regime {name}")
        methods = regime.get("methods")
        if not isinstance(methods, Mapping):
            raise MatrixSummaryError(f"Terminal target regime {name} has no methods")
        denominator = _nonnegative_count(
            regime.get("coverage_denominator_pose_targets", 0),
            field=f"pose_by_terminal_target_regime.{name}.denominator",
        )
        shared = _nonnegative_count(
            regime.get("shared_samples", 0),
            field=f"pose_by_terminal_target_regime.{name}.shared_samples",
        )
        if shared > denominator:
            raise MatrixSummaryError(
                f"Terminal target regime {name} shared samples exceed denominator"
            )
        prefix = f"terminal_fair_{name}"
        result[f"{prefix}_interpretation"] = regime.get("interpretation")
        result[f"{prefix}_pose_target_denominator"] = denominator
        result[f"{prefix}_shared_samples"] = shared
        result[f"{prefix}_coverage"] = shared / denominator if denominator else None
        for method in (
            "learned_oracle_hand",
            "learned_end_to_end",
            "persistence",
        ):
            metrics = methods.get(method)
            if not isinstance(metrics, Mapping):
                raise MatrixSummaryError(
                    f"Terminal target regime {name} lacks method {method}"
                )
            method_fields = _pose_fields(
                metrics, prefix=f"{prefix}_{method}"
            )
            if method_fields[f"{prefix}_{method}_samples"] != shared:
                raise MatrixSummaryError(
                    f"Terminal target regime {name} method {method} is not paired"
                )
            result.update(method_fields)
        denominator_sum += denominator
    if denominator_sum != expected_pose_target_denominator:
        raise MatrixSummaryError(
            "Terminal target-regime denominators do not partition pose targets"
        )
    result["terminal_fair_target_regime_denominator_sum"] = denominator_sum
    return result


def _grouped_candidates(
    root: Path, experiment_id: str, seed: int
) -> list[Path]:
    candidates = (
        root / f"{experiment_id}_seed{seed}" / GROUPED_REPORT_NAME,
        root / f"{experiment_id}_seed{seed}_grouped_metrics.json",
    )
    return [path for path in candidates if path.is_file()]


def _recompute_grouped_fair_common(predictions_path: Path) -> Mapping[str, Any]:
    try:
        raw = pd.read_csv(predictions_path)
        normalized, _ = prepare_prediction_frame(raw)
        methods = discover_pose_methods(normalized)
        fair = summarize_windows(normalized, methods).get("pose_fair_common")
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixSummaryError(
            f"Could not recompute grouped metrics from {predictions_path}: {exc}"
        ) from exc
    if not isinstance(fair, Mapping):
        raise MatrixSummaryError(
            "Prediction CSV cannot produce the required three-method fair-common set"
        )
    return fair


def _assert_grouped_fair_matches_csv(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    for key in (
        "shared_samples",
        "coverage_denominator_pose_targets",
        "coverage",
    ):
        if not _same_number(stored.get(key), recomputed.get(key)):
            raise MatrixSummaryError(
                f"Grouped JSON {key} differs from the hash-bound prediction CSV"
            )
    required = ("learned_oracle_hand", "persistence", "constant_velocity")
    stored_methods = stored.get("methods")
    recomputed_methods = recomputed.get("methods")
    if not isinstance(stored_methods, Mapping) or not isinstance(
        recomputed_methods, Mapping
    ):
        raise MatrixSummaryError("Grouped fair-common methods are missing")
    metric_fields = (
        "position_samples",
        "orientation_samples",
        "coverage_denominator_pose_targets",
        "coverage",
        "position_mean_cm",
        "position_median_cm",
        "position_rmse_cm",
        "orientation_mean_deg",
        "orientation_median_deg",
    )
    for method in required:
        stored_metric = stored_methods.get(method)
        recomputed_metric = recomputed_methods.get(method)
        if not isinstance(stored_metric, Mapping) or not isinstance(
            recomputed_metric, Mapping
        ):
            raise MatrixSummaryError(f"Grouped fair-common method is missing: {method}")
        for field in metric_fields:
            if not _same_number(stored_metric.get(field), recomputed_metric.get(field)):
                raise MatrixSummaryError(
                    f"Grouped JSON {method}.{field} differs from the hash-bound "
                    "prediction CSV"
                )
    stored_fingerprints = stored.get("method_sample_key_fingerprints")
    if stored_fingerprints is not None and stored_fingerprints != recomputed.get(
        "method_sample_key_fingerprints"
    ):
        raise MatrixSummaryError(
            "Grouped JSON sample-key fingerprints differ from the prediction CSV"
        )


def _grouped_t1_fields(
    path: Path | None,
    *,
    checkpoint_sha256: str,
    dataset_content_fingerprint: str | None,
    source_content_fingerprint: str,
    artifact_manifest_fingerprint: str,
    test_endpoint_fingerprint: str,
    test_endpoint_count: int,
    final_test_report_sha256: str,
    final_test_report_fingerprint: str,
    project_root: Path,
) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "t1_fair_common_status": "not_available",
        "t1_grouped_report": None,
        "t1_grouped_report_sha256": None,
        "t1_fair_receiving_hand_context": None,
        "t1_fair_shared_samples": None,
        "t1_fair_pose_target_denominator": None,
        "t1_fair_coverage": None,
        "t1_fair_sample_key_fingerprint": None,
        "t1_baseline_policy": None,
        "t1_baseline_policy_fingerprint": None,
    }
    for method in ("learned_model", "persistence", "constant_velocity"):
        empty.update(
            {
                f"t1_fair_{method}_position_mean_cm": None,
                f"t1_fair_{method}_position_median_cm": None,
                f"t1_fair_{method}_position_rmse_cm": None,
                f"t1_fair_{method}_orientation_mean_deg": None,
                f"t1_fair_{method}_orientation_median_deg": None,
                f"t1_fair_{method}_samples": None,
            }
        )
    if path is None:
        return empty
    report = read_object(path)
    if report.get("schema_version") != GROUPED_PROTOCOL:
        raise MatrixSummaryError(f"Unsupported grouped report schema: {path}")
    binding = report.get("checkpoint_binding")
    if not isinstance(binding, Mapping):
        raise MatrixSummaryError(f"Grouped report has no checkpoint binding: {path}")
    if binding.get("status") != "bound_single_checkpoint" or binding.get(
        "result_role"
    ) != "checkpoint_bound_grouped_primary":
        raise MatrixSummaryError(f"Grouped report is not a bound primary result: {path}")
    if str(binding.get("checkpoint_sha256", "")).lower() != checkpoint_sha256.lower():
        raise MatrixSummaryError("Grouped report comes from a different checkpoint")
    if binding.get("checkpoint_selection_split") != "validation":
        raise MatrixSummaryError("Grouped report checkpoint was not validation-selected")
    if binding.get("split") != "test":
        raise MatrixSummaryError("Grouped report is not for the test split")
    grouped_dataset = binding.get("dataset_content_fingerprint")
    if not dataset_content_fingerprint or grouped_dataset != dataset_content_fingerprint:
        raise MatrixSummaryError("Grouped report uses another dataset fingerprint")

    predictions_value = report.get("predictions_csv")
    if not isinstance(predictions_value, str) or not predictions_value:
        raise MatrixSummaryError("Grouped report has no source prediction CSV")
    predictions_path = Path(predictions_value).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = path.parent / predictions_path
    predictions_path = predictions_path.resolve()
    if not predictions_path.is_file():
        raise MatrixSummaryError(
            f"Grouped report source prediction CSV is unavailable: {predictions_path}"
        )
    predictions_hash = sha256_file(predictions_path)
    if str(report.get("predictions_csv_sha256", "")).lower() != predictions_hash:
        raise MatrixSummaryError("Grouped report prediction CSV hash is stale")
    if str(binding.get("predictions_csv_sha256", "")).lower() != predictions_hash:
        raise MatrixSummaryError("Grouped checkpoint binding has another prediction CSV")
    try:
        actual_prediction_rows = int(len(pd.read_csv(predictions_path)))
        declared_prediction_rows = int(report.get("prediction_rows", -1))
    except (OSError, TypeError, ValueError) as exc:
        raise MatrixSummaryError("Grouped report has an invalid prediction row count") from exc
    if declared_prediction_rows != actual_prediction_rows:
        raise MatrixSummaryError(
            "Grouped report row count differs from the bound prediction CSV"
        )

    sidecar_value = binding.get("source_prediction_report")
    if not isinstance(sidecar_value, str) or not sidecar_value:
        raise MatrixSummaryError("Grouped report has no source prediction sidecar")
    sidecar_path = Path(sidecar_value).expanduser()
    if not sidecar_path.is_absolute():
        sidecar_path = path.parent / sidecar_path
    sidecar_path = sidecar_path.resolve()
    if not sidecar_path.is_file():
        raise MatrixSummaryError(
            f"Grouped source prediction sidecar is unavailable: {sidecar_path}"
        )
    declared_sidecar_hash = str(
        binding.get("source_prediction_report_sha256", "")
    ).lower()
    if declared_sidecar_hash != sha256_file(sidecar_path):
        raise MatrixSummaryError(
            "Grouped checkpoint binding has a stale prediction sidecar hash"
        )
    sidecar = read_object(sidecar_path)
    if sidecar.get("schema_version") != 3 or sidecar.get(
        "report_fingerprint"
    ) != canonical_json_hash({**sidecar, "report_fingerprint": None}):
        raise MatrixSummaryError("Prediction sidecar fingerprint is invalid")
    if sidecar.get("result_role") != "primary_validation_selected_checkpoint":
        raise MatrixSummaryError("Prediction sidecar is not a primary result")
    if str(sidecar.get("checkpoint_sha256", "")).lower() != checkpoint_sha256.lower():
        raise MatrixSummaryError("Prediction sidecar comes from another checkpoint")
    if sidecar.get("checkpoint_selection_split") != "validation" or not str(
        sidecar.get("checkpoint_selection_metric", "")
    ).startswith("validation_"):
        raise MatrixSummaryError("Prediction sidecar checkpoint is not validation-selected")
    if sidecar.get("split") != "test":
        raise MatrixSummaryError("Prediction sidecar is not for test")
    if str(sidecar.get("predictions_csv_sha256", "")).lower() != predictions_hash:
        raise MatrixSummaryError("Prediction sidecar is bound to another CSV")
    if int(sidecar.get("rows", -1)) != actual_prediction_rows:
        raise MatrixSummaryError("Grouped report and prediction sidecar row counts differ")
    if sidecar.get("dataset_content_fingerprint") != dataset_content_fingerprint:
        raise MatrixSummaryError("Prediction sidecar uses another dataset fingerprint")
    if sidecar.get("source_content_fingerprint") != source_content_fingerprint:
        raise MatrixSummaryError("Prediction sidecar uses another source fingerprint")
    freeze_binding = sidecar.get("artifact_freeze")
    if not isinstance(freeze_binding, Mapping) or freeze_binding.get(
        "manifest_fingerprint"
    ) != artifact_manifest_fingerprint:
        raise MatrixSummaryError("Prediction sidecar uses another artifact freeze")
    if sidecar.get("full_split_export") is not True or sidecar.get(
        "sequence_filter"
    ) not in ([], None):
        raise MatrixSummaryError("Grouped primary predictions are a filtered test subset")
    if sidecar.get("frozen_split_endpoint_fingerprint") != test_endpoint_fingerprint:
        raise MatrixSummaryError("Prediction sidecar uses another frozen endpoint set")
    if sidecar.get("exported_endpoint_fingerprint") != test_endpoint_fingerprint:
        raise MatrixSummaryError("Prediction sidecar did not export the full endpoint set")
    if int(sidecar.get("frozen_split_endpoint_count", -1)) != test_endpoint_count:
        raise MatrixSummaryError("Prediction sidecar frozen endpoint count differs")
    if int(sidecar.get("exported_endpoint_count", -1)) != test_endpoint_count:
        raise MatrixSummaryError("Prediction sidecar is not a complete test export")
    endpoint_frame = pd.read_csv(predictions_path)
    if not {"sequence_id", "endpoint_timestamp_ns"}.issubset(
        endpoint_frame.columns
    ):
        raise MatrixSummaryError("Prediction CSV lacks endpoint identity columns")
    endpoint_payload = "\n".join(
        f"{sequence_id}:{int(timestamp)}"
        for sequence_id, timestamp in zip(
            endpoint_frame["sequence_id"].astype(str),
            pd.to_numeric(
                endpoint_frame["endpoint_timestamp_ns"], errors="raise"
            ),
        )
    )
    if hashlib.sha256(endpoint_payload.encode("utf-8")).hexdigest() != (
        test_endpoint_fingerprint
    ):
        raise MatrixSummaryError("Prediction CSV endpoint fingerprint differs")
    final_binding = sidecar.get("final_test_authorization")
    if not isinstance(final_binding, Mapping) or final_binding.get(
        "evaluation_protocol"
    ) != FINAL_TEST_PROTOCOL:
        raise MatrixSummaryError("Prediction sidecar lacks final-test authorization")
    final_path_value = final_binding.get("path")
    if not isinstance(final_path_value, str) or not final_path_value:
        raise MatrixSummaryError("Prediction sidecar has no final-test report path")
    final_path = Path(final_path_value).expanduser()
    if not final_path.is_absolute():
        final_path = sidecar_path.parent / final_path
    final_path = final_path.resolve()
    if not final_path.is_file() or final_binding.get("sha256") != sha256_file(
        final_path
    ):
        raise MatrixSummaryError("Prediction sidecar final-test report hash is stale")
    final_report = read_object(final_path)
    if final_report.get("report_fingerprint") != final_binding.get(
        "report_fingerprint"
    ) or final_report.get("report_fingerprint") != canonical_json_hash(
        {**final_report, "report_fingerprint": None}
    ):
        raise MatrixSummaryError("Prediction sidecar final-test fingerprint is invalid")
    if final_binding.get("sha256") != final_test_report_sha256 or final_binding.get(
        "report_fingerprint"
    ) != final_test_report_fingerprint:
        raise MatrixSummaryError(
            "Prediction sidecar is not bound to this seed's validated final report"
        )
    if str(final_report.get("checkpoint", {}).get("sha256", "")).lower() != (
        checkpoint_sha256.lower()
    ):
        raise MatrixSummaryError("Prediction sidecar final test used another checkpoint")
    try:
        validated_authorization = validate_embedded_final_test_authorization(
            final_report,
            authorization_base=final_path.parent,
            project_root=project_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise MatrixSummaryError(
            f"Prediction sidecar final-test authorization is invalid: {exc}"
        ) from exc
    if final_binding.get("matrix_authorization") != validated_authorization.get(
        "matrix_authorization"
    ):
        raise MatrixSummaryError(
            "Prediction sidecar matrix authorization differs from final test"
        )
    pose_comparison = sidecar.get("pose_comparison")
    baseline_policy = sidecar.get("baseline_policy")
    if not isinstance(pose_comparison, Mapping) or not isinstance(
        baseline_policy, Mapping
    ):
        raise MatrixSummaryError(
            "Prediction sidecar lacks fair-sample or baseline-policy provenance"
        )
    window = report.get("window_level")
    stored_fair = window.get("pose_fair_common") if isinstance(window, Mapping) else None
    if not isinstance(stored_fair, Mapping):
        raise MatrixSummaryError("Grouped report lacks the exact fair-common t+1 set")
    fair = _recompute_grouped_fair_common(predictions_path)
    _assert_grouped_fair_matches_csv(stored_fair, fair)
    methods = fair.get("methods")
    required = ("learned_oracle_hand", "persistence", "constant_velocity")
    if not isinstance(methods, Mapping) or any(
        not isinstance(methods.get(name), Mapping) for name in required
    ):
        raise MatrixSummaryError("Grouped fair-common t+1 methods are incomplete")
    shared = int(fair.get("shared_samples", 0))
    denominator = int(fair.get("coverage_denominator_pose_targets", 0))
    raw_frame = pd.read_csv(predictions_path)
    fair_mask = raw_frame["fair_common"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    )
    recomputed_fair_fingerprint = sample_key_fingerprint(
        raw_frame.loc[fair_mask, "sample_key"].astype(str).tolist()
    )
    if pose_comparison.get("fair_common_sample_key_fingerprint") != (
        recomputed_fair_fingerprint
    ):
        raise MatrixSummaryError(
            "Prediction sidecar fair-common fingerprint differs from its CSV"
        )
    baseline_policy_fingerprint = canonical_json_hash(dict(baseline_policy))
    result = dict(empty)
    result.update(
        {
            "t1_fair_common_status": "available_checkpoint_bound",
            "t1_grouped_report": str(path.resolve()),
            "t1_grouped_report_sha256": sha256_file(path),
            "t1_fair_receiving_hand_context": "ground_truth_receiving_hand",
            "t1_fair_shared_samples": shared,
            "t1_fair_pose_target_denominator": denominator,
            "t1_fair_coverage": shared / denominator if denominator else None,
            "t1_fair_sample_key_fingerprint": recomputed_fair_fingerprint,
            "t1_baseline_policy": dict(baseline_policy),
            "t1_baseline_policy_fingerprint": baseline_policy_fingerprint,
        }
    )
    aliases = {
        "learned_oracle_hand": "learned_model",
        "persistence": "persistence",
        "constant_velocity": "constant_velocity",
    }
    for source, target in aliases.items():
        metric = methods[source]
        method_fields = _pose_fields(metric, prefix=f"t1_fair_{target}")
        if method_fields[f"t1_fair_{target}_samples"] != shared:
            raise MatrixSummaryError(
                f"Grouped fair-common {source} does not use the shared samples"
            )
        result.update(method_fields)
    return result


def _validate_final_report(
    report: Mapping[str, Any],
    *,
    matrix: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_path: Path,
    experiment: Mapping[str, Any],
    seed: int,
    project_root: Path,
    artifact_validator: Callable[[Path], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    experiment_id = str(experiment["id"])
    if int(report.get("schema_version", -1)) != 2:
        raise MatrixSummaryError("Final report uses an obsolete schema")
    stored_report_fingerprint = report.get("report_fingerprint")
    if stored_report_fingerprint != canonical_json_hash(
        {**report, "report_fingerprint": None}
    ):
        raise MatrixSummaryError("Final report fingerprint mismatch")
    if report.get("evaluation_protocol") != FINAL_TEST_PROTOCOL:
        raise MatrixSummaryError("Final report has an unsupported evaluation protocol")
    if report.get("split") != "test":
        raise MatrixSummaryError("Final report does not evaluate test")
    if report.get("test_used_for_model_or_checkpoint_selection") is not False:
        raise MatrixSummaryError("Final report used test for model/checkpoint selection")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise MatrixSummaryError("Final report has no checkpoint identity")
    if checkpoint.get("name") != "best_intention":
        raise MatrixSummaryError("Main result is not the executable best_intention checkpoint")
    if checkpoint.get("selection_split") != "validation" or not str(
        checkpoint.get("selection_metric", "")
    ).startswith("validation_"):
        raise MatrixSummaryError("Final checkpoint was not validation-selected")
    checkpoint_hash = str(checkpoint.get("sha256", "")).lower()
    if len(checkpoint_hash) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_hash
    ):
        raise MatrixSummaryError("Final checkpoint SHA-256 is invalid")

    expected_run_relative = run_directory(matrix, experiment_id, int(seed))
    expected_run = _normalized_run_path(expected_run_relative, project_root=project_root)
    source_run = _normalized_run_path(
        str(report.get("source_run", "")), project_root=project_root
    )
    if source_run != expected_run:
        raise MatrixSummaryError("Final report source_run differs from the matrix run")
    fingerprint = str(report.get("source_artifact_manifest_fingerprint", ""))
    if not fingerprint:
        raise MatrixSummaryError("Final report has no artifact-manifest fingerprint")
    if report.get("dataset_identifier") != matrix.get("dataset_tag"):
        raise MatrixSummaryError("Final report dataset identifier differs from matrix")
    dataset_fingerprint = report.get("dataset_content_fingerprint")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint.strip():
        raise MatrixSummaryError("Final report has no dataset_content_fingerprint")

    authorization = report.get("matrix_authorization")
    if not isinstance(authorization, Mapping):
        raise MatrixSummaryError("Final report has no matrix authorization")
    if authorization.get("matrix_id") != matrix.get("matrix_id"):
        raise MatrixSummaryError("Final report matrix authorization differs from matrix")
    if authorization.get("experiment_id") != experiment_id:
        raise MatrixSummaryError("Final report authorization has another experiment")
    if int(authorization.get("seed", -1)) != int(seed):
        raise MatrixSummaryError("Final report authorization has another seed")
    selection_hash = sha256_file(selection_path)
    if str(authorization.get("selection_file_sha256", "")).lower() != selection_hash:
        raise MatrixSummaryError("Final report is bound to another selection file")
    if authorization.get("authorized_checkpoint_sha256") != checkpoint_hash:
        raise MatrixSummaryError("Final report mixes authorization and checkpoint hashes")
    if authorization.get("test_metrics_read_during_authorization") is not False:
        raise MatrixSummaryError("Final report authorization read test metrics")

    try:
        authorized = validate_final_test_authorization(
            dict(selection),
            experiment_id=experiment_id,
            seed=int(seed),
            run_dir=expected_run_relative,
            checkpoint_sha256=checkpoint_hash,
            artifact_manifest_fingerprint=fingerprint,
        )
    except (TypeError, ValueError) as exc:
        raise MatrixSummaryError(f"Final report is not exactly authorized: {exc}") from exc
    if int(checkpoint.get("epoch", -1)) != int(
        authorized.get("checkpoint_epoch", -2)
    ):
        raise MatrixSummaryError("Checkpoint epoch differs from validation authorization")
    if checkpoint.get("selection_metric") != authorized.get(
        "checkpoint_selection_metric"
    ):
        raise MatrixSummaryError(
            "Checkpoint selection_metric differs from validation authorization"
        )
    if not _same_number(
        checkpoint.get("selection_value"),
        authorized.get("checkpoint_selection_value"),
    ):
        raise MatrixSummaryError(
            "Checkpoint selection_value differs from validation authorization"
        )

    selection_checkpoint = Path(str(authorized.get("checkpoint_path", "")))
    if not selection_checkpoint.is_absolute():
        selection_checkpoint = expected_run / selection_checkpoint
    report_checkpoint = _normalized_run_path(
        str(checkpoint.get("path", "")), project_root=project_root
    )
    if selection_checkpoint.resolve() != report_checkpoint:
        raise MatrixSummaryError("Checkpoint path differs from validation authorization")
    manifest_binding = _validate_source_manifest_binding(
        report,
        expected_run=expected_run,
        checkpoint_path=report_checkpoint,
        checkpoint_sha256=checkpoint_hash,
        dataset_identifier=str(matrix["dataset_tag"]),
        dataset_content_fingerprint=str(dataset_fingerprint),
        artifact_validator=artifact_validator,
    )
    values = report.get("test_metrics")
    if not isinstance(values, Mapping):
        raise MatrixSummaryError("Final report has no test_metrics object")
    task_semantics = report.get("training_task_semantics")
    if not isinstance(task_semantics, Mapping) or not isinstance(
        task_semantics.get("future_pose_loss_enabled"), bool
    ):
        raise MatrixSummaryError(
            "Final report lacks explicit future-pose training semantics"
        )
    return values, {
        "expected_run_relative": expected_run_relative,
        "checkpoint_sha256": checkpoint_hash,
        "artifact_manifest_fingerprint": fingerprint,
        "selection_sha256": selection_hash,
        **manifest_binding,
        "future_pose_loss_enabled": bool(
            task_semantics["future_pose_loss_enabled"]
        ),
        "pose_metrics_role": task_semantics.get("pose_metrics_role"),
    }


def _seed_row(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    matrix: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_path: Path,
    experiment: Mapping[str, Any],
    seed: int,
    project_root: Path,
    grouped_path: Path | None,
    artifact_validator: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    values, identity = _validate_final_report(
        report,
        matrix=matrix,
        selection=selection,
        selection_path=selection_path,
        experiment=experiment,
        seed=seed,
        project_root=project_root,
        artifact_validator=artifact_validator,
    )
    task_role = (
        "secondary_terminal_endpose"
        if experiment.get("family") == "secondary_endpose"
        else "primary_t_plus_1_future_wrist"
    )
    checkpoint = report["checkpoint"]
    row: dict[str, Any] = {
        "result_semantics": SEED_RESULT_SEMANTICS,
        "matrix_id": matrix["matrix_id"],
        "dataset_tag": matrix["dataset_tag"],
        "experiment_id": str(experiment["id"]),
        "family": experiment.get("family"),
        "factor": experiment.get("factor"),
        "variant": experiment.get("variant"),
        "thesis_task_role": task_role,
        "seed": int(seed),
        "model_type": report.get("model_type"),
        "trainable_parameters": report.get("trainable_parameters"),
        "source_final_test_report": str(report_path.resolve()),
        "source_final_test_report_sha256": sha256_file(report_path),
        "source_run": identity["expected_run_relative"],
        "artifact_manifest_fingerprint": identity["artifact_manifest_fingerprint"],
        "dataset_content_fingerprint": report.get("dataset_content_fingerprint"),
        "source_content_fingerprint": identity["source_content_fingerprint"],
        "test_window_endpoint_fingerprint": identity[
            "test_window_endpoint_fingerprint"
        ],
        "test_window_endpoint_count": identity["test_window_endpoint_count"],
        "checkpoint_name": "best_intention",
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": checkpoint["selection_metric"],
        "checkpoint_selection_value": checkpoint.get("selection_value"),
        "selection_file_sha256": identity["selection_sha256"],
        "future_pose_loss_enabled": identity["future_pose_loss_enabled"],
        "pose_metrics_role": identity["pose_metrics_role"],
    }
    row.update(
        _classification_fields(
            values.get("assistance", {}),
            prefix="test_assistance",
            expected_names=ASSISTANCE_NAMES,
        )
    )
    row.update(
        _classification_fields(
            values.get("intention", {}),
            prefix="test_intention",
            expected_names=INTENTION_NAMES,
        )
    )
    row.update(
        _classification_fields(
            values.get("assistance_type", {}),
            prefix="test_assistance_type",
            expected_names=ASSISTANCE_TYPE_NAMES,
        )
    )
    row.update(
        _classification_fields(
            values.get("receiving_hand", {}),
            prefix="test_receiving_hand",
            expected_names=HAND_NAMES,
        )
    )

    end_to_end = values.get("pose_end_to_end")
    has_end_to_end = isinstance(end_to_end, Mapping)
    native_pose = end_to_end if has_end_to_end else values.get("pose", {})
    native_pose_mapping = native_pose if isinstance(native_pose, Mapping) else {}
    internal_fair = values.get("pose_fair_common")
    internal_fair_methods = (
        internal_fair.get("methods")
        if isinstance(internal_fair, Mapping)
        else None
    )
    fixed_pose = (
        internal_fair_methods.get("learned_end_to_end")
        if isinstance(internal_fair_methods, Mapping)
        else None
    )
    standard_fixed_pose = values.get("pose_fixed_both_references")
    if not isinstance(fixed_pose, Mapping) and isinstance(
        standard_fixed_pose, Mapping
    ):
        fixed_pose = standard_fixed_pose
    uses_fixed_pose_cohort = isinstance(fixed_pose, Mapping)
    fixed_pose_sample_fingerprint = (
        internal_fair.get("sample_key_fingerprint")
        if isinstance(internal_fair, Mapping)
        else (
            standard_fixed_pose.get("sample_key_fingerprint")
            if isinstance(standard_fixed_pose, Mapping)
            else None
        )
    )
    pose_mapping = fixed_pose if uses_fixed_pose_cohort else native_pose_mapping
    native_pose_fields = _pose_fields(
        native_pose_mapping, prefix="diagnostic_native_end_to_end_pose"
    )
    raw_pose_fields = _pose_fields(
        native_pose_mapping, prefix="diagnostic_untrained_pose"
    )
    if identity["future_pose_loss_enabled"]:
        row["pose_output_semantics"] = (
            "learned_end_to_end_fixed_both_reference_cohort"
            if uses_fixed_pose_cohort
            else (
                "learned_end_to_end_predicted_receiving_hand"
                if has_end_to_end
                else "standard_baseline_single_pose_head"
            )
        )
        row.update(_pose_fields(pose_mapping, prefix="test_pose"))
        if uses_fixed_pose_cohort:
            if not isinstance(fixed_pose_sample_fingerprint, str) or len(
                fixed_pose_sample_fingerprint
            ) != 64:
                raise MatrixSummaryError(
                    "Fixed pose cohort lacks a sample-key fingerprint"
                )
            row.update(native_pose_fields)
            row["test_pose_cross_model_comparability"] = (
                "fixed_model_independent_both_hand_references_valid_cohort"
            )
            row["test_pose_sample_key_fingerprint"] = (
                fixed_pose_sample_fingerprint
            )
        else:
            row["test_pose_cross_model_comparability"] = (
                "common_target_windows_for_standard_single_pose_head"
            )
    else:
        row["pose_output_semantics"] = (
            "untrained_pose_head_diagnostic_excluded_from_main_results"
        )
        row.update(_pose_fields({}, prefix="test_pose"))
        row.update(raw_pose_fields)
    coverage = values.get("pose_coverage", {})
    if not isinstance(coverage, Mapping):
        coverage = {}
    denominator = coverage.get("pose_targets", coverage.get("future_targets"))
    if denominator is None:
        raise MatrixSummaryError("Main pose result has no target denominator")
    denominator = _nonnegative_count(
        denominator, field="pose_coverage.pose_targets"
    )
    pose_samples_for_validation = (
        row["test_pose_samples"]
        if identity["future_pose_loss_enabled"]
        else raw_pose_fields["diagnostic_untrained_pose_samples"]
    )
    if pose_samples_for_validation is None or pose_samples_for_validation > denominator:
        raise MatrixSummaryError(
            "Main pose samples are outside the pose-target denominator"
        )
    oracle_reference_valid = coverage.get("oracle_reference_valid")
    if oracle_reference_valid is not None:
        oracle_reference_valid = _nonnegative_count(
            oracle_reference_valid,
            field="pose_coverage.oracle_reference_valid",
        )
        if oracle_reference_valid > denominator:
            raise MatrixSummaryError(
                "Oracle reference count exceeds the pose-target denominator"
            )
    predicted_reference_valid = coverage.get("predicted_reference_valid")
    if predicted_reference_valid is not None:
        predicted_reference_valid = _nonnegative_count(
            predicted_reference_valid,
            field="pose_coverage.predicted_reference_valid",
        )
        native_samples = native_pose_fields[
            "diagnostic_native_end_to_end_pose_samples"
        ]
        if predicted_reference_valid != native_samples:
            raise MatrixSummaryError(
                "End-to-end pose samples differ from predicted_reference_valid"
            )
    predicted_pose_valid = coverage.get("predicted_pose_valid")
    if predicted_pose_valid is not None:
        predicted_pose_valid = _nonnegative_count(
            predicted_pose_valid,
            field="pose_coverage.predicted_pose_valid",
        )
        if predicted_pose_valid != pose_samples_for_validation:
            raise MatrixSummaryError(
                "Standard pose samples differ from predicted_pose_valid"
            )
    stored_coverage = coverage.get("coverage")
    calculated_coverage = (
        pose_samples_for_validation / denominator if denominator else None
    )
    if (not uses_fixed_pose_cohort) and stored_coverage is not None and not _same_number(
        stored_coverage, calculated_coverage
    ):
        raise MatrixSummaryError("Stored main pose coverage is inconsistent")
    if uses_fixed_pose_cohort and identity["future_pose_loss_enabled"]:
        if isinstance(internal_fair, Mapping):
            fixed_shared = _nonnegative_count(
                internal_fair.get("shared_samples", 0),
                field="pose_fair_common.shared_samples",
            )
            fixed_denominator = _nonnegative_count(
                internal_fair.get("coverage_denominator_pose_targets", 0),
                field="pose_fair_common.coverage_denominator_pose_targets",
            )
        else:
            fixed_shared = _nonnegative_count(
                standard_fixed_pose.get("samples", 0),
                field="pose_fixed_both_references.samples",
            )
            fixed_denominator = _nonnegative_count(
                standard_fixed_pose.get("coverage_denominator_pose_targets", 0),
                field=(
                    "pose_fixed_both_references."
                    "coverage_denominator_pose_targets"
                ),
            )
        if fixed_denominator != denominator:
            raise MatrixSummaryError(
                "Fixed pose cohort denominator differs from main pose coverage"
            )
        if fixed_shared != pose_samples_for_validation:
            raise MatrixSummaryError(
                "Fixed pose cohort sample count differs from main pose fields"
            )
    row.update(
        {
            "test_pose_target_denominator": denominator,
            "test_pose_oracle_reference_valid": oracle_reference_valid,
            "test_pose_predicted_reference_valid": predicted_reference_valid,
            "test_pose_coverage": (
                calculated_coverage
                if identity["future_pose_loss_enabled"]
                else None
            ),
            "test_pose_coverage_status": (
                "reported_denominator"
                if identity["future_pose_loss_enabled"]
                else "untrained_pose_head_excluded"
            ),
            "diagnostic_untrained_pose_coverage": (
                None
                if identity["future_pose_loss_enabled"]
                else calculated_coverage
            ),
        }
    )
    if task_role == "secondary_terminal_endpose":
        if not identity["future_pose_loss_enabled"]:
            raise MatrixSummaryError(
                "Terminal main experiment has a disabled primary pose loss"
            )
        row.update(
            _terminal_fair_fields(
                values, expected_pose_target_denominator=denominator
            )
        )
        row.update(
            _terminal_regime_fields(
                values, expected_pose_target_denominator=denominator
            )
        )
        row.update(
            _grouped_t1_fields(
                None,
                checkpoint_sha256=identity["checkpoint_sha256"],
                dataset_content_fingerprint=report.get(
                    "dataset_content_fingerprint"
                ),
                source_content_fingerprint=identity[
                    "source_content_fingerprint"
                ],
                artifact_manifest_fingerprint=identity[
                    "artifact_manifest_fingerprint"
                ],
                test_endpoint_fingerprint=identity[
                    "test_window_endpoint_fingerprint"
                ],
                test_endpoint_count=identity["test_window_endpoint_count"],
                final_test_report_sha256=sha256_file(report_path),
                final_test_report_fingerprint=str(report["report_fingerprint"]),
                project_root=project_root,
            )
        )
    else:
        if identity["future_pose_loss_enabled"]:
            row.update(
                _grouped_t1_fields(
                    grouped_path,
                    checkpoint_sha256=identity["checkpoint_sha256"],
                    dataset_content_fingerprint=report.get("dataset_content_fingerprint"),
                    source_content_fingerprint=identity[
                        "source_content_fingerprint"
                    ],
                    artifact_manifest_fingerprint=identity[
                        "artifact_manifest_fingerprint"
                    ],
                    test_endpoint_fingerprint=identity[
                        "test_window_endpoint_fingerprint"
                    ],
                    test_endpoint_count=identity[
                        "test_window_endpoint_count"
                    ],
                    final_test_report_sha256=sha256_file(report_path),
                    final_test_report_fingerprint=str(
                        report["report_fingerprint"]
                    ),
                    project_root=project_root,
                )
            )
        else:
            row.update(
                _grouped_t1_fields(
                    None,
                    checkpoint_sha256=identity["checkpoint_sha256"],
                    dataset_content_fingerprint=report.get("dataset_content_fingerprint"),
                    source_content_fingerprint=identity[
                        "source_content_fingerprint"
                    ],
                    artifact_manifest_fingerprint=identity[
                        "artifact_manifest_fingerprint"
                    ],
                    test_endpoint_fingerprint=identity[
                        "test_window_endpoint_fingerprint"
                    ],
                    test_endpoint_count=identity[
                        "test_window_endpoint_count"
                    ],
                    final_test_report_sha256=sha256_file(report_path),
                    final_test_report_fingerprint=str(
                        report["report_fingerprint"]
                    ),
                    project_root=project_root,
                )
            )
            row["t1_fair_common_status"] = (
                "untrained_pose_head_diagnostic_excluded"
            )
    return row


def build_matrix_summary(
    *,
    matrix: Mapping[str, Any],
    matrix_path: Path,
    selection_path: Path,
    final_test_dir: Path,
    postprocess_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    artifact_validator: Callable[[Path], Mapping[str, Any]] = (
        validate_historical_artifact_freeze
    ),
) -> dict[str, Any]:
    """Validate all matrix cells and return seed plus aggregate result objects."""

    selection_path = selection_path.resolve()
    final_test_dir = final_test_dir.resolve()
    selection = read_object(selection_path)
    _validate_selection(
        matrix,
        selection,
        matrix_path=matrix_path.resolve(),
        project_root=project_root,
    )
    postprocessing = matrix.get("postprocessing", {})
    required_t1 = {
        str(value)
        for value in postprocessing.get("required_t1_experiments", [])
    }
    if required_t1 and postprocess_root is None:
        raise MatrixSummaryError(
            "Matrix requires checkpoint-bound t+1 postprocessing, but no "
            "postprocess root was provided"
        )
    expected_names = {
        f"{entry['id']}_seed{int(seed)}.json"
        for entry in matrix["training_experiments"]
        for seed in matrix["seeds"]
    }
    if not final_test_dir.is_dir():
        raise MatrixSummaryError(f"Final-test directory does not exist: {final_test_dir}")
    unexpected_executable = []
    for path in final_test_dir.glob("*.json"):
        if path.name in expected_names:
            continue
        try:
            candidate = read_object(path)
        except MatrixSummaryError:
            continue
        if candidate.get("evaluation_protocol") == FINAL_TEST_PROTOCOL:
            unexpected_executable.append(path.name)
    if unexpected_executable:
        raise MatrixSummaryError(
            "Unexpected/duplicate executable final reports: "
            + ", ".join(sorted(unexpected_executable))
        )

    rows: list[dict[str, Any]] = []
    for experiment in matrix["training_experiments"]:
        experiment_rows: list[dict[str, Any]] = []
        for seed_value in matrix["seeds"]:
            seed = int(seed_value)
            report_path = final_test_dir / f"{experiment['id']}_seed{seed}.json"
            if not report_path.is_file():
                raise MatrixSummaryError(f"Missing final-test report: {report_path}")
            grouped_path = None
            if postprocess_root is not None:
                candidates = _grouped_candidates(
                    postprocess_root.resolve(), str(experiment["id"]), seed
                )
                if len(candidates) > 1:
                    raise MatrixSummaryError(
                        f"Multiple grouped reports for {experiment['id']} seed {seed}"
                    )
                grouped_path = candidates[0] if candidates else None
            if str(experiment["id"]) in required_t1 and grouped_path is None:
                raise MatrixSummaryError(
                    "Missing required t+1 grouped report for "
                    f"{experiment['id']} seed {seed}"
                )
            experiment_rows.append(
                _seed_row(
                    read_object(report_path),
                    report_path=report_path,
                    matrix=matrix,
                    selection=selection,
                    selection_path=selection_path,
                    experiment=experiment,
                    seed=seed,
                    project_root=project_root,
                    grouped_path=grouped_path,
                    artifact_validator=artifact_validator,
                )
            )
        fingerprints = {
            str(row["dataset_content_fingerprint"]) for row in experiment_rows
        }
        if len(fingerprints) != 1:
            raise MatrixSummaryError(
                f"Dataset content fingerprint differs across seeds for "
                f"{experiment['id']}: {sorted(fingerprints)}"
            )
        rows.extend(experiment_rows)
    source_fingerprints = {
        str(row["source_content_fingerprint"]) for row in rows
    }
    if len(source_fingerprints) != 1:
        raise MatrixSummaryError(
            "Source content fingerprint differs across matrix cells: "
            + ", ".join(sorted(source_fingerprints))
        )
    endpoint_fingerprints = {
        str(row["test_window_endpoint_fingerprint"]) for row in rows
    }
    endpoint_counts = {int(row["test_window_endpoint_count"]) for row in rows}
    if len(endpoint_fingerprints) != 1 or len(endpoint_counts) != 1:
        raise MatrixSummaryError(
            "Test window endpoint set differs across matrix cells"
        )
    grouped_rows = [
        row
        for row in rows
        if row.get("t1_fair_common_status") == "available_checkpoint_bound"
    ]
    expected_required_grouped = len(required_t1) * len(matrix["seeds"])
    actual_required_grouped = sum(
        row.get("experiment_id") in required_t1
        and row.get("t1_fair_common_status") == "available_checkpoint_bound"
        for row in rows
    )
    if actual_required_grouped != expected_required_grouped:
        raise MatrixSummaryError(
            "Required t+1 postprocessing is incomplete: "
            f"{actual_required_grouped}/{expected_required_grouped}"
        )
    if grouped_rows:
        grouped_sample_fingerprints = {
            str(row["t1_fair_sample_key_fingerprint"]) for row in grouped_rows
        }
        grouped_policy_fingerprints = {
            str(row["t1_baseline_policy_fingerprint"]) for row in grouped_rows
        }
        if len(grouped_sample_fingerprints) != 1:
            raise MatrixSummaryError(
                "Grouped t+1 reports use different fair-common sample sets"
            )
        if len(grouped_policy_fingerprints) != 1:
            raise MatrixSummaryError(
                "Grouped t+1 reports use different baseline policies"
            )
    for task_role in {
        str(row["thesis_task_role"])
        for row in rows
        if row.get("future_pose_loss_enabled") is True
    }:
        pose_enabled_rows = [
            row
            for row in rows
            if row.get("future_pose_loss_enabled") is True
            and row.get("thesis_task_role") == task_role
        ]
        fixed_pose_fingerprints = {
            row.get("test_pose_sample_key_fingerprint")
            for row in pose_enabled_rows
        }
        fixed_pose_counts = {
            row.get("test_pose_samples") for row in pose_enabled_rows
        }
        if None in fixed_pose_fingerprints or len(fixed_pose_fingerprints) != 1:
            raise MatrixSummaryError(
                f"Pose-enabled {task_role} cells do not share one fixed-cohort "
                "fingerprint"
            )
        if None in fixed_pose_counts or len(fixed_pose_counts) != 1:
            raise MatrixSummaryError(
                f"Pose-enabled {task_role} cells do not share one fixed-cohort "
                "sample count"
            )
    expected_count = len(matrix["training_experiments"]) * len(matrix["seeds"])
    if len(rows) != expected_count:
        raise MatrixSummaryError("Final matrix is incomplete")
    aggregates = aggregate_seed_rows(rows)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reporting_protocol": "thesis_v2_authoritative_single_checkpoint_matrix_v1",
        "result_semantics": SEED_RESULT_SEMANTICS,
        "matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256_file(matrix_path.resolve()),
            "matrix_id": matrix["matrix_id"],
            "dataset_tag": matrix["dataset_tag"],
        },
        "validation_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selection_split": "validation",
            "test_metrics_read": False,
        },
        "one_row_one_checkpoint": True,
        "common_source_content_fingerprint": next(iter(source_fingerprints)),
        "common_test_window_endpoint_fingerprint": next(
            iter(endpoint_fingerprints)
        ),
        "common_test_window_endpoint_count": next(iter(endpoint_counts)),
        "seed_row_count": len(rows),
        "expected_seed_row_count": expected_count,
        "seed_rows": rows,
        "seed_aggregation": {
            "result_semantics": AGGREGATE_RESULT_SEMANTICS,
            "standard_deviation_definition": (
                "sample standard deviation (n-1) across independently trained seeds; "
                "not population uncertainty and not a cluster-bootstrap interval"
            ),
            "rows": aggregates,
        },
    }


def _numeric_metric_items(row: Mapping[str, Any]) -> Iterable[tuple[str, float]]:
    prefixes = ("test_", "terminal_fair_", "t1_fair_")
    excluded_suffixes = ("_confusion_matrix",)
    for key, value in row.items():
        if not key.startswith(prefixes) or key.endswith(excluded_suffixes):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            yield key, float(value)


def aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_id"])].append(row)
    aggregates = []
    for experiment_id, group in grouped.items():
        first = group[0]
        values: dict[str, list[float]] = defaultdict(list)
        for row in group:
            for key, value in _numeric_metric_items(row):
                values[key].append(value)
        metrics = {
            key: {
                "mean": statistics.fmean(items),
                "std": statistics.stdev(items) if len(items) >= 2 else None,
                "n": len(items),
            }
            for key, items in sorted(values.items())
        }
        summed_confusions = {}
        for key in (
            "test_assistance_confusion_matrix",
            "test_intention_confusion_matrix",
            "test_assistance_type_confusion_matrix",
            "test_receiving_hand_confusion_matrix",
        ):
            matrices = [row.get(key) for row in group]
            if all(isinstance(matrix, list) for matrix in matrices):
                size = len(matrices[0])
                summed_confusions[key] = [
                    [
                        sum(int(matrix[i][j]) for matrix in matrices)
                        for j in range(size)
                    ]
                    for i in range(size)
                ]
        aggregates.append(
            {
                "result_semantics": AGGREGATE_RESULT_SEMANTICS,
                "experiment_id": experiment_id,
                "family": first.get("family"),
                "factor": first.get("factor"),
                "variant": first.get("variant"),
                "thesis_task_role": first.get("thesis_task_role"),
                "seeds": sorted(int(row["seed"]) for row in group),
                "seed_count": len(group),
                "checkpoint_hashes": [str(row["checkpoint_sha256"]) for row in group],
                "metrics": metrics,
                "summed_confusion_matrices": summed_confusions,
                "grouped_t1_reports_available": sum(
                    row.get("t1_fair_common_status")
                    == "available_checkpoint_bound"
                    for row in group
                ),
            }
        )
    return aggregates


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def flattened_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        value = {
            key: item
            for key, item in row.items()
            if key not in {"metrics", "summed_confusion_matrices"}
        }
        for metric, summary in row["metrics"].items():
            value[f"{metric}_mean"] = summary["mean"]
            value[f"{metric}_std"] = summary["std"]
            value[f"{metric}_n"] = summary["n"]
        for name, matrix in row["summed_confusion_matrices"].items():
            value[f"summed_{name}"] = matrix
        flattened.append(value)
    return flattened


def _mean_std(row: Mapping[str, Any], metric: str) -> str:
    summary = row.get("metrics", {}).get(metric)
    if not isinstance(summary, Mapping) or summary.get("mean") is None:
        return "—"
    mean = float(summary["mean"])
    std = summary.get("std")
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {float(std):.4f}"


def markdown_report(
    summary: Mapping[str, Any], *, seed_results_sha256: str | None = None
) -> str:
    lines = [
        "# Thesis-v2 final matrix",
        "",
        (
            "Every seed row is bound to exactly one executable validation-selected "
            "`best_intention` checkpoint. Aggregate rows are mean ± sample SD across "
            "seeds and are not executable checkpoints."
        ),
        (
            f"Matrix SHA-256: `{summary['matrix']['sha256']}`; validation-selection "
            f"SHA-256: `{summary['validation_selection']['sha256']}`."
        ),
        (
            f"Checkpoint-coherent seed-results SHA-256: `{seed_results_sha256}`."
            if seed_results_sha256 is not None
            else ""
        ),
        "",
        "| Task | Experiment | Seeds | Intent macro-F1 | Assistance macro-F1 | Receiving-hand macro-F1 | Pose mean cm | Orientation mean deg | Pose coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["seed_aggregation"]["rows"]:
        terminal_row = row["thesis_task_role"] == "secondary_terminal_endpose"
        pose_metric = (
            "terminal_fair_strictly_before_aggregation_learned_end_to_end_position_mean_cm"
            if terminal_row
            else "test_pose_position_mean_cm"
        )
        orientation_metric = (
            "terminal_fair_strictly_before_aggregation_learned_end_to_end_orientation_mean_deg"
            if terminal_row
            else "test_pose_orientation_mean_deg"
        )
        coverage_metric = (
            "terminal_fair_strictly_before_aggregation_coverage"
            if terminal_row
            else "test_pose_coverage"
        )
        lines.append(
            "| {task} | {experiment} | {seeds} | {intent} | {assistance} | {hand} | {pose} | {orientation} | {coverage} |".format(
                task=str(row["thesis_task_role"]),
                experiment=str(row["experiment_id"]),
                seeds=int(row["seed_count"]),
                intent=_mean_std(row, "test_intention_macro_f1"),
                assistance=_mean_std(row, "test_assistance_macro_f1"),
                hand=_mean_std(row, "test_receiving_hand_macro_f1"),
                pose=_mean_std(row, pose_metric),
                orientation=_mean_std(row, orientation_metric),
                coverage=_mean_std(row, coverage_metric),
            )
        )
    terminal = [
        row
        for row in summary["seed_aggregation"]["rows"]
        if row["thesis_task_role"] == "secondary_terminal_endpose"
    ]
    if terminal:
        lines.extend(
            [
                "",
                "## Secondary terminal/endpose paired diagnostic",
                "",
                "The overview uses the strictly-before-aggregation (pure-future) regime. The pooled row below is diagnostic because it mixes pure forecasting and partial-target-evidence estimation.",
                "",
                "| Experiment | Learned GT-hand mean cm | Learned end-to-end mean cm | Persistence mean cm | Shared samples | Coverage |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in terminal:
            lines.append(
                "| {experiment} | {learned_oracle} | {learned_end_to_end} | {persistence} | {samples} | {coverage} |".format(
                    experiment=row["experiment_id"],
                    learned_oracle=_mean_std(
                        row, "terminal_fair_learned_oracle_hand_position_mean_cm"
                    ),
                    learned_end_to_end=_mean_std(
                        row, "terminal_fair_learned_end_to_end_position_mean_cm"
                    ),
                    persistence=_mean_std(
                        row, "terminal_fair_persistence_position_mean_cm"
                    ),
                    samples=_mean_std(row, "terminal_fair_shared_samples"),
                    coverage=_mean_std(row, "terminal_fair_coverage"),
                )
            )
        lines.extend(
            [
                "",
                "| Experiment | Target regime | Learned GT-hand mean cm | Learned end-to-end mean cm | Persistence (GT-hand) mean cm | Shared samples | Coverage |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in terminal:
            for regime, label in (
                ("strictly_before_aggregation", "pure future"),
                ("partially_overlapping_aggregation", "partial target evidence"),
            ):
                prefix = f"terminal_fair_{regime}"
                lines.append(
                    "| {experiment} | {label} | {learned_oracle} | {learned} | {persistence} | {samples} | {coverage} |".format(
                        experiment=row["experiment_id"],
                        label=label,
                        learned_oracle=_mean_std(
                            row, f"{prefix}_learned_oracle_hand_position_mean_cm"
                        ),
                        learned=_mean_std(
                            row, f"{prefix}_learned_end_to_end_position_mean_cm"
                        ),
                        persistence=_mean_std(
                            row, f"{prefix}_persistence_position_mean_cm"
                        ),
                        samples=_mean_std(row, f"{prefix}_shared_samples"),
                        coverage=_mean_std(row, f"{prefix}_coverage"),
                    )
                )
    lines.extend(
        [
            "",
            "## t+1 paired baseline availability",
            "",
            (
                "Persistence/constant-velocity/learned values are included only when a "
                "checkpoint-bound grouped prediction report exists. Missing sidecars "
                "remain unavailable and are not estimated from other reports."
            ),
            "",
            "| Experiment | Bound seed reports | Learned (GT hand) mean cm | Persistence mean cm | Constant velocity mean cm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["seed_aggregation"]["rows"]:
        if row["thesis_task_role"] != "primary_t_plus_1_future_wrist":
            continue
        lines.append(
            "| {experiment} | {available}/{total} | {learned} | {persistence} | {velocity} |".format(
                experiment=row["experiment_id"],
                available=row["grouped_t1_reports_available"],
                total=row["seed_count"],
                learned=_mean_std(
                    row, "t1_fair_learned_model_position_mean_cm"
                ),
                persistence=_mean_std(
                    row, "t1_fair_persistence_position_mean_cm"
                ),
                velocity=_mean_std(
                    row, "t1_fair_constant_velocity_position_mean_cm"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    summary: Mapping[str, Any], output_dir: Path, *, overwrite: bool = False
) -> dict[str, Path]:
    seed_json = output_dir / "final_seed_results.json"
    seed_csv = output_dir / "final_seed_results.csv"
    aggregate_json = output_dir / "final_seed_aggregates.json"
    aggregate_csv = output_dir / "final_seed_aggregates.csv"
    markdown = output_dir / "FINAL_MATRIX_SUMMARY.md"
    artifact_manifest = output_dir / "summary_artifact_manifest.json"
    targets = (
        seed_json,
        seed_csv,
        aggregate_json,
        aggregate_csv,
        markdown,
        artifact_manifest,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing thesis summary outputs: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_payload = {
        key: value
        for key, value in summary.items()
        if key not in {"seed_aggregation"}
    }
    seed_json.write_text(
        json.dumps(seed_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(seed_csv, summary["seed_rows"])
    aggregate_payload = {
        "schema_version": summary["schema_version"],
        "created_at": summary["created_at"],
        "reporting_protocol": summary["reporting_protocol"],
        "matrix": summary["matrix"],
        "validation_selection": summary["validation_selection"],
        "one_row_one_checkpoint": summary["one_row_one_checkpoint"],
        "source_seed_results_sha256": sha256_file(seed_json),
        **summary["seed_aggregation"],
    }
    aggregate_json.write_text(
        json.dumps(
            aggregate_payload, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        aggregate_csv,
        flattened_aggregate_rows(summary["seed_aggregation"]["rows"]),
    )
    markdown.write_text(
        markdown_report(summary, seed_results_sha256=sha256_file(seed_json)),
        encoding="utf-8",
    )
    outputs = {
        "seed_json": seed_json,
        "seed_csv": seed_csv,
        "aggregate_json": aggregate_json,
        "aggregate_csv": aggregate_csv,
        "markdown": markdown,
    }
    artifact_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reporting_protocol": summary["reporting_protocol"],
                "matrix": summary["matrix"],
                "validation_selection": summary["validation_selection"],
                "outputs": {
                    name: {"path": path.name, "sha256": sha256_file(path)}
                    for name, path in outputs.items()
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["artifact_manifest"] = artifact_manifest
    return outputs


def main() -> int:
    args = parse_args()
    try:
        matrix_path = resolve(args.matrix)
        matrix = validate_matrix(matrix_path)
        report_root = (
            PROJECT_ROOT
            / "Training"
            / "reports"
            / matrix["dataset_tag"]
            / matrix["matrix_id"]
        )
        selection_path = (
            resolve(args.selection)
            if args.selection is not None
            else report_root / "validation_selection.json"
        )
        final_test_dir = (
            resolve(args.final_test_dir)
            if args.final_test_dir is not None
            else report_root / "final_test"
        )
        output_dir = (
            resolve(args.output_dir)
            if args.output_dir is not None
            else report_root / "final_summary"
        )
        postprocess_root = (
            resolve(args.postprocess_root)
            if args.postprocess_root is not None
            else None
        )
        summary = build_matrix_summary(
            matrix=matrix,
            matrix_path=matrix_path,
            selection_path=selection_path,
            final_test_dir=final_test_dir,
            postprocess_root=postprocess_root,
        )
        paths = write_outputs(summary, output_dir, overwrite=args.overwrite)
    except (
        FileNotFoundError,
        KeyError,
        MatrixSummaryError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Validated {summary['seed_row_count']} single-checkpoint seed rows")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
