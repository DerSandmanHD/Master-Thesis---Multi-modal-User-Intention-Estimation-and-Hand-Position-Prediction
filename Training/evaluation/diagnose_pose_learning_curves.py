#!/usr/bin/env python3
"""Diagnose primary t+1 pose learning from frozen validation-run histories.

This is a retrospective reporting utility.  It never trains a model and it
refuses to overwrite an existing non-empty report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from artifact_freeze import canonical_json_hash, sha256_file  # noqa: E402


SCHEMA_VERSION = "primary_t1_pose_learning_curve_diagnosis_v1"
EXPECTED_DATASET = "dataset_v3_causal_20260815_n214_5d136a34"
UNDERFITTING_WARNING_CM = 14.0


class PoseLearningDiagnosisError(ValueError):
    """Raised when a frozen learning history cannot be diagnosed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Frozen residual_current_gate validation run (repeat per seed).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset", default=EXPECTED_DATASET)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PoseLearningDiagnosisError(f"Expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoseLearningDiagnosisError(message)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PoseLearningDiagnosisError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise PoseLearningDiagnosisError(f"{label} is not finite: {value!r}")
    return result


def _nested(row: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise PoseLearningDiagnosisError(f"Missing {label}: {'.'.join(path)}")
        value = value[key]
    return value


def trajectory_summary(values: Sequence[float], epochs: Sequence[int]) -> dict[str, Any]:
    _require(bool(values) and len(values) == len(epochs), "Invalid trajectory")
    _require(len(set(epochs)) == len(epochs), "Duplicate history epoch")
    minimum_index = min(range(len(values)), key=values.__getitem__)
    mean_epoch = statistics.fmean(epochs)
    mean_value = statistics.fmean(values)
    denominator = sum((epoch - mean_epoch) ** 2 for epoch in epochs)
    slope = (
        sum(
            (epoch - mean_epoch) * (value - mean_value)
            for epoch, value in zip(epochs, values)
        )
        / denominator
        if denominator
        else 0.0
    )
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "minimum": float(values[minimum_index]),
        "minimum_epoch": int(epochs[minimum_index]),
        "absolute_change_first_to_last": float(values[-1] - values[0]),
        "relative_change_first_to_last": (
            float((values[-1] - values[0]) / values[0]) if values[0] else None
        ),
        "least_squares_slope_per_epoch": float(slope),
        "decreasing_step_fraction": (
            sum(right < left for left, right in zip(values, values[1:]))
            / max(1, len(values) - 1)
        ),
    }


def diagnose_run(
    run_dir: Path,
    *,
    expected_dataset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    _require(metrics_path.is_file(), f"Missing metrics.json: {run_dir}")
    _require(config_path.is_file(), f"Missing config.json: {run_dir}")
    metrics = _read_object(metrics_path)
    config = _read_object(config_path)
    context = metrics.get("run_context")
    _require(isinstance(context, Mapping), f"Missing run_context: {metrics_path}")
    _require(
        context.get("dataset_tag") == expected_dataset,
        f"Run is not from active dataset {expected_dataset}: {run_dir}",
    )
    _require(
        context.get("experiment_tag") == "thesis_final_v2_validation",
        f"Run is not a frozen matrix-validation run: {run_dir}",
    )
    architecture = metrics.get("architecture")
    _require(isinstance(architecture, Mapping), f"Missing architecture: {metrics_path}")
    _require(
        architecture.get("fusion_mode") == "temporal_channel_gated",
        f"Run is not the primary current-gate architecture: {run_dir}",
    )
    training = config.get("training")
    _require(isinstance(training, Mapping), f"Missing training config: {config_path}")
    seed = int(training["seed"])
    position_config = dict(training.get("position_loss", {}))
    position_type = str(position_config.get("type", "smooth_l1_meters"))
    beta = float(position_config.get("beta_m", 1.0))
    _require(
        position_type in {"smooth_l1_meters", "normalized_smooth_l1"},
        f"Unsupported position loss: {position_type}",
    )
    checkpoints = metrics.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), f"Missing checkpoints: {metrics_path}")
    best_intention = checkpoints.get("best_intention")
    best_pose = checkpoints.get("best_pose")
    _require(
        isinstance(best_intention, Mapping) and isinstance(best_pose, Mapping),
        f"Missing best checkpoints: {metrics_path}",
    )
    best_intention_epoch = int(best_intention["epoch"])
    best_pose_epoch = int(best_pose["epoch"])
    raw_history = metrics.get("history")
    _require(isinstance(raw_history, list) and raw_history, f"Empty history: {metrics_path}")
    rows: list[dict[str, Any]] = []
    for raw in raw_history:
        _require(isinstance(raw, Mapping), f"Invalid history row: {metrics_path}")
        epoch = int(raw["epoch"])
        row: dict[str, Any] = {"seed": seed, "epoch": epoch}
        for split in ("train", "validation"):
            split_row = _nested(raw, (split,), f"epoch {epoch} {split}")
            row[f"{split}_position_loss"] = _finite(
                _nested(split_row, ("loss", "position"), "position loss"),
                f"epoch {epoch} {split} position loss",
            )
            row[f"{split}_orientation_loss"] = _finite(
                _nested(split_row, ("loss", "orientation"), "orientation loss"),
                f"epoch {epoch} {split} orientation loss",
            )
            row[f"{split}_position_error_cm"] = _finite(
                _nested(
                    split_row,
                    ("pose_oracle", "position_mean_euclidean_error_cm"),
                    "oracle position error",
                ),
                f"epoch {epoch} {split} position error",
            )
            row[f"{split}_intention_macro_f1"] = _finite(
                _nested(split_row, ("intention", "macro_f1"), "intention Macro-F1"),
                f"epoch {epoch} {split} intention Macro-F1",
            )
        rows.append(row)
    epochs = [int(row["epoch"]) for row in rows]
    _require(epochs == sorted(epochs), f"History epochs are not increasing: {run_dir}")
    by_epoch = {int(row["epoch"]): row for row in rows}
    _require(
        best_intention_epoch in by_epoch and best_pose_epoch in by_epoch,
        f"Selected checkpoint epoch is absent from history: {run_dir}",
    )
    best_intention_row = by_epoch[best_intention_epoch]
    best_pose_row = by_epoch[best_pose_epoch]
    selected_pose_value = _finite(best_pose["selection_value"], "best-pose value")
    _require(
        math.isclose(
            selected_pose_value,
            float(best_pose_row["validation_position_error_cm"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ),
        f"Best-pose checkpoint disagrees with history: {run_dir}",
    )
    fields = (
        "train_position_loss",
        "validation_position_loss",
        "train_orientation_loss",
        "validation_orientation_loss",
        "train_position_error_cm",
        "validation_position_error_cm",
        "train_intention_macro_f1",
        "validation_intention_macro_f1",
    )
    trajectories = {
        field: trajectory_summary([float(row[field]) for row in rows], epochs)
        for field in fields
    }
    pose_gain_after_intention = float(
        best_intention_row["validation_position_error_cm"]
        - best_pose_row["validation_position_error_cm"]
    )
    selection_lag = best_pose_epoch > best_intention_epoch and pose_gain_after_intention > 0
    report = {
        "seed": seed,
        "epochs_observed": len(rows),
        "position_loss": {
            "type": position_type,
            "beta_m": beta if position_type == "smooth_l1_meters" else None,
            "explicit_config_block": "position_loss" in training,
            "pose_loss_weight": float(training.get("pose_loss_weight", 1.0)),
            "orientation_loss_weight": float(training.get("orientation_loss_weight", 0.25)),
        },
        "checkpoints": {
            "best_intention_epoch": best_intention_epoch,
            "best_pose_epoch": best_pose_epoch,
            "best_intention_validation_macro_f1": float(
                best_intention_row["validation_intention_macro_f1"]
            ),
            "best_intention_validation_position_error_cm": float(
                best_intention_row["validation_position_error_cm"]
            ),
            "best_pose_validation_position_error_cm": float(
                best_pose_row["validation_position_error_cm"]
            ),
            "pose_improvement_after_best_intention_cm": pose_gain_after_intention,
            "later_pose_optimum_than_intention": selection_lag,
        },
        "trajectories": trajectories,
        "input_artifacts": {
            "metrics": _identity(metrics_path),
            "config": _identity(config_path),
        },
    }
    return report, rows


def build_report(
    run_dirs: Sequence[Path],
    *,
    expected_dataset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(bool(run_dirs), "At least one run directory is required")
    runs: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run, rows = diagnose_run(run_dir, expected_dataset=expected_dataset)
        runs.append(run)
        curve_rows.extend(rows)
    seeds = [int(run["seed"]) for run in runs]
    _require(len(seeds) == len(set(seeds)), "Duplicate seed in run inputs")
    runs.sort(key=lambda row: int(row["seed"]))
    curve_rows.sort(key=lambda row: (int(row["seed"]), int(row["epoch"])))
    final_train_errors = [
        float(run["trajectories"]["train_position_error_cm"]["last"])
        for run in runs
    ]
    minimum_train_errors = [
        float(run["trajectories"]["train_position_error_cm"]["minimum"])
        for run in runs
    ]
    case_a = all(value >= UNDERFITTING_WARNING_CM for value in minimum_train_errors)
    selection_lag_seeds = [
        int(run["seed"])
        for run in runs
        if run["checkpoints"]["later_pose_optimum_than_intention"]
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_fingerprint": None,
        "scope": {
            "dataset": expected_dataset,
            "experiment": "residual_current_gate",
            "target": "receiving-wrist pose at t+1 s",
            "endpose_included": False,
            "retrospective_only": True,
            "new_training_started": False,
        },
        "runs": runs,
        "decision": {
            "checklist_case_a_underfitting_signal": case_a,
            "underfitting_warning_threshold_cm": UNDERFITTING_WARNING_CM,
            "final_train_position_error_cm": final_train_errors,
            "minimum_train_position_error_cm": minimum_train_errors,
            "selection_timing_signal": bool(selection_lag_seeds),
            "selection_timing_seeds": selection_lag_seeds,
            "normalized_smooth_l1_sensitivity_run_recommended": case_a,
            "conclusion": (
                "All observed train-pose curves remain at or above the checklist "
                "underfitting warning threshold; a controlled normalized-loss "
                "sensitivity run is warranted."
                if case_a
                else "The checklist's 14–15 cm train-underfitting trigger is not "
                "met. Existing curves do not justify a new normalized-loss run; "
                "checkpoint-selection timing explains part of the pose gap where "
                "the validation pose optimum occurs after best_intention."
            ),
            "guardrail": (
                "Any normalized-loss run initiated after final-test inspection must "
                "be labelled post-hoc sensitivity analysis, never the original primary model."
            ),
        },
        "interpretation": {
            "position_loss_scale": (
                "smooth_l1_meters with beta_m=1.0 operates in the quadratic region "
                "for sub-metre residuals; its scalar magnitude is not itself evidence "
                "of a negligible parameter gradient."
            ),
            "pose_loss_composition": (
                "position loss + orientation_loss_weight × orientation loss, then "
                "multiplied by pose_loss_weight in the multitask objective."
            ),
            "train_validation_comparability": (
                "Training and validation summaries have different sample mixtures and "
                "training uses dropout while validation uses evaluation mode; their raw "
                "loss levels must not be interpreted as paired measurements."
            ),
        },
    }
    report["report_fingerprint"] = canonical_json_hash(report)
    return report, curve_rows


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Primary t+1 pose-learning diagnosis",
        "",
        (
            f"Dataset: `{report['scope']['dataset']}`. This is a retrospective "
            "diagnosis of the primary receiving-wrist pose at t+1 s; terminal "
            "endpose is excluded."
        ),
        "",
        "| Seed | Epochs | Best intention | Best pose | Train error first→last (cm) | Validation error at best intention→best pose (cm) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        checkpoints = run["checkpoints"]
        trajectory = run["trajectories"]["train_position_error_cm"]
        lines.append(
            f"| {run['seed']} | {run['epochs_observed']} | "
            f"{checkpoints['best_intention_epoch']} | {checkpoints['best_pose_epoch']} | "
            f"{trajectory['first']:.3f}→{trajectory['last']:.3f} | "
            f"{checkpoints['best_intention_validation_position_error_cm']:.3f}→"
            f"{checkpoints['best_pose_validation_position_error_cm']:.3f} |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(decision["conclusion"]),
            "",
            (
                "A new `normalized_smooth_l1` run is therefore "
                + ("recommended." if decision["normalized_smooth_l1_sensitivity_run_recommended"] else "not recommended.")
            ),
            "",
            "The raw position-loss magnitude alone is not used as evidence: the pose "
            "objective also contains the weighted orientation term, and metre-scale "
            "Smooth-L1 is quadratic for these sub-metre residuals.",
            "",
            f"Guardrail: {decision['guardrail']}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "No curve rows to write")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        run_dirs = [_resolve(path) for path in args.run_dir]
        output_dir = _resolve(args.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty pose diagnosis: {output_dir}"
            )
        report, curve_rows = build_report(
            run_dirs,
            expected_dataset=str(args.expected_dataset),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "pose_learning_diagnosis.json"
        curves_path = output_dir / "pose_learning_curves.csv"
        markdown_path = output_dir / "POSE_LEARNING_DIAGNOSIS.md"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(curves_path, curve_rows)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        manifest = {
            "schema_version": "primary_t1_pose_learning_artifacts_v1",
            "manifest_fingerprint": None,
            "report_fingerprint": report["report_fingerprint"],
            "inputs": [run["input_artifacts"] for run in report["runs"]],
            "outputs": {
                path.name: _identity(path)
                for path in (report_path, curves_path, markdown_path)
            },
        }
        manifest["manifest_fingerprint"] = canonical_json_hash(manifest)
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Primary t+1 pose-learning diagnosis: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
