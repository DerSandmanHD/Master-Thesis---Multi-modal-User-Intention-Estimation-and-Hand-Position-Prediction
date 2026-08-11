#!/usr/bin/env python3
"""Authorize final-test checkpoints from a complete validation-only matrix.

Every predeclared experiment/seed keeps its own validation-selected
``best_intention`` checkpoint so seed variability can be reported.  A separate
representative seed per experiment is selected only for deterministic
qualitative/postprocessing examples; it does not replace the three-seed result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from artifact_freeze import MANIFEST_NAME, sha256_file, validate_artifact_freeze
from experiment_matrix import DEFAULT_MATRIX, run_directory, validate_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
F1_TOLERANCE = 0.005
SELECTION_RULE = (
    "retain seeds within 0.005 of best validation intention macro-F1; then "
    "minimize validation executable end-to-end pose error; maximize validation "
    "receiving-hand macro-F1 when available; use lower seed as deterministic tie-break"
)
FINAL_TEST_RULE = (
    "authorize every predeclared experiment/seed best_intention checkpoint "
    "after all matrix validation runs are complete; test is never read during "
    "authorization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def value_at(values: dict, *keys: str, default: Any = None) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def candidate_from_run(
    *,
    experiment_id: str,
    seed: int,
    run_dir: Path,
    expected_config: Path,
    expected_dataset_tag: str,
    expected_experiment_tag: str,
) -> dict[str, Any]:
    metrics = read(run_dir / "metrics.json")
    if metrics.get("test_evaluation_skipped") is not True:
        raise ValueError("run is not validation-only")
    if "test" in metrics or "test_by_checkpoint" in metrics:
        raise ValueError("run contains test metrics")
    freeze = validate_artifact_freeze(run_dir / MANIFEST_NAME)
    validate_candidate_identity(
        freeze,
        seed=seed,
        expected_config=expected_config,
        expected_dataset_tag=expected_dataset_tag,
        expected_experiment_tag=expected_experiment_tag,
    )
    checkpoint = metrics["checkpoints"]["best_intention"]
    if not str(checkpoint["selection_metric"]).startswith("validation_"):
        raise ValueError("checkpoint selection metric is not validation-based")
    validation = metrics["validation_by_checkpoint"]["best_intention"]
    pose = validation.get("pose_end_to_end", validation.get("pose", {}))
    hand = validation.get("receiving_hand", {})
    checkpoint_identity = freeze["output_artifacts"]["checkpoints"][
        "best_intention"
    ]
    return {
        "experiment_id": experiment_id,
        "seed": int(seed),
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_name": "best_intention",
        "checkpoint_path": checkpoint_identity["path"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_metric": checkpoint["selection_metric"],
        "checkpoint_selection_value": float(checkpoint["selection_value"]),
        "artifact_manifest_fingerprint": freeze["manifest_fingerprint"],
        "validation_intention_macro_f1": float(
            validation["intention"]["macro_f1"]
        ),
        "validation_pose_mae_cm": (
            None
            if pose.get("position_mae_cm") is None
            else float(pose["position_mae_cm"])
        ),
        "validation_receiving_hand_macro_f1": (
            None
            if value_at(
                hand,
                "macro_f1_supported",
                default=hand.get("macro_f1"),
            )
            is None
            else float(
                value_at(
                    hand,
                    "macro_f1_supported",
                    default=hand.get("macro_f1"),
                )
            )
        ),
    }


def validate_candidate_identity(
    freeze: dict[str, Any],
    *,
    seed: int,
    expected_config: Path,
    expected_dataset_tag: str,
    expected_experiment_tag: str,
) -> None:
    if int(freeze.get("seed", -1)) != int(seed):
        raise ValueError("Frozen run seed differs from matrix seed")
    context = freeze.get("run_context", {})
    if context.get("dataset_tag") != expected_dataset_tag:
        raise ValueError("Frozen run dataset differs from matrix")
    if context.get("experiment_tag") != expected_experiment_tag:
        raise ValueError("Frozen run experiment tag differs from matrix")
    source = freeze.get("configuration", {}).get("source", {})
    source_path = Path(str(source.get("path", ""))).expanduser()
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    expected = resolve(expected_config)
    if source_path.resolve() != expected:
        raise ValueError("Frozen source config differs from matrix entry")
    if source.get("sha256") != sha256_file(expected):
        raise ValueError("Frozen source config hash differs from matrix entry")


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No validation candidates")
    best_f1 = max(row["validation_intention_macro_f1"] for row in candidates)
    eligible = [
        row
        for row in candidates
        if row["validation_intention_macro_f1"] >= best_f1 - F1_TOLERANCE
    ]
    return min(
        eligible,
        key=lambda row: (
            float("inf")
            if row["validation_pose_mae_cm"] is None
            else row["validation_pose_mae_cm"],
            float("inf")
            if row["validation_receiving_hand_macro_f1"] is None
            else -row["validation_receiving_hand_macro_f1"],
            row["seed"],
        ),
    )


def validate_final_test_authorization(
    report: dict[str, Any],
    *,
    experiment_id: str,
    seed: int,
    run_dir: str,
    checkpoint_sha256: str,
    artifact_manifest_fingerprint: str,
) -> dict[str, Any]:
    """Return the exact authorized row or reject a final-test invocation."""

    if report.get("complete") is not True:
        raise ValueError("Validation-selection manifest is incomplete")
    if report.get("selection_split") != "validation":
        raise ValueError("Final-test authorization is not validation-only")
    if report.get("test_metrics_read") is not False:
        raise ValueError("Final-test authorization read test metrics")
    expected_run = str(run_dir).replace("\\", "/").rstrip("/")
    matches = [
        row
        for row in report.get("final_test_runs", [])
        if row.get("experiment_id") == experiment_id
        and int(row.get("seed", -1)) == int(seed)
        and str(row.get("run_dir", "")).replace("\\", "/").rstrip("/")
        == expected_run
    ]
    if len(matches) != 1:
        raise ValueError("Run is not uniquely authorized by validation selection")
    row = matches[0]
    if row.get("checkpoint_name") != "best_intention":
        raise ValueError("Authorized main checkpoint is not best_intention")
    if row.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Authorized checkpoint hash differs from executable file")
    if row.get("artifact_manifest_fingerprint") != artifact_manifest_fingerprint:
        raise ValueError(
            "Authorized artifact manifest differs from the selected validation run"
        )
    return row


def main() -> int:
    args = parse_args()
    matrix_path = resolve(args.matrix)
    matrix = validate_matrix(matrix_path)
    output = (
        resolve(args.output)
        if args.output is not None
        else PROJECT_ROOT
        / "Training/reports"
        / matrix["dataset_tag"]
        / matrix["matrix_id"]
        / "validation_selection.json"
    )
    candidates = []
    errors = []
    selections = []
    for entry in matrix["training_experiments"]:
        group = []
        for seed in matrix["seeds"]:
            run_dir = PROJECT_ROOT / run_directory(matrix, entry["id"], int(seed))
            try:
                row = candidate_from_run(
                    experiment_id=entry["id"],
                    seed=int(seed),
                    run_dir=run_dir,
                    expected_config=Path(entry["config"]),
                    expected_dataset_tag=matrix["dataset_tag"],
                    expected_experiment_tag=matrix["validation_experiment_tag"],
                )
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "experiment_id": entry["id"],
                        "seed": int(seed),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                candidates.append(row)
                group.append(row)
        if len(group) == len(matrix["seeds"]):
            selected = dict(select_candidate(group))
            selected["selection_rule"] = SELECTION_RULE
            selections.append(selected)
    complete = (
        not errors
        and len(candidates)
        == len(matrix["training_experiments"]) * len(matrix["seeds"])
        and len(selections) == len(matrix["training_experiments"])
    )
    try:
        matrix_file = matrix_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        matrix_file = str(matrix_path)
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix_id": matrix["matrix_id"],
        "matrix_file": matrix_file,
        "matrix_sha256": sha256_file(matrix_path),
        "dataset_tag": matrix["dataset_tag"],
        "complete": complete,
        "selection_split": "validation",
        "test_metrics_read": False,
        "representative_selection_rule": SELECTION_RULE,
        "final_test_rule": FINAL_TEST_RULE,
        "f1_tolerance": F1_TOLERANCE,
        "errors": errors,
        "representative_by_experiment": selections,
        "selected": selections,
        "final_test_runs": candidates,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validation selection complete={complete}: {output}")
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
