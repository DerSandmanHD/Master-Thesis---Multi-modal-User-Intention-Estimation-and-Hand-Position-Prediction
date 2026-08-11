#!/usr/bin/env python3
"""Evaluate exported prediction windows with grouped scientific statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from grouped_metrics import build_grouped_evaluation, report_table_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Per-window prediction CSV from an executable checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="JSON output (default: <predictions>_grouped_metrics.json).",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=None,
        help="Long-form CSV output (default: <predictions>_grouped_metrics.csv).",
    )
    parser.add_argument(
        "--prediction-report",
        type=Path,
        default=None,
        help=(
            "JSON sidecar emitted with the prediction CSV; binds all grouped "
            "metrics to its single validation-selected executable checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-diagnostic-checkpoint",
        action="store_true",
        help=(
            "Allow an explicitly diagnostic/oracle result_role. Without this "
            "flag such a sidecar is refused as a primary grouped result."
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--sequence-bootstrap",
        action="store_true",
        help="Additionally resample whole sequences; participant clusters remain primary.",
    )
    return parser.parse_args()


def default_output(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_grouped_metrics{suffix}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_binding(
    path: Path | None,
    *,
    predictions_path: Path,
    prediction_rows: int,
    allow_diagnostic: bool,
) -> dict:
    if path is None:
        return {
            "status": "unverified_no_prediction_report",
            "result_role": "unbound_grouped_diagnostic",
            "warning": (
                "No prediction sidecar was supplied, so these metrics cannot be "
                "used as a checkpoint-bound main result row."
            ),
        }
    source = json.loads(path.read_text(encoding="utf-8"))
    role = str(source.get("result_role", "")).strip()
    if not role:
        raise ValueError("Prediction report must declare result_role")
    primary = role == "primary_validation_selected_checkpoint"
    diagnostic = not primary
    if diagnostic and not allow_diagnostic:
        raise ValueError(
            f"Prediction report result_role={role!r} is diagnostic/oracle; pass "
            "--allow-diagnostic-checkpoint to preserve it as diagnostic only"
        )
    selection_split = str(source.get("checkpoint_selection_split", "")).casefold()
    selection_metric = str(source.get("checkpoint_selection_metric", ""))
    if selection_split != "validation" or not selection_metric.casefold().startswith(
        "validation_"
    ):
        raise ValueError(
            "Prediction checkpoint must have an explicit validation-only selection rule"
        )
    required = (
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "predictions_csv_sha256",
    )
    missing = [name for name in required if source.get(name) in (None, "")]
    if missing:
        raise ValueError(
            f"Prediction report cannot bind one executable checkpoint; missing={missing}"
        )
    checkpoint_hash = str(source["checkpoint_sha256"])
    if len(checkpoint_hash) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in checkpoint_hash
    ):
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
    if source.get("rows") is not None and int(source["rows"]) != prediction_rows:
        raise ValueError(
            "Prediction report row count differs from the supplied prediction CSV"
        )
    predictions_hash = sha256_file(predictions_path)
    if str(source["predictions_csv_sha256"]).lower() != predictions_hash:
        raise ValueError(
            "Prediction CSV SHA-256 differs from its checkpoint-bound sidecar"
        )
    checkpoint_path = Path(str(source["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (path.parent / checkpoint_path).resolve()
    hash_verification = "not_locally_available"
    if checkpoint_path.is_file():
        if sha256_file(checkpoint_path) != checkpoint_hash.lower():
            raise ValueError("Local checkpoint SHA-256 differs from prediction report")
        hash_verification = "matched_local_checkpoint"
    return {
        "status": "bound_single_checkpoint",
        "result_role": (
            "checkpoint_bound_grouped_diagnostic"
            if diagnostic
            else "checkpoint_bound_grouped_primary"
        ),
        "source_prediction_report": str(path.resolve()),
        "source_prediction_report_sha256": sha256_file(path),
        "source_result_role": role,
        "predictions_csv_sha256": predictions_hash,
        "checkpoint": str(source["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash.lower(),
        "checkpoint_epoch": int(source["checkpoint_epoch"]),
        "checkpoint_hash_verification": hash_verification,
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_value": source.get("checkpoint_selection_value"),
        "dataset_content_fingerprint": source.get("dataset_content_fingerprint"),
        "split": source.get("split"),
    }


def main() -> int:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {args.predictions}")
    report_path = args.report_out or default_output(args.predictions, ".json")
    table_path = args.table_out or default_output(args.predictions, ".csv")
    frame = pd.read_csv(args.predictions)
    report = build_grouped_evaluation(
        frame,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        include_sequence_bootstrap=args.sequence_bootstrap,
    )
    report.update(
        {
            "predictions_csv": str(args.predictions.resolve()),
            "predictions_csv_sha256": sha256_file(args.predictions),
            "prediction_rows": int(len(frame)),
            "checkpoint_binding": checkpoint_binding(
                args.prediction_report,
                predictions_path=args.predictions,
                prediction_rows=len(frame),
                allow_diagnostic=args.allow_diagnostic_checkpoint,
            ),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(report_table_rows(report)).to_csv(table_path, index=False)
    print(f"Grouped report: {report_path}")
    print(f"Grouped table:  {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
