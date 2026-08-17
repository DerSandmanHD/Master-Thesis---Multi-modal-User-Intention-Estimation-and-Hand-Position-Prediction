#!/usr/bin/env python3
"""Validate completed postprocessing steps before a SLURM task resumes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from artifact_freeze import canonical_json_hash, sha256_file


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def csv_rows(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return sum(1 for _ in reader), header


def validate_prediction(args: argparse.Namespace) -> None:
    report = read_json(args.report)
    if report.get("schema_version") != 3:
        raise ValueError("Unsupported prediction-report schema")
    if report.get("report_fingerprint") != canonical_json_hash(
        {**report, "report_fingerprint": None}
    ):
        raise ValueError("Prediction-report fingerprint mismatch")
    if Path(str(report.get("run_dir", ""))).resolve() != args.run_dir.resolve():
        raise ValueError("Prediction report belongs to another run")
    if Path(str(report.get("predictions_csv", ""))).resolve() != args.csv.resolve():
        raise ValueError("Prediction report points to another CSV")
    if report.get("predictions_csv_sha256") != sha256_file(args.csv):
        raise ValueError("Prediction CSV hash mismatch")
    rows, _ = csv_rows(args.csv)
    if rows != int(report.get("rows", -1)):
        raise ValueError("Prediction CSV row count mismatch")
    checkpoint = Path(str(report.get("checkpoint", "")))
    if not checkpoint.is_file() or sha256_file(checkpoint) != report.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Prediction checkpoint hash mismatch")


def validate_grouped(args: argparse.Namespace) -> None:
    report = read_json(args.report)
    source = read_json(args.prediction_report)
    rows, header = csv_rows(args.table)
    if report.get("schema_version") != "grouped_prediction_evaluation_v1":
        raise ValueError("Unsupported grouped-report schema")
    if report.get("predictions_csv_sha256") != sha256_file(args.csv):
        raise ValueError("Grouped report prediction hash mismatch")
    if int(report.get("prediction_rows", -1)) != int(source.get("rows", -2)):
        raise ValueError("Grouped report row count mismatch")
    binding = report.get("checkpoint_binding", {})
    if binding.get("source_prediction_report_sha256") != sha256_file(
        args.prediction_report
    ):
        raise ValueError("Grouped report sidecar hash mismatch")
    if binding.get("checkpoint_sha256") != source.get("checkpoint_sha256"):
        raise ValueError("Grouped report checkpoint mismatch")
    if rows <= 0 or not header:
        raise ValueError("Grouped table is empty")


def validate_baseline(args: argparse.Namespace) -> None:
    report = read_json(args.report)
    rows, header = csv_rows(args.table)
    if report.get("schema_version") != 2 or set(report.get("splits", {})) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Incomplete pose-baseline report")
    if Path(str(report.get("config", ""))).resolve() != args.config.resolve():
        raise ValueError("Pose-baseline report belongs to another config")
    if rows <= 0 or not header:
        raise ValueError("Pose-baseline table is empty")


def validate_modality(args: argparse.Namespace) -> None:
    report = read_json(args.report)
    rows, header = csv_rows(args.table)
    prediction_rows, _ = csv_rows(args.csv)
    if report.get("schema_version") != 1 or not report.get("modality_names"):
        raise ValueError("Invalid modality-weight report")
    if rows != prediction_rows or rows != int(report.get("rows", -1)):
        raise ValueError("Modality-weight row count mismatch")
    if not any(name.startswith("modality_") and name.endswith("_weight") for name in header):
        raise ValueError("Modality-weight table lacks weight columns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("prediction", "grouped", "baseline", "modality"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prediction-report", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    required = {
        "prediction": ("csv", "run_dir"),
        "grouped": ("csv", "table", "prediction_report"),
        "baseline": ("table", "config"),
        "modality": ("csv", "table"),
    }[args.kind]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing arguments for validation: " + ", ".join(missing))
    globals()[f"validate_{args.kind}"](args)
    print(f"Validated resumable {args.kind} artifacts: {args.report}")


if __name__ == "__main__":
    main()
