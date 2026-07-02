#!/usr/bin/env python3
"""Build master datasets for all eligible sequences with resumable reporting."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from annotation_utils import parse_target_object_id, read_review_rows
from build_master_dataset import build_master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch wrapper for build_master_dataset.py.")
    parser.add_argument("--data-root", type=Path, default=Path("Data_collection"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--timestamps", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-report", type=Path, default=None)
    parser.add_argument("--sequence", action="append", default=[], help="Process only this sequence; repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-semantic-annotations",
        action="store_true",
        help="Only build sequences with target_object_id and a known receiving hand.",
    )
    parser.add_argument("--future-horizon-seconds", type=float, default=1.0)
    parser.add_argument("--hand-tolerance-ms", type=float, default=12.0)
    parser.add_argument("--slam-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--marker-tolerance-ms", type=float, default=20.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value) -> bool:
    return str(value).strip().lower() == "true"


def manifest_eligible(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not truthy(row.get("include_in_training")):
        reasons.append("excluded_from_training")
    for field in ("vrs_exists", "hand_tracking_exists", "slam_exists", "aruco_csv_exists"):
        if not truthy(row.get(field)):
            reasons.append(f"missing_{field.removesuffix('_exists')}")
    if not truthy(row.get("timestamps_exists")) or row.get("missing_commands", "").strip():
        reasons.append("incomplete_timestamps")
    blocking_issues = set(filter(None, row.get("issues", "").split(";")))
    if "missing_handover_hand_tracking" in blocking_issues:
        reasons.append("missing_handover_hand_tracking")
    return not reasons, reasons


def atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
        temp_path.replace(path)
        path.chmod(0o644)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (args.manifest or data_root / "dataset_manifest.csv").expanduser().resolve()
    timestamps_path = (
        args.timestamps or data_root / "Data_vrs" / "timestamps_summary.reviewed.json"
    ).expanduser().resolve()
    if not timestamps_path.exists() and args.timestamps is None:
        timestamps_path = data_root / "Data_vrs" / "timestamps_summary.json"
    annotations_path = (
        args.annotations or data_root / "manual_timestamp_review.csv"
    ).expanduser().resolve()
    output_dir = (args.output_dir or data_root / "master_datasets").expanduser().resolve()
    batch_report_path = (
        args.batch_report or output_dir / "master_batch_report.json"
    ).expanduser().resolve()

    try:
        manifest_rows = read_manifest(manifest_path)
        annotations = read_review_rows(annotations_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    requested = set(args.sequence)
    selected = []
    skipped = []
    for row in manifest_rows:
        sequence_id = row.get("sequence_id", "").strip()
        if requested and sequence_id not in requested:
            continue
        eligible, reasons = manifest_eligible(row)
        annotation = annotations.get(sequence_id, {})
        if annotation.get("decision") == "exclude":
            eligible = False
            reasons.append("manual_exclusion")
        if args.require_semantic_annotations:
            try:
                target_object_id = parse_target_object_id(annotation.get("target_object_id"))
            except ValueError:
                target_object_id = None
            if target_object_id is None:
                reasons.append("missing_target_object_id")
            if annotation.get("receiving_hand") not in {"left", "right"}:
                reasons.append("missing_receiving_hand")
            eligible = not reasons
        if eligible:
            selected.append(sequence_id)
        else:
            skipped.append({"sequence_id": sequence_id, "reasons": sorted(set(reasons))})

    if requested:
        found = {row.get("sequence_id", "") for row in manifest_rows}
        for missing in sorted(requested - found):
            skipped.append({"sequence_id": missing, "reasons": ["not_in_manifest"]})
    selected.sort()
    if args.limit is not None:
        selected = selected[:max(0, args.limit)]

    records = []
    for index, sequence_id in enumerate(selected, start=1):
        output_csv = output_dir / f"{sequence_id}_master.csv"
        report_out = output_dir / f"{sequence_id}_master_report.json"
        if output_csv.exists() and report_out.exists() and not args.overwrite:
            records.append({"sequence_id": sequence_id, "status": "already_exists"})
            print(f"[{index}/{len(selected)}] {sequence_id}: already exists")
            continue
        if args.dry_run:
            records.append({"sequence_id": sequence_id, "status": "would_build"})
            print(f"[{index}/{len(selected)}] {sequence_id}: would build")
            continue

        build_args = Namespace(
            sequence_id=sequence_id,
            data_root=data_root,
            timestamps=timestamps_path,
            marker_csv=None,
            output_csv=output_csv,
            report_out=report_out,
            gaze_csv=None,
            annotations=annotations_path,
            target_object_id=None,
            receiving_hand=None,
            future_horizon_seconds=args.future_horizon_seconds,
            hand_tolerance_ms=args.hand_tolerance_ms,
            slam_tolerance_ms=args.slam_tolerance_ms,
            marker_tolerance_ms=args.marker_tolerance_ms,
            overwrite=args.overwrite,
        )
        print(f"[{index}/{len(selected)}] {sequence_id}: building")
        try:
            _, report, _, _ = build_master(build_args)
            records.append(
                {
                    "sequence_id": sequence_id,
                    "status": "built",
                    "rows": report["rows"],
                    "warnings": report["warnings"],
                }
            )
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "sequence_id": sequence_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ERROR: {exc}")

    report = {
        "manifest": str(manifest_path),
        "timestamps": str(timestamps_path),
        "annotations": str(annotations_path),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "selected": len(selected),
        "built": sum(record["status"] == "built" for record in records),
        "already_exists": sum(record["status"] == "already_exists" for record in records),
        "errors": sum(record["status"] == "error" for record in records),
        "skipped": skipped,
        "records": records,
    }
    atomic_json(report, batch_report_path)
    print(f"Batch report: {batch_report_path}")
    print(
        f"Selected: {report['selected']}, built: {report['built']}, "
        f"existing: {report['already_exists']}, errors: {report['errors']}"
    )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
