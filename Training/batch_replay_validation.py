#!/usr/bin/env python3
"""Run causal replay for every sequence in an artifact-defined split.

This script orchestrates inference-only subprocesses. It cannot send robot
commands and never changes datasets or deployment checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from replay_stream_inference import build_replay_summary


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLAY_SCRIPT = Path(__file__).resolve().parent / "replay_stream_inference.py"
INTENTION_NAMES = {"continue", "fetch", "handover"}


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_plan(
    artifacts_dir: Path,
    master_dir: Path,
    split: str,
) -> list[tuple[str, Path]]:
    artifacts_dir = resolve_path(artifacts_dir)
    master_dir = resolve_path(master_dir)
    metadata = json.loads(
        (artifacts_dir / "data_metadata.json").read_text(encoding="utf-8")
    )
    split_sequences = metadata.get("split", {}).get("sequences", {})
    if split not in split_sequences:
        raise ValueError(
            f"Split {split!r} is not present in artifact metadata"
        )
    expected = [str(value) for value in split_sequences[split]]
    if len(expected) != len(set(expected)):
        raise ValueError(f"Artifact split {split!r} contains duplicate sequences")

    plan = []
    missing = []
    for sequence_id in expected:
        path = master_dir / f"{sequence_id}_master.csv"
        if not path.is_file():
            missing.append(sequence_id)
        else:
            plan.append((sequence_id, path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {split} master CSVs are missing from {master_dir}: "
            + ", ".join(missing)
        )
    return plan


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_replay_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            reasons = source.get("input_quality_reasons", "")
            parsed_reasons = json.loads(reasons) if reasons else []
            quality_value = source.get("input_quality_ok", "").strip().lower()
            row = {
                **source,
                "input_quality_ok": quality_value in {"1", "true", "yes"},
                "input_quality_reasons": parsed_reasons,
                "timestamp_ns": int(source["timestamp_ns"]),
                "endpoint_row": int(source["endpoint_row"]),
                "intention_inference_ms": float(
                    source["intention_inference_ms"]
                ),
                "pose_inference_ms": optional_float(
                    source.get("pose_inference_ms")
                ),
                "pose_position_error_cm": optional_float(
                    source.get("pose_position_error_cm")
                ),
                "pose_orientation_error_deg": optional_float(
                    source.get("pose_orientation_error_deg")
                ),
            }
            rows.append(row)
    return rows


def percentile(values: list[float], percentage: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def event_onset_summary(rows: list[dict], field: str) -> dict:
    """Measure first matching prediction after each labeled segment onset.

    Onsets are approximate at replay resolution: they are the first evaluated
    endpoint carrying the new label, not the original annotation command time.
    """

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)

    events: dict[str, list[float | None]] = {
        label: [] for label in INTENTION_NAMES
    }
    for sequence_rows in by_sequence.values():
        sequence_rows.sort(key=lambda row: int(row["endpoint_row"]))
        index = 0
        while index < len(sequence_rows):
            target = str(sequence_rows[index]["target_intention"])
            if target not in INTENTION_NAMES:
                index += 1
                continue
            start = index
            while (
                index + 1 < len(sequence_rows)
                and sequence_rows[index + 1]["target_intention"] == target
            ):
                index += 1
            segment = sequence_rows[start : index + 1]
            onset_ns = int(segment[0]["timestamp_ns"])
            first_match = next(
                (
                    row
                    for row in segment
                    if str(row.get(field)) == target
                ),
                None,
            )
            events[target].append(
                None
                if first_match is None
                else (int(first_match["timestamp_ns"]) - onset_ns) / 1e6
            )
            index += 1

    result = {}
    for label, values in events.items():
        detected = [float(value) for value in values if value is not None]
        result[label] = {
            "events": len(values),
            "detected_events": len(detected),
            "detection_rate": (
                len(detected) / len(values) if values else None
            ),
            "latency_ms_from_first_labeled_replay_endpoint": {
                "mean": statistics.fmean(detected) if detected else None,
                "median": statistics.median(detected) if detected else None,
                "p95": percentile(detected, 95.0),
                "maximum": max(detected) if detected else None,
            },
        }
    return result


def replay_command(
    args: argparse.Namespace,
    *,
    master_csv: Path,
    output_csv: Path,
    summary_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--artifacts-dir",
        str(resolve_path(args.artifacts_dir)),
        "--master-csv",
        str(master_csv),
        "--output-csv",
        str(output_csv),
        "--summary-json",
        str(summary_json),
        "--device",
        args.device,
        "--smoothing-window",
        str(args.smoothing_window),
        "--minimum-confidence",
        str(args.minimum_confidence),
        "--minimum-stable-predictions",
        str(args.minimum_stable_predictions),
        "--minimum-gaze-coverage",
        str(args.minimum_gaze_coverage),
        "--maximum-gaze-gap-ms",
        str(args.maximum_gaze_gap_ms),
        "--minimum-handover-hand-coverage",
        str(args.minimum_handover_hand_coverage),
        "--maximum-hand-age-ms",
        str(args.maximum_hand_age_ms),
        "--maximum-vio-age-ms",
        str(args.maximum_vio_age_ms),
        "--maximum-anchor-age-ms",
        str(args.maximum_anchor_age_ms),
        "--maximum-marker-age-ms",
        str(args.maximum_marker_age_ms),
        "--print-mode",
        "none",
    ]
    if args.step_size is not None:
        command.extend(["--step-size", str(args.step_size)])
    if args.max_predictions is not None:
        command.extend(["--max-predictions", str(args.max_predictions)])
    if args.allow_missing_features:
        command.append("--allow-missing-features")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("Training/final_clean_v1_residual_v2_seed44"),
    )
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=Path(
            "Data_collection/final_dataset_snapshot_20260729/master_datasets"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Training/evaluation/deployment_validation_runs/"
            "residual_v2_seed44_test"
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--minimum-confidence", type=float, default=0.65)
    parser.add_argument("--minimum-stable-predictions", type=int, default=2)
    parser.add_argument("--minimum-gaze-coverage", type=float, default=0.80)
    parser.add_argument("--maximum-gaze-gap-ms", type=float, default=500.0)
    parser.add_argument(
        "--minimum-handover-hand-coverage", type=float, default=0.50
    )
    parser.add_argument("--maximum-hand-age-ms", type=float, default=50.0)
    parser.add_argument("--maximum-vio-age-ms", type=float, default=10.0)
    parser.add_argument("--maximum-anchor-age-ms", type=float, default=500.0)
    parser.add_argument("--maximum-marker-age-ms", type=float, default=250.0)
    parser.add_argument("--max-predictions", type=int, default=None)
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace same-named generated replay reports, never source data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = resolve_path(args.artifacts_dir)
    master_dir = resolve_path(args.master_dir)
    output_dir = resolve_path(args.output_dir)
    plan = load_split_plan(artifacts_dir, master_dir, args.split)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or pass --overwrite."
        )
    per_sequence_dir = output_dir / "per_sequence"
    per_sequence_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    sequence_reports = []
    combined_rows: list[dict] = []
    for index, (sequence_id, master_csv) in enumerate(plan, start=1):
        output_csv = per_sequence_dir / f"{sequence_id}_predictions.csv"
        summary_json = per_sequence_dir / f"{sequence_id}_summary.json"
        log_path = per_sequence_dir / f"{sequence_id}.log"
        command = replay_command(
            args,
            master_csv=master_csv,
            output_csv=output_csv,
            summary_json=summary_json,
        )
        print(f"[{index:02d}/{len(plan):02d}] {sequence_id}", flush=True)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Replay failed for {sequence_id}; see {log_path}\n"
                + "\n".join(completed.stdout.splitlines()[-20:])
            )
        rows = load_replay_rows(output_csv)
        if not rows:
            raise RuntimeError(f"Replay emitted no predictions for {sequence_id}")
        combined_rows.extend(rows)
        sequence_summary = json.loads(
            summary_json.read_text(encoding="utf-8")
        )
        sequence_reports.append(
            {
                "sequence_id": sequence_id,
                "master_csv": str(master_csv),
                "master_sha256": sha256_file(master_csv),
                "predictions_csv": str(output_csv),
                "summary_json": str(summary_json),
                "predictions": len(rows),
                "summary": sequence_summary,
            }
        )

    aggregate = build_replay_summary(combined_rows)
    snapshot_report_path = master_dir.parent / "snapshot_validation.json"
    snapshot_report = None
    if snapshot_report_path.is_file():
        snapshot_report = json.loads(
            snapshot_report_path.read_text(encoding="utf-8")
        )

    report = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": str(artifacts_dir),
        "artifact_metadata_sha256": sha256_file(
            artifacts_dir / "data_metadata.json"
        ),
        "master_dir": str(master_dir),
        "split": args.split,
        "expected_sequences": len(plan),
        "completed_sequences": len(sequence_reports),
        "decision_parameters": {
            "step_size": args.step_size,
            "smoothing_window": args.smoothing_window,
            "minimum_confidence": args.minimum_confidence,
            "minimum_stable_predictions": args.minimum_stable_predictions,
            "minimum_gaze_coverage": args.minimum_gaze_coverage,
            "maximum_gaze_gap_ms": args.maximum_gaze_gap_ms,
            "minimum_handover_hand_coverage": (
                args.minimum_handover_hand_coverage
            ),
            "maximum_hand_age_ms": args.maximum_hand_age_ms,
            "maximum_vio_age_ms": args.maximum_vio_age_ms,
            "maximum_anchor_age_ms": args.maximum_anchor_age_ms,
            "maximum_marker_age_ms": args.maximum_marker_age_ms,
        },
        "diagnostic_overrides": {
            "allow_missing_features": args.allow_missing_features,
            "max_predictions": args.max_predictions,
        },
        "snapshot_validation": (
            {
                "path": str(snapshot_report_path),
                "status": snapshot_report.get("status"),
                "manifest_sha256": snapshot_report.get("manifest", {}).get(
                    "sha256"
                ),
                "sequence_fingerprint": snapshot_report.get(
                    "manifest", {}
                ).get("sequence_fingerprint"),
            }
            if snapshot_report is not None
            else None
        ),
        "aggregate": aggregate,
        "event_onset": {
            level: event_onset_summary(combined_rows, field)
            for level, field in (
                ("raw", "raw_intention"),
                ("stable", "stable_intention"),
                ("actionable", "actionable_intention"),
            )
        },
        "event_onset_definition": (
            "Latency starts at the first evaluated replay endpoint carrying "
            "a new contiguous target label. It is not capture-to-decision "
            "latency and can differ from the original annotation onset by up "
            "to the replay stride."
        ),
        "sequences": sequence_reports,
    }
    report_path = output_dir / "batch_replay_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Batch replay complete: sequences={len(sequence_reports)}, "
        f"predictions={aggregate['predictions']}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
