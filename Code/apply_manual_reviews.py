#!/usr/bin/env python3
"""Validate manual timestamp reviews and merge accepted corrections into JSON outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import wave
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from annotation_utils import COMMANDS, read_review_rows


MANUAL_COLUMNS = {
    "START": "manual_start_s",
    "SECOND": "manual_second_s",
    "DONE": "manual_done_s",
    "THIRD": "manual_third_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manual_timestamp_review.csv and apply manual command times."
    )
    parser.add_argument("--data-root", type=Path, default=Path("Data_collection"))
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--timestamps", type=Path, default=None)
    parser.add_argument("--debug", type=Path, default=None)
    parser.add_argument("--wav-dir", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--overrides-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace timestamps_summary.json after creating a timestamped backup.",
    )
    parser.add_argument("--min-command-gap-seconds", type=float, default=0.4)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.1)
    return parser.parse_args()


def read_json(path: Path, required: bool = True) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def atomic_json(data: dict, path: Path) -> None:
    path = path.expanduser().resolve()
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


def timestamp_key(sequence_id: str, timestamps: dict) -> str:
    if sequence_id in timestamps:
        return sequence_id
    vrs_key = f"{sequence_id}.vrs"
    if vrs_key in timestamps:
        return vrs_key
    return vrs_key


def parse_optional_time(value, field: str) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not a number: {value!r}") from exc
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def existing_relative_times(entry: dict) -> dict[str, float | None]:
    result = {}
    for command in COMMANDS:
        value = entry.get(command, {})
        relative = value.get("relative_seconds") if isinstance(value, dict) else None
        result[command] = float(relative) if isinstance(relative, (int, float)) else None
    return result


def derive_audio_start_ns(entry: dict, debug_entry: dict) -> int | None:
    debug_start = debug_entry.get("audio_start_timestamp_ns")
    if isinstance(debug_start, (int, float)):
        return int(debug_start)
    for command in COMMANDS:
        value = entry.get(command)
        if not isinstance(value, dict):
            continue
        timestamp_ns = value.get("timestamp_ns")
        relative_seconds = value.get("relative_seconds")
        if isinstance(timestamp_ns, (int, float)) and isinstance(relative_seconds, (int, float)):
            return int(round(float(timestamp_ns) - float(relative_seconds) * 1e9))
    return None


def wav_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return None


def sequence_duration(sequence_id: str, debug_entry: dict, wav_dir: Path) -> float | None:
    debug_duration = debug_entry.get("audio_duration_seconds")
    if isinstance(debug_duration, (int, float)) and debug_duration > 0:
        return float(debug_duration)
    return wav_duration(wav_dir / f"{sequence_id}.wav")


def validate_times(
    times: dict[str, float | None],
    duration_seconds: float | None,
    min_gap_seconds: float,
    duration_tolerance_seconds: float,
) -> list[str]:
    issues = [f"missing_{command.lower()}" for command in COMMANDS if times.get(command) is None]
    if issues:
        return issues
    ordered = [float(times[command]) for command in COMMANDS]
    for previous, current, command in zip(ordered, ordered[1:], COMMANDS[1:]):
        if current - previous < min_gap_seconds:
            issues.append(f"invalid_order_or_gap_before_{command.lower()}")
    if duration_seconds is not None:
        for command, value in zip(COMMANDS, ordered):
            if value > duration_seconds + duration_tolerance_seconds:
                issues.append(f"{command.lower()}_after_audio_end")
    return issues


def apply_reviews(
    reviews: dict[str, dict],
    timestamps: dict,
    debug: dict,
    existing_overrides: dict,
    wav_dir: Path,
    min_gap_seconds: float,
    duration_tolerance_seconds: float,
) -> tuple[dict, dict, dict]:
    merged = deepcopy(timestamps)
    overrides = deepcopy(existing_overrides)
    records = []

    for sequence_id, review in sorted(reviews.items()):
        decision = review.get("decision", "")
        record = {
            "sequence_id": sequence_id,
            "decision": decision,
            "status": "not_applied",
            "manual_commands": [],
            "issues": [],
        }
        if decision != "manual_fix":
            record["status"] = {
                "accept_auto": "accepted_auto",
                "exclude": "excluded",
                "uncertain": "needs_review",
                "": "not_reviewed",
            }.get(decision, "not_applied")
            records.append(record)
            continue

        key = timestamp_key(sequence_id, merged)
        entry = deepcopy(merged.get(key, {}))
        if not isinstance(entry, dict):
            entry = {}
        debug_entry = debug.get(key, debug.get(sequence_id, {}))
        if not isinstance(debug_entry, dict):
            debug_entry = {}

        try:
            manual_times = {
                command: parse_optional_time(review.get(column), column)
                for command, column in MANUAL_COLUMNS.items()
            }
        except ValueError as exc:
            record["issues"].append(str(exc))
            record["status"] = "rejected"
            records.append(record)
            continue

        record["manual_commands"] = [
            command for command, value in manual_times.items() if value is not None
        ]
        if not record["manual_commands"]:
            record["issues"].append("manual_fix_without_manual_times")
            record["status"] = "rejected"
            records.append(record)
            continue

        effective_times = existing_relative_times(entry)
        effective_times.update(
            {command: value for command, value in manual_times.items() if value is not None}
        )
        duration_seconds = sequence_duration(sequence_id, debug_entry, wav_dir)
        record["audio_duration_seconds"] = duration_seconds
        record["issues"].extend(
            validate_times(
                effective_times,
                duration_seconds,
                min_gap_seconds,
                duration_tolerance_seconds,
            )
        )
        audio_start_ns = derive_audio_start_ns(entry, debug_entry)
        if audio_start_ns is None:
            record["issues"].append("missing_audio_start_timestamp_ns")
        if record["issues"]:
            record["status"] = "rejected"
            records.append(record)
            continue

        sequence_overrides = dict(overrides.get(key, {}))
        for command, relative_seconds in manual_times.items():
            if relative_seconds is None:
                continue
            entry[command] = {
                "timestamp_ns": int(round(audio_start_ns + relative_seconds * 1e9)),
                "relative_seconds": round(relative_seconds, 3),
                "raw_word": "manual_review",
                "match_score": 1.0,
                "avg_logprob": None,
                "speech_end_seconds": None,
                "timestamp_source": "manual_review",
                "timestamp_uncertainty_ms": None,
            }
            sequence_overrides[command] = {"relative_seconds": round(relative_seconds, 3)}
        merged[key] = entry
        overrides[key] = sequence_overrides
        record["status"] = "applied"
        records.append(record)

    report = {
        "review_rows": len(records),
        "applied": sum(record["status"] == "applied" for record in records),
        "accepted_auto": sum(record["status"] == "accepted_auto" for record in records),
        "excluded": sum(record["status"] == "excluded" for record in records),
        "needs_review": sum(record["status"] == "needs_review" for record in records),
        "not_reviewed": sum(record["status"] == "not_reviewed" for record in records),
        "rejected": sum(record["status"] == "rejected" for record in records),
        "records": records,
    }
    return merged, overrides, report


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.before_manual_review_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    vrs_dir = data_root / "Data_vrs"
    review_csv = (args.review_csv or data_root / "manual_timestamp_review.csv").expanduser().resolve()
    timestamps_path = (args.timestamps or vrs_dir / "timestamps_summary.json").expanduser().resolve()
    debug_path = (args.debug or vrs_dir / "timestamps_debug.json").expanduser().resolve()
    wav_dir = (args.wav_dir or vrs_dir / "debug_audio").expanduser().resolve()
    if args.in_place and args.summary_out is not None:
        print("Error: --in-place and --summary-out cannot be combined")
        return 2
    summary_out = (
        timestamps_path
        if args.in_place
        else (args.summary_out or vrs_dir / "timestamps_summary.reviewed.json").expanduser().resolve()
    )
    if not args.in_place and summary_out == timestamps_path:
        print("Error: refusing to replace the input summary without --in-place")
        return 2
    overrides_out = (
        args.overrides_out or vrs_dir / "timestamps_manual_overrides.json"
    ).expanduser().resolve()
    report_out = (
        args.report_out or data_root / "manual_timestamp_review_report.json"
    ).expanduser().resolve()

    try:
        reviews = read_review_rows(review_csv)
        timestamps = read_json(timestamps_path)
        debug = read_json(debug_path, required=False)
        existing_overrides = read_json(overrides_out, required=False)
        merged, overrides, report = apply_reviews(
            reviews,
            timestamps,
            debug,
            existing_overrides,
            wav_dir,
            args.min_command_gap_seconds,
            args.duration_tolerance_seconds,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 2

    backup = backup_file(timestamps_path) if args.in_place else None
    report.update(
        {
            "review_csv": str(review_csv),
            "timestamps_input": str(timestamps_path),
            "timestamps_output": str(summary_out),
            "overrides_output": str(overrides_out),
            "backup": str(backup) if backup else None,
        }
    )
    atomic_json(merged, summary_out)
    atomic_json(overrides, overrides_out)
    atomic_json(report, report_out)

    print(f"Reviewed summary: {summary_out}")
    print(f"Manual overrides: {overrides_out}")
    print(f"Import report:    {report_out}")
    print(f"Applied: {report['applied']}, rejected: {report['rejected']}")
    return 1 if report["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
