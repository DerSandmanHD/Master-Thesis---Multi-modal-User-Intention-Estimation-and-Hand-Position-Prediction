#!/usr/bin/env python3
"""Create a QA manifest for Aria recordings and derived processing outputs."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_COMMANDS = ("START", "SECOND", "DONE", "THIRD")
MPS_PREFIX = "mps_"
MPS_SUFFIX = "_vrs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check VRS, MP4, MPS and trigger-label availability per recording."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data_collection"),
        help="Root directory containing Data_vrs and Data_mp4.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("BackUp_Videos"),
        help="Directory containing the backup VRS files.",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="Path to timestamps_summary.json. Defaults to Data_collection/Data_vrs/timestamps_summary.json.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="CSV output path. Defaults to Data_collection/dataset_manifest.csv.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="JSON report output path. Defaults to Data_collection/dataset_qa_report.json.",
    )
    parser.add_argument(
        "--min-phase-seconds",
        type=float,
        default=0.5,
        help="Warn when a labeled phase is shorter than this value.",
    )
    parser.add_argument(
        "--max-sequence-seconds",
        type=float,
        default=180.0,
        help="Warn when START->THIRD is longer than this value.",
    )
    parser.add_argument(
        "--min-handover-hand-valid-ratio",
        type=float,
        default=0.8,
        help="Warn when neither hand is valid in at least this fraction of DONE->THIRD rows. Default: 0.8.",
    )
    return parser.parse_args()


def sequence_id_from_mps_dir(path: Path) -> str | None:
    name = path.name
    if name.startswith(MPS_PREFIX) and name.endswith(MPS_SUFFIX):
        return name[len(MPS_PREFIX):-len(MPS_SUFFIX)]
    return None


def sequence_id_from_timestamp_key(key: str) -> str:
    return Path(key).stem


def participant_from_sequence(sequence_id: str) -> str:
    return sequence_id.split("_", 1)[0] if "_" in sequence_id else sequence_id


def training_exclusion_reason(sequence_id: str) -> str:
    participant = participant_from_sequence(sequence_id).lower()
    if participant == "test" or sequence_id.lower().startswith("test_"):
        return "test_recording"
    if participant == "unknown" or sequence_id.lower().startswith("unknown"):
        return "unknown_participant"
    return ""


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


def collect_sequences(data_root: Path, backup_dir: Path, timestamps: dict) -> set[str]:
    vrs_dir = data_root / "Data_vrs"
    mp4_dir = data_root / "Data_mp4"
    sequence_ids = set()

    if vrs_dir.exists():
        sequence_ids.update(path.stem for path in vrs_dir.glob("*.vrs"))
        for path in vrs_dir.iterdir():
            if path.is_dir():
                sequence_id = sequence_id_from_mps_dir(path)
                if sequence_id:
                    sequence_ids.add(sequence_id)

    if mp4_dir.exists():
        sequence_ids.update(path.stem for path in mp4_dir.glob("*.mp4"))

    if backup_dir.exists():
        sequence_ids.update(path.stem for path in backup_dir.glob("*.vrs"))

    sequence_ids.update(sequence_id_from_timestamp_key(key) for key in timestamps.keys())
    return sequence_ids


def timestamp_entry_for(sequence_id: str, timestamps: dict) -> dict:
    return timestamps.get(f"{sequence_id}.vrs", timestamps.get(sequence_id, {}))


def extract_command_seconds(timestamp_entry: dict) -> dict[str, float | None]:
    seconds = {}
    for command in EXPECTED_COMMANDS:
        value = timestamp_entry.get(command)
        if isinstance(value, dict):
            seconds[command] = value.get("relative_seconds")
        else:
            seconds[command] = None
    return seconds


def command_timestamp_ns(timestamp_entry: dict, command: str) -> int | None:
    value = timestamp_entry.get(command)
    if not isinstance(value, dict) or not isinstance(value.get("timestamp_ns"), (int, float)):
        return None
    return int(value["timestamp_ns"])


def hand_tracking_phase_stats(path: Path, start_ns: int | None, end_ns: int | None) -> dict:
    stats = {
        "rows": 0,
        "left_valid_rows": 0,
        "right_valid_rows": 0,
        "either_valid_rows": 0,
        "left_valid_ratio": None,
        "right_valid_ratio": None,
        "either_valid_ratio": None,
    }
    if not path.exists() or start_ns is None or end_ns is None or end_ns <= start_ns:
        return stats

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "tracking_timestamp_us",
                "left_tracking_confidence",
                "right_tracking_confidence",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return stats
            for row in reader:
                timestamp_ns = int(row["tracking_timestamp_us"]) * 1000
                if timestamp_ns < start_ns or timestamp_ns > end_ns:
                    continue
                left_valid = float(row["left_tracking_confidence"]) > 0.0
                right_valid = float(row["right_tracking_confidence"]) > 0.0
                stats["rows"] += 1
                stats["left_valid_rows"] += int(left_valid)
                stats["right_valid_rows"] += int(right_valid)
                stats["either_valid_rows"] += int(left_valid or right_valid)
    except (OSError, TypeError, ValueError):
        return stats

    if stats["rows"]:
        stats["left_valid_ratio"] = round(stats["left_valid_rows"] / stats["rows"], 4)
        stats["right_valid_ratio"] = round(stats["right_valid_rows"] / stats["rows"], 4)
        stats["either_valid_ratio"] = round(stats["either_valid_rows"] / stats["rows"], 4)
    return stats


def check_timestamps(command_seconds: dict, min_phase_seconds: float, max_sequence_seconds: float):
    issues = []
    warnings = []
    missing = [command for command in EXPECTED_COMMANDS if command_seconds.get(command) is None]

    if len(missing) == len(EXPECTED_COMMANDS):
        issues.append("missing_timestamps")
        return issues, warnings, missing, None, None, None

    if missing:
        issues.append("partial_timestamps")
        return issues, warnings, missing, None, None, None

    ordered_seconds = [float(command_seconds[command]) for command in EXPECTED_COMMANDS]
    if ordered_seconds != sorted(ordered_seconds):
        issues.append("bad_timestamp_order")

    phase_continue = ordered_seconds[1] - ordered_seconds[0]
    phase_fetch = ordered_seconds[2] - ordered_seconds[1]
    phase_handover = ordered_seconds[3] - ordered_seconds[2]
    sequence_duration = ordered_seconds[3] - ordered_seconds[0]

    phase_lengths = {
        "continue": phase_continue,
        "fetch": phase_fetch,
        "handover": phase_handover,
    }
    for name, duration in phase_lengths.items():
        if duration < min_phase_seconds:
            warnings.append(f"short_{name}_phase")

    if sequence_duration > max_sequence_seconds:
        warnings.append("long_sequence")

    return (
        issues,
        warnings,
        missing,
        round(phase_continue, 3),
        round(phase_fetch, 3),
        round(phase_handover, 3),
    )


def classify_status(issues: list[str], warnings: list[str]) -> str:
    blocking_priority = [
        "missing_vrs",
        "missing_mps",
        "missing_hand_tracking",
        "missing_handover_hand_tracking",
        "missing_slam",
        "missing_timestamps",
        "partial_timestamps",
        "bad_timestamp_order",
    ]
    for issue in blocking_priority:
        if issue in issues:
            return issue
    if warnings:
        return "valid_with_warnings"
    return "valid"


def choose_next_action(issues: list[str], warnings: list[str]) -> str:
    if "missing_vrs" in issues:
        return "download_vrs"
    if any(issue in issues for issue in ("missing_mps", "missing_hand_tracking", "missing_slam")):
        return "download_or_process_mps"
    if any(issue in issues for issue in ("missing_timestamps", "partial_timestamps", "bad_timestamp_order")):
        return "fix_timestamps"
    if "missing_handover_hand_tracking" in issues or "low_handover_hand_tracking" in warnings:
        return "review_or_exclude_sequence"
    if any(warning in warnings for warning in ("missing_aruco_csv", "aruco_timestamp_mismatch")):
        return "run_aruco_extraction"
    if "missing_mp4" in warnings:
        return "convert_mp4"
    if any(warning.startswith("short_") or warning == "long_sequence" for warning in warnings):
        return "manual_review"
    return "ready_for_master_merge"


def timestamp_ns_range(timestamp_entry: dict) -> tuple[int, int] | None:
    values = [
        value.get("timestamp_ns")
        for value in timestamp_entry.values()
        if isinstance(value, dict) and isinstance(value.get("timestamp_ns"), (int, float))
    ]
    if not values:
        return None
    return int(min(values)), int(max(values))


def csv_timestamp_ns_range(path: Path) -> tuple[int, int] | None:
    minimum = None
    maximum = None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "timestamp_ns" not in reader.fieldnames:
                return None
            for row in reader:
                value = row.get("timestamp_ns")
                if not value:
                    continue
                timestamp_ns = int(value)
                minimum = timestamp_ns if minimum is None else min(minimum, timestamp_ns)
                maximum = timestamp_ns if maximum is None else max(maximum, timestamp_ns)
    except (OSError, TypeError, ValueError):
        return None
    if minimum is None or maximum is None:
        return None
    return minimum, maximum


def ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def aruco_csv_candidates(data_root: Path, sequence_id: str) -> list[Path]:
    candidates = [data_root / f"aruco_poses_{sequence_id}.csv"]

    # Older outputs sometimes use only participant + trial, e.g. aruco_poses_Edu_5.csv.
    match = re.match(r"^(.+?_\d+)_\d{8}_\d{6}$", sequence_id)
    if match:
        candidates.append(data_root / f"aruco_poses_{match.group(1)}.csv")
    return candidates


def aruco_csv_for(
    data_root: Path,
    sequence_id: str,
    timestamp_entry: dict,
) -> tuple[Path | None, tuple[int, int] | None, list[Path]]:
    expected_range = timestamp_ns_range(timestamp_entry)
    rejected = []

    for index, candidate in enumerate(aruco_csv_candidates(data_root, sequence_id)):
        if not candidate.exists():
            continue
        actual_range = csv_timestamp_ns_range(candidate)

        # Exact full-sequence filenames remain usable without labels, but their CSV
        # must contain at least one valid timestamp. Legacy compact names are only
        # accepted after proving temporal overlap with this exact recording.
        is_exact = index == 0
        if actual_range is None:
            rejected.append(candidate)
            continue
        if expected_range is None:
            if is_exact:
                return candidate, actual_range, rejected
            rejected.append(candidate)
            continue
        if ranges_overlap(expected_range, actual_range):
            return candidate, actual_range, rejected
        rejected.append(candidate)

    return None, None, rejected


def build_row(sequence_id: str, data_root: Path, backup_dir: Path, timestamps: dict, args) -> dict:
    vrs_dir = data_root / "Data_vrs"
    mp4_dir = data_root / "Data_mp4"
    vrs_path = vrs_dir / f"{sequence_id}.vrs"
    backup_path = backup_dir / f"{sequence_id}.vrs"
    mp4_path = mp4_dir / f"{sequence_id}.mp4"
    mps_dir = vrs_dir / f"mps_{sequence_id}_vrs"
    hand_tracking_path = mps_dir / "hand_tracking" / "hand_tracking_results.csv"
    slam_path = mps_dir / "slam" / "closed_loop_trajectory.csv"

    timestamp_entry = timestamp_entry_for(sequence_id, timestamps)
    done_timestamp_ns = command_timestamp_ns(timestamp_entry, "DONE")
    third_timestamp_ns = command_timestamp_ns(timestamp_entry, "THIRD")
    handover_hand_stats = hand_tracking_phase_stats(
        hand_tracking_path,
        done_timestamp_ns,
        third_timestamp_ns,
    )
    aruco_path, aruco_timestamp_range, rejected_aruco_paths = aruco_csv_for(
        data_root,
        sequence_id,
        timestamp_entry,
    )
    command_seconds = extract_command_seconds(timestamp_entry)
    timestamp_issues, timestamp_warnings, missing_commands, continue_s, fetch_s, handover_s = check_timestamps(
        command_seconds,
        args.min_phase_seconds,
        args.max_sequence_seconds,
    )

    issues = []
    warnings = list(timestamp_warnings)

    if not vrs_path.exists():
        issues.append("missing_vrs")
    if backup_dir.exists() and not backup_path.exists():
        warnings.append("missing_backup_vrs")
    if not mp4_path.exists():
        warnings.append("missing_mp4")
    if not mps_dir.exists():
        issues.append("missing_mps")
    if not hand_tracking_path.exists():
        issues.append("missing_hand_tracking")
    elif done_timestamp_ns is not None and third_timestamp_ns is not None:
        if handover_hand_stats["rows"] == 0 or handover_hand_stats["either_valid_rows"] == 0:
            issues.append("missing_handover_hand_tracking")
        elif handover_hand_stats["either_valid_ratio"] < args.min_handover_hand_valid_ratio:
            warnings.append("low_handover_hand_tracking")
    if not slam_path.exists():
        issues.append("missing_slam")
    if aruco_path is None:
        if rejected_aruco_paths:
            warnings.append("aruco_timestamp_mismatch")
        else:
            warnings.append("missing_aruco_csv")

    issues.extend(timestamp_issues)
    exclusion_reason = training_exclusion_reason(sequence_id)
    if exclusion_reason:
        warnings.append(f"exclude_from_training:{exclusion_reason}")

    backup_size = backup_path.stat().st_size if backup_path.exists() else None
    vrs_size = vrs_path.stat().st_size if vrs_path.exists() else None
    backup_size_delta = None
    if backup_size is not None and vrs_size is not None:
        backup_size_delta = vrs_size - backup_size

    row = {
        "sequence_id": sequence_id,
        "participant": participant_from_sequence(sequence_id),
        "include_in_training": not bool(exclusion_reason),
        "exclusion_reason": exclusion_reason,
        "status": classify_status(issues, warnings),
        "next_action": choose_next_action(issues, warnings),
        "issues": ";".join(issues),
        "warnings": ";".join(warnings),
        "backup_vrs_exists": backup_path.exists(),
        "backup_vrs_size_mb": file_size_mb(backup_path),
        "backup_size_delta_bytes": backup_size_delta,
        "vrs_exists": vrs_path.exists(),
        "vrs_size_mb": file_size_mb(vrs_path),
        "mp4_exists": mp4_path.exists(),
        "mp4_size_mb": file_size_mb(mp4_path),
        "mps_exists": mps_dir.exists(),
        "hand_tracking_exists": hand_tracking_path.exists(),
        "hand_tracking_rows": count_csv_rows(hand_tracking_path),
        "handover_hand_rows": handover_hand_stats["rows"],
        "handover_left_valid_rows": handover_hand_stats["left_valid_rows"],
        "handover_right_valid_rows": handover_hand_stats["right_valid_rows"],
        "handover_either_valid_rows": handover_hand_stats["either_valid_rows"],
        "handover_left_valid_ratio": handover_hand_stats["left_valid_ratio"],
        "handover_right_valid_ratio": handover_hand_stats["right_valid_ratio"],
        "handover_either_valid_ratio": handover_hand_stats["either_valid_ratio"],
        "slam_exists": slam_path.exists(),
        "slam_rows": count_csv_rows(slam_path),
        "aruco_csv_exists": aruco_path is not None,
        "aruco_csv_path": str(aruco_path) if aruco_path else "",
        "aruco_rows": count_csv_rows(aruco_path) if aruco_path else None,
        "aruco_timestamp_start_ns": aruco_timestamp_range[0] if aruco_timestamp_range else None,
        "aruco_timestamp_end_ns": aruco_timestamp_range[1] if aruco_timestamp_range else None,
        "aruco_rejected_paths": ";".join(str(path) for path in rejected_aruco_paths),
        "timestamps_exists": bool(timestamp_entry),
        "missing_commands": ";".join(missing_commands),
        "start_s": command_seconds["START"],
        "second_s": command_seconds["SECOND"],
        "done_s": command_seconds["DONE"],
        "third_s": command_seconds["THIRD"],
        "continue_duration_s": continue_s,
        "fetch_duration_s": fetch_s,
        "handover_duration_s": handover_s,
    }
    return row


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "participant",
        "include_in_training",
        "exclusion_reason",
        "status",
        "next_action",
        "issues",
        "warnings",
        "backup_vrs_exists",
        "backup_vrs_size_mb",
        "backup_size_delta_bytes",
        "vrs_exists",
        "vrs_size_mb",
        "mp4_exists",
        "mp4_size_mb",
        "mps_exists",
        "hand_tracking_exists",
        "hand_tracking_rows",
        "handover_hand_rows",
        "handover_left_valid_rows",
        "handover_right_valid_rows",
        "handover_either_valid_rows",
        "handover_left_valid_ratio",
        "handover_right_valid_ratio",
        "handover_either_valid_ratio",
        "slam_exists",
        "slam_rows",
        "aruco_csv_exists",
        "aruco_csv_path",
        "aruco_rows",
        "aruco_timestamp_start_ns",
        "aruco_timestamp_end_ns",
        "aruco_rejected_paths",
        "timestamps_exists",
        "missing_commands",
        "start_s",
        "second_s",
        "done_s",
        "third_s",
        "continue_duration_s",
        "fetch_duration_s",
        "handover_duration_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_backup_sync_report(rows: list[dict]) -> dict:
    missing_in_data = [row["sequence_id"] for row in rows if row["backup_vrs_exists"] and not row["vrs_exists"]]
    missing_in_backup = [row["sequence_id"] for row in rows if row["vrs_exists"] and not row["backup_vrs_exists"]]
    size_mismatches = [
        {
            "sequence_id": row["sequence_id"],
            "delta_bytes": row["backup_size_delta_bytes"],
        }
        for row in rows
        if row["backup_size_delta_bytes"] not in (None, 0)
    ]
    return {
        "backup_vrs_count": sum(1 for row in rows if row["backup_vrs_exists"]),
        "data_vrs_count": sum(1 for row in rows if row["vrs_exists"]),
        "missing_in_data_vrs": missing_in_data,
        "missing_in_backup": missing_in_backup,
        "size_mismatch_count": len(size_mismatches),
        "size_mismatches": size_mismatches[:50],
    }


def build_mps_report(rows: list[dict]) -> dict:
    missing_mps = [row["sequence_id"] for row in rows if "missing_mps" in row["issues"].split(";")]
    incomplete_mps = [
        row["sequence_id"]
        for row in rows
        if row["mps_exists"] and (not row["hand_tracking_exists"] or not row["slam_exists"])
    ]
    return {
        "missing_mps_count": len(missing_mps),
        "missing_mps_sequence_ids": missing_mps,
        "incomplete_mps_count": len(incomplete_mps),
        "incomplete_mps_sequence_ids": incomplete_mps,
    }


def build_report(rows: list[dict], timestamps_path: Path, manifest_path: Path, backup_dir: Path) -> dict:
    status_counts = Counter(row["status"] for row in rows)
    action_counts = Counter(row["next_action"] for row in rows)
    issue_counts = Counter(
        issue
        for row in rows
        for issue in row["issues"].split(";")
        if issue
    )
    warning_counts = Counter(
        warning
        for row in rows
        for warning in row["warnings"].split(";")
        if warning
    )
    participant_counts = defaultdict(lambda: Counter())
    for row in rows:
        participant_counts[row["participant"]][row["status"]] += 1

    valid_rows = [row for row in rows if row["status"] in {"valid", "valid_with_warnings"}]
    included_rows = [row for row in rows if row["include_in_training"]]
    excluded_rows = [row for row in rows if not row["include_in_training"]]
    return {
        "total_sequences": len(rows),
        "training_include_candidates": len(included_rows),
        "training_excluded_candidates": len(excluded_rows),
        "excluded_sequence_ids": [row["sequence_id"] for row in excluded_rows],
        "valid_or_warning_sequences": len(valid_rows),
        "status_counts": dict(status_counts),
        "next_action_counts": dict(action_counts),
        "issue_counts": dict(issue_counts),
        "warning_counts": dict(warning_counts),
        "participants": {
            participant: dict(counts)
            for participant, counts in sorted(participant_counts.items())
        },
        "backup_sync": build_backup_sync_report(rows),
        "mps": build_mps_report(rows),
        "timestamps_source": str(timestamps_path),
        "manifest_csv": str(manifest_path),
        "backup_dir": str(backup_dir),
        "expected_command_order": list(EXPECTED_COMMANDS),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    data_root = args.data_root
    timestamps_path = args.timestamps or data_root / "Data_vrs" / "timestamps_summary.json"
    manifest_path = args.manifest_out or data_root / "dataset_manifest.csv"
    report_path = args.report_out or data_root / "dataset_qa_report.json"

    timestamps = read_json(timestamps_path)
    sequence_ids = sorted(collect_sequences(data_root, args.backup_dir, timestamps))
    rows = [build_row(sequence_id, data_root, args.backup_dir, timestamps, args) for sequence_id in sequence_ids]

    write_manifest(manifest_path, rows)
    report = build_report(rows, timestamps_path, manifest_path, args.backup_dir)
    write_report(report_path, report)

    print(f"Manifest written: {manifest_path}")
    print(f"Report written:   {report_path}")
    print(f"Sequences:        {report['total_sequences']}")
    print(f"Training incl.:   {report['training_include_candidates']}")
    print(f"Training excl.:   {report['training_excluded_candidates']}")
    print(f"Usable now:       {report['valid_or_warning_sequences']}")
    print(
        "Backup sync:      "
        f"{len(report['backup_sync']['missing_in_data_vrs'])} missing in Data_vrs, "
        f"{len(report['backup_sync']['missing_in_backup'])} missing in backup"
    )
    print("Status counts:")
    for status, count in sorted(report["status_counts"].items()):
        print(f"  {status}: {count}")
    print("Next actions:")
    for action, count in sorted(report["next_action_counts"].items()):
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()
