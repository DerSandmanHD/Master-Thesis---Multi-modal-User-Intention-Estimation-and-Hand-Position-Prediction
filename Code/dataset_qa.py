#!/usr/bin/env python3
"""Create a QA manifest for Aria recordings and derived processing outputs."""

import argparse
import csv
import json
import re
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path

from annotation_utils import COMMANDS, parse_target_object_id, read_review_rows

EXPECTED_COMMANDS = COMMANDS
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
        "--annotations",
        type=Path,
        default=None,
        help="Manual review/semantic annotation CSV. Defaults to Data_collection/manual_timestamp_review.csv.",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="WAV directory. Defaults to Data_collection/Data_vrs/debug_audio.",
    )
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=None,
        help="Master dataset directory. Defaults to Data_collection/master_datasets.",
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
        help="Warn when neither hand is valid in at least this fraction of THIRD->recording-end rows. Default: 0.8.",
    )
    parser.add_argument(
        "--max-media-duration-delta-seconds",
        type=float,
        default=0.1,
        help="Warn when matching MP4 and WAV durations differ by more than this value.",
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


def wav_duration_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            return round(wav_file.getnframes() / float(wav_file.getframerate()), 3)
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return None


def mp4_duration_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return round(float(completed.stdout.strip()), 3)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def collect_sequences(
    data_root: Path,
    backup_dir: Path,
    timestamps: dict,
    annotations: dict,
) -> set[str]:
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
    sequence_ids.update(annotations.keys())
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


def hand_tracking_phase_stats(path: Path, start_ns: int | None, end_ns: int | None = None) -> dict:
    stats = {
        "rows": 0,
        "left_valid_rows": 0,
        "right_valid_rows": 0,
        "either_valid_rows": 0,
        "left_valid_ratio": None,
        "right_valid_ratio": None,
        "either_valid_ratio": None,
    }
    if not path.exists() or start_ns is None or (end_ns is not None and end_ns <= start_ns):
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
                if timestamp_ns < start_ns or (end_ns is not None and timestamp_ns > end_ns):
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


def check_timestamps(
    command_seconds: dict,
    min_phase_seconds: float,
    max_sequence_seconds: float,
    recording_duration_seconds: float | None = None,
):
    issues = []
    warnings = []
    missing = [command for command in EXPECTED_COMMANDS if command_seconds.get(command) is None]

    if len(missing) == len(EXPECTED_COMMANDS):
        issues.append("missing_timestamps")
        return issues, warnings, missing, None, None, None, None

    if missing:
        issues.append("partial_timestamps")
        return issues, warnings, missing, None, None, None, None

    ordered_seconds = [float(command_seconds[command]) for command in EXPECTED_COMMANDS]
    if ordered_seconds != sorted(ordered_seconds):
        issues.append("bad_timestamp_order")

    phase_continue = ordered_seconds[1] - ordered_seconds[0]
    phase_fetch = ordered_seconds[2] - ordered_seconds[1]
    phase_transition = ordered_seconds[3] - ordered_seconds[2]
    phase_handover = (
        recording_duration_seconds - ordered_seconds[3]
        if recording_duration_seconds is not None
        else None
    )
    sequence_duration = (
        recording_duration_seconds - ordered_seconds[0]
        if recording_duration_seconds is not None
        else ordered_seconds[3] - ordered_seconds[0]
    )

    phase_lengths = {
        "continue": phase_continue,
        "fetch": phase_fetch,
        "transition": phase_transition,
    }
    if phase_handover is not None:
        phase_lengths["handover"] = phase_handover
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
        round(phase_transition, 3),
        round(phase_handover, 3) if phase_handover is not None else None,
    )


def manual_review_application_issues(annotation: dict, timestamp_entry: dict) -> list[str]:
    if annotation.get("decision") != "manual_fix":
        return []
    issues = []
    manual_columns = {
        "START": "manual_start_s",
        "SECOND": "manual_second_s",
        "DONE": "manual_done_s",
        "THIRD": "manual_third_s",
    }
    manual_commands = [
        command
        for command, column in manual_columns.items()
        if str(annotation.get(column, "")).strip()
    ]
    if not manual_commands:
        return ["manual_fix_without_manual_times"]
    for command in manual_commands:
        column = manual_columns[command]
        entry = timestamp_entry.get(command, {})
        source = entry.get("timestamp_source") if isinstance(entry, dict) else None
        if source not in {"manual_review", "manual_override"}:
            issues.append(f"manual_{command.lower()}_not_applied")
            continue
        try:
            expected = float(annotation[column])
            actual = float(entry["relative_seconds"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"manual_{command.lower()}_value_missing")
            continue
        if abs(expected - actual) > 0.001:
            issues.append(f"manual_{command.lower()}_value_mismatch")
    return issues


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
        "timestamp_review_uncertain",
        "manual_review_not_applied",
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
    if "timestamp_review_uncertain" in issues or "manual_review_not_applied" in issues:
        return "fix_timestamps"
    if "missing_handover_hand_tracking" in issues or "low_handover_hand_tracking" in warnings:
        return "review_or_exclude_sequence"
    if any(warning in warnings for warning in ("missing_aruco_csv", "aruco_timestamp_mismatch")):
        return "run_aruco_extraction"
    if "missing_mp4" in warnings or "invalid_mp4" in warnings:
        return "convert_mp4"
    if "missing_wav" in warnings or "invalid_wav" in warnings:
        return "extract_wav"
    if any(
        warning in warnings
        for warning in (
            "missing_target_object_annotation",
            "missing_receiving_hand_annotation",
            "uncertain_semantic_annotation",
            "target_object_not_detected",
            "receiving_hand_tracking_low",
        )
    ):
        return "annotate_sequence"
    if "missing_master_dataset" in warnings:
        return "build_master_dataset"
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


def csv_object_marker_ids(path: Path | None) -> list[int]:
    if path is None or not path.exists():
        return []
    marker_ids = set()
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"marker_family", "marker_id"}.issubset(reader.fieldnames):
                return []
            for row in reader:
                if row.get("marker_family") != "aruco_4x4_50":
                    continue
                marker_id = int(row["marker_id"])
                if marker_id in range(6, 15):
                    marker_ids.add(marker_id)
    except (OSError, TypeError, ValueError):
        return []
    return sorted(marker_ids)


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


def build_row(
    sequence_id: str,
    data_root: Path,
    backup_dir: Path,
    timestamps: dict,
    annotations: dict,
    wav_dir: Path,
    master_dir: Path,
    args,
) -> dict:
    vrs_dir = data_root / "Data_vrs"
    mp4_dir = data_root / "Data_mp4"
    vrs_path = vrs_dir / f"{sequence_id}.vrs"
    backup_path = backup_dir / f"{sequence_id}.vrs"
    mp4_path = mp4_dir / f"{sequence_id}.mp4"
    wav_path = wav_dir / f"{sequence_id}.wav"
    master_path = master_dir / f"{sequence_id}_master.csv"
    master_report_path = master_dir / f"{sequence_id}_master_report.json"
    mps_dir = vrs_dir / f"mps_{sequence_id}_vrs"
    hand_tracking_path = mps_dir / "hand_tracking" / "hand_tracking_results.csv"
    slam_path = mps_dir / "slam" / "closed_loop_trajectory.csv"

    timestamp_entry = timestamp_entry_for(sequence_id, timestamps)
    annotation = annotations.get(sequence_id, {})
    third_timestamp_ns = command_timestamp_ns(timestamp_entry, "THIRD")
    handover_hand_stats = hand_tracking_phase_stats(
        hand_tracking_path,
        third_timestamp_ns,
    )
    aruco_path, aruco_timestamp_range, rejected_aruco_paths = aruco_csv_for(
        data_root,
        sequence_id,
        timestamp_entry,
    )
    aruco_object_ids = csv_object_marker_ids(aruco_path)
    command_seconds = extract_command_seconds(timestamp_entry)
    wav_duration = wav_duration_seconds(wav_path)
    timestamp_issues, timestamp_warnings, missing_commands, continue_s, fetch_s, transition_s, handover_s = check_timestamps(
        command_seconds,
        args.min_phase_seconds,
        args.max_sequence_seconds,
        wav_duration,
    )

    issues = []
    warnings = list(timestamp_warnings)
    review_application_issues = manual_review_application_issues(annotation, timestamp_entry)
    review_decision = annotation.get("decision", "")
    target_object_id = parse_target_object_id(annotation.get("target_object_id"))
    receiving_hand = annotation.get("receiving_hand", "")
    annotation_confidence = annotation.get("annotation_confidence", "")
    mp4_duration = mp4_duration_seconds(mp4_path)
    media_duration_delta = (
        round(wav_duration - mp4_duration, 3)
        if wav_duration is not None and mp4_duration is not None
        else None
    )

    if not vrs_path.exists():
        issues.append("missing_vrs")
    if backup_dir.exists() and not backup_path.exists():
        warnings.append("missing_backup_vrs")
    if not mp4_path.exists():
        warnings.append("missing_mp4")
    elif mp4_duration is None:
        warnings.append("invalid_mp4")
    if not wav_path.exists():
        warnings.append("missing_wav")
    elif wav_duration is None:
        warnings.append("invalid_wav")
    if (
        media_duration_delta is not None
        and abs(media_duration_delta) > args.max_media_duration_delta_seconds
    ):
        warnings.append("mp4_wav_duration_mismatch")
    if not mps_dir.exists():
        issues.append("missing_mps")
    if not hand_tracking_path.exists():
        issues.append("missing_hand_tracking")
    elif third_timestamp_ns is not None:
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
    if review_decision == "uncertain":
        issues.append("timestamp_review_uncertain")
    if review_application_issues:
        issues.append("manual_review_not_applied")

    exclusion_reasons = []
    automatic_exclusion = training_exclusion_reason(sequence_id)
    if automatic_exclusion:
        exclusion_reasons.append(automatic_exclusion)
    if review_decision == "exclude":
        exclusion_reasons.append("manual_exclusion")
    exclusion_reason = ";".join(exclusion_reasons)
    if exclusion_reason:
        for reason in exclusion_reasons:
            warnings.append(f"exclude_from_training:{reason}")
    else:
        if target_object_id is None:
            warnings.append("missing_target_object_annotation")
        if receiving_hand not in {"left", "right"}:
            warnings.append("missing_receiving_hand_annotation")
        if annotation_confidence == "uncertain":
            warnings.append("uncertain_semantic_annotation")
        if target_object_id is not None and target_object_id not in aruco_object_ids:
            warnings.append("target_object_not_detected")
        if receiving_hand == "left" and (
            handover_hand_stats["left_valid_ratio"] is None
            or handover_hand_stats["left_valid_ratio"] < args.min_handover_hand_valid_ratio
        ):
            warnings.append("receiving_hand_tracking_low")
        if receiving_hand == "right" and (
            handover_hand_stats["right_valid_ratio"] is None
            or handover_hand_stats["right_valid_ratio"] < args.min_handover_hand_valid_ratio
        ):
            warnings.append("receiving_hand_tracking_low")
    if not master_path.exists() or not master_report_path.exists():
        warnings.append("missing_master_dataset")

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
        "mp4_duration_s": mp4_duration,
        "wav_exists": wav_path.exists(),
        "wav_size_mb": file_size_mb(wav_path),
        "wav_duration_s": wav_duration,
        "mp4_wav_duration_delta_s": media_duration_delta,
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
        "aruco_object_ids": ";".join(str(marker_id) for marker_id in aruco_object_ids),
        "aruco_timestamp_start_ns": aruco_timestamp_range[0] if aruco_timestamp_range else None,
        "aruco_timestamp_end_ns": aruco_timestamp_range[1] if aruco_timestamp_range else None,
        "aruco_rejected_paths": ";".join(str(path) for path in rejected_aruco_paths),
        "timestamps_exists": bool(timestamp_entry),
        "timestamp_sources": ";".join(
            sorted(
                {
                    str(value.get("timestamp_source", "unknown"))
                    for value in timestamp_entry.values()
                    if isinstance(value, dict) and "timestamp_ns" in value
                }
            )
        ),
        "review_decision": review_decision,
        "review_application_issues": ";".join(review_application_issues),
        "target_object_id": target_object_id,
        "receiving_hand": receiving_hand,
        "annotation_confidence": annotation_confidence,
        "annotation_notes": annotation.get("notes", ""),
        "missing_commands": ";".join(missing_commands),
        "start_s": command_seconds["START"],
        "second_s": command_seconds["SECOND"],
        "done_s": command_seconds["DONE"],
        "third_s": command_seconds["THIRD"],
        "continue_duration_s": continue_s,
        "fetch_duration_s": fetch_s,
        "transition_duration_s": transition_s,
        "handover_duration_s": handover_s,
        "master_csv_exists": master_path.exists(),
        "master_csv_rows": count_csv_rows(master_path),
        "master_report_exists": master_report_path.exists(),
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
        "mp4_duration_s",
        "wav_exists",
        "wav_size_mb",
        "wav_duration_s",
        "mp4_wav_duration_delta_s",
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
        "aruco_object_ids",
        "aruco_timestamp_start_ns",
        "aruco_timestamp_end_ns",
        "aruco_rejected_paths",
        "timestamps_exists",
        "timestamp_sources",
        "review_decision",
        "review_application_issues",
        "target_object_id",
        "receiving_hand",
        "annotation_confidence",
        "annotation_notes",
        "missing_commands",
        "start_s",
        "second_s",
        "done_s",
        "third_s",
        "continue_duration_s",
        "fetch_duration_s",
        "transition_duration_s",
        "handover_duration_s",
        "master_csv_exists",
        "master_csv_rows",
        "master_report_exists",
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


def build_report(
    rows: list[dict],
    timestamps_path: Path,
    annotations_path: Path,
    manifest_path: Path,
    backup_dir: Path,
) -> dict:
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
        "annotations": {
            "source": str(annotations_path),
            "review_decision_counts": dict(Counter(row["review_decision"] or "not_reviewed" for row in rows)),
            "target_object_labeled": sum(row["target_object_id"] is not None for row in rows),
            "receiving_hand_labeled": sum(row["receiving_hand"] in {"left", "right"} for row in rows),
            "uncertain": sum(row["annotation_confidence"] == "uncertain" for row in rows),
        },
        "media": {
            "wav_available": sum(row["wav_exists"] for row in rows),
            "mp4_available": sum(row["mp4_exists"] for row in rows),
            "duration_mismatches": sum(
                "mp4_wav_duration_mismatch" in row["warnings"].split(";") for row in rows
            ),
        },
        "master_datasets": {
            "csv_available": sum(row["master_csv_exists"] for row in rows),
            "reports_available": sum(row["master_report_exists"] for row in rows),
        },
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
    data_root = args.data_root.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()
    timestamps_path = args.timestamps or data_root / "Data_vrs" / "timestamps_summary.json"
    annotations_path = args.annotations or data_root / "manual_timestamp_review.csv"
    wav_dir = args.wav_dir or data_root / "Data_vrs" / "debug_audio"
    master_dir = args.master_dir or data_root / "master_datasets"
    manifest_path = args.manifest_out or data_root / "dataset_manifest.csv"
    report_path = args.report_out or data_root / "dataset_qa_report.json"

    timestamps = read_json(timestamps_path)
    annotations = read_review_rows(annotations_path)
    sequence_ids = sorted(collect_sequences(data_root, backup_dir, timestamps, annotations))
    rows = [
        build_row(
            sequence_id,
            data_root,
            backup_dir,
            timestamps,
            annotations,
            wav_dir,
            master_dir,
            args,
        )
        for sequence_id in sequence_ids
    ]

    write_manifest(manifest_path, rows)
    report = build_report(rows, timestamps_path, annotations_path, manifest_path, backup_dir)
    write_report(report_path, report)

    print(f"Manifest written: {manifest_path}")
    print(f"Report written:   {report_path}")
    print(f"Sequences:        {report['total_sequences']}")
    print(f"Training incl.:   {report['training_include_candidates']}")
    print(f"Training excl.:   {report['training_excluded_candidates']}")
    print(f"Usable now:       {report['valid_or_warning_sequences']}")
    print(f"Target labels:    {report['annotations']['target_object_labeled']}")
    print(f"Receiving hands:  {report['annotations']['receiving_hand_labeled']}")
    print(f"Master datasets:  {report['master_datasets']['csv_available']}")
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
