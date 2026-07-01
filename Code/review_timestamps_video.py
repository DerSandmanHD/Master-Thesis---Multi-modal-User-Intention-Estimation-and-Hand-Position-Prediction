#!/usr/bin/env python3
"""Interactive video review tool for command timestamps.

Reads Data_collection/dataset_manifest.csv, opens matching MP4 files, overlays
automatic timestamps, and writes manual corrections to
Data_collection/manual_timestamp_review.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2


COMMANDS = ["START", "SECOND", "DONE", "THIRD"]
TIME_COLUMNS = {
    "START": "start_s",
    "SECOND": "second_s",
    "DONE": "done_s",
    "THIRD": "third_s",
}

REVIEW_FIELDS = [
    "sequence_id",
    "decision",
    "auto_start_s",
    "auto_second_s",
    "auto_done_s",
    "auto_third_s",
    "manual_start_s",
    "manual_second_s",
    "manual_done_s",
    "manual_third_s",
    "missing_commands",
    "status",
    "next_action",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data_collection"),
        help="Path to Data_collection.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to dataset_manifest.csv. Default: Data_collection/dataset_manifest.csv",
    )
    parser.add_argument(
        "--mp4-dir",
        type=Path,
        default=None,
        help="Path to MP4 directory. Default: Data_collection/Data_mp4",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Manual review CSV. Default: Data_collection/manual_timestamp_review.csv",
    )
    parser.add_argument(
        "--only-next-action",
        type=str,
        default=None,
        help="Only review rows with this next_action, e.g. fix_timestamps.",
    )
    parser.add_argument(
        "--only-status",
        type=str,
        default=None,
        help="Only review rows with this status, e.g. partial_timestamps.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Review only one sequence_id.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Also include rows where include_in_training is False.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start at this index in the filtered review list.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_review_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            seq = row.get("sequence_id", "").strip()
            if seq:
                rows[seq] = row
        return rows


def write_review_csv(path: Path, rows_by_seq: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()

        for seq in sorted(rows_by_seq):
            row = {field: rows_by_seq[seq].get(field, "") for field in REVIEW_FIELDS}
            writer.writerow(row)


def to_float(value: str) -> Optional[float]:
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_time(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    return f"{value:.3f}s"


def find_video(mp4_dir: Path, sequence_id: str) -> Optional[Path]:
    candidates = [
        mp4_dir / f"{sequence_id}.mp4",
        mp4_dir / f"{sequence_id}.MP4",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(mp4_dir.glob(f"{sequence_id}*.mp4"))
    if matches:
        return matches[0]

    matches = sorted(mp4_dir.glob(f"{sequence_id}*.MP4"))
    if matches:
        return matches[0]

    return None


def initial_seek_time(row: Dict[str, str]) -> float:
    missing = row.get("missing_commands", "").strip()
    start_s = to_float(row.get("start_s", ""))
    second_s = to_float(row.get("second_s", ""))
    done_s = to_float(row.get("done_s", ""))
    third_s = to_float(row.get("third_s", ""))

    if missing:
        missing_set = {x.strip().upper() for x in missing.split(";") if x.strip()}

        if "DONE" in missing_set and second_s is not None:
            return max(0.0, second_s - 3.0)
        if "THIRD" in missing_set and done_s is not None:
            return max(0.0, done_s - 3.0)
        if "SECOND" in missing_set and start_s is not None:
            return max(0.0, start_s - 1.0)
        if "START" in missing_set:
            return 0.0

    first_available = next(
        (t for t in [start_s, second_s, done_s, third_s] if t is not None),
        0.0,
    )
    return max(0.0, first_available - 2.0)


def make_review_row(row: Dict[str, str], existing: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    review = {field: "" for field in REVIEW_FIELDS}

    if existing:
        review.update(existing)

    review["sequence_id"] = row.get("sequence_id", "")
    review["auto_start_s"] = row.get("start_s", "")
    review["auto_second_s"] = row.get("second_s", "")
    review["auto_done_s"] = row.get("done_s", "")
    review["auto_third_s"] = row.get("third_s", "")
    review["missing_commands"] = row.get("missing_commands", "")
    review["status"] = row.get("status", "")
    review["next_action"] = row.get("next_action", "")

    return review


def set_manual_time(review: Dict[str, str], command: str, time_s: float) -> None:
    column = {
        "START": "manual_start_s",
        "SECOND": "manual_second_s",
        "DONE": "manual_done_s",
        "THIRD": "manual_third_s",
    }[command]
    review[column] = f"{time_s:.3f}"
    if not review.get("decision"):
        review["decision"] = "manual_fix"


def put_text_lines(frame, lines: List[str]) -> None:
    x = 20
    y = 30
    line_height = 26

    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(
            frame,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def review_video(
    row: Dict[str, str],
    video_path: Path,
    review_rows: Dict[str, Dict[str, str]],
    review_csv_path: Path,
    index: int,
    total: int,
) -> str:
    sequence_id = row["sequence_id"]
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if frame_count > 0 else 0.0

    start_time = initial_seek_time(row)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    existing = review_rows.get(sequence_id)
    review = make_review_row(row, existing)
    review_rows[sequence_id] = review

    paused = False

    auto_times = {
        "START": to_float(row.get("start_s", "")),
        "SECOND": to_float(row.get("second_s", "")),
        "DONE": to_float(row.get("done_s", "")),
        "THIRD": to_float(row.get("third_s", "")),
    }

    print()
    print("============================================================")
    print(f"[{index + 1}/{total}] {sequence_id}")
    print(f"Video: {video_path}")
    print(f"Start review at: {start_time:.3f}s")
    print("Keys: 1 START, 2 SECOND, 3 DONE, 4 THIRD, space pause, n next, p previous, q quit")
    print("      a -1s, d +1s, z -5s, c +5s, v accept_auto, e exclude, u uncertain")
    print("============================================================")

    last_frame = None

    while True:
        if not paused or last_frame is None:
            ok, frame = cap.read()
            if not ok:
                return "next"
            last_frame = frame
        else:
            frame = last_frame.copy()

        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_s = current_msec / 1000.0

        lines = [
            f"[{index + 1}/{total}] {sequence_id}",
            f"t={current_s:.3f}s / {duration_s:.3f}s",
            f"status={row.get('status', '')} next={row.get('next_action', '')}",
            f"missing={row.get('missing_commands', '') or 'none'}",
            "",
            f"AUTO  START  {fmt_time(auto_times['START'])}",
            f"AUTO  SECOND {fmt_time(auto_times['SECOND'])}",
            f"AUTO  DONE   {fmt_time(auto_times['DONE'])}",
            f"AUTO  THIRD  {fmt_time(auto_times['THIRD'])}",
            "",
            f"MAN   START  {review.get('manual_start_s', '') or '-'}",
            f"MAN   SECOND {review.get('manual_second_s', '') or '-'}",
            f"MAN   DONE   {review.get('manual_done_s', '') or '-'}",
            f"MAN   THIRD  {review.get('manual_third_s', '') or '-'}",
            f"decision={review.get('decision', '') or '-'}",
        ]

        put_text_lines(frame, lines)
        cv2.imshow("Timestamp review", frame)

        delay = 30 if not paused else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == 255:
            continue

        if key == ord(" "):
            paused = not paused

        elif key == ord("q"):
            write_review_csv(review_csv_path, review_rows)
            cap.release()
            cv2.destroyAllWindows()
            return "quit"

        elif key == ord("n"):
            write_review_csv(review_csv_path, review_rows)
            cap.release()
            return "next"

        elif key == ord("p"):
            write_review_csv(review_csv_path, review_rows)
            cap.release()
            return "previous"

        elif key == ord("a"):
            new_t = max(0.0, current_s - 1.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_t * 1000.0)
            last_frame = None

        elif key == ord("d"):
            new_t = min(duration_s, current_s + 1.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_t * 1000.0)
            last_frame = None

        elif key == ord("z"):
            new_t = max(0.0, current_s - 5.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_t * 1000.0)
            last_frame = None

        elif key == ord("c"):
            new_t = min(duration_s, current_s + 5.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_t * 1000.0)
            last_frame = None

        elif key == ord("1"):
            set_manual_time(review, "START", current_s)
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: START -> {current_s:.3f}s")

        elif key == ord("2"):
            set_manual_time(review, "SECOND", current_s)
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: SECOND -> {current_s:.3f}s")

        elif key == ord("3"):
            set_manual_time(review, "DONE", current_s)
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: DONE -> {current_s:.3f}s")

        elif key == ord("4"):
            set_manual_time(review, "THIRD", current_s)
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: THIRD -> {current_s:.3f}s")

        elif key == ord("v"):
            review["decision"] = "accept_auto"
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: accept_auto")

        elif key == ord("e"):
            review["decision"] = "exclude"
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: exclude")

        elif key == ord("u"):
            review["decision"] = "uncertain"
            write_review_csv(review_csv_path, review_rows)
            print(f"{sequence_id}: uncertain")


def filter_rows(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    filtered = []

    for row in rows:
        seq = row.get("sequence_id", "")

        if args.sequence and seq != args.sequence:
            continue

        if not args.include_excluded:
            include = row.get("include_in_training", "").strip().lower() == "true"
            if not include:
                continue

        if args.only_next_action and row.get("next_action", "") != args.only_next_action:
            continue

        if args.only_status and row.get("status", "") != args.only_status:
            continue

        filtered.append(row)

    return filtered


def main() -> int:
    args = parse_args()

    data_root = args.data_root
    manifest_path = args.manifest or data_root / "dataset_manifest.csv"
    mp4_dir = args.mp4_dir or data_root / "Data_mp4"
    review_csv_path = args.review_csv or data_root / "manual_timestamp_review.csv"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 2

    if not mp4_dir.exists():
        print(f"MP4 dir not found: {mp4_dir}")
        return 2

    rows = read_manifest(manifest_path)
    rows = filter_rows(rows, args)

    if not rows:
        print("No rows to review.")
        return 0

    review_rows = read_review_csv(review_csv_path)

    i = max(0, min(args.start_index, len(rows) - 1))

    while 0 <= i < len(rows):
        row = rows[i]
        seq = row["sequence_id"]
        video_path = find_video(mp4_dir, seq)

        if video_path is None:
            print(f"[{i + 1}/{len(rows)}] Missing MP4 for {seq}, skipping.")
            i += 1
            continue

        action = review_video(
            row=row,
            video_path=video_path,
            review_rows=review_rows,
            review_csv_path=review_csv_path,
            index=i,
            total=len(rows),
        )

        if action == "quit":
            break
        if action == "previous":
            i = max(0, i - 1)
        else:
            i += 1

    write_review_csv(review_csv_path, review_rows)
    print(f"Review saved to: {review_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
