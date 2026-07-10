#!/usr/bin/env python3
"""Interactive MP4 + WAV timestamp review tool.

Shows the MP4 with OpenCV, plays the matching WAV with sounddevice, and lets you
write manual command timestamp corrections using keyboard shortcuts.

Inputs:
  Data_collection/dataset_manifest.csv
  Data_collection/Data_mp4/<sequence_id>.mp4
  Data_collection/Data_vrs/debug_audio/<sequence_id>.wav
  Data_collection/aruco_poses_<sequence_id>.csv (optional ID fallback)

Output:
  Data_collection/manual_timestamp_review.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import sounddevice as sd
import soundfile as sf

from annotation_utils import OBJECT_MARKER_IDS, REVIEW_FIELDS, read_review_rows, write_review_rows


COMMAND_TO_MANUAL_COLUMN = {
    "START": "manual_start_s",
    "SECOND": "manual_second_s",
    "DONE": "manual_done_s",
    "THIRD": "manual_third_s",
}

OBJECT_MARKER_IDS = set(OBJECT_MARKER_IDS)


class WavPlayer:
    def __init__(self, wav_path: Path):
        self.wav_path = wav_path
        self.audio, self.sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        self.duration_s = len(self.audio) / float(self.sample_rate)
        self.stream: Optional[sd.OutputStream] = None
        self.position_sample = 0
        self.next_sample = 0
        self.playing = False
        self.finished = False
        self.first_dac_time: Optional[float] = None
        self.first_dac_sample = 0
        self.last_status = ""

    def _callback(self, outdata, frames, time_info, status):
        if status:
            self.last_status = str(status)

        start = self.next_sample
        end = min(start + frames, len(self.audio))
        chunk = self.audio[start:end]

        outdata.fill(0)

        if len(chunk) > 0:
            outdata[: len(chunk), :] = chunk

        if self.first_dac_time is None:
            self.first_dac_time = float(time_info.outputBufferDacTime)
            self.first_dac_sample = start

        self.next_sample = end

        if self.next_sample >= len(self.audio):
            self.finished = True
            raise sd.CallbackStop

    def _finished_callback(self) -> None:
        if self.finished:
            self.position_sample = len(self.audio)
        self.playing = False

    def play(self) -> None:
        if self.playing:
            return

        self._close_stream()
        self.next_sample = self.position_sample
        self.first_dac_time = None
        self.first_dac_sample = self.position_sample
        self.finished = self.position_sample >= len(self.audio)
        if self.finished:
            return

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.audio.shape[1],
                dtype="float32",
                callback=self._callback,
                finished_callback=self._finished_callback,
                blocksize=0,
            )
            self.playing = True
            self.stream.start()
        except Exception:
            self.playing = False
            self._close_stream()
            raise

    def pause(self) -> None:
        current_sample = int(round(self.current_time() * self.sample_rate))
        self.playing = False
        self._close_stream()
        self.position_sample = max(0, min(current_sample, len(self.audio)))
        self.next_sample = self.position_sample
        self.first_dac_time = None
        self.finished = self.position_sample >= len(self.audio)

    def _close_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def seek(self, time_s: float) -> None:
        was_playing = self.playing
        self.pause()
        time_s = max(0.0, min(float(time_s), self.duration_s))
        self.position_sample = int(time_s * self.sample_rate)
        self.next_sample = self.position_sample
        self.finished = self.position_sample >= len(self.audio)
        if was_playing:
            self.play()

    def current_time(self) -> float:
        if self.playing and self.stream is not None and self.first_dac_time is not None:
            try:
                stream_time = float(self.stream.time)
                played_seconds = max(0.0, stream_time - self.first_dac_time)
                audible_sample = self.first_dac_sample + int(played_seconds * self.sample_rate)
                audible_sample = max(0, min(audible_sample, len(self.audio)))
                return audible_sample / float(self.sample_rate)
            except Exception:
                pass
        return self.position_sample / float(self.sample_rate)

    def close(self) -> None:
        self.pause()


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
        help="Default: Data_collection/dataset_manifest.csv",
    )
    parser.add_argument(
        "--mp4-dir",
        type=Path,
        default=None,
        help="Default: Data_collection/Data_mp4",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="Default: Data_collection/Data_vrs/debug_audio",
    )
    parser.add_argument(
        "--marker-dir",
        type=Path,
        default=None,
        help="Marker CSV directory. Defaults to Data_collection and its Aruco_CSV subdirectory.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Default: Data_collection/manual_timestamp_review.csv",
    )
    parser.add_argument(
        "--only-next-action",
        type=str,
        default=None,
        help="Example: fix_timestamps",
    )
    parser.add_argument(
        "--only-status",
        type=str,
        default=None,
        help="Example: partial_timestamps",
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
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable WAV audio playback.",
    )
    parser.add_argument(
        "--no-live-marker-detection",
        action="store_true",
        help="Disable live marker boxes; IDs from an existing marker CSV remain available.",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help="Resize large frames for display. Use 0 to keep the original width.",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=900,
        help="Resize large frames for display. Use 0 to keep the original height.",
    )

    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_review_csv(path: Path) -> Dict[str, Dict[str, str]]:
    return read_review_rows(path)


def write_review_csv(path: Path, rows_by_seq: Dict[str, Dict[str, str]]) -> None:
    write_review_rows(path, rows_by_seq)


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


def find_wav(wav_dir: Path, sequence_id: str) -> Optional[Path]:
    candidates = [
        wav_dir / f"{sequence_id}.wav",
        wav_dir / f"{sequence_id}.WAV",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(wav_dir.glob(f"{sequence_id}*.wav"))
    if matches:
        return matches[0]

    matches = sorted(wav_dir.glob(f"{sequence_id}*.WAV"))
    if matches:
        return matches[0]

    return None


def find_marker_csv(marker_dir: Path, sequence_id: str) -> Optional[Path]:
    filename = f"aruco_poses_{sequence_id}.csv"
    candidates = [
        marker_dir / filename,
        marker_dir / "Aruco_CSV" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_marker_frame_ids(path: Optional[Path]) -> Dict[int, set[int]]:
    if path is None:
        return {}
    frame_ids: Dict[int, set[int]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "marker_family", "marker_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Marker CSV has an incomplete schema: {path}")
        for row in reader:
            if row.get("marker_family") != "aruco_4x4_50":
                continue
            try:
                frame_index = int(row["frame_index"])
                marker_id = int(row["marker_id"])
            except (TypeError, ValueError):
                continue
            if marker_id in OBJECT_MARKER_IDS:
                frame_ids.setdefault(frame_index, set()).add(marker_id)
    return frame_ids


def create_live_marker_detectors():
    parameters = cv2.aruco.DetectorParameters()
    return (
        (
            "OBJECT",
            cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                parameters,
            ),
            OBJECT_MARKER_IDS,
            (0, 255, 255),
        ),
    )


def detect_and_draw_markers(frame, detectors, selected_target_id: int | None) -> set[int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    visible_object_ids = set()
    for family, detector, allowed_ids, default_color in detectors:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for marker_corners, marker_id_array in zip(corners, ids):
            marker_id = int(marker_id_array.item())
            if marker_id not in allowed_ids:
                continue
            points = marker_corners.reshape(4, 2).astype("int32")
            is_selected = family == "OBJECT" and marker_id == selected_target_id
            color = (0, 80, 255) if is_selected else default_color
            thickness = 4 if is_selected else 2
            cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
            anchor = (int(points[0, 0]), max(20, int(points[0, 1]) - 8))
            label = f"{family} ID {marker_id}"
            if is_selected:
                label += " SELECTED"
            cv2.putText(
                frame,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            if family == "OBJECT":
                visible_object_ids.add(marker_id)
    return visible_object_ids


def initial_seek_time(row: Dict[str, str]) -> float:
    missing = row.get("missing_commands", "").strip()

    start_s = to_float(row.get("start_s", ""))
    second_s = to_float(row.get("second_s", ""))
    done_s = to_float(row.get("done_s", ""))
    third_s = to_float(row.get("third_s", ""))

    if row.get("next_action", "") == "annotate_sequence" and second_s is not None:
        return max(0.0, second_s)

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
    column = COMMAND_TO_MANUAL_COLUMN[command]
    review[column] = f"{time_s:.3f}"
    review["decision"] = "manual_fix"


def cycle_target_object(review: Dict[str, str], direction: int) -> int:
    marker_ids = sorted(OBJECT_MARKER_IDS)
    current_text = str(review.get("target_object_id", "")).strip()
    if current_text and int(current_text) in marker_ids:
        current_index = marker_ids.index(int(current_text))
        marker_id = marker_ids[(current_index + direction) % len(marker_ids)]
    else:
        marker_id = marker_ids[0] if direction >= 0 else marker_ids[-1]
    review["target_object_id"] = str(marker_id)
    if not review.get("annotation_confidence"):
        review["annotation_confidence"] = "certain"
    return marker_id


def select_visible_target(review: Dict[str, str], visible_ids: set[int]) -> int | None:
    marker_ids = sorted(visible_ids)
    if not marker_ids:
        return None
    current_text = str(review.get("target_object_id", "")).strip()
    if current_text and int(current_text) in marker_ids:
        current_index = marker_ids.index(int(current_text))
        marker_id = marker_ids[(current_index + 1) % len(marker_ids)]
    else:
        marker_id = marker_ids[0]
    review["target_object_id"] = str(marker_id)
    if not review.get("annotation_confidence"):
        review["annotation_confidence"] = "certain"
    return marker_id


def set_receiving_hand(review: Dict[str, str], hand: str) -> None:
    review["receiving_hand"] = hand
    if not review.get("annotation_confidence"):
        review["annotation_confidence"] = "certain"


def put_text_lines(frame, lines: List[str]) -> None:
    x = 20
    y = 30
    line_height = 26

    for i, line in enumerate(lines):
        yy = y + i * line_height

        cv2.putText(
            frame,
            line,
            (x + 2, yy + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

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


def resize_for_display(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    width_scale = max_width / width if max_width > 0 and width > max_width else 1.0
    height_scale = max_height / height if max_height > 0 and height > max_height else 1.0
    scale = min(width_scale, height_scale)
    if scale >= 1.0:
        return frame
    display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def review_video(
    row: Dict[str, str],
    video_path: Path,
    wav_path: Optional[Path],
    marker_csv_path: Optional[Path],
    marker_frame_ids: Dict[int, set[int]],
    review_rows: Dict[str, Dict[str, str]],
    review_csv_path: Path,
    index: int,
    total: int,
    audio_enabled: bool,
    live_marker_detection: bool,
    max_display_width: int,
    max_display_height: int,
) -> str:
    sequence_id = row["sequence_id"]
    live_marker_detectors = create_live_marker_detectors() if live_marker_detection else ()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_s = frame_count / fps if frame_count > 0 else 0.0

    wav_player: Optional[WavPlayer] = None
    if audio_enabled and wav_path is not None:
        try:
            wav_player = WavPlayer(wav_path)
        except Exception as exc:
            print(f"WARNING: Could not open WAV audio: {wav_path}")
            print(f"Reason: {exc}")
            wav_player = None

    playback_duration_s = video_duration_s
    if wav_player is not None:
        playback_duration_s = min(video_duration_s, wav_player.duration_s)

    start_time = min(initial_seek_time(row), playback_duration_s)

    existing = review_rows.get(sequence_id)
    review = make_review_row(row, existing)
    review_rows[sequence_id] = review

    paused = True
    last_frame = None
    last_frame_index = -1
    timeline_s = start_time
    playback_start_wall = time.monotonic()

    auto_times = {
        "START": to_float(row.get("start_s", "")),
        "SECOND": to_float(row.get("second_s", "")),
        "DONE": to_float(row.get("done_s", "")),
        "THIRD": to_float(row.get("third_s", "")),
    }

    print()
    print("============================================================")
    print(f"[{index + 1}/{total}] {sequence_id}")
    print(f"MP4: {video_path}")
    print(f"WAV: {wav_path if wav_path else 'missing'}")
    print(f"Marker CSV: {marker_csv_path if marker_csv_path else 'missing; live detection only'}")
    print(f"Start review at: {start_time:.3f}s")
    print("Keys:")
    print("  1 = set START")
    print("  2 = set SECOND")
    print("  3 = set DONE")
    print("  4 = set THIRD")
    print("  space = pause/play")
    print("  a/d = -1s/+1s")
    print("  z/c = -5s/+5s")
    print("  v = accept_auto")
    print("  e = exclude")
    print("  u = uncertain")
    print("  5 or ] = next target object ID (6-14), [ = previous ID")
    print("  m = select/cycle an object ID visible in the current frame")
    print("  6/7/8/9 = set that target object ID directly")
    print("  l/r/b/0 = receiving hand left/right/both/uncertain")
    print("  i = toggle annotation confidence")
    print("  x = clear object/hand annotation")
    print("  n = next")
    print("  p = previous")
    print("  q = quit")
    print("============================================================")

    def seek_video(new_time_s: float) -> bool:
        nonlocal last_frame, last_frame_index

        target_index = min(
            max(0, int(new_time_s * fps)),
            max(0, frame_count - 1),
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, decoded_frame = cap.read()
        if not ok:
            last_frame = None
            return False

        last_frame = decoded_frame
        next_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        last_frame_index = max(target_index, next_index - 1)
        return True

    def current_time_s() -> float:
        if paused:
            return timeline_s
        if wav_player is not None:
            return min(playback_duration_s, wav_player.current_time())
        return min(playback_duration_s, timeline_s + (time.monotonic() - playback_start_wall))

    def start_playback(new_time_s: float) -> None:
        nonlocal paused, timeline_s, playback_start_wall, wav_player

        timeline_s = max(0.0, min(float(new_time_s), playback_duration_s))
        if wav_player is not None:
            try:
                wav_player.seek(timeline_s)
                wav_player.play()
            except Exception as exc:
                print(f"WARNING: Audio playback disabled: {exc}")
                wav_player.close()
                wav_player = None
        playback_start_wall = time.monotonic()
        paused = False

    def pause_playback() -> None:
        nonlocal paused, timeline_s

        timeline_s = current_time_s()
        if wav_player is not None:
            wav_player.pause()
            timeline_s = min(playback_duration_s, wav_player.current_time())
        paused = True

    def seek_all(new_time_s: float) -> None:
        nonlocal timeline_s

        was_paused = paused
        if not was_paused:
            pause_playback()
        new_time_s = max(0.0, min(float(new_time_s), playback_duration_s))
        timeline_s = new_time_s
        if wav_player is not None:
            wav_player.seek(new_time_s)
        seek_video(new_time_s)
        if not was_paused:
            start_playback(new_time_s)

    if not seek_video(start_time):
        print(f"Could not decode video: {video_path}")
        cap.release()
        return "next"
    start_playback(start_time)

    try:
        while True:
            current_s = current_time_s()
            if not paused and current_s >= playback_duration_s:
                return "next"

            target_frame_index = min(
                max(0, int(current_s * fps)),
                max(0, frame_count - 1),
            )
            frame_gap = target_frame_index - last_frame_index

            if frame_gap < 0 or frame_gap > max(5, int(fps / 2)):
                if not seek_video(current_s):
                    return "next"
            elif frame_gap > 0:
                for _ in range(frame_gap):
                    ok, decoded_frame = cap.read()
                    if not ok:
                        return "next"
                    last_frame = decoded_frame
                    last_frame_index += 1

            frame = resize_for_display(
                last_frame.copy(),
                max_display_width,
                max_display_height,
            )

            selected_text = str(review.get("target_object_id", "")).strip()
            selected_target_id = int(selected_text) if selected_text else None
            visible_object_ids = set(marker_frame_ids.get(last_frame_index, set()))
            if live_marker_detectors:
                visible_object_ids.update(
                    detect_and_draw_markers(frame, live_marker_detectors, selected_target_id)
                )

            lines = [
                f"[{index + 1}/{total}] {sequence_id}",
                f"t={current_s:.3f}s / video={video_duration_s:.3f}s",
                f"status={row.get('status', '')}",
                f"next={row.get('next_action', '')}",
                f"missing={row.get('missing_commands', '') or 'none'}",
                f"audio={'on' if wav_player is not None else 'off'} paused={'yes' if paused else 'no'}",
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
                "",
                f"TARGET ID={review.get('target_object_id', '') or '-'}",
                f"VISIBLE OBJECT IDs={','.join(map(str, sorted(visible_object_ids))) or '-'}",
                f"RECEIVING HAND={review.get('receiving_hand', '') or '-'}",
                f"ANNOTATION={review.get('annotation_confidence', '') or '-'}",
            ]

            put_text_lines(frame, lines)
            cv2.imshow("Timestamp review", frame)

            if paused:
                wait_ms = 0
            else:
                next_frame_s = (last_frame_index + 1) / fps
                wait_ms = max(1, min(33, int((next_frame_s - current_s) * 1000.0)))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == 255:
                continue

            if key == ord(" "):
                if paused:
                    start_playback(timeline_s)
                else:
                    pause_playback()

            elif key == ord("q"):
                write_review_csv(review_csv_path, review_rows)
                if wav_player is not None:
                    wav_player.close()
                cap.release()
                cv2.destroyAllWindows()
                return "quit"

            elif key == ord("n"):
                write_review_csv(review_csv_path, review_rows)
                if wav_player is not None:
                    wav_player.close()
                cap.release()
                return "next"

            elif key == ord("p"):
                write_review_csv(review_csv_path, review_rows)
                if wav_player is not None:
                    wav_player.close()
                cap.release()
                return "previous"

            elif key == ord("a"):
                seek_all(current_time_s() - 1.0)

            elif key == ord("d"):
                seek_all(current_time_s() + 1.0)

            elif key == ord("z"):
                seek_all(current_time_s() - 5.0)

            elif key == ord("c"):
                seek_all(current_time_s() + 5.0)

            elif key == ord("1"):
                t = current_time_s()
                set_manual_time(review, "START", t)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: START -> {t:.3f}s")

            elif key == ord("2"):
                t = current_time_s()
                set_manual_time(review, "SECOND", t)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: SECOND -> {t:.3f}s")

            elif key == ord("3"):
                t = current_time_s()
                set_manual_time(review, "DONE", t)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: DONE -> {t:.3f}s")

            elif key == ord("4"):
                t = current_time_s()
                set_manual_time(review, "THIRD", t)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: THIRD -> {t:.3f}s")

            elif key == ord("v"):
                review["decision"] = "accept_auto"
                for column in COMMAND_TO_MANUAL_COLUMN.values():
                    review[column] = ""
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

            elif key == ord("["):
                marker_id = cycle_target_object(review, -1)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: target_object_id -> {marker_id}")

            elif key in (ord("5"), ord("]")):
                marker_id = cycle_target_object(review, 1)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: target_object_id -> {marker_id}")

            elif key == ord("m"):
                marker_id = select_visible_target(review, visible_object_ids)
                if marker_id is None:
                    print(f"{sequence_id}: no object marker visible in the current frame")
                else:
                    write_review_csv(review_csv_path, review_rows)
                    print(f"{sequence_id}: target_object_id -> visible ID {marker_id}")

            elif key in (ord("6"), ord("7"), ord("8"), ord("9")):
                marker_id = int(chr(key))
                review["target_object_id"] = str(marker_id)
                if not review.get("annotation_confidence"):
                    review["annotation_confidence"] = "certain"
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: target_object_id -> {marker_id}")

            elif key in (ord("l"), ord("r"), ord("b"), ord("0")):
                hand = {
                    ord("l"): "left",
                    ord("r"): "right",
                    ord("b"): "both",
                    ord("0"): "uncertain",
                }[key]
                set_receiving_hand(review, hand)
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: receiving_hand -> {hand}")

            elif key == ord("i"):
                current = review.get("annotation_confidence", "")
                review["annotation_confidence"] = "certain" if current == "uncertain" else "uncertain"
                write_review_csv(review_csv_path, review_rows)
                print(
                    f"{sequence_id}: annotation_confidence -> "
                    f"{review['annotation_confidence']}"
                )

            elif key == ord("x"):
                review["target_object_id"] = ""
                review["receiving_hand"] = ""
                review["annotation_confidence"] = ""
                write_review_csv(review_csv_path, review_rows)
                print(f"{sequence_id}: object/hand annotation cleared")

    finally:
        if wav_player is not None:
            wav_player.close()
        cap.release()


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
    wav_dir = args.wav_dir or data_root / "Data_vrs" / "debug_audio"
    marker_dir = args.marker_dir or data_root
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

        wav_path = None
        if not args.no_audio:
            wav_path = find_wav(wav_dir, seq)
            if wav_path is None:
                print(f"[{i + 1}/{len(rows)}] Missing WAV for {seq}. Continuing without audio.")

        marker_csv_path = find_marker_csv(marker_dir, seq)
        try:
            marker_frame_ids = read_marker_frame_ids(marker_csv_path)
        except (OSError, ValueError) as exc:
            print(f"[{i + 1}/{len(rows)}] WARNING: Could not read marker CSV: {exc}")
            marker_csv_path = None
            marker_frame_ids = {}

        action = review_video(
            row=row,
            video_path=video_path,
            wav_path=wav_path,
            marker_csv_path=marker_csv_path,
            marker_frame_ids=marker_frame_ids,
            review_rows=review_rows,
            review_csv_path=review_csv_path,
            index=i,
            total=len(rows),
            audio_enabled=not args.no_audio,
            live_marker_detection=not args.no_live_marker_detection,
            max_display_width=args.max_display_width,
            max_display_height=args.max_display_height,
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
