#!/usr/bin/env python3
"""Render synchronized RGB videos with intention, hand, and robot-frame pose overlays."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTENTION_COLORS = {
    "continue": (80, 180, 255),
    "fetch": (255, 180, 50),
    "handover": (190, 80, 255),
}
GT_COLOR = (70, 220, 70)
PRED_COLOR = (60, 60, 240)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--video-dir", type=Path, default=Path("Data_collection/Data_mp4")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-prediction-age-s", type=float, default=0.5)
    parser.add_argument("--no-transcode", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def sequence_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sequence_id, group in frame.groupby("sequence_id"):
        pose = group.loc[as_bool(group["pose_valid"])]
        rows.append(
            {
                "sequence_id": sequence_id,
                "windows": len(group),
                "intention_accuracy": float(as_bool(group["intention_correct"]).mean()),
                "pose_windows": len(pose),
                "pose_mae_cm": (
                    float(pose["predicted_position_error_cm"].astype(float).mean())
                    if len(pose)
                    else np.nan
                ),
                "incorrect_windows": int((~as_bool(group["intention_correct"])).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_sequences(frame: pd.DataFrame, count: int) -> tuple[list[str], dict]:
    stats = sequence_statistics(frame)
    eligible = stats.loc[(stats["windows"] >= 3) & (stats["pose_windows"] >= 1)].copy()
    if eligible.empty:
        eligible = stats.copy()
    chosen: list[str] = []
    reasons = {}

    success = eligible.sort_values(
        ["intention_accuracy", "pose_mae_cm"],
        ascending=[False, True],
        na_position="last",
    ).iloc[0]
    chosen.append(str(success["sequence_id"]))
    reasons[chosen[-1]] = "success_example_high_accuracy_low_pose_error"

    error_pool = eligible.loc[~eligible["sequence_id"].isin(chosen)]
    if not error_pool.empty:
        failure = error_pool.sort_values(
            ["intention_accuracy", "pose_mae_cm"],
            ascending=[True, False],
            na_position="last",
        ).iloc[0]
        chosen.append(str(failure["sequence_id"]))
        reasons[chosen[-1]] = "failure_example_low_accuracy_or_high_pose_error"

    remaining = eligible.loc[~eligible["sequence_id"].isin(chosen)].copy()
    if not remaining.empty and len(chosen) < count:
        accuracy_median = float(eligible["intention_accuracy"].median())
        pose_median = float(eligible["pose_mae_cm"].median())
        remaining["median_distance"] = (
            (remaining["intention_accuracy"] - accuracy_median).abs()
            + (remaining["pose_mae_cm"] - pose_median).abs()
            / max(1.0, abs(pose_median))
        )
        typical = remaining.sort_values("median_distance").iloc[0]
        chosen.append(str(typical["sequence_id"]))
        reasons[chosen[-1]] = "representative_example_near_median_metrics"

    for sequence_id in stats["sequence_id"].astype(str):
        if len(chosen) >= count:
            break
        if sequence_id not in chosen:
            chosen.append(sequence_id)
            reasons[sequence_id] = "additional_available_example"
    return chosen[:count], reasons


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_probability_bars(
    canvas: np.ndarray,
    row: pd.Series,
    *,
    x: int,
    y: int,
    width: int,
    scale: float,
) -> None:
    for index, name in enumerate(("continue", "fetch", "handover")):
        probability = float(row[f"{name}_probability"])
        top = y + index * int(38 * scale)
        height = max(12, int(19 * scale))
        cv2.rectangle(canvas, (x, top), (x + width, top + height), (65, 65, 65), -1)
        cv2.rectangle(
            canvas,
            (x, top),
            (x + int(width * probability), top + height),
            INTENTION_COLORS[name],
            -1,
        )
        put_text(
            canvas,
            f"{name[:1].upper()} {probability:.3f}",
            (x + width + int(12 * scale), top + height),
            scale=0.48 * scale,
            thickness=max(1, int(1.5 * scale)),
        )


def pose_bounds(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = []
    valid = group.loc[as_bool(group["pose_valid"])]
    for prefix in ("target", "predicted"):
        columns = [f"{prefix}_{axis}_m" for axis in "xyz"]
        if set(columns).issubset(valid.columns):
            array = valid[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            values.append(array[np.isfinite(array).all(axis=1)])
    combined = np.concatenate([value for value in values if len(value)], axis=0) if any(len(value) for value in values) else np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    lower = np.nanpercentile(combined, 2, axis=0)
    upper = np.nanpercentile(combined, 98, axis=0)
    span = np.maximum(upper - lower, 0.2)
    return lower - 0.15 * span, upper + 0.15 * span


def draw_pose_inset(
    frame: np.ndarray,
    row: pd.Series,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    scale: float,
) -> None:
    height, width = frame.shape[:2]
    inset_w = int(min(430 * scale, width * 0.32))
    inset_h = int(min(265 * scale, height * 0.33))
    x0 = width - inset_w - int(22 * scale)
    y0 = height - inset_h - int(22 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + inset_w, y0 + inset_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    put_text(frame, "Future hand position (robot frame, XY)", (x0 + 12, y0 + int(30 * scale)), scale=0.48 * scale, thickness=1)
    lower, upper = bounds

    def map_xy(values: np.ndarray) -> tuple[int, int]:
        normalized = (values[:2] - lower[:2]) / np.maximum(upper[:2] - lower[:2], 1e-6)
        px = x0 + int(30 * scale) + int(normalized[0] * (inset_w - int(60 * scale)))
        py = y0 + inset_h - int(42 * scale) - int(normalized[1] * (inset_h - int(85 * scale)))
        return px, py

    cv2.rectangle(
        frame,
        (x0 + int(30 * scale), y0 + int(48 * scale)),
        (x0 + inset_w - int(30 * scale), y0 + inset_h - int(38 * scale)),
        (120, 120, 120),
        1,
    )
    if bool(row.get("pose_valid", False)):
        target = np.asarray([float(row[f"target_{axis}_m"]) for axis in "xyz"])
        prediction = np.asarray([float(row[f"predicted_{axis}_m"]) for axis in "xyz"])
        if np.isfinite(target).all() and np.isfinite(prediction).all():
            gt_point = map_xy(target)
            pred_point = map_xy(prediction)
            cv2.line(frame, gt_point, pred_point, (230, 230, 230), 2)
            cv2.circle(frame, gt_point, max(5, int(8 * scale)), GT_COLOR, -1)
            cv2.drawMarker(frame, pred_point, PRED_COLOR, cv2.MARKER_CROSS, max(12, int(18 * scale)), 3)
            put_text(frame, f"GT z={target[2]:+.2f} m", (x0 + 15, y0 + inset_h - int(14 * scale)), scale=0.39 * scale, color=GT_COLOR, thickness=1)
            put_text(frame, f"Pred z={prediction[2]:+.2f} m", (x0 + inset_w // 2, y0 + inset_h - int(14 * scale)), scale=0.39 * scale, color=PRED_COLOR, thickness=1)
            return
    put_text(frame, "No valid handover pose target", (x0 + 25, y0 + inset_h // 2), scale=0.48 * scale, color=(180, 180, 180), thickness=1)


def annotate_frame(
    frame: np.ndarray,
    row: pd.Series | None,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    prediction_age_s: float | None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max(0.75, width / 1800.0)
    panel_h = int(185 * scale)
    canvas = np.zeros((height + panel_h, width, 3), dtype=np.uint8)
    canvas[panel_h:] = frame
    if row is None:
        put_text(canvas, "Waiting for the first valid model window...", (25, int(80 * scale)), scale=0.8 * scale)
        return canvas
    gt = str(row["target_intention"])
    pred = str(row["predicted_intention"])
    correct = bool(row["intention_correct"])
    put_text(canvas, f"GT: {gt.upper()}", (25, int(48 * scale)), scale=0.72 * scale, color=GT_COLOR)
    put_text(canvas, f"Pred: {pred.upper()}", (25, int(92 * scale)), scale=0.72 * scale, color=GT_COLOR if correct else PRED_COLOR)
    gt_hand = str(row.get("target_receiving_hand", "")).strip() or "n/a"
    pred_hand = str(row.get("predicted_receiving_hand", ""))
    hand_probability = float(row.get("predicted_receiving_hand_probability", 0.0))
    hand_head_active = gt == "handover" or pred == "handover"
    predicted_hand_text = (
        f"{pred_hand} ({hand_probability:.2f})"
        if hand_head_active
        else "n/a (handover only)"
    )
    pose_error = row.get("predicted_position_error_cm")
    pose_text = "pose n/a" if pd.isna(pose_error) else f"pose error {float(pose_error):.1f} cm"
    put_text(
        canvas,
        f"GT hand: {gt_hand} | Pred hand: {predicted_hand_text} | {pose_text}",
        (25, int(139 * scale)),
        scale=0.52 * scale,
        thickness=max(1, int(1.5 * scale)),
    )
    age_text = "" if prediction_age_s is None else f"prediction age {prediction_age_s * 1000:.0f} ms"
    put_text(canvas, age_text, (25, int(174 * scale)), scale=0.38 * scale, color=(170, 170, 170), thickness=1)
    draw_probability_bars(
        canvas,
        row,
        x=int(width * 0.51),
        y=int(25 * scale),
        width=int(width * 0.27),
        scale=scale,
    )
    camera = canvas[panel_h:]
    draw_pose_inset(camera, row, bounds, scale=scale)
    cv2.rectangle(camera, (15, 15), (int(245 * scale), int(48 * scale)), (0, 0, 0), -1)
    put_text(camera, "GT = green circle", (25, int(39 * scale)), scale=0.42 * scale, color=GT_COLOR, thickness=1)
    put_text(camera, "Pred = red cross", (25, int(68 * scale)), scale=0.42 * scale, color=PRED_COLOR, thickness=1)
    return canvas


def still_rows(group: pd.DataFrame) -> dict[str, float]:
    result = {}
    correct = group.loc[as_bool(group["intention_correct"])]
    if not correct.empty:
        result["success"] = float(correct.iloc[len(correct) // 2]["video_time_s"])
    incorrect = group.loc[~as_bool(group["intention_correct"])]
    if not incorrect.empty:
        confidence = incorrect[["continue_probability", "fetch_probability", "handover_probability"]].max(axis=1)
        result["error"] = float(incorrect.loc[confidence.idxmax()]["video_time_s"])
    handover = group.loc[group["target_intention"].astype(str) == "handover"]
    if not handover.empty:
        result["handover"] = float(handover.iloc[len(handover) // 2]["video_time_s"])
    return result


def transcode(temp_path: Path, source_path: Path, output_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        os.replace(temp_path, output_path)
        return "opencv_mp4v_no_ffmpeg_available"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    temp_path.unlink()
    return "h264_yuv420p_with_optional_source_audio"


def render_sequence(
    sequence_id: str,
    group: pd.DataFrame,
    video_path: Path,
    output_dir: Path,
    *,
    max_prediction_age_s: float,
    use_transcode: bool,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {video_path}")
    scale = max(0.75, width / 1800.0)
    panel_h = int(185 * scale)
    temp_path = output_dir / f".{sequence_id}.mp4v.mp4"
    output_path = output_dir / f"{sequence_id}_prediction_overlay.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height + panel_h),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {temp_path}")

    group = group.sort_values("video_time_s").reset_index(drop=True)
    times = group["video_time_s"].to_numpy(np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Prediction times are not strictly increasing: {sequence_id}")
    group["intention_correct"] = as_bool(group["intention_correct"])
    group["pose_valid"] = as_bool(group["pose_valid"])
    bounds = pose_bounds(group)
    requested_stills = still_rows(group)
    saved_stills = {}
    ages = []
    future_matches = 0
    rendered_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_time = rendered_frames / fps
        index = int(np.searchsorted(times, frame_time, side="right") - 1)
        row = None
        age = None
        if index >= 0:
            age = frame_time - times[index]
            if age < -1e-9:
                future_matches += 1
            elif age <= max_prediction_age_s:
                row = group.iloc[index]
                ages.append(age)
        annotated = annotate_frame(frame, row, bounds, prediction_age_s=age if row is not None else None)
        writer.write(annotated)
        for label, target_time in requested_stills.items():
            if label not in saved_stills and frame_time >= target_time:
                still_path = output_dir / f"{sequence_id}_{label}.png"
                cv2.imwrite(str(still_path), annotated)
                saved_stills[label] = str(still_path)
        rendered_frames += 1
    capture.release()
    writer.release()
    if rendered_frames == 0 or not temp_path.is_file():
        raise RuntimeError(f"No output frames rendered for {sequence_id}")
    codec = (
        transcode(temp_path, video_path, output_path)
        if use_transcode
        else "opencv_mp4v"
    )
    if not use_transcode:
        os.replace(temp_path, output_path)
    ages_ms = np.asarray(ages, dtype=float) * 1000.0
    return {
        "sequence_id": sequence_id,
        "source_video": str(video_path),
        "output_video": str(output_path),
        "codec": codec,
        "source_fps": fps,
        "source_reported_frames": frame_count,
        "rendered_frames": rendered_frames,
        "duration_s": rendered_frames / fps,
        "prediction_windows": len(group),
        "prediction_first_s": float(times[0]),
        "prediction_last_s": float(times[-1]),
        "prediction_times_strictly_increasing": True,
        "alignment_policy": "latest prediction with video_time_s <= frame_time",
        "max_allowed_prediction_age_ms": max_prediction_age_s * 1000.0,
        "matched_video_frames": len(ages),
        "mean_prediction_age_ms": float(ages_ms.mean()) if len(ages_ms) else None,
        "max_prediction_age_ms": float(ages_ms.max()) if len(ages_ms) else None,
        "future_prediction_matches": future_matches,
        "robot_frame_pose_inset": True,
        "rgb_projection_claimed": False,
        "stills": saved_stills,
    }


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.max_prediction_age_s <= 0:
        raise ValueError("count and max-prediction-age-s must be positive")
    predictions_path = resolve(args.predictions).resolve()
    video_dir = resolve(args.video_dir).resolve()
    output_dir = resolve(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(predictions_path)
    required = {
        "sequence_id",
        "video_time_s",
        "target_intention",
        "predicted_intention",
        "intention_correct",
        "continue_probability",
        "fetch_probability",
        "handover_probability",
        "pose_valid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {missing}")
    available = set(frame["sequence_id"].astype(str))
    if args.sequence:
        selected = list(dict.fromkeys(args.sequence))
        missing_sequences = sorted(set(selected) - available)
        if missing_sequences:
            raise ValueError(f"Sequences absent from predictions: {missing_sequences}")
        reasons = {sequence_id: "manually_selected" for sequence_id in selected}
    else:
        selected, reasons = choose_sequences(frame, args.count)
    reports = []
    for sequence_id in selected:
        video_path = video_dir / f"{sequence_id}.mp4"
        report = render_sequence(
            sequence_id,
            frame.loc[frame["sequence_id"].astype(str) == sequence_id].copy(),
            video_path,
            output_dir,
            max_prediction_age_s=args.max_prediction_age_s,
            use_transcode=not args.no_transcode,
        )
        report["selection_reason"] = reasons[sequence_id]
        if report["future_prediction_matches"]:
            raise ValueError(f"Future predictions used while rendering {sequence_id}")
        reports.append(report)
        print(f"Rendered: {report['output_video']}")
    summary = {
        "schema_version": 1,
        "predictions": str(predictions_path),
        "selected_sequences": selected,
        "selection_reasons": reasons,
        "videos": reports,
        "synchronization_valid": all(
            report["future_prediction_matches"] == 0
            and report["prediction_times_strictly_increasing"]
            for report in reports
        ),
        "pose_visualization": (
            "Separate robot-frame XY inset; no 3D-to-RGB projection is claimed "
            "because a validated time-varying camera projection was not available."
        ),
    }
    (output_dir / "overlay_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Overlay report: {output_dir / 'overlay_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
