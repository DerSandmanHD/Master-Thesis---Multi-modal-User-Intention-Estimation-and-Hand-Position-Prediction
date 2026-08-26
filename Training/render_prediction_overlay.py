#!/usr/bin/env python3
"""Render synchronized RGB videos with intention, hand, and robot-frame pose overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping

try:
    import cv2
except ImportError:  # Pure alignment/selection tests do not require OpenCV.
    cv2 = None
import numpy as np
import pandas as pd

from artifact_freeze import canonical_json_hash, validate_artifact_freeze
from select_matrix_checkpoints import validate_embedded_final_test_authorization

from video_alignment import (
    VIDEO_ALIGNMENT_FILE_SUFFIX,
    VIDEO_ALIGNMENT_SCHEMA_VERSION,
    file_identity,
    first_rgb_frame_at_or_after,
    load_video_alignment_sidecar,
    prediction_indices_for_rgb_frames,
    sha256_file,
    validate_visual_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTENTION_COLORS = {
    "continue": (80, 180, 255),
    "fetch": (255, 180, 50),
    "handover": (190, 80, 255),
}
GT_COLOR = (70, 220, 70)
PRED_COLOR = (60, 60, 240)
OVERLAY_SCHEMA_VERSION = "qualitative_overlay_device_time_v2"
QUALITATIVE_SELECTION_VERSION = "qualitative_good_typical_failure_v1"
POSE_COMPONENTS = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


def validate_historical_artifact_freeze(
    manifest_path: Path,
) -> dict[str, object]:
    """Validate an immutable training run from a later reporting checkout."""

    return validate_artifact_freeze(
        manifest_path, require_current_git_state=False
    )


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required to render qualitative overlays; use the "
            "pinned Singularity environment or install opencv-python."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--visual-cache-manifest", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument(
        "--video-dir", type=Path, default=Path("Data_collection/Data_mp4")
    )
    parser.add_argument(
        "--vrs-dir", type=Path, default=Path("Data_collection/Data_vrs")
    )
    parser.add_argument(
        "--master-dir", type=Path, default=Path("Data_collection/master_datasets")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-prediction-age-s", type=float, default=0.5)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--no-transcode", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def row_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _stable_tie_breaker(row: pd.Series, seed: int) -> str:
    identity = str(
        row.get(
            "sample_key",
            f"{row['sequence_id']}|{int(row['endpoint_timestamp_ns'])}",
        )
    )
    return hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()


def choose_qualitative_cases(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, pd.Series]:
    """Select distinct reproducible good, typical, and failure pose windows."""

    available = as_bool(frame["learned_end_to_end_available"])
    errors = pd.to_numeric(frame["predicted_position_error_cm"], errors="coerce")
    eligible = frame.loc[available & errors.notna()].copy()
    if len(eligible) < 3:
        raise ValueError(
            "At least three learned_end_to_end_available pose windows are required "
            "for good/typical/failure qualitative cases"
        )
    eligible["_pose_error"] = pd.to_numeric(
        eligible["predicted_position_error_cm"], errors="raise"
    )
    eligible["_confidence"] = eligible[
        ["continue_probability", "fetch_probability", "handover_probability"]
    ].max(axis=1)
    eligible["_correct"] = as_bool(eligible["intention_correct"])
    eligible["_tie"] = [
        _stable_tie_breaker(row, seed) for _, row in eligible.iterrows()
    ]

    cases: dict[str, pd.Series] = {}
    good_pool = eligible.loc[eligible["_correct"]]
    if good_pool.empty:
        good_pool = eligible
    good = good_pool.sort_values(
        ["_pose_error", "_confidence", "_tie"],
        ascending=[True, False, True],
    ).iloc[0]
    cases["good"] = good

    remaining = eligible.drop(index=good.name)
    failure_pool = remaining.loc[~remaining["_correct"]]
    if failure_pool.empty:
        failure_pool = remaining
    failure = failure_pool.sort_values(
        ["_correct", "_pose_error", "_confidence", "_tie"],
        ascending=[True, False, False, True],
    ).iloc[0]
    cases["failure"] = failure

    remaining = remaining.drop(index=failure.name).copy()
    median_error = float(eligible["_pose_error"].median())
    remaining["_median_distance"] = (
        remaining["_pose_error"] - median_error
    ).abs()
    typical = remaining.sort_values(
        ["_median_distance", "_tie"], ascending=[True, True]
    ).iloc[0]
    cases["typical"] = typical
    return cases


def choose_sequences_from_cases(
    cases: dict[str, pd.Series],
    frame: pd.DataFrame,
    count: int,
) -> tuple[list[str], dict[str, str]]:
    chosen: list[str] = []
    reasons: dict[str, str] = {}
    for label in ("good", "typical", "failure"):
        sequence_id = str(cases[label]["sequence_id"])
        if sequence_id not in chosen:
            chosen.append(sequence_id)
            reasons[sequence_id] = f"contains_{label}_qualitative_case"
    for sequence_id in sorted(frame["sequence_id"].astype(str).unique()):
        if len(chosen) >= count:
            break
        if sequence_id not in chosen:
            chosen.append(sequence_id)
            reasons[sequence_id] = "additional_deterministic_sequence"
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
    require_opencv()
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
    valid = group.loc[as_bool(group["learned_end_to_end_available"])]
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
    if row_bool(row.get("learned_end_to_end_available", False)):
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
    participant = str(row.get("participant", ""))
    sequence = str(row.get("sequence_id", ""))
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
    age_text = (
        f"{participant} / {sequence}"
        if prediction_age_s is None
        else (
            f"{participant} / {sequence} | prediction age "
            f"{prediction_age_s * 1000:.0f} ms"
        )
    )
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


def validate_prediction_report(
    report: dict,
    *,
    report_path: Path,
    predictions_path: Path,
    prediction_rows: int,
    artifact_validator: Callable[[Path], Mapping[str, object]] = (
        validate_historical_artifact_freeze
    ),
) -> dict:
    if report.get("schema_version") != 3 or report.get(
        "report_fingerprint"
    ) != canonical_json_hash({**report, "report_fingerprint": None}):
        raise ValueError("Prediction report fingerprint is invalid")
    if report.get("result_role") != "primary_validation_selected_checkpoint":
        raise ValueError(
            "Qualitative main evidence requires the primary validation-selected "
            "checkpoint, not an oracle/pose-selected diagnostic"
        )
    if report.get("checkpoint_selection_split") != "validation" or not str(
        report.get("checkpoint_selection_metric", "")
    ).startswith("validation_"):
        raise ValueError("Qualitative checkpoint selection must use validation only")
    required = (
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "predictions_csv_sha256",
        "dataset_content_fingerprint",
        "source_content_fingerprint",
        "artifact_freeze",
        "final_test_authorization",
        "visual_artifacts",
        "architecture",
    )
    missing = [name for name in required if report.get(name) in (None, "")]
    if missing:
        raise ValueError(f"Prediction report lacks provenance fields: {missing}")
    if report.get("rows") is not None and int(report["rows"]) != prediction_rows:
        raise ValueError("Prediction report and CSV row counts differ")
    predictions_hash = sha256_file(predictions_path)
    if str(report["predictions_csv_sha256"]).lower() != predictions_hash:
        raise ValueError("Prediction report and CSV SHA-256 differ")
    if report.get("split") != "test" or report.get("full_split_export") is not True:
        raise ValueError("Qualitative evidence requires the complete frozen test split")
    if report.get("sequence_filter") not in ([], None):
        raise ValueError("Qualitative prediction report is a filtered test subset")
    checkpoint_hash = str(report["checkpoint_sha256"]).lower()
    if len(checkpoint_hash) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_hash
    ):
        raise ValueError("Prediction report checkpoint SHA-256 is invalid")
    checkpoint_path = Path(str(report["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (report_path.parent / checkpoint_path).resolve()
    verification = "not_locally_available"
    if checkpoint_path.is_file():
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise ValueError("Prediction report checkpoint hash mismatch")
        verification = "matched_local_checkpoint"
    freeze_binding = report["artifact_freeze"]
    if not isinstance(freeze_binding, Mapping):
        raise ValueError("Prediction report artifact_freeze is invalid")
    manifest_path = Path(str(freeze_binding.get("manifest", ""))).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = report_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = artifact_validator(manifest_path)
    if manifest.get("manifest_fingerprint") != freeze_binding.get(
        "manifest_fingerprint"
    ):
        raise ValueError("Prediction report artifact freeze fingerprint differs")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get(
        "dataset_content_fingerprint"
    ) != report["dataset_content_fingerprint"] or dataset.get(
        "source_content_fingerprint"
    ) != report["source_content_fingerprint"]:
        raise ValueError("Prediction report dataset/source differs from artifact freeze")
    frozen_checkpoint = (
        manifest.get("output_artifacts", {})
        .get("checkpoints", {})
        .get("best_intention")
    )
    if not isinstance(frozen_checkpoint, Mapping) or frozen_checkpoint.get(
        "sha256"
    ) != checkpoint_hash:
        raise ValueError("Prediction report checkpoint differs from artifact freeze")
    final_binding = report["final_test_authorization"]
    if not isinstance(final_binding, Mapping):
        raise ValueError("Prediction report final-test authorization is invalid")
    final_path = Path(str(final_binding.get("path", ""))).expanduser()
    if not final_path.is_absolute():
        final_path = report_path.parent / final_path
    final_path = final_path.resolve()
    if not final_path.is_file() or final_binding.get("sha256") != sha256_file(
        final_path
    ):
        raise ValueError("Prediction report final-test authorization hash differs")
    final_report = json.loads(final_path.read_text(encoding="utf-8"))
    if final_report.get("report_fingerprint") != final_binding.get(
        "report_fingerprint"
    ) or final_report.get("report_fingerprint") != canonical_json_hash(
        {**final_report, "report_fingerprint": None}
    ):
        raise ValueError("Prediction report final-test fingerprint differs")
    if (
        final_report.get("evaluation_protocol")
        != "validation_frozen_checkpoint_single_test_v2"
        or final_report.get("checkpoint", {}).get("sha256") != checkpoint_hash
        or final_report.get("source_artifact_manifest_fingerprint")
        != freeze_binding.get("manifest_fingerprint")
    ):
        raise ValueError("Prediction report final-test authorization is mismatched")
    validated_authorization = validate_embedded_final_test_authorization(
        final_report,
        authorization_base=final_path.parent,
        project_root=PROJECT_ROOT,
    )
    if final_binding.get("matrix_authorization") != validated_authorization.get(
        "matrix_authorization"
    ):
        raise ValueError(
            "Prediction report embedded matrix authorization differs from final test"
        )
    return {
        "prediction_report": str(report_path),
        "prediction_report_sha256": sha256_file(report_path),
        "predictions_csv": str(predictions_path),
        "predictions_csv_sha256": predictions_hash,
        "checkpoint": str(report["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_verification": verification,
        "checkpoint_epoch": int(report["checkpoint_epoch"]),
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": report["checkpoint_selection_metric"],
        "checkpoint_selection_value": report.get("checkpoint_selection_value"),
        "dataset_content_fingerprint": report["dataset_content_fingerprint"],
        "source_content_fingerprint": report["source_content_fingerprint"],
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_fingerprint": freeze_binding[
            "manifest_fingerprint"
        ],
        "final_test_authorization": dict(final_binding),
        "visual_artifacts": report["visual_artifacts"],
        "architecture": report["architecture"],
        "split": report.get("split"),
    }


def validate_visual_artifact_binding(
    checkpoint_provenance: Mapping[str, object],
    *,
    visual_manifest_path: Path,
    visual_manifest_sha256: str,
    visual_manifest: Mapping[str, object],
) -> str:
    visual = checkpoint_provenance.get("visual_artifacts")
    if not isinstance(visual, Mapping):
        raise ValueError("Checkpoint provenance has no visual-artifact declaration")
    if not visual.get("enabled"):
        return "alignment_only_sensor_model"
    if visual.get("cache_manifest_sha256") != visual_manifest_sha256:
        raise ValueError("Overlay visual cache differs from the model visual cache")
    if visual.get("alignment_fingerprint") != visual_manifest.get(
        "alignment_fingerprint"
    ):
        raise ValueError("Overlay and model use different visual alignment")
    declared_path = Path(str(visual.get("cache_manifest_path", ""))).expanduser()
    if declared_path.is_file() and declared_path.resolve() != visual_manifest_path.resolve():
        raise ValueError("Overlay visual cache path differs from model provenance")
    return "model_visual_cache_exact_hash_match"


def validate_qualitative_columns(frame: pd.DataFrame) -> None:
    required = {
        "participant",
        "sequence_id",
        "endpoint_timestamp_ns",
        "target_intention",
        "predicted_intention",
        "intention_correct",
        "continue_probability",
        "fetch_probability",
        "handover_probability",
        "sequence_receiving_hand",
        "target_receiving_hand",
        "predicted_receiving_hand",
        "predicted_receiving_hand_probability",
        "pose_valid",
        "learned_end_to_end_available",
        "predicted_position_error_cm",
        "predicted_orientation_error_deg",
        *(f"target_{component}" for component in POSE_COMPONENTS),
        *(f"predicted_{component}" for component in POSE_COMPONENTS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing qualitative fields: {missing}")
    weight_names = {
        column.removeprefix("modality_").removesuffix("_weight")
        for column in frame.columns
        if column.startswith("modality_") and column.endswith("_weight")
    }
    availability_names = {
        column.removeprefix("modality_").removesuffix("_available")
        for column in frame.columns
        if column.startswith("modality_") and column.endswith("_available")
    }
    if weight_names != availability_names:
        raise ValueError(
            "Every qualitative modality weight requires a matching availability flag"
        )


def qualitative_case_record(
    label: str,
    row: pd.Series,
    *,
    rgb_capture_timestamps_ns: np.ndarray,
    checkpoint_provenance: dict,
) -> dict:
    if not row_bool(row["learned_end_to_end_available"]):
        raise ValueError("Qualitative pose cases require learned end-to-end availability")
    endpoint_timestamp_ns = int(row["endpoint_timestamp_ns"])
    rgb_index = first_rgb_frame_at_or_after(
        rgb_capture_timestamps_ns, endpoint_timestamp_ns
    )
    rgb_timestamp_ns = int(rgb_capture_timestamps_ns[rgb_index])
    modality_names = sorted(
        column.removeprefix("modality_").removesuffix("_weight")
        for column in row.index
        if column.startswith("modality_") and column.endswith("_weight")
    )
    modality_weights = {
        name: float(row[f"modality_{name}_weight"])
        for name in modality_names
    }
    modality_available = {
        name: row_bool(row[f"modality_{name}_available"])
        for name in modality_names
    }

    def pose(prefix: str) -> dict:
        values = {
            component: float(row[f"{prefix}_{component}"])
            for component in POSE_COMPONENTS
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Selected qualitative {prefix} pose is non-finite")
        return values

    return {
        "case": label,
        "selection_version": QUALITATIVE_SELECTION_VERSION,
        "sample_key": row.get("sample_key"),
        "participant": str(row["participant"]),
        "sequence_id": str(row["sequence_id"]),
        "endpoint_timestamp_ns": endpoint_timestamp_ns,
        "display_rgb_frame_index": rgb_index,
        "display_rgb_capture_timestamp_ns": rgb_timestamp_ns,
        "display_after_prediction_ms": (
            rgb_timestamp_ns - endpoint_timestamp_ns
        )
        / 1e6,
        "ground_truth_intention": str(row["target_intention"]),
        "predicted_intention": str(row["predicted_intention"]),
        "intention_correct": row_bool(row["intention_correct"]),
        "class_probabilities": {
            name: float(row[f"{name}_probability"])
            for name in ("continue", "fetch", "handover")
        },
        "ground_truth_receiving_hand": str(
            row.get("target_receiving_hand", "")
        ),
        "sequence_receiving_hand": str(row.get("sequence_receiving_hand", "")),
        "predicted_receiving_hand": str(row["predicted_receiving_hand"]),
        "predicted_receiving_hand_probability": float(
            row["predicted_receiving_hand_probability"]
        ),
        "ground_truth_future_wrist": pose("target"),
        "predicted_future_wrist": pose("predicted"),
        "position_error_cm": float(row["predicted_position_error_cm"]),
        "orientation_error_deg": (
            float(row["predicted_orientation_error_deg"])
            if not pd.isna(row["predicted_orientation_error_deg"])
            else None
        ),
        "learned_end_to_end_available": True,
        "modality_weights": modality_weights,
        "modality_available": modality_available,
        "available_modalities": [
            name for name in modality_names if modality_available[name]
        ],
        "missing_modalities": [
            name for name in modality_names if not modality_available[name]
        ],
        "checkpoint_provenance": checkpoint_provenance,
    }


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
    rgb_capture_timestamps_ns: np.ndarray,
    alignment_sidecar: dict,
    alignment_sidecar_path: Path,
    requested_stills: dict[str, Mapping[str, object]],
    max_prediction_age_s: float,
    use_transcode: bool,
) -> dict:
    require_opencv()
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
    if frame_count != len(rgb_capture_timestamps_ns):
        capture.release()
        raise ValueError(
            f"MP4/VRS RGB frame count mismatch for {sequence_id}: "
            f"{frame_count} != {len(rgb_capture_timestamps_ns)}"
        )
    scale = max(0.75, width / 1800.0)
    panel_h = int(185 * scale)
    temp_path = output_dir / f".{sequence_id}.mp4v.mp4"
    output_path = output_dir / f"{sequence_id}_device_time_v2_prediction_overlay.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height + panel_h),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {temp_path}")

    group = group.sort_values("endpoint_timestamp_ns").reset_index(drop=True)
    times = pd.to_numeric(
        group["endpoint_timestamp_ns"], errors="raise"
    ).to_numpy(np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Prediction times are not strictly increasing: {sequence_id}")
    frame_prediction_indices = prediction_indices_for_rgb_frames(
        times, rgb_capture_timestamps_ns
    )
    group["intention_correct"] = as_bool(group["intention_correct"])
    group["pose_valid"] = as_bool(group["pose_valid"])
    group["learned_end_to_end_available"] = as_bool(
        group["learned_end_to_end_available"]
    )
    if group["sample_key"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate prediction sample keys: {sequence_id}")
    rows_by_sample_key = {
        str(row["sample_key"]): row for _, row in group.iterrows()
    }
    bounds = pose_bounds(group)
    saved_stills = {}
    ages = []
    future_matches = 0
    rendered_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if rendered_frames >= len(rgb_capture_timestamps_ns):
            capture.release()
            writer.release()
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"MP4 decoded more frames than VRS RGB for {sequence_id}")
        frame_timestamp_ns = int(rgb_capture_timestamps_ns[rendered_frames])
        index = int(frame_prediction_indices[rendered_frames])
        row = None
        age = None
        if index >= 0:
            age = (frame_timestamp_ns - int(times[index])) / 1e9
            if age < -1e-9:
                future_matches += 1
            elif age <= max_prediction_age_s:
                row = group.iloc[index]
                ages.append(age)
        annotated = annotate_frame(frame, row, bounds, prediction_age_s=age if row is not None else None)
        writer.write(annotated)
        for label, request in requested_stills.items():
            requested_index = int(request["display_rgb_frame_index"])
            if label not in saved_stills and rendered_frames == requested_index:
                expected_rgb_timestamp = int(
                    request["display_rgb_capture_timestamp_ns"]
                )
                if frame_timestamp_ns != expected_rgb_timestamp:
                    raise ValueError(
                        f"Selected still RGB timestamp changed for {label}"
                    )
                requested_sample_key = str(request["sample_key"])
                if requested_sample_key not in rows_by_sample_key:
                    raise ValueError(
                        f"Selected still sample is absent: {requested_sample_key}"
                    )
                still_row = rows_by_sample_key[requested_sample_key]
                still_age_s = (
                    frame_timestamp_ns
                    - int(still_row["endpoint_timestamp_ns"])
                ) / 1e9
                if still_age_s < 0 or still_age_s > max_prediction_age_s:
                    raise ValueError(
                        f"Selected still sample is not causally displayable: {label}"
                    )
                # A later prediction endpoint may lie between the selected
                # endpoint and its first RGB frame. The still is intentionally
                # re-annotated with the exact selected row; the video itself
                # retains the latest-causal-row policy.
                still_annotated = annotate_frame(
                    frame,
                    still_row,
                    bounds,
                    prediction_age_s=still_age_s,
                )
                still_path = (
                    output_dir / f"{sequence_id}_{label}_device_time_v2.png"
                )
                if not cv2.imwrite(str(still_path), still_annotated):
                    raise RuntimeError(f"Could not write still: {still_path}")
                saved_stills[label] = {
                    "path": str(still_path),
                    "sample_key": requested_sample_key,
                    "endpoint_timestamp_ns": int(
                        still_row["endpoint_timestamp_ns"]
                    ),
                    "rgb_frame_index": rendered_frames,
                    "rgb_capture_timestamp_ns": frame_timestamp_ns,
                }
        rendered_frames += 1
    capture.release()
    writer.release()
    if rendered_frames == 0 or not temp_path.is_file():
        raise RuntimeError(f"No output frames rendered for {sequence_id}")
    if rendered_frames != len(rgb_capture_timestamps_ns):
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Decoded MP4/VRS RGB frame count mismatch for {sequence_id}: "
            f"{rendered_frames} != {len(rgb_capture_timestamps_ns)}"
        )
    missing_stills = sorted(set(requested_stills) - set(saved_stills))
    if missing_stills:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            "Selected qualitative stills were not rendered: "
            + ", ".join(missing_stills)
        )
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
        "duration_s": float(
            (int(rgb_capture_timestamps_ns[-1]) - int(rgb_capture_timestamps_ns[0]))
            / 1e9
        ),
        "prediction_windows": len(group),
        "prediction_first_timestamp_ns": int(times[0]),
        "prediction_last_timestamp_ns": int(times[-1]),
        "rgb_first_capture_timestamp_ns": int(rgb_capture_timestamps_ns[0]),
        "rgb_last_capture_timestamp_ns": int(rgb_capture_timestamps_ns[-1]),
        "prediction_times_strictly_increasing": True,
        "alignment_policy": (
            "latest endpoint_timestamp_ns at or before VRS RGB "
            "image_record.capture_timestamp_ns"
        ),
        "time_basis": alignment_sidecar["time_basis"],
        "clip_alignment_version": alignment_sidecar["clip_alignment_version"],
        "clip_alignment_fingerprint": alignment_sidecar[
            "clip_alignment_fingerprint"
        ],
        "video_alignment_schema_version": alignment_sidecar["schema_version"],
        "video_alignment_sidecar": str(alignment_sidecar_path),
        "video_alignment_sidecar_sha256": sha256_file(alignment_sidecar_path),
        "video_alignment_sidecar_fingerprint": alignment_sidecar[
            "sidecar_fingerprint"
        ],
        "source_video_sha256": sha256_file(video_path),
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
    if args.count < 3 or args.max_prediction_age_s <= 0:
        raise ValueError("count must be at least three and prediction age positive")
    predictions_path = resolve(args.predictions).resolve()
    prediction_report_path = resolve(args.prediction_report).resolve()
    visual_manifest_path = resolve(args.visual_cache_manifest).resolve()
    alignment_dir = resolve(args.alignment_dir).resolve()
    video_dir = resolve(args.video_dir).resolve()
    vrs_dir = resolve(args.vrs_dir).resolve()
    master_dir = resolve(args.master_dir).resolve()
    output_dir = resolve(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Qualitative output directory is not empty: {output_dir}. "
            "Partial or historical artifacts are never reused."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(predictions_path)
    validate_qualitative_columns(frame)
    prediction_report = json.loads(
        prediction_report_path.read_text(encoding="utf-8")
    )
    checkpoint_provenance = validate_prediction_report(
        prediction_report,
        report_path=prediction_report_path,
        predictions_path=predictions_path,
        prediction_rows=len(frame),
    )
    visual_manifest = json.loads(
        visual_manifest_path.read_text(encoding="utf-8")
    )
    alignment_spec, clip_alignment_fingerprint = validate_visual_manifest(
        visual_manifest
    )
    visual_manifest_sha256 = sha256_file(visual_manifest_path)
    visual_binding_status = validate_visual_artifact_binding(
        checkpoint_provenance,
        visual_manifest_path=visual_manifest_path,
        visual_manifest_sha256=visual_manifest_sha256,
        visual_manifest=visual_manifest,
    )
    available = set(frame["sequence_id"].astype(str))
    if args.sequence:
        selected = list(dict.fromkeys(args.sequence))
        missing_sequences = sorted(set(selected) - available)
        if missing_sequences:
            raise ValueError(f"Sequences absent from predictions: {missing_sequences}")
        reasons = {sequence_id: "manually_selected" for sequence_id in selected}
        case_frame = frame.loc[frame["sequence_id"].astype(str).isin(selected)]
    else:
        case_frame = frame
    cases = choose_qualitative_cases(case_frame, seed=args.selection_seed)
    if not args.sequence:
        selected, reasons = choose_sequences_from_cases(cases, frame, args.count)

    alignments: dict[str, tuple[dict, np.ndarray, Path, Path]] = {}
    for sequence_id in selected:
        video_path = video_dir / f"{sequence_id}.mp4"
        source_files = {
            "master": file_identity(master_dir / f"{sequence_id}_master.csv"),
            "vrs": file_identity(vrs_dir / f"{sequence_id}.vrs"),
            "mp4": file_identity(video_path),
        }
        sidecar_path = alignment_dir / (
            f"{sequence_id}{VIDEO_ALIGNMENT_FILE_SUFFIX}"
        )
        sidecar, rgb_timestamps = load_video_alignment_sidecar(
            sidecar_path,
            sequence_id=sequence_id,
            expected_source_files=source_files,
            visual_manifest=visual_manifest,
            visual_manifest_sha256=visual_manifest_sha256,
        )
        alignments[sequence_id] = (
            sidecar,
            rgb_timestamps,
            sidecar_path,
            video_path,
        )

    qualitative_cases = []
    for label in ("good", "typical", "failure"):
        row = cases[label]
        sequence_id = str(row["sequence_id"])
        if sequence_id not in alignments:
            raise ValueError(f"Selected case sequence was not prepared: {sequence_id}")
        qualitative_cases.append(
            qualitative_case_record(
                label,
                row,
                rgb_capture_timestamps_ns=alignments[sequence_id][1],
                checkpoint_provenance=checkpoint_provenance,
            )
        )

    stills_by_sequence: dict[str, dict[str, dict[str, object]]] = {
        sequence_id: {} for sequence_id in selected
    }
    for case in qualitative_cases:
        stills_by_sequence[case["sequence_id"]][case["case"]] = {
            "sample_key": case["sample_key"],
            "display_rgb_frame_index": case["display_rgb_frame_index"],
            "display_rgb_capture_timestamp_ns": case[
                "display_rgb_capture_timestamp_ns"
            ],
        }
    reports = []
    for sequence_id in selected:
        sidecar, rgb_timestamps, sidecar_path, video_path = alignments[sequence_id]
        report = render_sequence(
            sequence_id,
            frame.loc[frame["sequence_id"].astype(str) == sequence_id].copy(),
            video_path,
            output_dir,
            rgb_capture_timestamps_ns=rgb_timestamps,
            alignment_sidecar=sidecar,
            alignment_sidecar_path=sidecar_path,
            requested_stills=stills_by_sequence[sequence_id],
            max_prediction_age_s=args.max_prediction_age_s,
            use_transcode=not args.no_transcode,
        )
        report["selection_reason"] = reasons[sequence_id]
        if report["future_prediction_matches"]:
            raise ValueError(f"Future predictions used while rendering {sequence_id}")
        reports.append(report)
        print(f"Rendered: {report['output_video']}")
    summary = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "checkpoint_provenance": checkpoint_provenance,
        "visual_cache_manifest": str(visual_manifest_path),
        "visual_cache_manifest_sha256": visual_manifest_sha256,
        "visual_model_binding_status": visual_binding_status,
        "clip_alignment_version": alignment_spec["version"],
        "clip_alignment_fingerprint": clip_alignment_fingerprint,
        "time_basis": alignment_spec["time_basis"],
        "video_alignment_schema_version": VIDEO_ALIGNMENT_SCHEMA_VERSION,
        "video_time_s_role": (
            "display-only START-relative field; never used for RGB/prediction matching"
        ),
        "selected_sequences": selected,
        "selection_reasons": reasons,
        "selection_seed": int(args.selection_seed),
        "qualitative_selection_version": QUALITATIVE_SELECTION_VERSION,
        "qualitative_cases": qualitative_cases,
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
        "legacy_overlay_artifacts_valid": False,
        "legacy_invalidation_reason": (
            "Schema-v1 overlays compared START-relative video_time_s with MP4 time. "
            "Only device-time-v2 sidecar-bound artifacts are valid."
        ),
    }
    cases_path = output_dir / "qualitative_cases_device_time_v2.json"
    cases_path.write_text(
        json.dumps(qualitative_cases, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat_cases = []
    for case in qualitative_cases:
        flat = {
            key: value
            for key, value in case.items()
            if not isinstance(value, dict) and not isinstance(value, list)
        }
        for name, value in case["class_probabilities"].items():
            flat[f"probability_{name}"] = value
        for prefix, source in (
            ("ground_truth_future_wrist", case["ground_truth_future_wrist"]),
            ("predicted_future_wrist", case["predicted_future_wrist"]),
        ):
            for component, value in source.items():
                flat[f"{prefix}_{component}"] = value
        for name, value in case["modality_weights"].items():
            flat[f"modality_{name}_weight"] = value
        for name, value in case["modality_available"].items():
            flat[f"modality_{name}_available"] = value
        flat_cases.append(flat)
    cases_csv_path = output_dir / "qualitative_cases_device_time_v2.csv"
    pd.DataFrame(flat_cases).to_csv(cases_csv_path, index=False)
    overlay_report_path = output_dir / "overlay_report.json"
    overlay_report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_paths = [cases_path, cases_csv_path, overlay_report_path]
    for video_report in reports:
        output_paths.append(Path(video_report["output_video"]))
        output_paths.extend(
            Path(still["path"])
            for still in video_report["stills"].values()
        )
    artifact_manifest = {
        "schema_version": "qualitative_artifact_manifest_v1",
        "manifest_fingerprint": None,
        "prediction_report_fingerprint": prediction_report[
            "report_fingerprint"
        ],
        "artifact_manifest_fingerprint": checkpoint_provenance[
            "artifact_manifest_fingerprint"
        ],
        "visual_cache_manifest_sha256": visual_manifest_sha256,
        "outputs": {
            path.relative_to(output_dir).as_posix(): file_identity(path)
            for path in output_paths
        },
    }
    artifact_manifest["manifest_fingerprint"] = canonical_json_hash(
        artifact_manifest
    )
    artifact_manifest_path = output_dir / "qualitative_artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Overlay report: {overlay_report_path}")
    print(f"Artifact manifest: {artifact_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
