#!/usr/bin/env python3
"""Build a timestamp-aligned multimodal training table for one Aria sequence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from annotation_utils import normalize_receiving_hand, parse_target_object_id, read_review_rows
from extract_multimodal_data import default_gaze_output, extract_vrs_tracking, write_gaze_csv


INTENT_TO_ID = {"continue": 0, "fetch": 1, "handover": 2}
HAND_COORDINATE_COLUMNS = {
    "left": ("hand_tx_left_device_wrist", "hand_ty_left_device_wrist", "hand_tz_left_device_wrist"),
    "right": ("hand_tx_right_device_wrist", "hand_ty_right_device_wrist", "hand_tz_right_device_wrist"),
}
HAND_QUATERNION_COLUMNS = {
    "left": (
        "hand_qx_left_device_wrist",
        "hand_qy_left_device_wrist",
        "hand_qz_left_device_wrist",
        "hand_qw_left_device_wrist",
    ),
    "right": (
        "hand_qx_right_device_wrist",
        "hand_qy_right_device_wrist",
        "hand_qz_right_device_wrist",
        "hand_qw_right_device_wrist",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge gaze, MPS hand/SLAM, calibrated markers, and audio-derived labels."
    )
    parser.add_argument("--sequence-id", required=True, help="Full sequence ID without .vrs.")
    parser.add_argument("--data-root", type=Path, default=Path("Data_collection"))
    parser.add_argument("--timestamps", type=Path, default=None, help="Override timestamps_summary.json.")
    parser.add_argument("--marker-csv", type=Path, default=None, help="Override calibrated marker CSV.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output master CSV.")
    parser.add_argument("--report-out", type=Path, default=None, help="Output validation JSON.")
    parser.add_argument("--gaze-csv", type=Path, default=None, help="Native gaze CSV cache path.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Sequence annotations CSV. Default: Data_collection/manual_timestamp_review.csv when present.",
    )
    parser.add_argument("--target-object-id", type=int, default=None, help="Ground-truth object marker ID, if known.")
    parser.add_argument(
        "--receiving-hand",
        choices=("left", "right", "both", "uncertain"),
        default=None,
        help="Receiving hand override. Otherwise read from the annotation CSV.",
    )
    parser.add_argument("--future-horizon-seconds", type=float, default=1.0)
    parser.add_argument("--hand-tolerance-ms", type=float, default=12.0)
    parser.add_argument("--slam-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--marker-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated outputs.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def command_entry(timestamp_data: dict, sequence_id: str) -> dict:
    entry = timestamp_data.get(f"{sequence_id}.vrs", timestamp_data.get(sequence_id))
    if not isinstance(entry, dict):
        raise ValueError(f"No timestamp entry for {sequence_id}")
    missing = [command for command in ("START", "SECOND", "DONE", "THIRD") if command not in entry]
    if missing:
        raise ValueError(f"Incomplete timestamps for {sequence_id}: {', '.join(missing)}")
    return entry


def sequence_annotation(
    sequence_id: str,
    annotation_path: Path | None,
    target_object_override: int | None,
    receiving_hand_override: str | None,
) -> dict:
    row = {}
    if annotation_path is not None and annotation_path.exists():
        row = read_review_rows(annotation_path).get(sequence_id, {})
    if row.get("decision") == "exclude":
        raise ValueError(f"Sequence is excluded by manual annotation: {sequence_id}")

    target_object_id = (
        parse_target_object_id(target_object_override)
        if target_object_override is not None
        else parse_target_object_id(row.get("target_object_id"))
    )
    receiving_hand = normalize_receiving_hand(
        receiving_hand_override
        if receiving_hand_override is not None
        else row.get("receiving_hand")
    )
    annotation_confidence = str(row.get("annotation_confidence", "")).strip().lower()
    return {
        "target_object_id": target_object_id,
        "receiving_hand": receiving_hand,
        "annotation_confidence": annotation_confidence,
        "review_decision": row.get("decision", ""),
        "annotation_source": str(annotation_path) if row else "",
    }


def label_timeline(frame: pd.DataFrame, commands: dict) -> pd.DataFrame:
    start = int(commands["START"]["timestamp_ns"])
    second = int(commands["SECOND"]["timestamp_ns"])
    done = int(commands["DONE"]["timestamp_ns"])
    third = int(commands["THIRD"]["timestamp_ns"])
    if not start < second < done < third:
        raise ValueError("Expected START < SECOND < DONE < THIRD")

    frame = frame.loc[(frame["timestamp_ns"] >= start) & (frame["timestamp_ns"] <= third)].copy()
    frame["time_since_start_s"] = (frame["timestamp_ns"] - start) / 1e9
    frame["intent_label"] = np.select(
        [frame["timestamp_ns"] < second, frame["timestamp_ns"] < done],
        ["continue", "fetch"],
        default="handover",
    )
    frame["intent_id"] = frame["intent_label"].map(INTENT_TO_ID).astype("int8")
    return frame


def nearest_merge(
    timeline: pd.DataFrame,
    source: pd.DataFrame,
    source_timestamp: str,
    tolerance_ms: float,
) -> pd.DataFrame:
    if source.empty:
        return timeline
    timeline = timeline.sort_values("timestamp_ns")
    source = source.sort_values(source_timestamp)
    return pd.merge_asof(
        timeline,
        source,
        left_on="timestamp_ns",
        right_on=source_timestamp,
        direction="nearest",
        tolerance=int(tolerance_ms * 1e6),
    )


def load_hand_data(path: Path) -> pd.DataFrame:
    hand = pd.read_csv(path)
    hand["hand_timestamp_ns"] = hand["tracking_timestamp_us"].astype("int64") * 1000
    hand["left_valid"] = (hand["left_tracking_confidence"] > 0).astype("int8")
    hand["right_valid"] = (hand["right_tracking_confidence"] > 0).astype("int8")

    for side in ("left", "right"):
        invalid = hand[f"{side}_valid"] == 0
        coordinate_columns = [column for column in hand.columns if f"_{side}_" in column]
        hand.loc[invalid, coordinate_columns] = np.nan

    rename = {
        column: f"hand_{column}"
        for column in hand.columns
        if column not in {"hand_timestamp_ns"}
    }
    hand = hand.rename(columns=rename)
    return hand


def load_slam_data(path: Path) -> pd.DataFrame:
    slam = pd.read_csv(path)
    slam["slam_timestamp_ns"] = slam["tracking_timestamp_us"].astype("int64") * 1000
    rename = {
        column: f"slam_{column}"
        for column in slam.columns
        if column not in {"slam_timestamp_ns"}
    }
    return slam.rename(columns=rename)


def marker_prefix(family: str, marker_id: int) -> str:
    if family == "apriltag_36h11":
        return f"apriltag_{marker_id}"
    if family == "aruco_4x4_50":
        return f"aruco_{marker_id}"
    return f"marker_{marker_id}"


def merge_markers(timeline: pd.DataFrame, marker_path: Path, tolerance_ms: float) -> tuple[pd.DataFrame, list[str]]:
    markers = pd.read_csv(marker_path)
    required = {
        "timestamp_ns",
        "marker_family",
        "marker_id",
        "tx_camera_m",
        "ty_camera_m",
        "tz_camera_m",
        "qx_camera_marker",
        "qy_camera_marker",
        "qz_camera_marker",
        "qw_camera_marker",
    }
    if not required.issubset(markers.columns):
        missing = sorted(required - set(markers.columns))
        raise ValueError(f"Marker CSV uses the legacy/incomplete schema; missing: {', '.join(missing)}")

    marker_keys = []
    pose_columns = [
        "tx_camera_m",
        "ty_camera_m",
        "tz_camera_m",
        "qx_camera_marker",
        "qy_camera_marker",
        "qz_camera_marker",
        "qw_camera_marker",
        "reprojection_error_px",
        "marker_area_px2",
    ]
    groups = markers.groupby(["marker_family", "marker_id"], sort=True)
    for (family, marker_id), group in groups:
        prefix = marker_prefix(str(family), int(marker_id))
        marker_keys.append(prefix)
        source = group[["timestamp_ns", *pose_columns]].copy()
        source = source.drop_duplicates("timestamp_ns", keep="first")
        source = source.rename(
            columns={
                "timestamp_ns": f"{prefix}_timestamp_ns",
                **{column: f"{prefix}_{column}" for column in pose_columns},
            }
        )
        source[f"{prefix}_valid"] = 1
        timeline = nearest_merge(timeline, source, f"{prefix}_timestamp_ns", tolerance_ms)
        timeline[f"{prefix}_valid"] = timeline[f"{prefix}_valid"].fillna(0).astype("int8")
        timeline[f"{prefix}_time_offset_ms"] = (
            timeline[f"{prefix}_timestamp_ns"] - timeline["timestamp_ns"]
        ) / 1e6
    return timeline, marker_keys


def pose_matrix(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray | None:
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion_xyzw)):
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def add_coordinate_transforms(
    master: pd.DataFrame,
    marker_keys: list[str],
    transform_device_camera: np.ndarray,
) -> pd.DataFrame:
    derived: dict[str, np.ndarray] = {}
    row_count = len(master)

    def output(name: str) -> np.ndarray:
        if name not in derived:
            derived[name] = np.full(row_count, np.nan, dtype=np.float64)
        return derived[name]

    object_keys = [key for key in marker_keys if key.startswith("aruco_")]

    for row_index, (_, row) in enumerate(master.iterrows()):
        slam_translation = row[["slam_tx_world_device", "slam_ty_world_device", "slam_tz_world_device"]].to_numpy(dtype=float)
        slam_quaternion = row[
            ["slam_qx_world_device", "slam_qy_world_device", "slam_qz_world_device", "slam_qw_world_device"]
        ].to_numpy(dtype=float)
        transform_world_device = pose_matrix(slam_translation, slam_quaternion)

        robot_key = "apriltag_0"
        robot_translation = row[
            [f"{robot_key}_tx_camera_m", f"{robot_key}_ty_camera_m", f"{robot_key}_tz_camera_m"]
        ].to_numpy(dtype=float)
        robot_quaternion = row[
            [
                f"{robot_key}_qx_camera_marker",
                f"{robot_key}_qy_camera_marker",
                f"{robot_key}_qz_camera_marker",
                f"{robot_key}_qw_camera_marker",
            ]
        ].to_numpy(dtype=float)
        transform_camera_robot = pose_matrix(robot_translation, robot_quaternion)
        transform_device_robot = (
            transform_device_camera @ transform_camera_robot if transform_camera_robot is not None else None
        )
        transform_robot_device = (
            np.linalg.inv(transform_device_robot) if transform_device_robot is not None else None
        )

        if transform_world_device is not None and transform_device_robot is not None:
            transform_world_robot = transform_world_device @ transform_device_robot
            robot_world_quaternion = matrix_quaternion(transform_world_robot)
            for axis, value in zip("xyz", transform_world_robot[:3, 3]):
                output(f"robot_marker_world_{axis}_m")[row_index] = value
            for component, value in zip("xyzw", robot_world_quaternion):
                output(f"robot_marker_world_q{component}")[row_index] = value

        for side in ("left", "right"):
            wrist_device = row[list(HAND_COORDINATE_COLUMNS[side])].to_numpy(dtype=float)
            wrist_quaternion_device = row[list(HAND_QUATERNION_COLUMNS[side])].to_numpy(dtype=float)
            transform_device_wrist = pose_matrix(wrist_device, wrist_quaternion_device)

            if transform_device_wrist is not None and transform_world_device is not None:
                transform_world_wrist = transform_world_device @ transform_device_wrist
                quaternion_world = matrix_quaternion(transform_world_wrist)
                for axis, value in zip("xyz", transform_world_wrist[:3, 3]):
                    output(f"{side}_wrist_world_{axis}_m")[row_index] = value
                for component, value in zip("xyzw", quaternion_world):
                    output(f"{side}_wrist_world_q{component}")[row_index] = value

            if transform_device_wrist is not None and transform_robot_device is not None:
                transform_robot_wrist = transform_robot_device @ transform_device_wrist
                quaternion_robot = matrix_quaternion(transform_robot_wrist)
                for axis, value in zip("xyz", transform_robot_wrist[:3, 3]):
                    output(f"{side}_wrist_robot_{axis}_m")[row_index] = value
                for component, value in zip("xyzw", quaternion_robot):
                    output(f"{side}_wrist_robot_q{component}")[row_index] = value

        gaze_origin_device = row[
            ["gaze_origin_device_x_m", "gaze_origin_device_y_m", "gaze_origin_device_z_m"]
        ].to_numpy(dtype=float)
        gaze_direction_device = row[
            ["gaze_direction_device_x", "gaze_direction_device_y", "gaze_direction_device_z"]
        ].to_numpy(dtype=float)
        if transform_world_device is not None and np.all(np.isfinite(gaze_origin_device)):
            gaze_origin_world = transform_world_device[:3, :3] @ gaze_origin_device + transform_world_device[:3, 3]
            gaze_direction_world = transform_world_device[:3, :3] @ gaze_direction_device
            for axis, value in zip("xyz", gaze_origin_world):
                output(f"gaze_origin_world_{axis}_m")[row_index] = value
            for axis, value in zip("xyz", gaze_direction_world):
                output(f"gaze_direction_world_{axis}")[row_index] = value
        if transform_robot_device is not None and np.all(np.isfinite(gaze_origin_device)):
            gaze_origin_robot = transform_robot_device[:3, :3] @ gaze_origin_device + transform_robot_device[:3, 3]
            gaze_direction_robot = transform_robot_device[:3, :3] @ gaze_direction_device
            for axis, value in zip("xyz", gaze_origin_robot):
                output(f"gaze_origin_robot_{axis}_m")[row_index] = value
            for axis, value in zip("xyz", gaze_direction_robot):
                output(f"gaze_direction_robot_{axis}")[row_index] = value

        for object_key in object_keys:
            translation_camera = row[
                [
                    f"{object_key}_tx_camera_m",
                    f"{object_key}_ty_camera_m",
                    f"{object_key}_tz_camera_m",
                ]
            ].to_numpy(dtype=float)
            quaternion_camera = row[
                [
                    f"{object_key}_qx_camera_marker",
                    f"{object_key}_qy_camera_marker",
                    f"{object_key}_qz_camera_marker",
                    f"{object_key}_qw_camera_marker",
                ]
            ].to_numpy(dtype=float)
            transform_camera_object = pose_matrix(translation_camera, quaternion_camera)
            if transform_camera_object is None:
                continue
            transform_device_object = transform_device_camera @ transform_camera_object
            for axis, value in zip("xyz", transform_device_object[:3, 3]):
                output(f"{object_key}_device_{axis}_m")[row_index] = value

            if transform_world_device is not None:
                transform_world_object = transform_world_device @ transform_device_object
                for axis, value in zip("xyz", transform_world_object[:3, 3]):
                    output(f"{object_key}_world_{axis}_m")[row_index] = value

            if transform_robot_device is not None:
                transform_robot_object = transform_robot_device @ transform_device_object
                quaternion_robot_object = matrix_quaternion(transform_robot_object)
                for axis, value in zip("xyz", transform_robot_object[:3, 3]):
                    output(f"{object_key}_robot_{axis}_m")[row_index] = value
                for component, value in zip("xyzw", quaternion_robot_object):
                    output(f"{object_key}_robot_q{component}")[row_index] = value

            if np.all(np.isfinite(gaze_origin_device)) and np.all(np.isfinite(gaze_direction_device)):
                to_object = transform_device_object[:3, 3] - gaze_origin_device
                distance = float(np.linalg.norm(to_object))
                if distance > 1e-12:
                    cosine = float(np.clip(np.dot(gaze_direction_device, to_object / distance), -1.0, 1.0))
                    output(f"{object_key}_gaze_angle_rad")[row_index] = float(np.arccos(cosine))
                    output(f"{object_key}_gaze_distance_m")[row_index] = distance

    return pd.concat([master.reset_index(drop=True), pd.DataFrame(derived)], axis=1)


def add_future_targets(master: pd.DataFrame, horizon_seconds: float, tolerance_ms: float) -> pd.DataFrame:
    desired_timestamp = master["timestamp_ns"] + int(horizon_seconds * 1e9)
    query = pd.DataFrame({"timestamp_ns": master["timestamp_ns"], "future_query_ns": desired_timestamp})

    target_columns = []
    for side in ("left", "right"):
        target_columns.extend(
            [
                f"{side}_wrist_robot_{axis}_m" for axis in "xyz"
            ] + [f"{side}_wrist_robot_q{component}" for component in "xyzw"]
        )
    target_source = master[["timestamp_ns", *target_columns]].copy()
    rename = {column: f"future_{horizon_seconds:g}s_{column}" for column in target_columns}
    target_source = target_source.rename(columns={"timestamp_ns": "future_target_timestamp_ns", **rename})

    aligned = pd.merge_asof(
        query.sort_values("future_query_ns"),
        target_source.sort_values("future_target_timestamp_ns"),
        left_on="future_query_ns",
        right_on="future_target_timestamp_ns",
        direction="nearest",
        tolerance=int(tolerance_ms * 1e6),
    ).sort_values("timestamp_ns")
    aligned[f"future_{horizon_seconds:g}s_time_error_ms"] = (
        aligned["future_target_timestamp_ns"] - aligned["future_query_ns"]
    ) / 1e6
    for side in ("left", "right"):
        aligned[f"future_{horizon_seconds:g}s_{side}_wrist_valid"] = aligned[
            f"future_{horizon_seconds:g}s_{side}_wrist_robot_x_m"
        ].notna().astype("int8")

    return master.merge(aligned.drop(columns=["future_query_ns"]), on="timestamp_ns", how="left")


def add_receiving_hand_target(
    master: pd.DataFrame,
    horizon_seconds: float,
    receiving_hand: str,
) -> pd.DataFrame:
    prefix = f"future_{horizon_seconds:g}s_"
    selected_hand = receiving_hand if receiving_hand in {"left", "right"} else None
    for axis in "xyz":
        output_column = f"{prefix}receiving_wrist_robot_{axis}_m"
        master[output_column] = (
            master[f"{prefix}{selected_hand}_wrist_robot_{axis}_m"]
            if selected_hand
            else np.nan
        )
    for component in "xyzw":
        output_column = f"{prefix}receiving_wrist_robot_q{component}"
        master[output_column] = (
            master[f"{prefix}{selected_hand}_wrist_robot_q{component}"]
            if selected_hand
            else np.nan
        )
    master[f"{prefix}receiving_wrist_valid"] = (
        master[f"{prefix}{selected_hand}_wrist_valid"]
        if selected_hand
        else 0
    )
    return master


def atomic_dataframe_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        temp_path.replace(path)
        path.chmod(0o644)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def atomic_json(data: dict, path: Path, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {path}")
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


def build_report(
    master: pd.DataFrame,
    sequence_id: str,
    marker_keys: list[str],
    output_csv: Path,
    target_object_id: int | None,
    receiving_hand: str,
    annotation_confidence: str,
    annotation_source: str,
    horizon_seconds: float,
) -> dict:
    marker_valid_ratios = {
        key: round(float(master[f"{key}_valid"].mean()), 4)
        for key in marker_keys
    }
    warnings = []
    if target_object_id is None:
        warnings.append("target_object_id_unknown")
    elif f"aruco_{target_object_id}" not in marker_keys:
        warnings.append("target_object_marker_not_detected")
    elif marker_valid_ratios[f"aruco_{target_object_id}"] == 0.0:
        warnings.append("target_object_marker_not_visible_in_labeled_window")
    if receiving_hand not in {"left", "right"}:
        warnings.append("receiving_hand_not_labeled")
    if annotation_confidence == "uncertain":
        warnings.append("sequence_annotation_uncertain")

    return {
        "sequence_id": sequence_id,
        "rows": len(master),
        "columns": len(master.columns),
        "timestamp_start_ns": int(master["timestamp_ns"].min()),
        "timestamp_end_ns": int(master["timestamp_ns"].max()),
        "duration_s": round((master["timestamp_ns"].max() - master["timestamp_ns"].min()) / 1e9, 3),
        "intent_counts": {key: int(value) for key, value in master["intent_label"].value_counts().items()},
        "gaze_valid_ratio": round(float(master["gaze_valid"].mean()), 4),
        "hand_left_valid_ratio": round(float(master["hand_left_valid"].fillna(0).mean()), 4),
        "hand_right_valid_ratio": round(float(master["hand_right_valid"].fillna(0).mean()), 4),
        "slam_match_ratio": round(float(master["slam_timestamp_ns"].notna().mean()), 4),
        "robot_marker_valid_ratio": round(float(master["apriltag_0_valid"].mean()), 4),
        "marker_valid_ratios": marker_valid_ratios,
        "future_horizon_seconds": horizon_seconds,
        "future_left_wrist_valid_ratio": round(
            float(master[f"future_{horizon_seconds:g}s_left_wrist_valid"].mean()), 4
        ),
        "future_right_wrist_valid_ratio": round(
            float(master[f"future_{horizon_seconds:g}s_right_wrist_valid"].mean()), 4
        ),
        "target_object_id": target_object_id,
        "target_object_known": target_object_id is not None,
        "receiving_hand": receiving_hand or None,
        "receiving_hand_known": receiving_hand in {"left", "right"},
        "annotation_confidence": annotation_confidence or None,
        "annotation_source": annotation_source or None,
        "future_receiving_wrist_valid_ratio": round(
            float(master[f"future_{horizon_seconds:g}s_receiving_wrist_valid"].mean()), 4
        ),
        "warnings": warnings,
        "coordinate_frames": {
            "hand_input": "device and world; robot-marker frame when AprilTag 0 is visible",
            "gaze_input": "CPF, device, world, and robot-marker frame",
            "marker_input": "linear RGB camera; objects additionally in device/world/robot-marker frames",
            "pose_target": "robot-marker frame defined by AprilTag 0; physical robot-base offset not yet applied",
        },
        "master_csv": str(output_csv),
    }


def build_master(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, Path, Path]:
    data_root = args.data_root.expanduser().resolve()
    sequence_id = args.sequence_id
    default_annotations = data_root / "manual_timestamp_review.csv"
    annotations_arg = getattr(args, "annotations", None)
    annotation_path = (
        annotations_arg.expanduser().resolve()
        if annotations_arg is not None
        else default_annotations if default_annotations.exists() else None
    )
    annotation = sequence_annotation(
        sequence_id,
        annotation_path,
        getattr(args, "target_object_id", None),
        getattr(args, "receiving_hand", None),
    )
    target_object_id = annotation["target_object_id"]
    receiving_hand = annotation["receiving_hand"]
    vrs_dir = data_root / "Data_vrs"
    vrs_path = require_file(vrs_dir / f"{sequence_id}.vrs", "VRS")
    mps_dir = vrs_dir / f"mps_{sequence_id}_vrs"
    hand_path = require_file(mps_dir / "hand_tracking" / "hand_tracking_results.csv", "MPS hand tracking")
    slam_path = require_file(mps_dir / "slam" / "closed_loop_trajectory.csv", "MPS SLAM")
    marker_path = require_file(
        args.marker_csv or data_root / f"aruco_poses_{sequence_id}.csv",
        "calibrated marker CSV",
    )
    timestamps_path = require_file(
        args.timestamps or vrs_dir / "timestamps_summary.json",
        "timestamp summary",
    )
    output_dir = data_root / "master_datasets"
    output_csv = (args.output_csv or output_dir / f"{sequence_id}_master.csv").expanduser().resolve()
    report_out = (args.report_out or output_dir / f"{sequence_id}_master_report.json").expanduser().resolve()
    gaze_csv = (args.gaze_csv or default_gaze_output(vrs_path)).expanduser().resolve()

    if args.future_horizon_seconds <= 0.0:
        raise ValueError("--future-horizon-seconds must be greater than zero")
    if min(args.hand_tolerance_ms, args.slam_tolerance_ms, args.marker_tolerance_ms) <= 0.0:
        raise ValueError("All merge tolerances must be greater than zero")
    if target_object_id is not None and target_object_id not in range(6, 15):
        raise ValueError("--target-object-id must be an object marker ID from 6 through 14")
    if not args.overwrite:
        existing = [path for path in (output_csv, report_out) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists (pass --overwrite): {existing[0]}")

    with timestamps_path.open("r", encoding="utf-8") as handle:
        timestamp_data = json.load(handle)
    commands = command_entry(timestamp_data, sequence_id)

    tracking = extract_vrs_tracking(vrs_path)
    if not gaze_csv.exists() or args.overwrite:
        write_gaze_csv(gaze_csv, tracking.gaze_records, overwrite=args.overwrite)
    gaze = pd.DataFrame(tracking.gaze_records).sort_values("timestamp_ns")
    master = label_timeline(gaze, commands)
    master.insert(0, "sequence_id", sequence_id)
    master.insert(1, "participant", sequence_id.split("_", 1)[0])
    master["target_object_id"] = target_object_id if target_object_id is not None else -1
    master["target_object_known"] = int(target_object_id is not None)
    master["receiving_hand"] = receiving_hand or "unknown"
    master["receiving_hand_id"] = {"left": 0, "right": 1, "both": 2}.get(receiving_hand, -1)
    master["annotation_confidence"] = annotation["annotation_confidence"] or "unknown"

    hand = load_hand_data(hand_path)
    master = nearest_merge(master, hand, "hand_timestamp_ns", args.hand_tolerance_ms)
    master["hand_time_offset_ms"] = (master["hand_timestamp_ns"] - master["timestamp_ns"]) / 1e6

    slam = load_slam_data(slam_path)
    master = nearest_merge(master, slam, "slam_timestamp_ns", args.slam_tolerance_ms)
    master["slam_time_offset_ms"] = (master["slam_timestamp_ns"] - master["timestamp_ns"]) / 1e6

    master, marker_keys = merge_markers(master, marker_path, args.marker_tolerance_ms)
    if "apriltag_0" not in marker_keys:
        raise ValueError("AprilTag 0 (robot anchor) is required for robot-relative coordinates")

    master = add_coordinate_transforms(master, marker_keys, tracking.transform_device_camera)
    master = add_future_targets(master, args.future_horizon_seconds, args.hand_tolerance_ms)
    master = add_receiving_hand_target(master, args.future_horizon_seconds, receiving_hand)
    master = master.sort_values("timestamp_ns").reset_index(drop=True)

    report = build_report(
        master,
        sequence_id,
        marker_keys,
        output_csv,
        target_object_id,
        receiving_hand,
        annotation["annotation_confidence"],
        annotation["annotation_source"],
        args.future_horizon_seconds,
    )
    atomic_dataframe_csv(master, output_csv, args.overwrite)
    atomic_json(report, report_out, args.overwrite)
    return master, report, output_csv, report_out


def main() -> int:
    args = parse_args()
    try:
        _, report, output_csv, report_out = build_master(args)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Master CSV: {output_csv}")
    print(f"Report:     {report_out}")
    print(f"Rows:       {report['rows']}")
    print(f"Columns:    {report['columns']}")
    print(f"Intentions: {report['intent_counts']}")
    print(f"Warnings:   {', '.join(report['warnings']) if report['warnings'] else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
