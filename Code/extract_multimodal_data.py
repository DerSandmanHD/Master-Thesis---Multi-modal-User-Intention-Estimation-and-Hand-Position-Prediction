#!/usr/bin/env python3
"""Extract native-rate eye-gaze features and static VRS calibration metadata."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np


EYE_STREAM_ID = "373-1"
RGB_STREAM_LABEL = "camera-rgb"

GAZE_FIELDS = [
    "timestamp_ns",
    "gaze_valid",
    "gaze_yaw_rad",
    "gaze_pitch_rad",
    "gaze_depth_m",
    "gaze_point_cpf_x_m",
    "gaze_point_cpf_y_m",
    "gaze_point_cpf_z_m",
    "gaze_direction_cpf_x",
    "gaze_direction_cpf_y",
    "gaze_direction_cpf_z",
    "gaze_origin_device_x_m",
    "gaze_origin_device_y_m",
    "gaze_origin_device_z_m",
    "gaze_point_device_x_m",
    "gaze_point_device_y_m",
    "gaze_point_device_z_m",
    "gaze_direction_device_x",
    "gaze_direction_device_y",
    "gaze_direction_device_z",
    "left_gaze_yaw_rad",
    "right_gaze_yaw_rad",
    "left_gaze_pitch_rad",
    "right_gaze_pitch_rad",
]


@dataclass
class VrsTrackingData:
    gaze_records: list[dict]
    transform_device_cpf: np.ndarray
    transform_device_camera: np.ndarray


def timedelta_to_ns(value: timedelta) -> int:
    return int(value // timedelta(microseconds=1)) * 1000


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return vector / norm


def _empty_gaze_features(timestamp_ns: int) -> dict:
    return {field: (timestamp_ns if field == "timestamp_ns" else 0 if field == "gaze_valid" else None) for field in GAZE_FIELDS}


def gaze_record(eye_data, transform_device_cpf: np.ndarray) -> dict:
    from projectaria_tools.core import mps

    timestamp_ns = timedelta_to_ns(eye_data.tracking_timestamp)
    record = _empty_gaze_features(timestamp_ns)
    is_valid = bool(eye_data.combined_gaze_valid)
    record["gaze_valid"] = int(is_valid)
    if not is_valid:
        return record

    yaw = float(eye_data.yaw)
    pitch = float(eye_data.pitch)
    depth = float(eye_data.depth)
    direction_depth = depth if np.isfinite(depth) and depth > 0.0 else 1.0
    point_cpf = np.asarray(mps.get_eyegaze_point_at_depth(yaw, pitch, direction_depth), dtype=np.float64).reshape(3)
    direction_cpf = normalized(point_cpf)

    rotation_device_cpf = transform_device_cpf[:3, :3]
    origin_device = transform_device_cpf[:3, 3]
    point_device = rotation_device_cpf @ point_cpf + origin_device
    direction_device = normalized(rotation_device_cpf @ direction_cpf)

    vergence = eye_data.vergence
    record.update(
        {
            "gaze_yaw_rad": yaw,
            "gaze_pitch_rad": pitch,
            "gaze_depth_m": depth if np.isfinite(depth) and depth > 0.0 else None,
            "gaze_point_cpf_x_m": float(point_cpf[0]),
            "gaze_point_cpf_y_m": float(point_cpf[1]),
            "gaze_point_cpf_z_m": float(point_cpf[2]),
            "gaze_direction_cpf_x": float(direction_cpf[0]),
            "gaze_direction_cpf_y": float(direction_cpf[1]),
            "gaze_direction_cpf_z": float(direction_cpf[2]),
            "gaze_origin_device_x_m": float(origin_device[0]),
            "gaze_origin_device_y_m": float(origin_device[1]),
            "gaze_origin_device_z_m": float(origin_device[2]),
            "gaze_point_device_x_m": float(point_device[0]),
            "gaze_point_device_y_m": float(point_device[1]),
            "gaze_point_device_z_m": float(point_device[2]),
            "gaze_direction_device_x": float(direction_device[0]),
            "gaze_direction_device_y": float(direction_device[1]),
            "gaze_direction_device_z": float(direction_device[2]),
            "left_gaze_yaw_rad": float(vergence.left_yaw),
            "right_gaze_yaw_rad": float(vergence.right_yaw),
            "left_gaze_pitch_rad": float(vergence.left_pitch),
            "right_gaze_pitch_rad": float(vergence.right_pitch),
        }
    )
    return record


def extract_vrs_tracking(input_vrs: Path, max_samples: int | None = None) -> VrsTrackingData:
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    input_vrs = Path(input_vrs).expanduser().resolve()
    if not input_vrs.is_file():
        raise FileNotFoundError(f"VRS file not found: {input_vrs}")

    provider = data_provider.create_vrs_data_provider(str(input_vrs))
    if provider is None:
        raise RuntimeError(f"Could not create VRS provider: {input_vrs}")

    device_calibration = provider.get_device_calibration()
    transform_device_cpf = np.asarray(device_calibration.get_transform_device_cpf().to_matrix(), dtype=np.float64)
    rgb_calibration = device_calibration.get_camera_calib(RGB_STREAM_LABEL)
    transform_device_camera = np.asarray(rgb_calibration.get_transform_device_camera().to_matrix(), dtype=np.float64)

    stream_id = StreamId(EYE_STREAM_ID)
    available_samples = provider.get_num_data(stream_id)
    sample_count = available_samples
    if max_samples is not None:
        sample_count = min(available_samples, max(max_samples, 0))
    if sample_count == 0:
        raise RuntimeError(f"No eye-gaze samples found in {input_vrs}")

    records = []
    for index in range(sample_count):
        eye_data = provider.get_eye_gaze_data_by_index(stream_id, index)
        if eye_data is not None:
            records.append(gaze_record(eye_data, transform_device_cpf))

    return VrsTrackingData(
        gaze_records=records,
        transform_device_cpf=transform_device_cpf,
        transform_device_camera=transform_device_camera,
    )


def default_gaze_output(input_vrs: Path) -> Path:
    input_vrs = input_vrs.expanduser().resolve()
    output_dir = input_vrs.parent.parent if input_vrs.parent.name == "Data_vrs" else input_vrs.parent
    return output_dir / f"gaze_{input_vrs.stem}.csv"


def write_gaze_csv(path: Path, records: list[dict], overwrite: bool = False) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    try:
        with temp_handle:
            writer = csv.DictWriter(temp_handle, fieldnames=GAZE_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        temp_path.replace(path)
        path.chmod(0o644)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract native-rate eye-gaze features from an Aria VRS file.")
    parser.add_argument("--input-vrs", type=Path, required=True, help="Input Project Aria .vrs file.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Flat gaze CSV output path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Extract at most N samples for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv or default_gaze_output(args.input_vrs)
    try:
        tracking = extract_vrs_tracking(args.input_vrs, args.max_samples)
        write_gaze_csv(output_csv, tracking.gaze_records, args.overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    valid_count = sum(record["gaze_valid"] for record in tracking.gaze_records)
    print(f"Gaze CSV: {output_csv}")
    print(f"Samples: {len(tracking.gaze_records)}")
    print(f"Valid: {valid_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
