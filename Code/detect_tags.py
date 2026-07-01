#!/usr/bin/env python3
"""Extract calibrated AprilTag and ArUco poses from a Project Aria VRS file."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


RGB_STREAM_LABEL = "camera-rgb"
RGB_STREAM_ID = "214-1"
APRILTAG_FAMILY = "apriltag_36h11"
ARUCO_FAMILY = "aruco_4x4_50"
DEFAULT_APRIL_IDS = frozenset(range(0, 6))
DEFAULT_ARUCO_IDS = frozenset(range(6, 15))

CSV_FIELDS = [
    "sequence_id",
    "frame_index",
    "timestamp_ns",
    "marker_family",
    "marker_id",
    "marker_role",
    "marker_size_m",
    "tx_camera_m",
    "ty_camera_m",
    "tz_camera_m",
    "rvec_x_rad",
    "rvec_y_rad",
    "rvec_z_rad",
    "qx_camera_marker",
    "qy_camera_marker",
    "qz_camera_marker",
    "qw_camera_marker",
    "reprojection_error_px",
    "marker_area_px2",
]


def parse_marker_ids(value: str) -> frozenset[int]:
    """Parse comma-separated IDs and inclusive ranges such as ``0-5,8``."""
    marker_ids: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid descending marker range: {token}")
            marker_ids.update(range(start, end + 1))
        else:
            marker_ids.add(int(token))
    if not marker_ids:
        raise argparse.ArgumentTypeError("At least one marker ID is required")
    return frozenset(marker_ids)


def default_output_csv(input_vrs: Path) -> Path:
    if input_vrs.parent.name == "Data_vrs":
        output_dir = input_vrs.parent.parent
    else:
        output_dir = input_vrs.parent
    return output_dir / f"aruco_poses_{input_vrs.stem}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rectify the Aria RGB stream, detect configured AprilTag/ArUco IDs, "
            "and export timestamped 6-DoF marker poses."
        )
    )
    parser.add_argument("--input-vrs", type=Path, required=True, help="Input Project Aria .vrs file.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV. Default: Data_collection/aruco_poses_<full-sequence-id>.csv",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional annotated MP4 output. No video is written when omitted.",
    )
    parser.add_argument(
        "--april-ids",
        type=parse_marker_ids,
        default=DEFAULT_APRIL_IDS,
        help="Allowed AprilTag IDs/ranges. Default: 0-5 (robot and table anchors).",
    )
    parser.add_argument(
        "--aruco-ids",
        type=parse_marker_ids,
        default=DEFAULT_ARUCO_IDS,
        help="Allowed ArUco IDs/ranges. Default: 6-14 (object markers).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames. Intended for smoke tests.",
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="Annotated video FPS. Default: 30.")
    parser.add_argument("--progress-every", type=int, default=100, help="Progress interval in frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def marker_role(family: str, marker_id: int) -> str:
    if family == APRILTAG_FAMILY:
        return "robot" if marker_id == 0 else "table"
    return "object"


def marker_size_m(family: str, marker_id: int) -> float:
    if family == APRILTAG_FAMILY:
        return 0.10 if marker_id == 0 else 0.08
    return 0.05


def marker_object_points(size_m: float) -> np.ndarray:
    half_size = size_m / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return a normalized quaternion in x, y, z, w order."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def pose_row(
    sequence_id: str,
    frame_index: int,
    timestamp_ns: int,
    family: str,
    marker_id: int,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray] | None:
    size_m = marker_size_m(family, marker_id)
    object_points = marker_object_points(size_m)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(tvec[2, 0]) <= 0.0:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    projected = projected.reshape(-1, 2)
    reprojection_error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)

    row = {
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "timestamp_ns": timestamp_ns,
        "marker_family": family,
        "marker_id": marker_id,
        "marker_role": marker_role(family, marker_id),
        "marker_size_m": round(size_m, 5),
        "tx_camera_m": round(float(tvec[0, 0]), 7),
        "ty_camera_m": round(float(tvec[1, 0]), 7),
        "tz_camera_m": round(float(tvec[2, 0]), 7),
        "rvec_x_rad": round(float(rvec[0, 0]), 8),
        "rvec_y_rad": round(float(rvec[1, 0]), 8),
        "rvec_z_rad": round(float(rvec[2, 0]), 8),
        "qx_camera_marker": round(qx, 9),
        "qy_camera_marker": round(qy, 9),
        "qz_camera_marker": round(qz, 9),
        "qw_camera_marker": round(qw, 9),
        "reprojection_error_px": round(reprojection_error, 5),
        "marker_area_px2": round(abs(float(cv2.contourArea(image_points.astype(np.float32)))), 2),
    }
    return row, rvec, tvec


def make_linear_calibration(source_calibration, width: int, height: int):
    from projectaria_tools.core import calibration

    focal_lengths = source_calibration.get_focal_lengths()
    focal_length = float(np.mean(focal_lengths))
    return calibration.get_linear_camera_calibration(
        width,
        height,
        focal_length,
        "camera-rgb-linear",
        source_calibration.get_transform_device_camera(),
    )


def camera_matrix_from_calibration(camera_calibration) -> np.ndarray:
    focal_x, focal_y = camera_calibration.get_focal_lengths()
    principal_x, principal_y = camera_calibration.get_principal_point()
    return np.array(
        [[focal_x, 0.0, principal_x], [0.0, focal_y, principal_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def validate_output_path(path: Path | None, overwrite: bool) -> None:
    if path is not None and path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {path}")


def extract_marker_poses(args: argparse.Namespace) -> int:
    from projectaria_tools.core import calibration, data_provider
    from projectaria_tools.core.stream_id import StreamId

    input_vrs = args.input_vrs.expanduser().resolve()
    output_csv = (args.output_csv or default_output_csv(input_vrs)).expanduser().resolve()
    output_video = args.output_video.expanduser().resolve() if args.output_video else None

    if not input_vrs.is_file():
        print(f"Error: VRS file not found: {input_vrs}", file=sys.stderr)
        return 2
    if input_vrs.suffix.lower() != ".vrs":
        print(f"Error: input must be a .vrs file: {input_vrs}", file=sys.stderr)
        return 2
    try:
        validate_output_path(output_csv, args.overwrite)
        validate_output_path(output_video, args.overwrite)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening: {input_vrs}")
    provider = data_provider.create_vrs_data_provider(str(input_vrs))
    if provider is None:
        print(f"Error: could not create a VRS data provider for {input_vrs}", file=sys.stderr)
        return 2

    stream_id = StreamId(RGB_STREAM_ID)
    available_frames = provider.get_num_data(stream_id)
    frame_count = available_frames
    if args.max_frames is not None:
        frame_count = min(available_frames, max(args.max_frames, 0))
    if frame_count == 0:
        print("Error: no RGB frames selected.", file=sys.stderr)
        return 2

    first_image = provider.get_image_data_by_index(stream_id, 0)[0].to_numpy_array()
    height, width = first_image.shape[:2]
    source_calibration = provider.get_device_calibration().get_camera_calib(RGB_STREAM_LABEL)
    linear_calibration = make_linear_calibration(source_calibration, width, height)
    camera_matrix = camera_matrix_from_calibration(linear_calibration)
    distortion = np.zeros((4, 1), dtype=np.float64)

    april_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    aruco_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    detector_specs = [
        (APRILTAG_FAMILY, april_detector, args.april_ids),
        (ARUCO_FAMILY, aruco_detector, args.aruco_ids),
    ]

    video_writer = None
    if output_video:
        video_writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (width, height),
        )
        if not video_writer.isOpened():
            print(f"Error: could not open video writer: {output_video}", file=sys.stderr)
            return 2

    detection_counts: Counter[tuple[str, int]] = Counter()
    rejected_counts: Counter[tuple[str, int]] = Counter()
    failed_pose_count = 0
    first_timestamp_ns = None
    last_timestamp_ns = None

    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_csv.name}.",
        suffix=".tmp",
        dir=output_csv.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)

    try:
        with temp_handle:
            writer = csv.DictWriter(temp_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for frame_index in range(frame_count):
                if args.progress_every > 0 and (
                    frame_index % args.progress_every == 0 or frame_index == frame_count - 1
                ):
                    print(f"Frame {frame_index + 1}/{frame_count}")

                image_data, image_record = provider.get_image_data_by_index(stream_id, frame_index)
                timestamp_ns = int(image_record.capture_timestamp_ns)
                first_timestamp_ns = timestamp_ns if first_timestamp_ns is None else first_timestamp_ns
                last_timestamp_ns = timestamp_ns

                rgb_image = image_data.to_numpy_array()
                rectified_rgb = calibration.distort_by_calibration(
                    rgb_image,
                    linear_calibration,
                    source_calibration,
                )
                bgr_image = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                for family, detector, allowed_ids in detector_specs:
                    corners, ids, _ = detector.detectMarkers(gray_image)
                    if ids is None:
                        continue

                    for marker_corners, marker_id_array in zip(corners, ids):
                        marker_id = int(marker_id_array[0])
                        if marker_id not in allowed_ids:
                            rejected_counts[(family, marker_id)] += 1
                            continue

                        image_points = marker_corners.reshape(4, 2).astype(np.float64)
                        result = pose_row(
                            input_vrs.stem,
                            frame_index,
                            timestamp_ns,
                            family,
                            marker_id,
                            image_points,
                            camera_matrix,
                            distortion,
                        )
                        if result is None:
                            failed_pose_count += 1
                            continue

                        row, rvec, tvec = result
                        writer.writerow(row)
                        detection_counts[(family, marker_id)] += 1

                        if video_writer:
                            cv2.polylines(
                                bgr_image,
                                [image_points.astype(np.int32)],
                                True,
                                (0, 255, 0),
                                2,
                            )
                            cv2.drawFrameAxes(
                                bgr_image,
                                camera_matrix,
                                distortion,
                                rvec,
                                tvec,
                                marker_size_m(family, marker_id) / 2.0,
                                2,
                            )
                            anchor = tuple(image_points[0].astype(int))
                            cv2.putText(
                                bgr_image,
                                f"{family}:{marker_id}",
                                anchor,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                if video_writer:
                    video_writer.write(bgr_image)

        temp_path.replace(output_csv)
        output_csv.chmod(0o644)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        if video_writer:
            video_writer.release()

    total_detections = sum(detection_counts.values())
    print("\nExtraction summary")
    print(f"  sequence: {input_vrs.stem}")
    print(f"  frames: {frame_count}/{available_frames}")
    print(f"  timestamp range ns: {first_timestamp_ns} .. {last_timestamp_ns}")
    print(f"  accepted marker poses: {total_detections}")
    print(f"  rejected IDs: {sum(rejected_counts.values())}")
    print(f"  failed pose estimates: {failed_pose_count}")
    for (family, marker_id), count in sorted(detection_counts.items()):
        print(f"    {family}:{marker_id} -> {count}")
    if rejected_counts:
        print("  rejected ID details:")
        for (family, marker_id), count in sorted(rejected_counts.items()):
            print(f"    {family}:{marker_id} -> {count}")
    print(f"  CSV: {output_csv}")
    if output_video:
        print(f"  video: {output_video}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return extract_marker_poses(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
