#!/usr/bin/env python3
"""Run inference-only residual-v2 predictions from an Aria Gen2 live stream.

The process receives RGB, eye gaze, hand tracking and VIO, reconstructs the
training feature schema in the AprilTag-0 frame and prints predictions. It has
no robot-control interface and cannot command a Franka arm.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from data import RECEIVING_HAND_NAMES
from live_decision import (
    GazeTargetSelector,
    InputQualityGate,
    PerceptionWorkflow,
)
from online_inference import OnlineInferenceEngine


APRILTAG_FAMILY = "apriltag_36h11"
ARUCO_FAMILY = "aruco_4x4_50"
OBJECT_MARKER_IDS = tuple(range(6, 15))
RGB_CAMERA_LABEL = "camera-rgb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("Training/final_clean_v1_residual_v2_seed44"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--profile-name", default="profile9")
    parser.add_argument(
        "--interface",
        choices=("usb", "wifi_sta", "wifi_sap"),
        default="usb",
    )
    parser.add_argument("--serial", default="")
    parser.add_argument("--server-address", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=6768)

    parser.add_argument(
        "--receiver-only",
        action="store_true",
        help=(
            "Do not connect to/start the device; receive an already-started "
            "stream."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate SDK and deployment artifacts without starting a stream.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop automatically after this duration; zero runs until Ctrl-C.",
    )

    parser.add_argument(
        "--hand-tolerance-ms",
        type=float,
        default=50.0,
        help="Maximum age of the latest causal hand sample.",
    )
    parser.add_argument(
        "--vio-tolerance-ms",
        type=float,
        default=10.0,
        help="Maximum age of the latest causal VIO sample.",
    )
    parser.add_argument(
        "--marker-tolerance-ms",
        type=float,
        default=500.0,
        help="Maximum age of the latest causal marker observation.",
    )

    parser.add_argument("--minimum-anchor-samples", type=int, default=8)
    parser.add_argument("--anchor-history", type=int, default=300)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--minimum-confidence", type=float, default=0.65)
    parser.add_argument("--minimum-stable-predictions", type=int, default=2)
    parser.add_argument(
        "--minimum-gaze-coverage",
        type=float,
        default=0.80,
        help=(
            "Minimum valid-gaze fraction in the complete model window. "
            "Otherwise decision_intention is insufficient_input."
        ),
    )
    parser.add_argument(
        "--maximum-gaze-gap-ms",
        type=float,
        default=500.0,
        help=(
            "Maximum continuous invalid-gaze interval in the model window."
        ),
    )
    parser.add_argument(
        "--minimum-handover-hand-coverage",
        type=float,
        default=0.50,
        help=(
            "Minimum per-side hand coverage required to release handover."
        ),
    )
    parser.add_argument(
        "--target-fixation-ms",
        type=float,
        default=1000.0,
        help="Required continuous gaze fixation before selecting an object.",
    )
    parser.add_argument(
        "--target-maximum-angle-rad",
        type=float,
        default=0.35,
        help="Maximum gaze-to-object angle for target selection.",
    )
    parser.add_argument(
        "--target-minimum-margin-rad",
        type=float,
        default=0.05,
        help="Minimum angular margin between the best two target objects.",
    )
    parser.add_argument(
        "--workflow-confirmation-predictions",
        type=int,
        default=2,
        help=(
            "Consecutive validated predictions required for workflow "
            "confirmation."
        ),
    )
    parser.add_argument(
        "--fetch-context-timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum age of a confirmed fetch context before handover.",
    )

    parser.add_argument(
        "--print-mode",
        choices=("changes", "all", "none"),
        default="changes",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional append-only prediction log.",
    )
    parser.add_argument(
        "--debug-features-jsonl",
        type=Path,
        default=None,
        help="Optional append-only log containing the exact model feature rows.",
    )
    parser.add_argument(
        "--debug-every-frames",
        type=int,
        default=30,
        help="Print feature availability every N assembled frames; 0 disables it.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (Path(__file__).resolve().parent.parent / path).resolve()


def resolve_output_path(path: Path) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path

    return (Path.cwd() / path).resolve()


def timestamp_ns(value) -> int:
    return int(
        round(
            float(value.tracking_timestamp.total_seconds())
            * 1e9
        )
    )


def transform_matrix(value) -> np.ndarray:
    if callable(value):
        value = value()

    matrix = np.asarray(
        value.to_matrix(),
        dtype=np.float64,
    )

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(
            f"Invalid rigid transform shape/content: {matrix.shape}"
        )

    return matrix


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    ).reshape(3)

    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        return np.full(
            3,
            np.nan,
            dtype=np.float64,
        )

    return vector / norm


def quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(
        matrix[:3, :3]
    ).as_quat()

    if quaternion[3] < 0:
        quaternion *= -1

    return quaternion


def marker_size_m(family: str, marker_id: int) -> float:
    if family == APRILTAG_FAMILY:
        return 0.10 if marker_id == 0 else 0.08

    return 0.05


def marker_object_points(size_m: float) -> np.ndarray:
    half = size_m / 2.0

    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def marker_pose(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    family: str,
    marker_id: int,
) -> np.ndarray | None:
    success, rvec, tvec = cv2.solvePnP(
        marker_object_points(
            marker_size_m(family, marker_id)
        ),
        np.asarray(
            image_points,
            dtype=np.float64,
        ).reshape(4, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success or float(tvec[2, 0]) <= 0:
        return None

    rotation, _ = cv2.Rodrigues(rvec)

    matrix = np.eye(
        4,
        dtype=np.float64,
    )
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(tvec).reshape(3)

    return matrix


def latest_before_item(
    queue: deque,
    query_ns: int,
    max_age_ns: int,
):
    """Return the newest sample at or before query_ns.

    Future samples are never used. A sample is rejected when it is older than
    max_age_ns.
    """

    for sample_timestamp, sample_value in reversed(queue):
        if sample_timestamp > query_ns:
            continue

        age_ns = query_ns - sample_timestamp

        if age_ns > max_age_ns:
            return None

        return sample_timestamp, sample_value

    return None


class LiveFeatureAssembler:
    """Create causal feature rows from asynchronous Aria callbacks."""

    def __init__(
        self,
        engine: OnlineInferenceEngine,
        *,
        hand_tolerance_ms: float,
        vio_tolerance_ms: float,
        marker_tolerance_ms: float,
        minimum_anchor_samples: int,
        anchor_history: int,
        quality_gate: InputQualityGate,
        target_selector: GazeTargetSelector,
        perception_workflow: PerceptionWorkflow,
        on_prediction: Callable[[dict], None],
        debug_features_jsonl: Path | None,
        debug_every_frames: int,
    ) -> None:
        if min(
            hand_tolerance_ms,
            vio_tolerance_ms,
            marker_tolerance_ms,
        ) <= 0:
            raise ValueError(
                "All synchronization tolerances must be positive"
            )

        if (
            minimum_anchor_samples <= 0
            or anchor_history < minimum_anchor_samples
        ):
            raise ValueError(
                "Invalid anchor sample/history configuration"
            )

        if debug_every_frames < 0:
            raise ValueError(
                "debug_every_frames cannot be negative"
            )

        self.engine = engine
        self.hand_tolerance_ns = int(
            hand_tolerance_ms * 1e6
        )
        self.vio_tolerance_ns = int(
            vio_tolerance_ms * 1e6
        )
        self.marker_tolerance_ns = int(
            marker_tolerance_ms * 1e6
        )
        self.minimum_anchor_samples = minimum_anchor_samples
        self.quality_gate = quality_gate
        self.target_selector = target_selector
        self.perception_workflow = perception_workflow
        self.on_prediction = on_prediction
        self.debug_every_frames = debug_every_frames

        self.debug_features_handle = None
        if debug_features_jsonl is not None:
            debug_features_jsonl.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.debug_features_handle = (
                debug_features_jsonl.open(
                    "a",
                    encoding="utf-8",
                )
            )

        self.lock = threading.Lock()

        self.hand_pose: deque = deque(maxlen=200)
        self.vio: deque = deque(maxlen=2000)

        marker_keys = [
            (APRILTAG_FAMILY, 0)
        ]
        marker_keys.extend(
            (ARUCO_FAMILY, marker_id)
            for marker_id in OBJECT_MARKER_IDS
        )

        self.marker_observations = {
            key: deque(maxlen=60)
            for key in marker_keys
        }

        self.anchor_candidates: deque[np.ndarray] = deque(
            maxlen=anchor_history
        )
        self.static_odometry_robot: np.ndarray | None = None

        self.device_calibration = None
        self.source_rgb_calibration = None

        self.transform_device_cpf: np.ndarray | None = None
        self.transform_device_camera: np.ndarray | None = None

        self.linear_rgb_calibration = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion = np.zeros(
            (4, 1),
            dtype=np.float64,
        )
        self.calibration_image_shape: tuple[int, int] | None = None

        self.stats = Counter()
        self.last_error: str | None = None
        self.latest_quality: dict | None = None
        self.latest_target_selection: dict | None = None
        self.latest_workflow: dict | None = None
        self.latest_anchor_diagnostics: dict | None = None

        self.april_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            ),
            cv2.aruco.DetectorParameters(),
        )

        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50
            ),
            cv2.aruco.DetectorParameters(),
        )

    def close(self) -> None:
        if self.debug_features_handle is not None:
            self.debug_features_handle.close()
            self.debug_features_handle = None

    def calibration_callback(self, device_calibration) -> None:
        try:
            source = device_calibration.get_camera_calib(
                RGB_CAMERA_LABEL
            )

            transform_device_cpf = transform_matrix(
                device_calibration.get_transform_device_cpf()
            )

            transform_device_camera = transform_matrix(
                source.get_transform_device_camera()
            )

            with self.lock:
                self.device_calibration = device_calibration
                self.source_rgb_calibration = source
                self.transform_device_cpf = transform_device_cpf
                self.transform_device_camera = transform_device_camera

                self.linear_rgb_calibration = None
                self.camera_matrix = None
                self.calibration_image_shape = None

                self.stats["calibration"] += 1

        except Exception as exc:
            self._record_error(
                "calibration",
                exc,
            )

    def eye_gaze_callback(self, value) -> None:
        try:
            timestamp = timestamp_ns(value)

            with self.lock:
                self.stats["eye_gaze"] += 1

            self.process_tick(
                value,
                timestamp,
            )

        except Exception as exc:
            self._record_error(
                "eye_gaze",
                exc,
            )

    def hand_pose_callback(self, value) -> None:
        try:
            timestamp = timestamp_ns(value)

            with self.lock:
                self.hand_pose.append(
                    (
                        timestamp,
                        value,
                    )
                )
                self.stats["hand_pose"] += 1

        except Exception as exc:
            self._record_error(
                "hand_pose",
                exc,
            )

    def vio_callback(self, value) -> None:
        try:
            values = (
                value
                if isinstance(value, (list, tuple))
                else (value,)
            )

            with self.lock:
                for item in values:
                    self.vio.append(
                        (
                            timestamp_ns(item),
                            item,
                        )
                    )
                    self.stats["vio"] += 1

        except Exception as exc:
            self._record_error(
                "vio",
                exc,
            )

    def rgb_callback(
        self,
        image_data,
        image_record,
    ) -> None:
        try:
            image = np.asarray(
                image_data.to_numpy_array()
            )

            timestamp = int(
                image_record.capture_timestamp_ns
            )

            self.process_rgb(
                image,
                timestamp,
            )

        except Exception as exc:
            self._record_error(
                "rgb",
                exc,
            )

    def _record_error(
        self,
        source: str,
        exc: Exception,
    ) -> None:
        with self.lock:
            self.stats[f"{source}_errors"] += 1
            self.last_error = (
                f"{source}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    def _ensure_linear_calibration(
        self,
        image: np.ndarray,
    ) -> None:
        from projectaria_tools.core import calibration

        height, width = image.shape[:2]
        shape = (height, width)

        with self.lock:
            source = self.source_rgb_calibration
            current_shape = self.calibration_image_shape
            camera_matrix_exists = self.camera_matrix is not None

        if source is None:
            raise RuntimeError(
                "Device calibration has not arrived"
            )

        if (
            current_shape == shape
            and camera_matrix_exists
        ):
            return

        focal_length = float(
            np.mean(
                source.get_focal_lengths()
            )
        )

        linear = calibration.get_linear_camera_calibration(
            width,
            height,
            focal_length,
            "camera-rgb-linear-live",
            source.get_transform_device_camera(),
        )

        focal_x, focal_y = linear.get_focal_lengths()
        principal_x, principal_y = linear.get_principal_point()

        camera_matrix = np.asarray(
            [
                [focal_x, 0.0, principal_x],
                [0.0, focal_y, principal_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        with self.lock:
            self.linear_rgb_calibration = linear
            self.camera_matrix = camera_matrix
            self.calibration_image_shape = shape

    def _detect_markers(
        self,
        image: np.ndarray,
    ) -> dict[tuple[str, int], np.ndarray]:
        from projectaria_tools.core import calibration

        self._ensure_linear_calibration(image)

        with self.lock:
            source = self.source_rgb_calibration
            linear = self.linear_rgb_calibration

            if self.camera_matrix is None:
                raise RuntimeError(
                    "Camera matrix has not been initialized"
                )

            camera_matrix = self.camera_matrix.copy()

        rectified = calibration.distort_by_calibration(
            image,
            linear,
            source,
        )

        gray = cv2.cvtColor(
            rectified,
            cv2.COLOR_RGB2GRAY,
        )

        found: dict[
            tuple[str, int],
            np.ndarray,
        ] = {}

        detector_configurations = (
            (
                APRILTAG_FAMILY,
                self.april_detector,
                {0},
            ),
            (
                ARUCO_FAMILY,
                self.aruco_detector,
                set(OBJECT_MARKER_IDS),
            ),
        )

        for (
            family,
            detector,
            allowed_ids,
        ) in detector_configurations:
            corners, ids, _ = detector.detectMarkers(
                gray
            )

            if ids is None:
                continue

            for (
                marker_corners,
                marker_id_array,
            ) in zip(corners, ids):
                marker_id = int(
                    np.asarray(
                        marker_id_array
                    ).reshape(-1)[0]
                )

                if marker_id not in allowed_ids:
                    continue

                pose = marker_pose(
                    marker_corners.reshape(4, 2),
                    camera_matrix,
                    self.distortion,
                    family,
                    marker_id,
                )

                if pose is not None:
                    found[
                        (
                            family,
                            marker_id,
                        )
                    ] = pose

        return found

    def _update_anchor(
        self,
        candidate: np.ndarray,
    ) -> None:
        self.anchor_candidates.append(candidate)

        if (
            len(self.anchor_candidates)
            < self.minimum_anchor_samples
        ):
            return

        candidates = list(
            self.anchor_candidates
        )

        translations = np.stack(
            [
                matrix[:3, 3]
                for matrix in candidates
            ]
        )

        center = np.median(
            translations,
            axis=0,
        )

        distances = np.linalg.norm(
            translations - center,
            axis=1,
        )

        median_distance = float(
            np.median(distances)
        )

        mad = float(
            np.median(
                np.abs(
                    distances - median_distance
                )
            )
        )

        threshold = max(
            0.02,
            median_distance
            + 3.0
            * 1.4826
            * mad,
        )

        inliers = [
            matrix
            for matrix, distance in zip(
                candidates,
                distances,
            )
            if distance <= threshold
        ]

        if not inliers:
            inliers = candidates

        anchor = np.eye(
            4,
            dtype=np.float64,
        )

        anchor[:3, 3] = np.median(
            np.stack(
                [
                    matrix[:3, 3]
                    for matrix in inliers
                ]
            ),
            axis=0,
        )

        anchor[:3, :3] = (
            Rotation.from_matrix(
                np.stack(
                    [
                        matrix[:3, :3]
                        for matrix in inliers
                    ]
                )
            )
            .mean()
            .as_matrix()
        )

        self.static_odometry_robot = anchor

    def _empty_features(
        self,
    ) -> dict[str, float]:
        values = {
            name: float("nan")
            for name in self.engine.feature_columns
        }

        for name in self.engine.feature_columns:
            if name.endswith(
                (
                    "_valid",
                    "_interpolated",
                )
            ):
                values[name] = 0.0

        return values

    @staticmethod
    def _gaze_features(
        eye_gaze,
        transform_device_cpf: np.ndarray,
        transform_robot_device: np.ndarray,
    ) -> tuple[
        dict[str, float],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        from projectaria_tools.core import mps

        values: dict[str, float] = {
            "gaze_valid": 0.0
        }

        if (
            eye_gaze is None
            or not bool(
                eye_gaze.combined_gaze_valid
            )
        ):
            return values, None, None

        yaw = float(eye_gaze.yaw)
        pitch = float(eye_gaze.pitch)
        depth = float(eye_gaze.depth)

        direction_depth = (
            depth
            if np.isfinite(depth) and depth > 0
            else 1.0
        )

        point_cpf = np.asarray(
            mps.get_eyegaze_point_at_depth(
                yaw,
                pitch,
                direction_depth,
            ),
            dtype=np.float64,
        ).reshape(3)

        direction_cpf = normalized(point_cpf)

        origin_device = transform_device_cpf[:3, 3]

        direction_device = normalized(
            transform_device_cpf[:3, :3]
            @ direction_cpf
        )

        origin_robot = (
            transform_robot_device[:3, :3]
            @ origin_device
            + transform_robot_device[:3, 3]
        )

        direction_robot = normalized(
            transform_robot_device[:3, :3]
            @ direction_device
        )

        values.update(
            {
                "gaze_valid": 1.0,
                "gaze_yaw_rad": yaw,
                "gaze_pitch_rad": pitch,
                "gaze_depth_m": (
                    depth
                    if np.isfinite(depth)
                    and depth > 0
                    else np.nan
                ),
                **{
                    f"gaze_origin_robot_{axis}_m": float(value)
                    for axis, value in zip(
                        "xyz",
                        origin_robot,
                    )
                },
                **{
                    f"gaze_direction_robot_{axis}": float(value)
                    for axis, value in zip(
                        "xyz",
                        direction_robot,
                    )
                },
            }
        )

        return (
            values,
            origin_device,
            direction_device,
        )

    @staticmethod
    def _hand_features(
        hand_pose,
        transform_robot_device: np.ndarray,
    ) -> dict[str, float]:
        values: dict[str, float] = {}

        for side in RECEIVING_HAND_NAMES:
            one_side = (
                getattr(
                    hand_pose,
                    f"{side}_hand",
                    None,
                )
                if hand_pose is not None
                else None
            )

            confidence = (
                float(one_side.confidence)
                if one_side is not None
                else -1.0
            )

            values[
                f"hand_{side}_tracking_confidence"
            ] = confidence

            values[
                f"hand_{side}_valid"
            ] = float(confidence > 0.0)

            if (
                one_side is None
                or confidence <= 0.0
            ):
                continue

            transform = getattr(
                one_side,
                "transform_device_wrist",
                None,
            )

            if transform is None:
                values[
                    f"hand_{side}_valid"
                ] = 0.0
                continue

            transform_device_wrist = transform_matrix(
                transform
            )

            transform_robot_wrist = (
                transform_robot_device
                @ transform_device_wrist
            )

            for axis, value in zip(
                "xyz",
                transform_robot_wrist[:3, 3],
            ):
                values[
                    f"{side}_wrist_robot_{axis}_m"
                ] = float(value)

            for component, value in zip(
                "xyzw",
                quaternion_xyzw(
                    transform_robot_wrist
                ),
            ):
                values[
                    f"{side}_wrist_robot_q{component}"
                ] = float(value)

        return values

    @staticmethod
    def _vio_features(
        vio,
    ) -> dict[str, float]:
        transform_odometry_device = transform_matrix(
            vio.transform_odometry_device
        )

        velocity_odometry = np.asarray(
            vio.device_linear_velocity_odometry,
            dtype=np.float64,
        ).reshape(3)

        velocity_device = (
            transform_odometry_device[:3, :3].T
            @ velocity_odometry
        )

        angular_device = np.asarray(
            vio.angular_velocity_device,
            dtype=np.float64,
        ).reshape(3)

        values = {
            f"slam_device_linear_velocity_{axis}_device": float(value)
            for axis, value in zip(
                "xyz",
                velocity_device,
            )
        }

        values.update(
            {
                f"slam_angular_velocity_{axis}_device": float(value)
                for axis, value in zip(
                    "xyz",
                    angular_device,
                )
            }
        )

        values["slam_quality_score"] = float(
            vio.quality_score
        )

        return values

    def process_rgb(
        self,
        image: np.ndarray,
        timestamp: int,
    ) -> None:
        with self.lock:
            self.stats["rgb"] += 1

            vio_item = latest_before_item(
                self.vio,
                timestamp,
                self.vio_tolerance_ns,
            )

            transform_device_camera = (
                None
                if self.transform_device_camera is None
                else self.transform_device_camera.copy()
            )

        if transform_device_camera is None:
            with self.lock:
                self.stats[
                    "rgb_waiting_calibration"
                ] += 1
            return

        markers = self._detect_markers(image)

        if vio_item is None:
            with self.lock:
                self.stats[
                    "rgb_waiting_vio"
                ] += 1
            return

        vio_timestamp, vio = vio_item

        transform_odometry_device = transform_matrix(
            vio.transform_odometry_device
        )

        with self.lock:
            self.stats[
                "rgb_vio_age_ns_sum"
            ] += timestamp - vio_timestamp

            self.stats[
                "rgb_vio_matches"
            ] += 1

            for key, marker in markers.items():
                self.marker_observations[
                    key
                ].append(
                    (
                        timestamp,
                        (
                            marker,
                            transform_odometry_device.copy(),
                        ),
                    )
                )

                family, marker_id = key

                if family == ARUCO_FAMILY:
                    self.stats[
                        f"aruco_{marker_id}"
                    ] += 1

            self.stats["marker_frames"] += 1

        tag_pose = markers.get(
            (
                APRILTAG_FAMILY,
                0,
            )
        )

        if tag_pose is not None:
            candidate = (
                transform_odometry_device
                @ transform_device_camera
                @ tag_pose
            )

            with self.lock:
                self._update_anchor(candidate)
                self.stats["apriltag_0"] += 1

    def _write_debug_features(
        self,
        timestamp: int,
        values: dict[str, float],
    ) -> None:
        if self.debug_features_handle is None:
            return

        debug_row = {
            "timestamp_ns": int(timestamp),
        }

        for name in self.engine.feature_columns:
            raw_value = values.get(
                name,
                np.nan,
            )

            try:
                numeric_value = float(raw_value)
            except (
                TypeError,
                ValueError,
            ):
                numeric_value = np.nan

            debug_row[name] = (
                float(numeric_value)
                if np.isfinite(numeric_value)
                else None
            )

        self.debug_features_handle.write(
            json.dumps(
                debug_row,
                ensure_ascii=False,
            )
            + "\n"
        )
        self.debug_features_handle.flush()

    def process_tick(
        self,
        eye_gaze,
        timestamp: int,
    ) -> None:
        with self.lock:
            hand_item = latest_before_item(
                self.hand_pose,
                timestamp,
                self.hand_tolerance_ns,
            )

            vio_item = latest_before_item(
                self.vio,
                timestamp,
                self.vio_tolerance_ns,
            )

            marker_observations = {
                key: latest_before_item(
                    queue,
                    timestamp,
                    self.marker_tolerance_ns,
                )
                for key, queue
                in self.marker_observations.items()
            }

            transform_device_cpf = (
                None
                if self.transform_device_cpf is None
                else self.transform_device_cpf.copy()
            )

            transform_device_camera = (
                None
                if self.transform_device_camera is None
                else self.transform_device_camera.copy()
            )

            anchor = (
                None
                if self.static_odometry_robot is None
                else self.static_odometry_robot.copy()
            )

            anchor_samples = len(
                self.anchor_candidates
            )

            assembled_frame_index = int(
                self.stats["model_frames"]
            )

        if (
            vio_item is None
            or transform_device_cpf is None
            or transform_device_camera is None
        ):
            with self.lock:
                self.stats[
                    "ticks_waiting_core_streams"
                ] += 1
            return

        if anchor is None:
            with self.lock:
                self.stats[
                    "ticks_waiting_anchor"
                ] += 1
            return

        hand_pose = (
            None
            if hand_item is None
            else hand_item[1]
        )

        vio_timestamp, vio = vio_item

        tag_observation = marker_observations[
            (
                APRILTAG_FAMILY,
                0,
            )
        ]

        tag_age_ns = (
            None
            if tag_observation is None
            else timestamp - tag_observation[0]
        )

        tag_is_frame_aligned = (
            tag_age_ns is not None
            and tag_age_ns <= int(20e6)
        )
        anchor_diagnostics = {
            "anchor_ready": True,
            "apriltag_0_recent": tag_observation is not None,
            "apriltag_0_frame_aligned": tag_is_frame_aligned,
            "apriltag_0_age_ms": (
                None
                if tag_age_ns is None
                else float(tag_age_ns / 1e6)
            ),
            "anchor_samples": anchor_samples,
        }

        transform_odometry_device = transform_matrix(
            vio.transform_odometry_device
        )

        transform_robot_odometry = np.linalg.inv(
            anchor
        )

        transform_robot_device = (
            transform_robot_odometry
            @ transform_odometry_device
        )

        values = self._empty_features()

        values.update(
            self._vio_features(vio)
        )

        (
            gaze_values,
            gaze_origin_device,
            gaze_direction_device,
        ) = self._gaze_features(
            eye_gaze,
            transform_device_cpf,
            transform_robot_device,
        )

        values.update(gaze_values)

        values.update(
            self._hand_features(
                hand_pose,
                transform_robot_device,
            )
        )

        values["apriltag_0_valid"] = float(
            tag_is_frame_aligned
        )
        values["robot_frame_valid"] = 1.0
        values["robot_anchor_interpolated"] = float(
            not tag_is_frame_aligned
        )

        for marker_id in OBJECT_MARKER_IDS:
            prefix = f"aruco_{marker_id}"

            observation = marker_observations[
                (
                    ARUCO_FAMILY,
                    marker_id,
                )
            ]

            values[
                f"{prefix}_valid"
            ] = float(
                observation is not None
            )

            if observation is None:
                continue

            _, (
                marker,
                marker_odometry_device,
            ) = observation

            transform_robot_object = (
                transform_robot_odometry
                @ marker_odometry_device
                @ transform_device_camera
                @ marker
            )

            for axis, value in zip(
                "xyz",
                transform_robot_object[:3, 3],
            ):
                values[
                    f"{prefix}_robot_{axis}_m"
                ] = float(value)

            if (
                gaze_origin_device is not None
                and gaze_direction_device is not None
            ):
                transform_device_object = (
                    np.linalg.inv(
                        transform_robot_device
                    )
                    @ transform_robot_object
                )

                to_object = (
                    transform_device_object[:3, 3]
                    - gaze_origin_device
                )

                distance = float(
                    np.linalg.norm(to_object)
                )

                if distance > 1e-12:
                    cosine = float(
                        np.clip(
                            np.dot(
                                gaze_direction_device,
                                to_object / distance,
                            ),
                            -1.0,
                            1.0,
                        )
                    )

                    values[
                        f"{prefix}_gaze_angle_rad"
                    ] = float(
                        np.arccos(cosine)
                    )

                    values[
                        f"{prefix}_gaze_distance_m"
                    ] = distance

        stream_reset = self.quality_gate.push_frame(
            timestamp,
            values,
        )
        if stream_reset:
            self.target_selector.reset()
            self.perception_workflow.reset()

        target_selection = self.target_selector.update(
            timestamp,
            values,
        )

        if (
            self.debug_every_frames > 0
            and assembled_frame_index
            % self.debug_every_frames
            == 0
        ):
            visible_objects = [
                marker_id
                for marker_id in OBJECT_MARKER_IDS
                if values.get(
                    f"aruco_{marker_id}_valid",
                    0.0,
                )
                > 0.5
            ]

            hand_age_ms = (
                None
                if hand_item is None
                else (
                    timestamp - hand_item[0]
                )
                / 1e6
            )

            vio_age_ms = (
                timestamp - vio_timestamp
            ) / 1e6

            print(
                f"DEBUG objects={visible_objects}, "
                f"left_valid={values.get('hand_left_valid')}, "
                f"right_valid={values.get('hand_right_valid')}, "
                f"gaze_valid={values.get('gaze_valid')}, "
                f"hand_age_ms={hand_age_ms}, "
                f"vio_age_ms={vio_age_ms:.3f}"
            )

        self._write_debug_features(
            timestamp,
            values,
        )

        prediction = self.engine.push_frame(
            timestamp,
            values,
        )

        if prediction is not None:
            quality = self.quality_gate.assess(
                prediction["stable_intention"]
            )
            decision_intention = (
                prediction["stable_intention"]
                if quality["ok"]
                else "insufficient_input"
            )
            workflow = self.perception_workflow.update(
                timestamp,
                decision_intention,
                target_selection,
            )
            prediction.update(
                {
                    "decision_intention": decision_intention,
                    "input_quality_ok": quality["ok"],
                    "input_quality": quality,
                    "target_selection": target_selection,
                    "perception_workflow": workflow,
                    "anchor_diagnostics": anchor_diagnostics,
                    "external_action_requested": False,
                }
            )

        with self.lock:
            self.stats["model_frames"] += 1
            self.stats[
                "anchor_samples"
            ] = anchor_samples

            if hand_item is None:
                self.stats[
                    "model_frames_without_hand_sample"
                ] += 1
            else:
                self.stats[
                    "hand_age_ns_sum"
                ] += timestamp - hand_item[0]

                self.stats[
                    "hand_matches"
                ] += 1

            self.stats[
                "vio_age_ns_sum"
            ] += timestamp - vio_timestamp

            self.stats[
                "vio_matches"
            ] += 1

            if prediction is not None:
                self.stats["predictions"] += 1
                if not prediction["input_quality_ok"]:
                    self.stats[
                        "predictions_blocked_by_input_quality"
                    ] += 1

            self.latest_target_selection = target_selection
            self.latest_anchor_diagnostics = anchor_diagnostics
            if prediction is not None:
                self.latest_quality = prediction["input_quality"]
                self.latest_workflow = prediction["perception_workflow"]

        if prediction is not None:
            self.on_prediction(prediction)

    def status(self) -> dict:
        with self.lock:
            status = {
                **dict(self.stats),
                "buffer_frames": self.engine.ready_frames,
                "required_frames": self.engine.required_frames,
                "anchor_ready": (
                    self.static_odometry_robot
                    is not None
                ),
                "last_error": self.last_error,
                "input_quality": self.latest_quality,
                "target_selection": self.latest_target_selection,
                "perception_workflow": self.latest_workflow,
                "anchor_diagnostics": self.latest_anchor_diagnostics,
            }

            hand_matches = int(
                self.stats.get(
                    "hand_matches",
                    0,
                )
            )

            if hand_matches:
                status["mean_hand_age_ms"] = (
                    self.stats[
                        "hand_age_ns_sum"
                    ]
                    / hand_matches
                    / 1e6
                )

            vio_matches = int(
                self.stats.get(
                    "vio_matches",
                    0,
                )
            )

            if vio_matches:
                status["mean_vio_age_ms"] = (
                    self.stats[
                        "vio_age_ns_sum"
                    ]
                    / vio_matches
                    / 1e6
                )

            rgb_vio_matches = int(
                self.stats.get(
                    "rgb_vio_matches",
                    0,
                )
            )

            if rgb_vio_matches:
                status["mean_rgb_vio_age_ms"] = (
                    self.stats[
                        "rgb_vio_age_ns_sum"
                    ]
                    / rgb_vio_matches
                    / 1e6
                )

            return status


def prediction_printer(
    print_mode: str,
    output_jsonl: Path | None,
) -> tuple[
    Callable[[dict], None],
    Callable[[], None],
]:
    previous_signature = None
    handle = None

    if output_jsonl is not None:
        output_jsonl = (
            output_jsonl
            .expanduser()
            .resolve()
        )

        output_jsonl.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handle = output_jsonl.open(
            "a",
            encoding="utf-8",
        )

    def emit(prediction: dict) -> None:
        nonlocal previous_signature

        model_label = prediction[
            "stable_intention"
        ]
        decision_label = prediction.get(
            "decision_intention",
            model_label,
        )
        workflow = prediction.get(
            "perception_workflow",
            {},
        )
        signature = (
            decision_label,
            workflow.get("state"),
            workflow.get("selected_object_id"),
        )

        should_print = (
            print_mode == "all"
            or (
                print_mode == "changes"
                and (
                    signature != previous_signature
                    or decision_label == "handover"
                )
            )
        )

        if should_print:
            pose = prediction[
                "predicted_pose_robot"
            ]

            pose_text = ""

            if (
                pose is not None
                and decision_label == "handover"
            ):
                pose_text = (
                    f" | hand="
                    f"{prediction['predicted_receiving_hand']} | "
                    f"xyz=("
                    f"{pose[0]:+.3f}, "
                    f"{pose[1]:+.3f}, "
                    f"{pose[2]:+.3f}"
                    f") m"
                )
            elif decision_label == "handover":
                pose_text = (
                    " | pose=no valid hand reference"
                )

            quality_text = ""
            if not prediction.get(
                "input_quality_ok",
                True,
            ):
                reasons = prediction.get(
                    "input_quality",
                    {},
                ).get("reasons", [])
                quality_text = (
                    " | blocked="
                    + ",".join(reasons)
                )

            target_id = workflow.get(
                "selected_object_id"
            )
            target_text = (
                ""
                if target_id is None
                else f" | target=aruco_{target_id}"
            )
            workflow_text = (
                ""
                if workflow.get("state") is None
                else f" | state={workflow['state']}"
            )

            print(
                f"[live {prediction['prediction_index']:05d}] "
                f"raw={prediction['raw_intention']} "
                f"({prediction['raw_confidence']:.3f}) | "
                f"probs=["
                f"continue={prediction['p_continue']:.3f}, "
                f"fetch={prediction['p_fetch']:.3f}, "
                f"handover={prediction['p_handover']:.3f}"
                f"] | "
                f"model={model_label} "
                f"({prediction['stable_confidence']:.3f}) | "
                f"decision={decision_label} | "
                f"intent={prediction['intention_inference_ms']:.2f} ms"
                f"{quality_text}"
                f"{target_text}"
                f"{workflow_text}"
                f"{pose_text}"
            )

            previous_signature = signature

        if handle is not None:
            handle.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

    def close() -> None:
        if handle is not None:
            handle.close()

    return emit, close


def check_environment(
    args: argparse.Namespace,
) -> tuple[
    OnlineInferenceEngine,
    object,
]:
    try:
        import aria.sdk_gen2 as sdk_gen2
        import aria.stream_receiver  # noqa: F401
        from projectaria_tools.core import calibration  # noqa: F401

    except ImportError as exc:
        raise RuntimeError(
            "Aria Gen2 client SDK is unavailable. "
            "Run with the aria_conda environment. "
            f"Original error: {exc}"
        ) from exc

    engine = OnlineInferenceEngine(
        resolve_path(args.artifacts_dir),
        device=args.device,
        smoothing_window=args.smoothing_window,
        minimum_confidence=args.minimum_confidence,
        minimum_stable_predictions=(
            args.minimum_stable_predictions
        ),
    )

    return engine, sdk_gen2


def run_live(
    args: argparse.Namespace,
) -> int:
    if args.duration_seconds < 0:
        raise ValueError(
            "duration_seconds cannot be negative"
        )

    if (
        args.server_port <= 0
        or args.server_port > 65535
    ):
        raise ValueError(
            "server_port must be between 1 and 65535"
        )

    engine, sdk_gen2 = check_environment(args)

    device_client = sdk_gen2.DeviceClient()

    discovery_error = None

    try:
        targets = (
            device_client.usb_network_devices()
        )
    except RuntimeError as exc:
        targets = []
        discovery_error = str(exc)

    print(
        f"SDK ready: "
        f"model_features={len(engine.feature_columns)}, "
        f"window={engine.required_frames}, "
        f"connected_usb_devices={len(targets)}"
    )

    if discovery_error is not None:
        print(
            "Device discovery: no reachable "
            f"USB device ({discovery_error})"
        )

    if args.check_only:
        for target in targets:
            print(
                f"Device: "
                f"serial={target.serial}, "
                f"ip={target.ip}"
            )

        return 0

    import aria.stream_receiver as receiver

    output_path = None
    if args.output_jsonl is not None:
        output_path = resolve_output_path(
            args.output_jsonl
        )

    debug_features_path = None
    if args.debug_features_jsonl is not None:
        debug_features_path = resolve_output_path(
            args.debug_features_jsonl
        )

    (
        emit_prediction,
        close_predictions,
    ) = prediction_printer(
        args.print_mode,
        output_path,
    )

    quality_gate = InputQualityGate(
        window_size=engine.required_frames,
        max_timestamp_gap_ns=(
            engine.artifacts.max_timestamp_gap_ns
        ),
        minimum_gaze_coverage=(
            args.minimum_gaze_coverage
        ),
        maximum_gaze_gap_ms=(
            args.maximum_gaze_gap_ms
        ),
        minimum_handover_hand_coverage=(
            args.minimum_handover_hand_coverage
        ),
    )
    target_selector = GazeTargetSelector(
        object_ids=OBJECT_MARKER_IDS,
        fixation_ms=args.target_fixation_ms,
        maximum_angle_rad=(
            args.target_maximum_angle_rad
        ),
        minimum_angle_margin_rad=(
            args.target_minimum_margin_rad
        ),
    )
    perception_workflow = PerceptionWorkflow(
        confirmation_predictions=(
            args.workflow_confirmation_predictions
        ),
        fetch_context_timeout_seconds=(
            args.fetch_context_timeout_seconds
        ),
    )

    assembler = LiveFeatureAssembler(
        engine,
        hand_tolerance_ms=(
            args.hand_tolerance_ms
        ),
        vio_tolerance_ms=(
            args.vio_tolerance_ms
        ),
        marker_tolerance_ms=(
            args.marker_tolerance_ms
        ),
        minimum_anchor_samples=(
            args.minimum_anchor_samples
        ),
        anchor_history=(
            args.anchor_history
        ),
        quality_gate=quality_gate,
        target_selector=target_selector,
        perception_workflow=(
            perception_workflow
        ),
        on_prediction=emit_prediction,
        debug_features_jsonl=(
            debug_features_path
        ),
        debug_every_frames=(
            args.debug_every_frames
        ),
    )

    stream_receiver = receiver.StreamReceiver(
        enable_image_decoding=True,
        enable_raw_stream=False,
    )

    stream_receiver.set_rgb_queue_size(2)
    stream_receiver.set_vio_high_freq_queue_size(
        200
    )
    stream_receiver.set_vio_high_freq_batch_queue_size(
        1
    )
    stream_receiver.set_eye_gaze_queue_size(
        20
    )
    stream_receiver.set_hand_pose_queue_size(
        20
    )

    stream_receiver.register_device_calib_callback(
        assembler.calibration_callback
    )
    stream_receiver.register_rgb_callback(
        assembler.rgb_callback
    )
    stream_receiver.register_eye_gaze_callback(
        assembler.eye_gaze_callback
    )
    stream_receiver.register_hand_pose_callback(
        assembler.hand_pose_callback
    )
    stream_receiver.register_vio_high_frequency_callback(
        assembler.vio_callback
    )

    server_config = sdk_gen2.HttpServerConfig()
    server_config.address = args.server_address
    server_config.port = args.server_port

    stream_receiver.set_server_config(
        server_config
    )

    device = None
    started_device = False
    started_server = False

    try:
        if not args.receiver_only:
            config = sdk_gen2.DeviceClientConfig()

            device_client.set_client_config(
                config
            )

            target = sdk_gen2.DeviceTarget(
                serial=args.serial
            )

            device = device_client.connect(
                target
            )

            interface_map = {
                "usb": (
                    sdk_gen2.StreamingInterface.USB_NCM
                ),
                "wifi_sta": (
                    sdk_gen2.StreamingInterface.WIFI_STA
                ),
                "wifi_sap": (
                    sdk_gen2.StreamingInterface.WIFI_SAP
                ),
            }

            streaming_config = (
                sdk_gen2.HttpStreamingConfig()
            )

            streaming_config.profile_name = (
                args.profile_name
            )

            streaming_config.streaming_interface = (
                interface_map[args.interface]
            )

            device.set_streaming_config(
                streaming_config
            )

            device.start_streaming()
            started_device = True

            print(
                f"Aria stream started: "
                f"serial={device.serial()}, "
                f"profile={args.profile_name}, "
                f"interface={args.interface}"
            )

        stream_receiver.start_server()
        started_server = True

        if args.receiver_only:
            print(
                f"Receiver listening on "
                f"{args.server_address}:"
                f"{args.server_port}; "
                "waiting for an externally "
                "started stream"
            )

        print(
            "INFERENCE ONLY: no robot commands "
            "are produced. Keep AprilTag 0 visible "
            f"until at least "
            f"{args.minimum_anchor_samples} "
            "anchor samples. decision_intention is "
            "released only when the complete input "
            "quality window passes."
        )

        started = time.monotonic()
        next_status = started

        while True:
            now = time.monotonic()

            if (
                args.duration_seconds
                and now - started
                >= args.duration_seconds
            ):
                break

            if now >= next_status:
                status = assembler.status()

                print(
                    "Status: "
                    + json.dumps(
                        status,
                        ensure_ascii=False,
                    )
                )

                next_status = now + 5.0

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping after Ctrl-C")

    finally:
        if (
            started_device
            and device is not None
        ):
            try:
                device.stop_streaming()
            except Exception as exc:
                print(
                    "Warning: device stop failed: "
                    f"{exc}"
                )

        try:
            if started_server:
                stream_receiver.stop_server()

        finally:
            assembler.close()
            close_predictions()

            if device is not None:
                try:
                    device_client.disconnect(
                        device
                    )
                except Exception:
                    pass

    return 0


def main() -> int:
    args = parse_args()

    try:
        return run_live(args)

    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            f"ERROR: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
