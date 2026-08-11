#!/usr/bin/env python3
"""Pure, timestamp-aware baselines for receiving-wrist pose prediction.

The master timeline is an alignment grid.  Wrist velocity must therefore be
estimated from the original hand-capture timestamps (``hand_timestamp_ns``),
not from the timestamps of rows to which a capture was aligned.  This module
contains no dataset splitting or model code so the scientific invariants can be
tested independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


POSE_COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw")
POSITION_ERROR_DEFINITION = (
    "Euclidean norm of the 3D position error in centimetres"
)
ORIENTATION_ERROR_DEFINITION = (
    "sign-invariant geodesic quaternion error in degrees"
)


def truthy(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def normalize_quaternion(values: np.ndarray) -> np.ndarray | None:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-6:
        return None
    return quaternion / norm


def pose_columns(side: str) -> list[str]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown hand side: {side!r}")
    return [
        *(f"{side}_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{side}_wrist_robot_q{component}" for component in "xyzw"),
    ]


def observation_pose(
    frame: pd.DataFrame,
    row_index: int,
    side: str,
) -> np.ndarray | None:
    row = frame.iloc[row_index]
    if not truthy(row[f"hand_{side}_valid"]) or not truthy(
        row["robot_frame_valid"]
    ):
        return None
    values = pd.to_numeric(
        row[pose_columns(side)], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    quaternion = normalize_quaternion(values[3:7])
    if quaternion is None:
        return None
    return np.concatenate((values[:3], quaternion)).astype(np.float32)


@dataclass(frozen=True)
class ObservationSeries:
    """Unique, causal hand captures contained in one observation window."""

    row_indices: np.ndarray
    capture_timestamps_ns: np.ndarray
    poses: np.ndarray
    duplicate_captures_removed: int = 0
    noncausal_captures_removed: int = 0

    @classmethod
    def empty(
        cls,
        *,
        duplicate_captures_removed: int = 0,
        noncausal_captures_removed: int = 0,
    ) -> "ObservationSeries":
        return cls(
            row_indices=np.empty(0, dtype=np.int64),
            capture_timestamps_ns=np.empty(0, dtype=np.int64),
            poses=np.empty((0, 7), dtype=np.float32),
            duplicate_captures_removed=duplicate_captures_removed,
            noncausal_captures_removed=noncausal_captures_removed,
        )


@dataclass(frozen=True)
class BaselineEstimate:
    """One method's prediction and explicit availability diagnostics."""

    pose: np.ndarray | None
    available: bool
    reason: str
    latest_capture_timestamp_ns: int | None = None
    observation_age_seconds: float | None = None
    fit_samples: int = 0
    fit_span_seconds: float | None = None
    estimated_speed_m_s: float | None = None


@dataclass(frozen=True)
class TargetTiming:
    """Nominal, aligned-master, and physical hand-capture target times."""

    nominal_timestamp_ns: int
    aligned_master_timestamp_ns: int
    actual_hand_capture_timestamp_ns: int
    target_row_index: int
    master_alignment_error_ms: float
    actual_capture_error_ms: float

    @property
    def prediction_horizon_timestamp_ns(self) -> int:
        """Inference-time horizon; never expose future capture jitter to a baseline."""

        return self.nominal_timestamp_ns


def extract_hand_observations(
    frame: pd.DataFrame,
    start: int,
    endpoint: int,
    side: str,
    *,
    source_timestamp_column: str = "hand_timestamp_ns",
    require_causal: bool = True,
) -> ObservationSeries:
    """Extract valid captures and deduplicate merge-asof row repetitions.

    A single hand capture can be aligned to multiple master rows.  The last
    aligned row for each source timestamp is retained, then captures are sorted
    by their actual device timestamp.  With ``require_causal=True`` captures
    after the endpoint timestamp are excluded, even if a nearest-neighbour
    merge placed them on the endpoint row.
    """

    if side not in {"left", "right"}:
        return ObservationSeries.empty()
    if not 0 <= start <= endpoint < len(frame):
        raise IndexError(
            f"Invalid window rows start={start}, endpoint={endpoint}, rows={len(frame)}"
        )
    if source_timestamp_column not in frame:
        raise ValueError(
            f"Missing real hand-capture timestamp column: {source_timestamp_column}"
        )

    endpoint_timestamp_ns = int(frame.iloc[endpoint]["timestamp_ns"])
    by_timestamp: dict[int, tuple[int, np.ndarray]] = {}
    valid_rows = 0
    noncausal = 0
    for row_index in range(start, endpoint + 1):
        pose = observation_pose(frame, row_index, side)
        if pose is None:
            continue
        raw_timestamp = pd.to_numeric(
            pd.Series([frame.iloc[row_index][source_timestamp_column]]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(raw_timestamp):
            continue
        capture_timestamp_ns = int(raw_timestamp)
        valid_rows += 1
        if require_causal and capture_timestamp_ns > endpoint_timestamp_ns:
            noncausal += 1
            continue
        # Keeping the last aligned row is deterministic and preserves the pose
        # visible closest to the window endpoint when a source capture repeats.
        by_timestamp[capture_timestamp_ns] = (row_index, pose)

    duplicate_count = max(0, valid_rows - noncausal - len(by_timestamp))
    if not by_timestamp:
        return ObservationSeries.empty(
            duplicate_captures_removed=duplicate_count,
            noncausal_captures_removed=noncausal,
        )

    ordered = sorted(by_timestamp.items())
    return ObservationSeries(
        row_indices=np.asarray(
            [row_and_pose[0] for _, row_and_pose in ordered], dtype=np.int64
        ),
        capture_timestamps_ns=np.asarray(
            [timestamp for timestamp, _ in ordered], dtype=np.int64
        ),
        poses=np.asarray(
            [row_and_pose[1] for _, row_and_pose in ordered], dtype=np.float32
        ),
        duplicate_captures_removed=duplicate_count,
        noncausal_captures_removed=noncausal,
    )


def persistence_pose(
    observations: ObservationSeries,
    endpoint_timestamp_ns: int,
    *,
    maximum_age_seconds: float,
) -> BaselineEstimate:
    """Predict the latest current receiving-wrist pose without fallback."""

    if maximum_age_seconds <= 0.0:
        raise ValueError("maximum_age_seconds must be greater than zero")
    if not len(observations.poses):
        return BaselineEstimate(None, False, "no_valid_observation")

    latest_timestamp_ns = int(observations.capture_timestamps_ns[-1])
    age_seconds = (int(endpoint_timestamp_ns) - latest_timestamp_ns) / 1e9
    if age_seconds < -1e-9:
        return BaselineEstimate(
            None,
            False,
            "noncausal_latest_observation",
            latest_capture_timestamp_ns=latest_timestamp_ns,
            observation_age_seconds=age_seconds,
        )
    if age_seconds > maximum_age_seconds:
        return BaselineEstimate(
            None,
            False,
            "stale_latest_observation",
            latest_capture_timestamp_ns=latest_timestamp_ns,
            observation_age_seconds=age_seconds,
        )
    return BaselineEstimate(
        observations.poses[-1].copy(),
        True,
        "available",
        latest_capture_timestamp_ns=latest_timestamp_ns,
        observation_age_seconds=age_seconds,
        fit_samples=1,
    )


def constant_velocity_pose(
    observations: ObservationSeries,
    endpoint_timestamp_ns: int,
    target_timestamp_ns: int,
    *,
    maximum_age_seconds: float,
    lookback_seconds: float,
    minimum_fit_span_seconds: float,
) -> BaselineEstimate:
    """Fit linear position velocity on real captures and extrapolate to target.

    Orientation uses a zero-angular-velocity model: the latest normalized
    quaternion is held constant.  This is robust, sign invariant during
    evaluation, and directly comparable to positional persistence.
    """

    if lookback_seconds <= 0.0:
        raise ValueError("lookback_seconds must be greater than zero")
    if minimum_fit_span_seconds <= 0.0:
        raise ValueError("minimum_fit_span_seconds must be greater than zero")
    if target_timestamp_ns <= endpoint_timestamp_ns:
        raise ValueError("target_timestamp_ns must be after endpoint_timestamp_ns")

    latest = persistence_pose(
        observations,
        endpoint_timestamp_ns,
        maximum_age_seconds=maximum_age_seconds,
    )
    if not latest.available:
        return BaselineEstimate(
            None,
            False,
            latest.reason,
            latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
            observation_age_seconds=latest.observation_age_seconds,
        )

    earliest_timestamp_ns = int(endpoint_timestamp_ns) - int(
        round(lookback_seconds * 1e9)
    )
    selected = observations.capture_timestamps_ns >= earliest_timestamp_ns
    timestamps_ns = observations.capture_timestamps_ns[selected]
    poses = observations.poses[selected]
    if len(timestamps_ns) < 2:
        return BaselineEstimate(
            None,
            False,
            "insufficient_unique_observations",
            latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
            observation_age_seconds=latest.observation_age_seconds,
            fit_samples=int(len(timestamps_ns)),
        )

    fit_span_seconds = (
        int(timestamps_ns[-1]) - int(timestamps_ns[0])
    ) / 1e9
    if fit_span_seconds < minimum_fit_span_seconds:
        return BaselineEstimate(
            None,
            False,
            "insufficient_fit_span",
            latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
            observation_age_seconds=latest.observation_age_seconds,
            fit_samples=int(len(timestamps_ns)),
            fit_span_seconds=fit_span_seconds,
        )

    latest_timestamp_ns = int(timestamps_ns[-1])
    times_seconds = (
        timestamps_ns.astype(np.float64) - float(latest_timestamp_ns)
    ) / 1e9
    positions = poses[:, :3].astype(np.float64)
    centered = times_seconds - times_seconds.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return BaselineEstimate(
            None,
            False,
            "degenerate_fit_timestamps",
            latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
            observation_age_seconds=latest.observation_age_seconds,
            fit_samples=int(len(timestamps_ns)),
            fit_span_seconds=fit_span_seconds,
        )
    velocity = np.sum(centered[:, None] * positions, axis=0) / denominator
    delta_seconds = (int(target_timestamp_ns) - latest_timestamp_ns) / 1e9
    predicted_position = poses[-1, :3].astype(np.float64) + velocity * delta_seconds
    prediction = np.concatenate((predicted_position, poses[-1, 3:7])).astype(
        np.float32
    )
    if not np.isfinite(prediction).all():
        return BaselineEstimate(
            None,
            False,
            "nonfinite_prediction",
            latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
            observation_age_seconds=latest.observation_age_seconds,
            fit_samples=int(len(timestamps_ns)),
            fit_span_seconds=fit_span_seconds,
        )
    return BaselineEstimate(
        prediction,
        True,
        "available",
        latest_capture_timestamp_ns=latest.latest_capture_timestamp_ns,
        observation_age_seconds=latest.observation_age_seconds,
        fit_samples=int(len(timestamps_ns)),
        fit_span_seconds=fit_span_seconds,
        estimated_speed_m_s=float(np.linalg.norm(velocity)),
    )


def resolve_target_timing(
    frame: pd.DataFrame,
    endpoint: int,
    *,
    horizon_seconds: float,
    source_timestamp_column: str = "hand_timestamp_ns",
) -> TargetTiming:
    """Resolve the physical capture time of the already aligned future target."""

    if horizon_seconds <= 0.0:
        raise ValueError("horizon_seconds must be greater than zero")
    for column in ("timestamp_ns", "future_target_timestamp_ns", source_timestamp_column):
        if column not in frame:
            raise ValueError(f"Missing target-timing column: {column}")
    endpoint_timestamp_ns = int(frame.iloc[endpoint]["timestamp_ns"])
    nominal_timestamp_ns = endpoint_timestamp_ns + int(
        round(horizon_seconds * 1e9)
    )
    raw_aligned = pd.to_numeric(
        pd.Series([frame.iloc[endpoint]["future_target_timestamp_ns"]]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(raw_aligned):
        raise ValueError("A valid pose target has no aligned target timestamp")
    aligned_timestamp_ns = int(raw_aligned)
    matching_rows = np.flatnonzero(
        pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(np.int64)
        == aligned_timestamp_ns
    )
    if len(matching_rows) != 1:
        raise ValueError(
            "Aligned target timestamp must identify exactly one master row; "
            f"timestamp={aligned_timestamp_ns}, matches={len(matching_rows)}"
        )
    target_row = int(matching_rows[0])
    raw_capture = pd.to_numeric(
        pd.Series([frame.iloc[target_row][source_timestamp_column]]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(raw_capture):
        raise ValueError("A valid pose target has no physical hand-capture timestamp")
    actual_capture_timestamp_ns = int(raw_capture)
    if actual_capture_timestamp_ns <= endpoint_timestamp_ns:
        raise ValueError(
            "Future target capture must occur after the observation endpoint"
        )
    return TargetTiming(
        nominal_timestamp_ns=nominal_timestamp_ns,
        aligned_master_timestamp_ns=aligned_timestamp_ns,
        actual_hand_capture_timestamp_ns=actual_capture_timestamp_ns,
        target_row_index=target_row,
        master_alignment_error_ms=(aligned_timestamp_ns - nominal_timestamp_ns)
        / 1e6,
        actual_capture_error_ms=(
            actual_capture_timestamp_ns - nominal_timestamp_ns
        )
        / 1e6,
    )


def pose_matches(
    first: np.ndarray,
    second: np.ndarray,
    *,
    position_atol: float = 1e-5,
    quaternion_atol: float = 1e-5,
) -> bool:
    """Compare poses with sign-invariant normalized quaternion semantics."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (7,) or second.shape != (7,):
        return False
    first_q = normalize_quaternion(first[3:7])
    second_q = normalize_quaternion(second[3:7])
    if first_q is None or second_q is None:
        return False
    return bool(
        np.allclose(first[:3], second[:3], atol=position_atol, rtol=0.0)
        and (
            np.allclose(first_q, second_q, atol=quaternion_atol, rtol=0.0)
            or np.allclose(first_q, -second_q, atol=quaternion_atol, rtol=0.0)
        )
    )


def pose_error_values(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1:] != (7,):
        raise ValueError("predictions must have shape (N, 7)")
    if targets.shape != predictions.shape:
        raise ValueError("targets must have the same shape as predictions")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Pose metrics require finite predictions and targets")

    position_errors_cm = np.linalg.norm(
        predictions[:, :3] - targets[:, :3], axis=1
    ) * 100.0
    predicted_norms = np.linalg.norm(predictions[:, 3:7], axis=1)
    target_norms = np.linalg.norm(targets[:, 3:7], axis=1)
    if np.any(predicted_norms <= 1e-6) or np.any(target_norms <= 1e-6):
        raise ValueError("Pose metrics require valid quaternions")
    predicted_q = predictions[:, 3:7] / predicted_norms[:, None]
    target_q = targets[:, 3:7] / target_norms[:, None]
    cosine = np.clip(np.abs(np.sum(predicted_q * target_q, axis=1)), 0.0, 1.0)
    orientation_errors_deg = np.degrees(2.0 * np.arccos(cosine))
    return position_errors_cm, orientation_errors_deg


def pose_metric_summary(
    predictions: list[np.ndarray] | np.ndarray,
    targets: list[np.ndarray] | np.ndarray,
    *,
    coverage_denominator: int,
) -> dict:
    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    if coverage_denominator < 0:
        raise ValueError("coverage_denominator cannot be negative")
    if predictions_array.size == 0:
        predictions_array = np.empty((0, 7), dtype=np.float32)
        targets_array = np.empty((0, 7), dtype=np.float32)
    position_errors_cm, orientation_errors_deg = pose_error_values(
        predictions_array, targets_array
    )
    samples = int(len(predictions_array))
    coverage = (
        float(samples / coverage_denominator) if coverage_denominator else None
    )
    if not samples:
        return {
            "samples": 0,
            "position_mean_euclidean_error_cm": None,
            "position_median_euclidean_error_cm": None,
            "position_root_mean_square_euclidean_error_cm": None,
            "position_mae_cm": None,
            "position_rmse_cm": None,
            "orientation_mean_deg": None,
            "orientation_median_deg": None,
            "coverage_numerator": 0,
            "coverage_denominator": int(coverage_denominator),
            "coverage": coverage,
            "position_error_definition": POSITION_ERROR_DEFINITION,
            "orientation_error_definition": ORIENTATION_ERROR_DEFINITION,
        }
    mean_position = float(np.mean(position_errors_cm))
    rms_position = float(np.sqrt(np.mean(np.square(position_errors_cm))))
    return {
        "samples": samples,
        "position_mean_euclidean_error_cm": mean_position,
        "position_median_euclidean_error_cm": float(
            np.median(position_errors_cm)
        ),
        "position_root_mean_square_euclidean_error_cm": rms_position,
        # Explicit compatibility aliases for existing report consumers.
        "position_mae_cm": mean_position,
        "position_rmse_cm": rms_position,
        "orientation_mean_deg": float(np.mean(orientation_errors_deg)),
        "orientation_median_deg": float(np.median(orientation_errors_deg)),
        "coverage_numerator": samples,
        "coverage_denominator": int(coverage_denominator),
        "coverage": coverage,
        "position_error_definition": POSITION_ERROR_DEFINITION,
        "orientation_error_definition": ORIENTATION_ERROR_DEFINITION,
    }


def sample_key_fingerprint(sample_keys: list[str]) -> str:
    payload = json.dumps(sorted(sample_keys), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def single_pose_errors(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    position, orientation = pose_error_values(
        np.asarray([prediction], dtype=np.float32),
        np.asarray([target], dtype=np.float32),
    )
    return float(position[0]), float(orientation[0])


def rms(values: np.ndarray) -> float:
    """Small public helper retained for report-side sanity checks."""

    values = np.asarray(values, dtype=np.float64)
    return float(math.sqrt(float(np.mean(np.square(values)))))
