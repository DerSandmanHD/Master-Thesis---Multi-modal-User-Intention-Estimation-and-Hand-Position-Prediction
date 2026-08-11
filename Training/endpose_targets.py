#!/usr/bin/env python3
"""Robust, sequence-level terminal receiving-hand pose targets.

The target is intentionally independent of model predictions.  For every
sequence containing a handover segment (the rows after the THIRD annotation),
we search backwards for the latest stable window of receiving-hand tracking.
Quality checks operate on unique physical hand captures, not on repeated rows
created when the hand stream is nearest-merged onto the master timeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


TERMINAL_ENDPOSE_MODE = "terminal_endpose"
TARGET_DEFINITION_VERSION = "terminal_endpose_unique_hand_capture_v2"
CAPTURE_TIMESTAMP_BASIS = "hand_timestamp_ns"
DEFAULT_TERMINAL_ENDPOSE_CONFIG = {
    "mode": TERMINAL_ENDPOSE_MODE,
    "target_definition_version": TARGET_DEFINITION_VERSION,
    "capture_timestamp_basis": CAPTURE_TIMESTAMP_BASIS,
    "aggregation_window_seconds": 0.5,
    "minimum_valid_samples": 8,
    "minimum_valid_ratio": 0.70,
    "minimum_valid_span_seconds": 0.35,
    "maximum_position_p90_deviation_m": 0.05,
    "maximum_orientation_p90_deviation_deg": 25.0,
    "maximum_stable_window_lag_seconds": 1.0,
}


@dataclass(frozen=True)
class TerminalEndposeEstimate:
    target_definition_version: str
    capture_timestamp_basis: str
    status: str
    eligible: bool
    reasons: tuple[str, ...]
    receiving_hand_id: int | None
    receiving_hand: str | None
    handover_start_timestamp_ns: int | None
    handover_end_timestamp_ns: int | None
    handover_event_start_timestamp_ns: int | None
    handover_event_end_timestamp_ns: int | None
    handover_duration_seconds: float | None
    aggregation_start_timestamp_ns: int | None
    aggregation_end_timestamp_ns: int | None
    aggregation_event_start_timestamp_ns: int | None
    aggregation_event_end_timestamp_ns: int | None
    aggregation_capture_start_timestamp_ns: int | None
    aggregation_capture_end_timestamp_ns: int | None
    target_capture_timestamp_ns: int | None
    target_capture_aligned_event_timestamp_ns: int | None
    target_capture_lag_seconds: float | None
    stable_window_lag_seconds: float | None
    candidate_rows: int
    candidate_unique_captures: int
    valid_samples: int
    valid_ratio: float | None
    valid_span_seconds: float | None
    position_p90_deviation_m: float | None
    orientation_p90_deviation_deg: float | None
    pose: tuple[float, ...] | None

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_terminal_endpose_config(config: dict | None) -> dict:
    values = dict(DEFAULT_TERMINAL_ENDPOSE_CONFIG)
    if config:
        values.update(config)
    if values.get("mode") != TERMINAL_ENDPOSE_MODE:
        raise ValueError(
            f"Expected pose_target.mode={TERMINAL_ENDPOSE_MODE!r}, "
            f"got {values.get('mode')!r}"
        )
    if values.get("target_definition_version") != TARGET_DEFINITION_VERSION:
        raise ValueError(
            "Unsupported terminal target_definition_version: "
            f"{values.get('target_definition_version')!r}"
        )
    if values.get("capture_timestamp_basis") != CAPTURE_TIMESTAMP_BASIS:
        raise ValueError(
            "Terminal end-pose targets require capture_timestamp_basis="
            f"{CAPTURE_TIMESTAMP_BASIS!r}"
        )
    positive = (
        "aggregation_window_seconds",
        "minimum_valid_samples",
        "minimum_valid_span_seconds",
        "maximum_position_p90_deviation_m",
        "maximum_orientation_p90_deviation_deg",
        "maximum_stable_window_lag_seconds",
    )
    for key in positive:
        if float(values[key]) <= 0:
            raise ValueError(f"pose_target.{key} must be greater than zero")
    ratio = float(values["minimum_valid_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("pose_target.minimum_valid_ratio must be in (0, 1]")
    values["minimum_valid_samples"] = int(values["minimum_valid_samples"])
    if values["minimum_valid_samples"] <= 0:
        raise ValueError("pose_target.minimum_valid_samples must be positive")
    for key in positive:
        if key != "minimum_valid_samples":
            values[key] = float(values[key])
    values["minimum_valid_ratio"] = ratio
    return values


def quaternion_average(quaternions: np.ndarray) -> np.ndarray:
    """Return a sign-invariant unit-quaternion average in xyzw convention."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("Expected at least one quaternion with shape [N, 4]")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms <= 1e-8):
        raise ValueError("Cannot average invalid quaternions")
    values = values / norms[:, None]
    accumulator = np.einsum("ni,nj->ij", values, values)
    _, eigenvectors = np.linalg.eigh(accumulator)
    mean = eigenvectors[:, -1]
    if mean[3] < 0.0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def quaternion_angular_errors_deg(
    quaternions: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    reference = reference / np.linalg.norm(reference)
    cosine = np.clip(np.abs(values @ reference), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine))


def robust_pose(poses: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values):
        raise ValueError("Expected at least one valid pose with shape [N, 7]")
    if not np.isfinite(values).all():
        raise ValueError("Pose aggregation received non-finite values")
    position = np.median(values[:, :3], axis=0)
    quaternion = quaternion_average(values[:, 3:7])
    position_deviation = np.linalg.norm(values[:, :3] - position, axis=1)
    orientation_deviation = quaternion_angular_errors_deg(
        values[:, 3:7], quaternion
    )
    return (
        np.concatenate((position, quaternion)).astype(np.float32),
        float(np.percentile(position_deviation, 90)),
        float(np.percentile(orientation_deviation, 90)),
    )


def _empty_estimate(status: str, reasons: tuple[str, ...]) -> TerminalEndposeEstimate:
    return TerminalEndposeEstimate(
        target_definition_version=TARGET_DEFINITION_VERSION,
        capture_timestamp_basis=CAPTURE_TIMESTAMP_BASIS,
        status=status,
        eligible=False,
        reasons=reasons,
        receiving_hand_id=None,
        receiving_hand=None,
        handover_start_timestamp_ns=None,
        handover_end_timestamp_ns=None,
        handover_event_start_timestamp_ns=None,
        handover_event_end_timestamp_ns=None,
        handover_duration_seconds=None,
        aggregation_start_timestamp_ns=None,
        aggregation_end_timestamp_ns=None,
        aggregation_event_start_timestamp_ns=None,
        aggregation_event_end_timestamp_ns=None,
        aggregation_capture_start_timestamp_ns=None,
        aggregation_capture_end_timestamp_ns=None,
        target_capture_timestamp_ns=None,
        target_capture_aligned_event_timestamp_ns=None,
        target_capture_lag_seconds=None,
        stable_window_lag_seconds=None,
        candidate_rows=0,
        candidate_unique_captures=0,
        valid_samples=0,
        valid_ratio=None,
        valid_span_seconds=None,
        position_p90_deviation_m=None,
        orientation_p90_deviation_deg=None,
        pose=None,
    )


def _representative_capture_rows(
    candidate_rows: np.ndarray,
    event_timestamps_ns: np.ndarray,
    capture_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Return one best-aligned master row per physical hand capture.

    A physical hand sample can occur on several master rows after a nearest
    merge.  The row whose event timestamp is closest to the source capture is
    the least distorted representation of that capture; ties are resolved by
    event time and then row index for deterministic artifacts.
    """

    rows = np.asarray(candidate_rows, dtype=np.int64)
    if not len(rows):
        return rows
    representatives = []
    for capture_timestamp in np.unique(capture_timestamps_ns[rows]):
        matching = rows[capture_timestamps_ns[rows] == capture_timestamp]
        representative = min(
            matching.tolist(),
            key=lambda row: (
                abs(int(event_timestamps_ns[row]) - int(capture_timestamp)),
                int(event_timestamps_ns[row]),
                int(row),
            ),
        )
        representatives.append(int(representative))
    representatives.sort(
        key=lambda row: (int(capture_timestamps_ns[row]), int(row))
    )
    return np.asarray(representatives, dtype=np.int64)


def future_terminal_target_mask(
    endpoint_event_timestamps_ns: np.ndarray,
    target_capture_timestamp_ns: int,
) -> np.ndarray:
    """Mark endpoints for which the physical target capture is still future."""

    timestamps = np.asarray(endpoint_event_timestamps_ns, dtype=np.int64)
    target_timestamp = int(target_capture_timestamp_ns)
    if target_timestamp < 0:
        raise ValueError("target_capture_timestamp_ns must be non-negative")
    return timestamps < target_timestamp


def estimate_terminal_endpose(
    *,
    timestamps_ns: np.ndarray,
    intention_ids: np.ndarray,
    receiving_hand_ids: np.ndarray,
    hand_poses: np.ndarray,
    hand_pose_valid: np.ndarray,
    hand_timestamps_ns: np.ndarray,
    handover_intent_id: int,
    receiving_hand_names: tuple[str, str] | list[str],
    config: dict | None,
) -> TerminalEndposeEstimate:
    """Estimate the latest stable receiving-hand pose after THIRD.

    Candidate 0.5-second windows are considered from the handover end backwards.
    The first window satisfying all predeclared quality limits is used.  Searching
    is bounded by ``maximum_stable_window_lag_seconds`` so an early stationary
    pose cannot silently become a terminal target.
    """

    cfg = normalized_terminal_endpose_config(config)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    hand_timestamps = np.asarray(hand_timestamps_ns, dtype=np.int64)
    intentions = np.asarray(intention_ids, dtype=np.int64)
    hand_ids = np.asarray(receiving_hand_ids, dtype=np.int64)
    poses = np.asarray(hand_poses, dtype=np.float32)
    pose_valid = np.asarray(hand_pose_valid, dtype=bool)
    count = len(timestamps)
    if not (
        intentions.shape == (count,)
        and hand_ids.shape == (count,)
        and hand_timestamps.shape == (count,)
        and poses.shape == (count, 2, 7)
        and pose_valid.shape == (count, 2)
    ):
        raise ValueError("Terminal end-pose arrays have inconsistent shapes")
    if count and np.any(np.diff(timestamps) < 0):
        raise ValueError("Timestamps must be sorted")
    available_capture_timestamps = hand_timestamps[hand_timestamps >= 0]
    if len(available_capture_timestamps) and np.any(
        np.diff(available_capture_timestamps) < 0
    ):
        raise ValueError("Physical hand capture timestamps must be sorted")

    handover_rows = np.flatnonzero(intentions == int(handover_intent_id))
    if not len(handover_rows):
        return _empty_estimate("not_applicable", ("no_handover_rows",))

    known_hands = hand_ids[handover_rows]
    known_hands = known_hands[np.isin(known_hands, (0, 1))]
    if not len(known_hands):
        base = _empty_estimate("rejected", ("unknown_receiving_hand",))
        return TerminalEndposeEstimate(
            **{
                **base.to_dict(),
                "handover_start_timestamp_ns": int(timestamps[handover_rows[0]]),
                "handover_end_timestamp_ns": int(timestamps[handover_rows[-1]]),
                "handover_event_start_timestamp_ns": int(
                    timestamps[handover_rows[0]]
                ),
                "handover_event_end_timestamp_ns": int(
                    timestamps[handover_rows[-1]]
                ),
                "handover_duration_seconds": float(
                    (timestamps[handover_rows[-1]] - timestamps[handover_rows[0]])
                    / 1e9
                ),
            }
        )
    unique_hands, hand_counts = np.unique(known_hands, return_counts=True)
    if len(unique_hands) != 1:
        base = _empty_estimate("rejected", ("inconsistent_receiving_hand",))
        return TerminalEndposeEstimate(
            **{
                **base.to_dict(),
                "handover_start_timestamp_ns": int(timestamps[handover_rows[0]]),
                "handover_end_timestamp_ns": int(timestamps[handover_rows[-1]]),
                "handover_event_start_timestamp_ns": int(
                    timestamps[handover_rows[0]]
                ),
                "handover_event_end_timestamp_ns": int(
                    timestamps[handover_rows[-1]]
                ),
                "handover_duration_seconds": float(
                    (timestamps[handover_rows[-1]] - timestamps[handover_rows[0]])
                    / 1e9
                ),
            }
        )
    hand_id = int(unique_hands[np.argmax(hand_counts)])
    hand_name = str(receiving_hand_names[hand_id])
    handover_start = int(timestamps[handover_rows[0]])
    handover_end = int(timestamps[handover_rows[-1]])
    window_ns = int(round(cfg["aggregation_window_seconds"] * 1e9))
    maximum_lag_ns = int(round(cfg["maximum_stable_window_lag_seconds"] * 1e9))

    latest_diagnostics: dict | None = None
    for endpoint_row in handover_rows[::-1]:
        endpoint_timestamp = int(timestamps[endpoint_row])
        lag_ns = handover_end - endpoint_timestamp
        if lag_ns > maximum_lag_ns:
            break
        start_timestamp = endpoint_timestamp - window_ns
        candidate_mask = (
            (intentions == handover_intent_id)
            & (hand_timestamps >= 0)
            & (hand_timestamps >= start_timestamp)
            & (hand_timestamps <= endpoint_timestamp)
        )
        candidate_rows = np.flatnonzero(candidate_mask)
        unique_capture_rows = _representative_capture_rows(
            candidate_rows, timestamps, hand_timestamps
        )
        valid_rows = unique_capture_rows[
            pose_valid[unique_capture_rows, hand_id]
            & (hand_ids[unique_capture_rows] == hand_id)
        ]
        valid_ratio = len(valid_rows) / max(1, len(unique_capture_rows))
        valid_span = (
            float(
                (
                    hand_timestamps[valid_rows[-1]]
                    - hand_timestamps[valid_rows[0]]
                )
                / 1e9
            )
            if len(valid_rows) >= 2
            else 0.0
        )
        capture_start = (
            int(hand_timestamps[valid_rows[0]]) if len(valid_rows) else None
        )
        capture_end = (
            int(hand_timestamps[valid_rows[-1]]) if len(valid_rows) else None
        )
        target_aligned_event = (
            int(timestamps[valid_rows[-1]]) if len(valid_rows) else None
        )
        diagnostics = {
            "aggregation_start_timestamp_ns": start_timestamp,
            "aggregation_end_timestamp_ns": endpoint_timestamp,
            "aggregation_event_start_timestamp_ns": start_timestamp,
            "aggregation_event_end_timestamp_ns": endpoint_timestamp,
            "aggregation_capture_start_timestamp_ns": capture_start,
            "aggregation_capture_end_timestamp_ns": capture_end,
            "target_capture_timestamp_ns": capture_end,
            "target_capture_aligned_event_timestamp_ns": target_aligned_event,
            "target_capture_lag_seconds": (
                float((handover_end - capture_end) / 1e9)
                if capture_end is not None
                else None
            ),
            "stable_window_lag_seconds": float(lag_ns / 1e9),
            "candidate_rows": int(len(candidate_rows)),
            "candidate_unique_captures": int(len(unique_capture_rows)),
            "valid_samples": int(len(valid_rows)),
            "valid_ratio": float(valid_ratio),
            "valid_span_seconds": valid_span,
            "position_p90_deviation_m": None,
            "orientation_p90_deviation_deg": None,
            "pose": None,
        }
        reasons = []
        if len(valid_rows) < cfg["minimum_valid_samples"]:
            reasons.append("insufficient_valid_samples")
        if valid_ratio < cfg["minimum_valid_ratio"]:
            reasons.append("low_valid_ratio")
        if valid_span < cfg["minimum_valid_span_seconds"]:
            reasons.append("insufficient_valid_span")
        if not reasons:
            pose, position_p90, orientation_p90 = robust_pose(
                poses[valid_rows, hand_id]
            )
            diagnostics.update(
                {
                    "position_p90_deviation_m": position_p90,
                    "orientation_p90_deviation_deg": orientation_p90,
                    "pose": tuple(float(value) for value in pose),
                }
            )
            if position_p90 > cfg["maximum_position_p90_deviation_m"]:
                reasons.append("unstable_position")
            if orientation_p90 > cfg["maximum_orientation_p90_deviation_deg"]:
                reasons.append("unstable_orientation")
        if latest_diagnostics is None:
            latest_diagnostics = {**diagnostics, "reasons": tuple(reasons)}
        if not reasons:
            return TerminalEndposeEstimate(
                target_definition_version=TARGET_DEFINITION_VERSION,
                capture_timestamp_basis=CAPTURE_TIMESTAMP_BASIS,
                status="accepted",
                eligible=True,
                reasons=(),
                receiving_hand_id=hand_id,
                receiving_hand=hand_name,
                handover_start_timestamp_ns=handover_start,
                handover_end_timestamp_ns=handover_end,
                handover_event_start_timestamp_ns=handover_start,
                handover_event_end_timestamp_ns=handover_end,
                handover_duration_seconds=float((handover_end - handover_start) / 1e9),
                **diagnostics,
            )

    diagnostics = latest_diagnostics or {
        "aggregation_start_timestamp_ns": None,
        "aggregation_end_timestamp_ns": None,
        "aggregation_event_start_timestamp_ns": None,
        "aggregation_event_end_timestamp_ns": None,
        "aggregation_capture_start_timestamp_ns": None,
        "aggregation_capture_end_timestamp_ns": None,
        "target_capture_timestamp_ns": None,
        "target_capture_aligned_event_timestamp_ns": None,
        "target_capture_lag_seconds": None,
        "stable_window_lag_seconds": None,
        "candidate_rows": 0,
        "candidate_unique_captures": 0,
        "valid_samples": 0,
        "valid_ratio": None,
        "valid_span_seconds": None,
        "position_p90_deviation_m": None,
        "orientation_p90_deviation_deg": None,
        "pose": None,
        "reasons": ("no_candidate_window",),
    }
    reasons = tuple(diagnostics.pop("reasons"))
    return TerminalEndposeEstimate(
        target_definition_version=TARGET_DEFINITION_VERSION,
        capture_timestamp_basis=CAPTURE_TIMESTAMP_BASIS,
        status="rejected",
        eligible=False,
        reasons=reasons,
        receiving_hand_id=hand_id,
        receiving_hand=hand_name,
        handover_start_timestamp_ns=handover_start,
        handover_end_timestamp_ns=handover_end,
        handover_event_start_timestamp_ns=handover_start,
        handover_event_end_timestamp_ns=handover_end,
        handover_duration_seconds=float((handover_end - handover_start) / 1e9),
        **diagnostics,
    )
