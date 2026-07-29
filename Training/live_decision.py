#!/usr/bin/env python3
"""Robot-free validation and workflow logic for live intent inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


INTENTIONS = {"continue", "fetch", "handover", "uncertain"}
RECEIVING_HANDS = {"left", "right"}
DEFAULT_MINIMUM_GAZE_COVERAGE = 0.80
DEFAULT_MAXIMUM_GAZE_GAP_MS = 500.0
DEFAULT_MINIMUM_HANDOVER_HAND_COVERAGE = 0.50
DEFAULT_MAXIMUM_HAND_AGE_MS = 50.0
DEFAULT_MAXIMUM_VIO_AGE_MS = 10.0
DEFAULT_MAXIMUM_ANCHOR_AGE_MS = 500.0
DEFAULT_MAXIMUM_MARKER_AGE_MS = 250.0


def _is_valid(values: Mapping[str, float | int | None], name: str) -> bool:
    raw = values.get(name, 0.0)
    try:
        return bool(float(raw) > 0.5)
    except (TypeError, ValueError):
        return False


def _optional_nonnegative_float(
    values: Mapping[str, float | int | None],
    name: str,
) -> float | None:
    raw = values.get(name)
    try:
        numeric = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) and numeric >= 0.0 else None


@dataclass(frozen=True)
class QualityFrame:
    timestamp_ns: int
    gaze_valid: bool
    left_hand_valid: bool
    right_hand_valid: bool
    robot_frame_valid: bool
    hand_age_ms: float | None
    vio_age_ms: float | None
    anchor_age_ms: float | None
    minimum_visible_marker_age_ms: float | None
    visible_marker_count: int


class InputQualityGate:
    """Validate the complete causal model window before releasing an intent."""

    def __init__(
        self,
        *,
        window_size: int,
        max_timestamp_gap_ns: int,
        minimum_gaze_coverage: float = DEFAULT_MINIMUM_GAZE_COVERAGE,
        maximum_gaze_gap_ms: float = DEFAULT_MAXIMUM_GAZE_GAP_MS,
        minimum_handover_hand_coverage: float = (
            DEFAULT_MINIMUM_HANDOVER_HAND_COVERAGE
        ),
        maximum_hand_age_ms: float = DEFAULT_MAXIMUM_HAND_AGE_MS,
        maximum_vio_age_ms: float = DEFAULT_MAXIMUM_VIO_AGE_MS,
        maximum_anchor_age_ms: float = DEFAULT_MAXIMUM_ANCHOR_AGE_MS,
        maximum_marker_age_ms: float = DEFAULT_MAXIMUM_MARKER_AGE_MS,
    ) -> None:
        if window_size <= 0 or max_timestamp_gap_ns <= 0:
            raise ValueError("Window size and timestamp gap must be positive")
        for name, value in (
            ("minimum_gaze_coverage", minimum_gaze_coverage),
            (
                "minimum_handover_hand_coverage",
                minimum_handover_hand_coverage,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if maximum_gaze_gap_ms <= 0:
            raise ValueError("maximum_gaze_gap_ms must be positive")
        for name, value in (
            ("maximum_hand_age_ms", maximum_hand_age_ms),
            ("maximum_vio_age_ms", maximum_vio_age_ms),
            ("maximum_anchor_age_ms", maximum_anchor_age_ms),
            ("maximum_marker_age_ms", maximum_marker_age_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.window_size = int(window_size)
        self.max_timestamp_gap_ns = int(max_timestamp_gap_ns)
        self.minimum_gaze_coverage = float(minimum_gaze_coverage)
        self.maximum_gaze_gap_ns = int(maximum_gaze_gap_ms * 1e6)
        self.minimum_handover_hand_coverage = float(
            minimum_handover_hand_coverage
        )
        self.maximum_hand_age_ms = float(maximum_hand_age_ms)
        self.maximum_vio_age_ms = float(maximum_vio_age_ms)
        self.maximum_anchor_age_ms = float(maximum_anchor_age_ms)
        self.maximum_marker_age_ms = float(maximum_marker_age_ms)
        self.frames: deque[QualityFrame] = deque(
            maxlen=self.window_size
        )
        self.last_timestamp_ns: int | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.last_timestamp_ns = None

    def push_frame(
        self,
        timestamp_ns: int,
        values: Mapping[str, float | int | None],
    ) -> bool:
        """Append one frame and return whether a stream-gap reset occurred."""

        timestamp_ns = int(timestamp_ns)
        stream_reset = False
        if self.last_timestamp_ns is not None:
            if timestamp_ns <= self.last_timestamp_ns:
                raise ValueError("Quality-gate timestamps must increase")
            if timestamp_ns - self.last_timestamp_ns > self.max_timestamp_gap_ns:
                self.reset()
                stream_reset = True

        gaze_valid = _is_valid(values, "gaze_valid")
        left_valid = _is_valid(values, "hand_left_valid")
        right_valid = _is_valid(values, "hand_right_valid")
        robot_frame_valid = _is_valid(values, "robot_frame_valid")
        visible_marker_count_raw = _optional_nonnegative_float(
            values,
            "_quality_visible_marker_count",
        )
        self.frames.append(
            QualityFrame(
                timestamp_ns=timestamp_ns,
                gaze_valid=gaze_valid,
                left_hand_valid=left_valid,
                right_hand_valid=right_valid,
                robot_frame_valid=robot_frame_valid,
                hand_age_ms=_optional_nonnegative_float(
                    values,
                    "_quality_hand_age_ms",
                ),
                vio_age_ms=_optional_nonnegative_float(
                    values,
                    "_quality_vio_age_ms",
                ),
                anchor_age_ms=_optional_nonnegative_float(
                    values,
                    "_quality_anchor_age_ms",
                ),
                minimum_visible_marker_age_ms=_optional_nonnegative_float(
                    values,
                    "_quality_minimum_visible_marker_age_ms",
                ),
                visible_marker_count=(
                    int(visible_marker_count_raw)
                    if visible_marker_count_raw is not None
                    else 0
                ),
            )
        )
        self.last_timestamp_ns = timestamp_ns
        return stream_reset

    @staticmethod
    def _maximum_invalid_duration_ns(
        timestamps: list[int],
        validity: list[bool],
    ) -> int:
        maximum = 0
        start: int | None = None
        for timestamp, valid in zip(timestamps, validity):
            if not valid and start is None:
                start = timestamp
            elif valid and start is not None:
                maximum = max(maximum, timestamp - start)
                start = None
        if start is not None and timestamps:
            maximum = max(maximum, timestamps[-1] - start)
        return maximum

    def assess(
        self,
        intention: str,
        *,
        predicted_receiving_hand: str | None = None,
    ) -> dict:
        if intention not in INTENTIONS:
            raise ValueError(f"Unknown intention for quality gate: {intention}")

        frame_count = len(self.frames)
        ready = frame_count == self.window_size
        timestamps = [frame.timestamp_ns for frame in self.frames]
        gaze = [frame.gaze_valid for frame in self.frames]
        left = [frame.left_hand_valid for frame in self.frames]
        right = [frame.right_hand_valid for frame in self.frames]
        robot_frame = [frame.robot_frame_valid for frame in self.frames]
        vio_ages = [frame.vio_age_ms for frame in self.frames]
        anchor_ages = [frame.anchor_age_ms for frame in self.frames]

        denominator = max(frame_count, 1)
        gaze_coverage = sum(gaze) / denominator
        left_coverage = sum(left) / denominator
        right_coverage = sum(right) / denominator
        robot_frame_coverage = sum(robot_frame) / denominator
        max_gaze_gap_ns = self._maximum_invalid_duration_ns(
            timestamps,
            gaze,
        )

        reasons: list[str] = []
        if not ready:
            reasons.append("quality_window_not_ready")
        if gaze_coverage < self.minimum_gaze_coverage:
            reasons.append("gaze_coverage_too_low")
        if max_gaze_gap_ns > self.maximum_gaze_gap_ns:
            reasons.append("gaze_missing_too_long")
        if robot_frame_coverage < 1.0:
            reasons.append("robot_reference_missing")
        maximum_vio_age = (
            max(age for age in vio_ages if age is not None)
            if any(age is not None for age in vio_ages)
            else None
        )
        maximum_anchor_age = (
            max(age for age in anchor_ages if age is not None)
            if any(age is not None for age in anchor_ages)
            else None
        )
        if frame_count and any(age is None for age in vio_ages):
            reasons.append("vio_age_unavailable")
        elif (
            maximum_vio_age is not None
            and maximum_vio_age > self.maximum_vio_age_ms
        ):
            reasons.append("vio_data_too_old")
        if frame_count and any(age is None for age in anchor_ages):
            reasons.append("robot_anchor_age_unavailable")
        elif (
            maximum_anchor_age is not None
            and maximum_anchor_age > self.maximum_anchor_age_ms
        ):
            reasons.append("robot_anchor_too_old")

        predicted_hand_coverage: float | None = None
        predicted_hand_valid_currently: bool | None = None
        predicted_hand_maximum_age_ms: float | None = None
        if intention == "handover":
            if predicted_receiving_hand not in RECEIVING_HANDS:
                reasons.append("handover_predicted_hand_unavailable")
            else:
                predicted_validity = (
                    left if predicted_receiving_hand == "left" else right
                )
                predicted_hand_coverage = (
                    left_coverage
                    if predicted_receiving_hand == "left"
                    else right_coverage
                )
                predicted_hand_valid_currently = bool(
                    predicted_validity[-1]
                ) if frame_count else False
                if (
                    predicted_hand_coverage
                    < self.minimum_handover_hand_coverage
                ):
                    reasons.append(
                        "handover_predicted_hand_coverage_too_low"
                    )
                if not predicted_hand_valid_currently:
                    reasons.append(
                        "handover_predicted_hand_missing_currently"
                    )
                predicted_hand_ages = [
                    frame.hand_age_ms
                    for frame, valid in zip(
                        self.frames,
                        predicted_validity,
                    )
                    if valid
                ]
                if any(age is None for age in predicted_hand_ages):
                    reasons.append(
                        "handover_predicted_hand_age_unavailable"
                    )
                elif predicted_hand_ages:
                    predicted_hand_maximum_age_ms = max(
                        float(age)
                        for age in predicted_hand_ages
                        if age is not None
                    )
                    if (
                        predicted_hand_maximum_age_ms
                        > self.maximum_hand_age_ms
                    ):
                        reasons.append(
                            "handover_predicted_hand_too_old"
                        )

        current_visible_marker_count = (
            self.frames[-1].visible_marker_count if frame_count else 0
        )
        current_minimum_marker_age = (
            self.frames[-1].minimum_visible_marker_age_ms
            if frame_count
            else None
        )
        if intention == "fetch" and current_visible_marker_count > 0:
            if current_minimum_marker_age is None:
                reasons.append("fetch_marker_age_unavailable")
            elif current_minimum_marker_age > self.maximum_marker_age_ms:
                reasons.append("fetch_visible_markers_too_old")

        return {
            "ok": not reasons,
            "reasons": reasons,
            "frame_count": frame_count,
            "required_frames": self.window_size,
            "gaze_coverage": float(gaze_coverage),
            "maximum_gaze_gap_ms": float(max_gaze_gap_ns / 1e6),
            "left_hand_coverage": float(left_coverage),
            "right_hand_coverage": float(right_coverage),
            "robot_reference_coverage": float(robot_frame_coverage),
            "predicted_receiving_hand": predicted_receiving_hand,
            "predicted_hand_coverage": predicted_hand_coverage,
            "predicted_hand_valid_currently": (
                predicted_hand_valid_currently
            ),
            "predicted_hand_maximum_age_ms": (
                predicted_hand_maximum_age_ms
            ),
            "maximum_vio_age_ms": maximum_vio_age,
            "maximum_anchor_age_ms": maximum_anchor_age,
            "current_visible_marker_count": current_visible_marker_count,
            "current_minimum_visible_marker_age_ms": (
                current_minimum_marker_age
            ),
            "age_limits_ms": {
                "hand": self.maximum_hand_age_ms,
                "vio": self.maximum_vio_age_ms,
                "anchor": self.maximum_anchor_age_ms,
                "marker": self.maximum_marker_age_ms,
            },
        }


def evaluate_actionability(
    quality_gate: InputQualityGate,
    *,
    stable_intention: str,
    predicted_receiving_hand: str | None,
) -> dict:
    """Apply the shared post-model quality gate without changing model output."""

    quality = quality_gate.assess(
        stable_intention,
        predicted_receiving_hand=predicted_receiving_hand,
    )
    actionable_intention = (
        stable_intention if quality["ok"] else "insufficient_input"
    )
    return {
        "input_quality_ok": bool(quality["ok"]),
        "input_quality_reasons": list(quality["reasons"]),
        "actionable_intention": actionable_intention,
        "input_quality": quality,
    }


class GazeTargetSelector:
    """Select a visible ArUco object after a sustained unambiguous fixation."""

    def __init__(
        self,
        *,
        object_ids: tuple[int, ...],
        fixation_ms: float = 1000.0,
        maximum_angle_rad: float = 0.35,
        minimum_angle_margin_rad: float = 0.05,
        maximum_sample_gap_ms: float = 200.0,
        maximum_marker_age_ms: float = DEFAULT_MAXIMUM_MARKER_AGE_MS,
        minimum_samples: int = 10,
    ) -> None:
        if not object_ids:
            raise ValueError("At least one target object id is required")
        if min(
            fixation_ms,
            maximum_angle_rad,
            minimum_angle_margin_rad,
            maximum_sample_gap_ms,
            maximum_marker_age_ms,
        ) <= 0:
            raise ValueError("Target-selection thresholds must be positive")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")

        self.object_ids = tuple(int(value) for value in object_ids)
        self.fixation_ns = int(fixation_ms * 1e6)
        self.maximum_angle_rad = float(maximum_angle_rad)
        self.minimum_angle_margin_rad = float(minimum_angle_margin_rad)
        self.maximum_sample_gap_ns = int(maximum_sample_gap_ms * 1e6)
        self.maximum_marker_age_ms = float(maximum_marker_age_ms)
        self.minimum_samples = int(minimum_samples)
        self.reset()

    def reset(self) -> None:
        self.candidate_id: int | None = None
        self.candidate_started_ns: int | None = None
        self.candidate_last_ns: int | None = None
        self.candidate_samples = 0

    def _clear_candidate(self) -> None:
        self.candidate_id = None
        self.candidate_started_ns = None
        self.candidate_last_ns = None
        self.candidate_samples = 0

    def update(
        self,
        timestamp_ns: int,
        values: Mapping[str, float | int | None],
    ) -> dict:
        timestamp_ns = int(timestamp_ns)
        status = "tracking"
        candidates: list[tuple[float, int]] = []
        stale_marker_count = 0

        if not _is_valid(values, "gaze_valid"):
            status = "no_gaze"
        else:
            for marker_id in self.object_ids:
                if not _is_valid(values, f"aruco_{marker_id}_valid"):
                    continue
                marker_age = _optional_nonnegative_float(
                    values,
                    f"_quality_aruco_{marker_id}_age_ms",
                )
                if (
                    marker_age is None
                    or marker_age > self.maximum_marker_age_ms
                ):
                    stale_marker_count += 1
                    continue
                raw_angle = values.get(
                    f"aruco_{marker_id}_gaze_angle_rad",
                    np.nan,
                )
                try:
                    angle = float(raw_angle)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(angle):
                    candidates.append((angle, marker_id))

            candidates.sort()
            if not candidates:
                status = (
                    "stale_visible_objects"
                    if stale_marker_count
                    else "no_visible_object"
                )
            elif candidates[0][0] > self.maximum_angle_rad:
                status = "gaze_off_object"
            elif (
                len(candidates) > 1
                and candidates[1][0] - candidates[0][0]
                < self.minimum_angle_margin_rad
            ):
                status = "ambiguous"

        can_pause_candidate = (
            status in {"no_gaze", "no_visible_object"}
            and self.candidate_id is not None
            and self.candidate_last_ns is not None
            and timestamp_ns > self.candidate_last_ns
            and timestamp_ns - self.candidate_last_ns
            <= self.maximum_sample_gap_ns
        )
        if can_pause_candidate:
            return {
                "status": "temporarily_missing",
                "candidate_object_id": self.candidate_id,
                "selected_object_id": None,
                "fixation_ms": float(
                    (timestamp_ns - int(self.candidate_started_ns)) / 1e6
                ),
                "samples": self.candidate_samples,
                "best_angle_rad": None,
                "angle_margin_rad": None,
                "selection_score": 0.0,
                "visible_candidate_count": len(candidates),
                "stale_marker_count": stale_marker_count,
            }

        if status != "tracking":
            self._clear_candidate()
            return {
                "status": status,
                "candidate_object_id": None,
                "selected_object_id": None,
                "fixation_ms": 0.0,
                "samples": 0,
                "best_angle_rad": (
                    float(candidates[0][0]) if candidates else None
                ),
                "angle_margin_rad": (
                    float(candidates[1][0] - candidates[0][0])
                    if len(candidates) > 1
                    else None
                ),
                "selection_score": 0.0,
                "visible_candidate_count": len(candidates),
                "stale_marker_count": stale_marker_count,
            }

        best_angle, best_id = candidates[0]
        angle_margin = (
            candidates[1][0] - best_angle
            if len(candidates) > 1
            else self.maximum_angle_rad
        )
        continues_candidate = (
            self.candidate_id == best_id
            and self.candidate_last_ns is not None
            and timestamp_ns > self.candidate_last_ns
            and timestamp_ns - self.candidate_last_ns
            <= self.maximum_sample_gap_ns
        )
        if not continues_candidate:
            self.candidate_id = best_id
            self.candidate_started_ns = timestamp_ns
            self.candidate_samples = 1
        else:
            self.candidate_samples += 1
        self.candidate_last_ns = timestamp_ns

        fixation_ns = timestamp_ns - int(self.candidate_started_ns)
        selected = (
            fixation_ns >= self.fixation_ns
            and self.candidate_samples >= self.minimum_samples
        )
        duration_score = min(fixation_ns / self.fixation_ns, 1.0)
        angle_score = max(0.0, 1.0 - best_angle / self.maximum_angle_rad)
        margin_score = min(
            angle_margin / max(self.minimum_angle_margin_rad * 2.0, 1e-9),
            1.0,
        )
        score = duration_score * (0.7 * angle_score + 0.3 * margin_score)

        return {
            "status": "selected" if selected else "fixating",
            "candidate_object_id": best_id,
            "selected_object_id": best_id if selected else None,
            "fixation_ms": float(fixation_ns / 1e6),
            "samples": self.candidate_samples,
            "best_angle_rad": float(best_angle),
            "angle_margin_rad": float(angle_margin),
            "selection_score": float(score),
            "visible_candidate_count": len(candidates),
            "stale_marker_count": stale_marker_count,
        }


class PerceptionWorkflow:
    """Track a semantic intent sequence without triggering external actions."""

    def __init__(
        self,
        *,
        confirmation_predictions: int = 2,
        fetch_context_timeout_seconds: float = 30.0,
    ) -> None:
        if confirmation_predictions <= 0:
            raise ValueError("confirmation_predictions must be positive")
        if fetch_context_timeout_seconds <= 0:
            raise ValueError("fetch context timeout must be positive")
        self.confirmation_predictions = int(confirmation_predictions)
        self.fetch_context_timeout_ns = int(
            fetch_context_timeout_seconds * 1e9
        )
        self.reset()

    def reset(self) -> None:
        self.state = "observing"
        self.last_intention: str | None = None
        self.consecutive_predictions = 0
        self.fetch_context_timestamp_ns: int | None = None
        self.selected_object_id: int | None = None

    def update(
        self,
        timestamp_ns: int,
        decision_intention: str,
        target_selection: Mapping[str, object],
    ) -> dict:
        timestamp_ns = int(timestamp_ns)
        if decision_intention not in INTENTIONS | {"insufficient_input"}:
            raise ValueError(
                f"Unknown decision intention: {decision_intention}"
            )

        if (
            self.fetch_context_timestamp_ns is not None
            and timestamp_ns - self.fetch_context_timestamp_ns
            > self.fetch_context_timeout_ns
        ):
            self.fetch_context_timestamp_ns = None
            self.selected_object_id = None

        if decision_intention == self.last_intention:
            self.consecutive_predictions += 1
        else:
            self.last_intention = decision_intention
            self.consecutive_predictions = 1

        sequence_consistent: bool | None = None
        if decision_intention == "insufficient_input":
            self.state = "insufficient_input"
            self.fetch_context_timestamp_ns = None
            self.selected_object_id = None
        elif decision_intention == "uncertain":
            self.state = "uncertain"
        elif decision_intention == "continue":
            if self.consecutive_predictions >= self.confirmation_predictions:
                self.state = "observing"
                self.fetch_context_timestamp_ns = None
                self.selected_object_id = None
            else:
                self.state = "continue_candidate"
        elif decision_intention == "fetch":
            if self.consecutive_predictions < self.confirmation_predictions:
                self.state = "fetch_candidate"
            else:
                self.fetch_context_timestamp_ns = timestamp_ns
                target_id = target_selection.get("selected_object_id")
                if target_id is not None:
                    self.selected_object_id = int(target_id)
                if self.selected_object_id is not None:
                    self.state = "fetch_confirmed_target_selected"
                else:
                    self.state = "fetch_confirmed_no_target"
        elif decision_intention == "handover":
            has_fetch_context = self.fetch_context_timestamp_ns is not None
            sequence_consistent = has_fetch_context
            if self.consecutive_predictions < self.confirmation_predictions:
                self.state = "handover_candidate"
            elif has_fetch_context:
                self.state = "handover_confirmed"
            else:
                self.state = "handover_without_fetch_context"

        return {
            "state": self.state,
            "decision_intention": decision_intention,
            "consecutive_predictions": self.consecutive_predictions,
            "fetch_context_active": self.fetch_context_timestamp_ns is not None,
            "selected_object_id": self.selected_object_id,
            "sequence_consistent": sequence_consistent,
            "external_action_requested": False,
        }
