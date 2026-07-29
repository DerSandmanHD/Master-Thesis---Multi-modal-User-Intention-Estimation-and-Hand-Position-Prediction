#!/usr/bin/env python3
"""Fast deterministic checks for the robot-free live decision layer."""

from __future__ import annotations

from live_decision import (
    GazeTargetSelector,
    InputQualityGate,
    PerceptionWorkflow,
    evaluate_actionability,
)


def frame(
    *,
    gaze: bool = True,
    left: bool = True,
    right: bool = False,
    angle_6: float | None = 0.10,
    angle_14: float | None = 0.30,
    hand_age_ms: float | None = 10.0,
    vio_age_ms: float | None = 2.0,
    anchor_age_ms: float | None = 100.0,
    marker_age_ms: float | None = 10.0,
) -> dict[str, float | None]:
    visible_count = sum(
        angle is not None for angle in (angle_6, angle_14)
    )
    values = {
        "gaze_valid": float(gaze),
        "hand_left_valid": float(left),
        "hand_right_valid": float(right),
        "robot_frame_valid": 1.0,
        "aruco_6_valid": float(angle_6 is not None),
        "aruco_14_valid": float(angle_14 is not None),
        "_quality_hand_age_ms": hand_age_ms,
        "_quality_vio_age_ms": vio_age_ms,
        "_quality_anchor_age_ms": anchor_age_ms,
        "_quality_visible_marker_count": float(visible_count),
        "_quality_minimum_visible_marker_age_ms": (
            marker_age_ms if visible_count else None
        ),
        "_quality_maximum_visible_marker_age_ms": (
            marker_age_ms if visible_count else None
        ),
    }
    if angle_6 is not None:
        values["aruco_6_gaze_angle_rad"] = angle_6
        values["_quality_aruco_6_age_ms"] = marker_age_ms
    if angle_14 is not None:
        values["aruco_14_gaze_angle_rad"] = angle_14
        values["_quality_aruco_14_age_ms"] = marker_age_ms
    return values


def test_quality_gate() -> None:
    gate = InputQualityGate(
        window_size=60,
        max_timestamp_gap_ns=int(1e9),
        minimum_gaze_coverage=0.80,
        maximum_gaze_gap_ms=500.0,
        minimum_handover_hand_coverage=0.50,
    )
    for index in range(60):
        gate.push_frame(index * 33_333_333, frame(gaze=index >= 6))
    result = gate.assess("fetch")
    assert result["ok"], result

    for index in range(60, 80):
        gate.push_frame(index * 33_333_333, frame(gaze=False))
    result = gate.assess("continue")
    assert not result["ok"]
    assert "gaze_missing_too_long" in result["reasons"]


def filled_gate(
    *,
    gaze: bool = True,
    left: bool = True,
    right: bool = False,
    hand_age_ms: float | None = 10.0,
    vio_age_ms: float | None = 2.0,
    anchor_age_ms: float | None = 100.0,
    marker_age_ms: float | None = 10.0,
) -> InputQualityGate:
    gate = InputQualityGate(
        window_size=60,
        max_timestamp_gap_ns=int(1e9),
    )
    for index in range(60):
        gate.push_frame(
            index * 33_333_333,
            frame(
                gaze=gaze,
                left=left,
                right=right,
                hand_age_ms=hand_age_ms,
                vio_age_ms=vio_age_ms,
                anchor_age_ms=anchor_age_ms,
                marker_age_ms=marker_age_ms,
            ),
        )
    return gate


def test_actionability_and_gaze_failure() -> None:
    released = evaluate_actionability(
        filled_gate(),
        stable_intention="fetch",
        predicted_receiving_hand=None,
    )
    assert released["input_quality_ok"], released
    assert released["input_quality_reasons"] == []
    assert released["actionable_intention"] == "fetch"

    prediction = {
        "raw_intention": "handover",
        "stable_intention": "fetch",
    }
    prediction.update(
        evaluate_actionability(
            filled_gate(gaze=False),
            stable_intention=prediction["stable_intention"],
            predicted_receiving_hand=None,
        )
    )
    assert prediction["raw_intention"] == "handover"
    assert prediction["stable_intention"] == "fetch"
    assert not prediction["input_quality_ok"]
    assert "gaze_coverage_too_low" in prediction["input_quality_reasons"]
    assert prediction["actionable_intention"] == "insufficient_input"


def assert_predicted_hand_blocked(result: dict) -> None:
    assert not result["input_quality_ok"], result
    assert result["actionable_intention"] == "insufficient_input"
    assert (
        "handover_predicted_hand_coverage_too_low"
        in result["input_quality_reasons"]
    )
    assert (
        "handover_predicted_hand_missing_currently"
        in result["input_quality_reasons"]
    )


def test_predicted_handover_hand_quality() -> None:
    left_ok = evaluate_actionability(
        filled_gate(left=True, right=False),
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    assert left_ok["input_quality_ok"], left_ok
    assert left_ok["actionable_intention"] == "handover"
    assert left_ok["input_quality"]["predicted_hand_coverage"] == 1.0

    left_blocked = evaluate_actionability(
        filled_gate(left=False, right=True),
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    assert_predicted_hand_blocked(left_blocked)

    right_ok = evaluate_actionability(
        filled_gate(left=False, right=True),
        stable_intention="handover",
        predicted_receiving_hand="right",
    )
    assert right_ok["input_quality_ok"], right_ok
    assert right_ok["actionable_intention"] == "handover"
    assert right_ok["input_quality"]["predicted_hand_coverage"] == 1.0

    right_blocked = evaluate_actionability(
        filled_gate(left=True, right=False),
        stable_intention="handover",
        predicted_receiving_hand="right",
    )
    assert_predicted_hand_blocked(right_blocked)

    current_left_missing_gate = InputQualityGate(
        window_size=60,
        max_timestamp_gap_ns=int(1e9),
    )
    for index in range(60):
        current_left_missing_gate.push_frame(
            index * 33_333_333,
            frame(left=index < 59, right=True),
        )
    current_left_missing = evaluate_actionability(
        current_left_missing_gate,
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    assert not current_left_missing["input_quality_ok"]
    assert (
        "handover_predicted_hand_coverage_too_low"
        not in current_left_missing["input_quality_reasons"]
    )
    assert (
        "handover_predicted_hand_missing_currently"
        in current_left_missing["input_quality_reasons"]
    )

    missing_prediction = evaluate_actionability(
        filled_gate(left=True, right=True),
        stable_intention="handover",
        predicted_receiving_hand=None,
    )
    assert not missing_prediction["input_quality_ok"]
    assert missing_prediction["actionable_intention"] == "insufficient_input"
    assert missing_prediction["input_quality_reasons"] == [
        "handover_predicted_hand_unavailable"
    ]


def test_non_handover_does_not_require_predicted_hand() -> None:
    for intention in ("continue", "fetch"):
        result = evaluate_actionability(
            filled_gate(left=False, right=False),
            stable_intention=intention,
            predicted_receiving_hand=None,
        )
        assert result["input_quality_ok"], result
        assert result["actionable_intention"] == intention


def test_sensor_and_reference_freshness() -> None:
    stale_vio = evaluate_actionability(
        filled_gate(vio_age_ms=11.0),
        stable_intention="continue",
        predicted_receiving_hand=None,
    )
    assert "vio_data_too_old" in stale_vio["input_quality_reasons"]

    stale_anchor = evaluate_actionability(
        filled_gate(anchor_age_ms=501.0),
        stable_intention="continue",
        predicted_receiving_hand=None,
    )
    assert "robot_anchor_too_old" in stale_anchor["input_quality_reasons"]

    missing_ages = evaluate_actionability(
        filled_gate(vio_age_ms=None, anchor_age_ms=None),
        stable_intention="continue",
        predicted_receiving_hand=None,
    )
    assert "vio_age_unavailable" in missing_ages["input_quality_reasons"]
    assert (
        "robot_anchor_age_unavailable"
        in missing_ages["input_quality_reasons"]
    )

    stale_hand = evaluate_actionability(
        filled_gate(left=True, hand_age_ms=51.0),
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    assert (
        "handover_predicted_hand_too_old"
        in stale_hand["input_quality_reasons"]
    )

    stale_marker = evaluate_actionability(
        filled_gate(marker_age_ms=251.0),
        stable_intention="fetch",
        predicted_receiving_hand=None,
    )
    assert (
        "fetch_visible_markers_too_old"
        in stale_marker["input_quality_reasons"]
    )
    continue_with_stale_marker = evaluate_actionability(
        filled_gate(marker_age_ms=251.0),
        stable_intention="continue",
        predicted_receiving_hand=None,
    )
    assert continue_with_stale_marker["input_quality_ok"]


def test_target_selector() -> dict:
    selector = GazeTargetSelector(
        object_ids=(6, 14),
        fixation_ms=500.0,
        minimum_samples=10,
    )
    result = {}
    for index in range(20):
        result = selector.update(
            index * 33_333_333,
            frame(gaze=index != 8),
        )
    assert result["status"] == "selected", result
    assert result["selected_object_id"] == 6

    ambiguous = selector.update(
        20 * 33_333_333,
        frame(angle_6=0.10, angle_14=0.12),
    )
    assert ambiguous["status"] == "ambiguous", ambiguous
    assert ambiguous["selected_object_id"] is None
    stale = selector.update(
        21 * 33_333_333,
        frame(marker_age_ms=251.0),
    )
    assert stale["status"] == "stale_visible_objects", stale
    return result


def test_workflow(target: dict) -> None:
    workflow = PerceptionWorkflow(
        confirmation_predictions=2,
        fetch_context_timeout_seconds=30.0,
    )
    empty_target = {"selected_object_id": None}
    assert workflow.update(0, "continue", empty_target)["state"] == (
        "continue_candidate"
    )
    assert workflow.update(1, "continue", empty_target)["state"] == "observing"
    assert workflow.update(2, "fetch", target)["state"] == "fetch_candidate"
    fetch = workflow.update(3, "fetch", target)
    assert fetch["state"] == "fetch_confirmed_target_selected"
    assert fetch["selected_object_id"] == 6
    assert workflow.update(4, "handover", empty_target)["state"] == (
        "handover_candidate"
    )
    handover = workflow.update(5, "handover", empty_target)
    assert handover["state"] == "handover_confirmed"
    assert handover["sequence_consistent"] is True
    blocked = workflow.update(6, "insufficient_input", empty_target)
    assert blocked["state"] == "insufficient_input"
    assert blocked["external_action_requested"] is False


def main() -> int:
    test_quality_gate()
    test_actionability_and_gaze_failure()
    test_predicted_handover_hand_quality()
    test_non_handover_does_not_require_predicted_hand()
    test_sensor_and_reference_freshness()
    target = test_target_selector()
    test_workflow(target)
    print("live decision smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
