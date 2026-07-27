#!/usr/bin/env python3
"""Fast deterministic checks for the robot-free live decision layer."""

from __future__ import annotations

from live_decision import GazeTargetSelector, InputQualityGate, PerceptionWorkflow


def frame(
    *,
    gaze: bool = True,
    left: bool = True,
    right: bool = False,
    angle_6: float | None = 0.10,
    angle_14: float | None = 0.30,
) -> dict[str, float]:
    values = {
        "gaze_valid": float(gaze),
        "hand_left_valid": float(left),
        "hand_right_valid": float(right),
        "robot_frame_valid": 1.0,
        "aruco_6_valid": float(angle_6 is not None),
        "aruco_14_valid": float(angle_14 is not None),
    }
    if angle_6 is not None:
        values["aruco_6_gaze_angle_rad"] = angle_6
    if angle_14 is not None:
        values["aruco_14_gaze_angle_rad"] = angle_14
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

    gate.reset()
    for index in range(60):
        gate.push_frame(
            index * 33_333_333,
            frame(gaze=True, left=index < 10, right=False),
        )
    result = gate.assess("handover")
    assert "handover_hand_coverage_too_low" in result["reasons"]
    assert "handover_hand_missing_currently" in result["reasons"]


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
    target = test_target_selector()
    test_workflow(target)
    print("live decision smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
