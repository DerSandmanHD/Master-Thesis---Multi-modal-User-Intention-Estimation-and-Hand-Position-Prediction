#!/usr/bin/env python3
"""Fast deterministic checks for shared replay/live intention decisions."""

from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from aria_live_inference import (
    LiveFeatureAssembler,
    TimedSample,
    latest_before_item,
    prediction_printer,
)
import torch

import online_inference
from data import INTENTION_NAMES, Normalizer
from live_decision import InputQualityGate, evaluate_actionability
from online_inference import OnlineInferenceEngine
from replay_stream_inference import (
    TemporalDecisionFilter,
    build_replay_summary,
    hierarchical_intention_id,
    intention_id_from_probabilities,
    joint_intention_probabilities,
    push_replay_quality_frames,
    replay_quality_diagnostics,
)


def test_raw_prediction_printer_only_reports_label_changes() -> None:
    emit, close = prediction_printer("raw", None)
    output = StringIO()
    base = {
        "stable_intention": "continue",
        "actionable_intention": "continue",
        "perception_workflow": {},
    }
    with redirect_stdout(output):
        emit({**base, "raw_intention": "continue"})
        emit({**base, "raw_intention": "continue"})
        emit({**base, "raw_intention": "fetch"})
    close()
    assert output.getvalue().splitlines() == [
        "raw_intention=continue",
        "raw_intention=fetch",
    ]


def test_only_confident_candidates_count() -> None:
    decision_filter = TemporalDecisionFilter(
        smoothing_window=1,
        minimum_confidence=0.65,
        minimum_stable_predictions=2,
    )
    low_confidence = np.asarray([0.34, 0.33, 0.33])
    high_confidence = np.asarray([0.80, 0.10, 0.10])

    for _ in range(2):
        label, _, _ = decision_filter.update(low_confidence)
        assert label == "uncertain"
        assert decision_filter.candidate is None
        assert decision_filter.candidate_count == 0

    label, _, _ = decision_filter.update(high_confidence)
    assert label == "uncertain"
    assert decision_filter.candidate_count == 1

    label, _, _ = decision_filter.update(high_confidence)
    assert label == "continue"
    assert decision_filter.candidate_count == 2


def test_unconfident_prediction_resets_confident_run() -> None:
    decision_filter = TemporalDecisionFilter(
        smoothing_window=1,
        minimum_confidence=0.65,
        minimum_stable_predictions=2,
    )
    fetch = np.asarray([0.10, 0.80, 0.10])
    uncertain_fetch = np.asarray([0.20, 0.60, 0.20])

    assert decision_filter.update(fetch)[0] == "uncertain"
    assert decision_filter.candidate_count == 1
    assert decision_filter.update(uncertain_fetch)[0] == "uncertain"
    assert decision_filter.candidate is None
    assert decision_filter.candidate_count == 0
    assert decision_filter.update(fetch)[0] == "uncertain"
    assert decision_filter.candidate_count == 1
    assert decision_filter.update(fetch)[0] == "fetch"


def divergent_outputs() -> dict[str, torch.Tensor]:
    return {
        "assistance_logits": torch.log(torch.tensor([[0.49, 0.51]])),
        "assistance_type_logits": torch.log(torch.tensor([[0.51, 0.49]])),
    }


def test_joint_probabilities_define_raw_and_stable_class() -> None:
    outputs = divergent_outputs()
    probabilities = joint_intention_probabilities(outputs)
    joint_id = intention_id_from_probabilities(probabilities)

    assert hierarchical_intention_id(outputs) == 1
    assert joint_id == 0
    assert INTENTION_NAMES[joint_id] == "continue"

    decision_filter = TemporalDecisionFilter(
        smoothing_window=1,
        minimum_confidence=0.0,
        minimum_stable_predictions=1,
    )
    stable_label, _, smoothed = decision_filter.update(probabilities)
    assert stable_label == INTENTION_NAMES[joint_id]
    assert intention_id_from_probabilities(smoothed) == joint_id


def test_online_engine_uses_shared_joint_decision() -> None:
    engine = OnlineInferenceEngine.__new__(OnlineInferenceEngine)
    engine.artifacts = SimpleNamespace(
        window_size=1,
        step_size=1,
        minimum_observed_fraction=0.0,
        max_timestamp_gap_ns=int(1e9),
        feature_columns=["gaze_valid"],
        normalizer=Normalizer(
            mean=np.asarray([0.0], dtype=np.float32),
            std=np.asarray([1.0], dtype=np.float32),
            feature_names=["gaze_valid"],
        ),
        model=object(),
        checkpoint_path=Path("best_intention_model.pt"),
        checkpoint_epoch=1,
        checkpoint_selection_metric="validation_intention_macro_f1",
        device=torch.device("cpu"),
    )
    engine.filter = TemporalDecisionFilter(
        smoothing_window=1,
        minimum_confidence=0.0,
        minimum_stable_predictions=1,
    )
    engine.timestamps = deque(maxlen=1)
    engine.features = deque(maxlen=1)
    engine.hand_poses = deque(maxlen=1)
    engine.hand_valid = deque(maxlen=1)
    engine.frames_since_prediction = 0
    engine.last_timestamp_ns = None
    engine.prediction_index = 0

    original_timed_forward = online_inference.timed_forward
    online_inference.timed_forward = lambda *args, **kwargs: (
        divergent_outputs(),
        0.0,
    )
    try:
        prediction = engine.push_frame(1, {"gaze_valid": 1.0})
    finally:
        online_inference.timed_forward = original_timed_forward

    assert prediction is not None
    assert prediction["raw_intention"] == "continue"
    assert prediction["stable_intention"] == "continue"
    assert prediction["hierarchical_raw_intention"] == "fetch"
    assert prediction["raw_confidence"] == prediction["raw_p_continue"]
    assert {
        "engine_push_started_host_ns",
        "intention_inference_started_host_ns",
        "intention_inference_ended_host_ns",
        "raw_decision_host_ns",
        "stable_decision_host_ns",
        "engine_prediction_ready_host_ns",
    } <= set(prediction["pipeline_timestamps"])


def test_online_engine_warms_single_executable_checkpoint_once() -> None:
    dummy_model = SimpleNamespace(input_dim=4)
    artifacts = SimpleNamespace(
        window_size=3,
        step_size=1,
        minimum_observed_fraction=0.0,
        max_timestamp_gap_ns=100,
        feature_columns=["a", "b"],
        model=dummy_model,
        device=torch.device("cpu"),
    )
    calls = []
    original_load = online_inference.load_artifacts
    original_forward = online_inference.timed_forward
    online_inference.load_artifacts = lambda *args, **kwargs: artifacts
    online_inference.timed_forward = lambda model, features, refs, device: (
        calls.append((tuple(features.shape), tuple(refs.shape)))
        or ({}, 1.25)
    )
    try:
        engine = OnlineInferenceEngine(
            Path("."),
            warm_up_models=True,
        )
    finally:
        online_inference.load_artifacts = original_load
        online_inference.timed_forward = original_forward
    assert calls == [((1, 3, 4), (1, 2, 7))]
    assert engine.warmup_latency_ms == {"model": 1.25}


def test_replay_and_live_use_same_quality_gate_semantics() -> None:
    columns = [
        "gaze_valid",
        "hand_left_valid",
        "hand_right_valid",
        "robot_frame_valid",
    ]
    timestamps = np.asarray([0, 10, 20, 30], dtype=np.int64)
    features = np.asarray(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    quality_diagnostics = [
        {
            "_quality_hand_age_ms": 10.0,
            "_quality_vio_age_ms": 2.0,
            "_quality_anchor_age_ms": 100.0,
            "_quality_visible_marker_count": 0.0,
            "_quality_minimum_visible_marker_age_ms": None,
        }
        for _ in timestamps
    ]
    live_gate = InputQualityGate(
        window_size=4,
        max_timestamp_gap_ns=100,
    )
    replay_gate = InputQualityGate(
        window_size=4,
        max_timestamp_gap_ns=100,
    )
    for timestamp, values in zip(timestamps, features):
        live_values = dict(zip(columns, values))
        live_values.update(
            quality_diagnostics[len(live_gate.frames)]
        )
        live_gate.push_frame(
            int(timestamp),
            live_values,
        )
    push_replay_quality_frames(
        replay_gate,
        timestamps_ns=timestamps,
        features=features,
        feature_columns=columns,
        quality_diagnostics=quality_diagnostics,
        start=0,
        stop=len(features),
    )

    live_decision = evaluate_actionability(
        live_gate,
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    replay_decision = evaluate_actionability(
        replay_gate,
        stable_intention="handover",
        predicted_receiving_hand="left",
    )
    assert replay_decision == live_decision
    assert replay_decision["actionable_intention"] == "handover"


def test_gap_resets_quality_window_and_causal_lookup() -> None:
    gate = InputQualityGate(
        window_size=2,
        max_timestamp_gap_ns=100,
    )
    values = {
        "gaze_valid": 1.0,
        "hand_left_valid": 1.0,
        "hand_right_valid": 0.0,
        "robot_frame_valid": 1.0,
        "_quality_hand_age_ms": 1.0,
        "_quality_vio_age_ms": 1.0,
        "_quality_anchor_age_ms": 1.0,
        "_quality_visible_marker_count": 0.0,
    }
    assert not gate.push_frame(0, values)
    assert not gate.push_frame(10, values)
    assert gate.assess("continue")["ok"]
    assert gate.push_frame(1000, values)
    reset_result = gate.assess("continue")
    assert not reset_result["ok"]
    assert "quality_window_not_ready" in reset_result["reasons"]

    queue = deque(
        [
            TimedSample(100, "past", 1000),
            TimedSample(300, "future", 1100),
        ]
    )
    selected = latest_before_item(queue, 200, 150)
    assert selected is not None and selected.value == "past"
    assert latest_before_item(queue, 260, 150) is None


def test_live_missing_hand_values_match_offline_missing_semantics() -> None:
    values = LiveFeatureAssembler._hand_features(
        None,
        np.eye(4),
    )
    for side in ("left", "right"):
        assert np.isnan(values[f"hand_{side}_tracking_confidence"])
        assert values[f"hand_{side}_valid"] == 0.0
        assert not any(
            name.startswith(f"{side}_wrist_robot_")
            for name in values
        )


def test_live_anchor_is_frozen_after_warmup() -> None:
    assembler = LiveFeatureAssembler.__new__(LiveFeatureAssembler)
    assembler.minimum_anchor_samples = 2
    assembler.anchor_candidates = deque(maxlen=10)
    assembler.static_odometry_robot = None
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.02
    assembler._update_anchor(first)
    assert assembler.static_odometry_robot is None
    assembler._update_anchor(second)
    frozen = assembler.static_odometry_robot.copy()
    moved = np.eye(4)
    moved[0, 3] = 2.0
    assembler._update_anchor(moved)
    assert np.array_equal(assembler.static_odometry_robot, frozen)
    assert len(assembler.anchor_candidates) == 2


def test_replay_freshness_is_causal_and_summary_is_layered() -> None:
    timestamps = np.asarray(
        [100_000_000, 200_000_000],
        dtype=np.int64,
    )
    frame = pd.DataFrame(
        {
            "hand_time_offset_ms": [-5.0, 5.0],
            "slam_time_offset_ms": [-2.0, -3.0],
            "apriltag_0_timestamp_ns": [
                90_000_000,
                210_000_000,
            ],
            "apriltag_0_valid": [1.0, 1.0],
            "aruco_6_timestamp_ns": [
                95_000_000,
                205_000_000,
            ],
            "aruco_6_valid": [1.0, 1.0],
        }
    )
    diagnostics = replay_quality_diagnostics(frame, timestamps)
    assert diagnostics[0]["_quality_hand_age_ms"] == 5.0
    assert diagnostics[1]["_quality_hand_age_ms"] is None
    assert diagnostics[0]["_quality_anchor_age_ms"] == 10.0
    assert diagnostics[1]["_quality_anchor_age_ms"] == 110.0
    assert diagnostics[0]["_quality_aruco_6_age_ms"] == 5.0
    assert diagnostics[1]["_quality_aruco_6_age_ms"] is None

    rows = []
    for index, (
        target,
        stable,
        actionable,
        quality_ok,
    ) in enumerate(
        (
            ("continue", "continue", "continue", True),
            ("fetch", "uncertain", "insufficient_input", False),
            ("handover", "handover", "handover", True),
        )
    ):
        rows.append(
            {
                "target_intention": target,
                "raw_intention": target,
                "stable_intention": stable,
                "actionable_intention": actionable,
                "input_quality_ok": quality_ok,
                "input_quality_reasons": (
                    [] if quality_ok else ["gaze_coverage_too_low"]
                ),
                "intention_inference_ms": float(index + 1),
                "pose_inference_ms": None,
                "pose_position_error_cm": None,
                "pose_orientation_error_deg": None,
            }
        )
    summary = build_replay_summary(rows)
    assert summary["decision_levels"]["raw"]["coverage"] == 1.0
    assert summary["decision_levels"]["raw"][
        "end_to_end_accuracy"
    ] == 1.0
    assert summary["decision_levels"]["stable"]["coverage"] == 2 / 3
    assert summary["decision_levels"]["actionable"]["coverage"] == 2 / 3
    assert summary["input_quality"]["blocked_windows"] == 1
    assert summary["input_quality"]["reason_counts"] == {
        "gaze_coverage_too_low": 1
    }


def main() -> int:
    test_raw_prediction_printer_only_reports_label_changes()
    test_only_confident_candidates_count()
    test_unconfident_prediction_resets_confident_run()
    test_joint_probabilities_define_raw_and_stable_class()
    test_online_engine_uses_shared_joint_decision()
    test_online_engine_warms_single_executable_checkpoint_once()
    test_replay_and_live_use_same_quality_gate_semantics()
    test_gap_resets_quality_window_and_causal_lookup()
    test_live_missing_hand_values_match_offline_missing_semantics()
    test_live_anchor_is_frozen_after_warmup()
    test_replay_freshness_is_causal_and_summary_is_layered()
    print("inference decision smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
