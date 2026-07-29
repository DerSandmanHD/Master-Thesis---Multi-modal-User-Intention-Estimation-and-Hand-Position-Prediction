#!/usr/bin/env python3
"""Short test for live event parsing and analysis."""

from __future__ import annotations

from analyze_live_validation import analyze
from live_event_marker import parse_start_command


def prediction(
    host_ns: int,
    *,
    raw: str,
    stable: str,
    actionable: str,
    quality: bool,
) -> dict:
    return {
        "raw_intention": raw,
        "stable_intention": stable,
        "actionable_intention": actionable,
        "input_quality_ok": quality,
        "input_quality_reasons": [] if quality else ["gaze_coverage_too_low"],
        "sensor_ages_ms": {"hand": 10.0, "vio": 1.0, "anchor": 100.0},
        "intention_inference_ms": 4.0,
        "pose_inference_ms": None,
        "pipeline_timestamps": {
            "gaze_callback_received_host_ns": host_ns - 5_000_000,
            "feature_assembly_started_host_ns": host_ns - 4_000_000,
            "output_ready_host_ns": host_ns,
        },
    }


def main() -> int:
    parsed = parse_start_command("start fetch_combined fetch true")
    assert parsed == {
        "scenario_id": "fetch_combined",
        "expected_intention": "fetch",
        "expected_quality_ok": True,
    }
    unscored = parse_start_command("start gaze_only unscored any")
    assert unscored["expected_intention"] is None
    assert unscored["expected_quality_ok"] is None

    events = [
        {
            "event": "start",
            "scenario_id": "neutral",
            "expected_intention": "continue",
            "expected_quality_ok": True,
            "host_monotonic_ns": 1_000_000_000,
        },
        {
            "event": "end",
            "scenario_id": "neutral",
            "expected_intention": "continue",
            "expected_quality_ok": True,
            "host_monotonic_ns": 2_000_000_000,
        },
        {
            "event": "start",
            "scenario_id": "gaze_dropout",
            "expected_intention": None,
            "expected_quality_ok": False,
            "host_monotonic_ns": 3_000_000_000,
        },
        {
            "event": "end",
            "scenario_id": "gaze_dropout",
            "expected_intention": None,
            "expected_quality_ok": False,
            "host_monotonic_ns": 4_000_000_000,
        },
    ]
    predictions = [
        prediction(
            1_200_000_000,
            raw="continue",
            stable="continue",
            actionable="continue",
            quality=True,
        ),
        prediction(
            1_500_000_000,
            raw="continue",
            stable="continue",
            actionable="continue",
            quality=True,
        ),
        prediction(
            3_500_000_000,
            raw="handover",
            stable="handover",
            actionable="insufficient_input",
            quality=False,
        ),
    ]
    report = analyze(predictions, events)
    assert report["status"] == "complete"
    assert (
        report["overall_scored"]["decision_levels"]["actionable"]["accuracy"]
        == 1.0
    )
    assert report["scenarios"][0]["decision_levels"]["stable"][
        "time_to_first_expected_ms"
    ] == 200.0
    assert report["scenarios"][1]["input_quality"][
        "expected_match_fraction"
    ] == 1.0
    assert report["latency_ms"]["gaze_callback_to_output_ready"]["mean"] == 5.0

    print("Live validation smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
