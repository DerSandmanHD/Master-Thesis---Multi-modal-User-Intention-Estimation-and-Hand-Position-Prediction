#!/usr/bin/env python3
"""Analyze controlled live predictions against monotonic event annotations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


INTENTION_NAMES = ("continue", "fetch", "handover")
LEVEL_FIELDS = {
    "raw": "raw_intention",
    "stable": "stable_intention",
    "actionable": "actionable_intention",
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object in {path} line {line_number}"
                )
            records.append(value)
    return records


def build_intervals(events: list[dict]) -> list[dict]:
    intervals = []
    active: dict | None = None
    for record in events:
        event = record.get("event")
        if event == "note":
            continue
        if event == "start":
            if active is not None:
                raise ValueError(
                    f"Overlapping event after {active.get('scenario_id')}"
                )
            active = record
        elif event == "end":
            if active is None:
                raise ValueError("Encountered an end event without a start")
            if record.get("scenario_id") != active.get("scenario_id"):
                raise ValueError(
                    "Start/end scenario IDs differ: "
                    f"{active.get('scenario_id')} vs "
                    f"{record.get('scenario_id')}"
                )
            start_ns = int(active["host_monotonic_ns"])
            end_ns = int(record["host_monotonic_ns"])
            if end_ns <= start_ns:
                raise ValueError(
                    f"Non-positive interval for {active.get('scenario_id')}"
                )
            intervals.append(
                {
                    "scenario_id": str(active["scenario_id"]),
                    "expected_intention": active.get("expected_intention"),
                    "expected_quality_ok": active.get(
                        "expected_quality_ok"
                    ),
                    "start_host_monotonic_ns": start_ns,
                    "end_host_monotonic_ns": end_ns,
                    "duration_seconds": (end_ns - start_ns) / 1e9,
                }
            )
            active = None
        else:
            raise ValueError(f"Unknown annotation event: {event!r}")
    if active is not None:
        raise ValueError(
            f"Scenario {active.get('scenario_id')!r} has no end event"
        )
    if not intervals:
        raise ValueError("No complete live validation intervals found")
    return intervals


def prediction_host_ns(prediction: dict) -> int | None:
    phases = prediction.get("pipeline_timestamps")
    if not isinstance(phases, dict):
        return None
    for key in (
        "output_ready_host_ns",
        "output_emit_started_host_ns",
        "engine_prediction_ready_host_ns",
    ):
        value = phases.get(key)
        if value is not None:
            return int(value)
    return None


def percentile(values: list[float], percentage: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def numeric_summary(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "samples": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(finite, 95.0),
        "maximum": max(finite) if finite else None,
    }


def level_summary(
    predictions: list[dict],
    *,
    field: str,
    expected: str | None,
    start_ns: int,
) -> dict:
    labels = [str(prediction.get(field, "unavailable")) for prediction in predictions]
    counts = Counter(labels)
    summary = {
        "counts": dict(sorted(counts.items())),
        "eligible_predictions": sum(label in INTENTION_NAMES for label in labels),
        "expected_intention": expected,
        "accuracy": None,
        "time_to_first_expected_ms": None,
    }
    if expected is None:
        return summary
    summary["accuracy"] = (
        sum(label == expected for label in labels) / len(labels)
        if labels
        else None
    )
    for prediction, label in zip(predictions, labels):
        if label == expected:
            host_ns = prediction_host_ns(prediction)
            if host_ns is not None:
                summary["time_to_first_expected_ms"] = (
                    host_ns - start_ns
                ) / 1e6
            break
    return summary


def phase_delta_ms(
    prediction: dict,
    start_key: str,
    end_key: str,
) -> float | None:
    phases = prediction.get("pipeline_timestamps")
    if not isinstance(phases, dict):
        return None
    start = phases.get(start_key)
    end = phases.get(end_key)
    if start is None or end is None:
        return None
    delta = (int(end) - int(start)) / 1e6
    return delta if delta >= 0.0 else None


def sensor_age_values(predictions: list[dict], key: str) -> list[float]:
    values = []
    for prediction in predictions:
        ages = prediction.get("sensor_ages_ms")
        value = ages.get(key) if isinstance(ages, dict) else None
        if value is not None:
            values.append(float(value))
    return values


def analyze(
    predictions: list[dict],
    events: list[dict],
) -> dict:
    intervals = build_intervals(events)
    host_pairs = [
        (prediction_host_ns(prediction), prediction)
        for prediction in predictions
    ]
    missing_host_timestamps = sum(host is None for host, _ in host_pairs)
    usable = [
        (int(host), prediction)
        for host, prediction in host_pairs
        if host is not None
    ]
    if not usable:
        raise ValueError(
            "Predictions have no host-monotonic pipeline timestamps. "
            "Create a fresh log with the current aria_live_inference.py."
        )
    usable.sort(key=lambda item: item[0])

    scenario_reports = []
    all_annotated: list[dict] = []
    warnings = []
    for interval in intervals:
        selected = [
            prediction
            for host_ns, prediction in usable
            if interval["start_host_monotonic_ns"]
            <= host_ns
            <= interval["end_host_monotonic_ns"]
        ]
        all_annotated.extend(selected)
        if not selected:
            warnings.append(
                f"No predictions in scenario {interval['scenario_id']}"
            )

        reason_counts = Counter(
            reason
            for prediction in selected
            for reason in prediction.get("input_quality_reasons", [])
        )
        quality_values = [
            prediction.get("input_quality_ok")
            for prediction in selected
            if isinstance(prediction.get("input_quality_ok"), bool)
        ]
        expected_quality = interval["expected_quality_ok"]
        quality_match = None
        if expected_quality is not None and quality_values:
            quality_match = sum(
                value is expected_quality for value in quality_values
            ) / len(quality_values)

        scenario_reports.append(
            {
                **interval,
                "predictions": len(selected),
                "decision_levels": {
                    level: level_summary(
                        selected,
                        field=field,
                        expected=interval["expected_intention"],
                        start_ns=interval["start_host_monotonic_ns"],
                    )
                    for level, field in LEVEL_FIELDS.items()
                },
                "input_quality": {
                    "available_predictions": len(quality_values),
                    "accepted_predictions": sum(
                        value is True for value in quality_values
                    ),
                    "blocked_predictions": sum(
                        value is False for value in quality_values
                    ),
                    "expected_quality_ok": expected_quality,
                    "expected_match_fraction": quality_match,
                    "reason_counts": dict(sorted(reason_counts.items())),
                },
                "sensor_ages_ms": {
                    key: numeric_summary(sensor_age_values(selected, key))
                    for key in (
                        "hand",
                        "vio",
                        "anchor",
                        "visible_marker_minimum",
                        "visible_marker_maximum",
                    )
                },
            }
        )

    scored = [
        (interval, report)
        for interval, report in zip(intervals, scenario_reports)
        if interval["expected_intention"] is not None
    ]
    overall_levels = {}
    for level, field in LEVEL_FIELDS.items():
        total = 0
        correct = 0
        confusion: dict[str, dict[str, int]] = {}
        for interval, report in scored:
            expected = str(interval["expected_intention"])
            scenario_predictions = [
                prediction
                for host_ns, prediction in usable
                if interval["start_host_monotonic_ns"]
                <= host_ns
                <= interval["end_host_monotonic_ns"]
            ]
            for prediction in scenario_predictions:
                predicted = str(prediction.get(field, "unavailable"))
                total += 1
                correct += predicted == expected
                confusion.setdefault(expected, {})
                confusion[expected][predicted] = (
                    confusion[expected].get(predicted, 0) + 1
                )
        overall_levels[level] = {
            "scored_predictions": total,
            "accuracy": correct / total if total else None,
            "confusion": confusion,
        }

    false_assist = 0
    continue_predictions = 0
    for interval, _ in scored:
        if interval["expected_intention"] != "continue":
            continue
        for host_ns, prediction in usable:
            if (
                interval["start_host_monotonic_ns"]
                <= host_ns
                <= interval["end_host_monotonic_ns"]
            ):
                continue_predictions += 1
                if prediction.get("actionable_intention") in {
                    "fetch",
                    "handover",
                }:
                    false_assist += 1

    schema_fields = (
        "raw_intention",
        "stable_intention",
        "actionable_intention",
        "input_quality_ok",
        "input_quality_reasons",
        "sensor_ages_ms",
        "pipeline_timestamps",
        "intention_inference_ms",
    )
    missing_field_counts = {
        field: sum(field not in prediction for prediction in predictions)
        for field in schema_fields
    }

    return {
        "status": "complete" if not warnings else "complete_with_warnings",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_total": len(predictions),
        "predictions_with_host_timestamp": len(usable),
        "predictions_missing_host_timestamp": missing_host_timestamps,
        "annotation_intervals": len(intervals),
        "schema_missing_field_counts": missing_field_counts,
        "scenarios": scenario_reports,
        "overall_scored": {
            "decision_levels": overall_levels,
            "false_actionable_assistance_during_continue": false_assist,
            "continue_predictions": continue_predictions,
            "false_assistance_fraction": (
                false_assist / continue_predictions
                if continue_predictions
                else None
            ),
        },
        "latency_ms": {
            "intention_inference": numeric_summary(
                [
                    float(prediction["intention_inference_ms"])
                    for prediction in all_annotated
                    if prediction.get("intention_inference_ms") is not None
                ]
            ),
            "pose_inference": numeric_summary(
                [
                    float(prediction["pose_inference_ms"])
                    for prediction in all_annotated
                    if prediction.get("pose_inference_ms") is not None
                ]
            ),
            "gaze_callback_to_output_ready": numeric_summary(
                [
                    value
                    for prediction in all_annotated
                    if (
                        value := phase_delta_ms(
                            prediction,
                            "gaze_callback_received_host_ns",
                            "output_ready_host_ns",
                        )
                    )
                    is not None
                ]
            ),
            "feature_assembly_to_output_ready": numeric_summary(
                [
                    value
                    for prediction in all_annotated
                    if (
                        value := phase_delta_ms(
                            prediction,
                            "feature_assembly_started_host_ns",
                            "output_ready_host_ns",
                        )
                    )
                    is not None
                ]
            ),
            "capture_to_host_note": (
                "Not derived: device and host clocks require an explicit "
                "clock mapping before their timestamps may be subtracted."
            ),
        },
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions_jsonl.expanduser().resolve()
    events_path = args.events_jsonl.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    report = analyze(
        read_jsonl(predictions_path),
        read_jsonl(events_path),
    )
    report["predictions_jsonl"] = str(predictions_path)
    report["events_jsonl"] = str(events_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Live validation {report['status']}: "
        f"scenarios={report['annotation_intervals']}, "
        f"predictions={report['predictions_total']}"
    )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
