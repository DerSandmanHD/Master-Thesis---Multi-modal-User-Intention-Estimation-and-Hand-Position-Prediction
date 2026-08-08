#!/usr/bin/env python3
"""Summarize available Mac live-session inference and host-pipeline latencies."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aria_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "aria_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--realtime-threshold-ms",
        type=float,
        default=1000.0 / 30.0,
        help="Predeclared budget; default is one 30-Hz frame.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object at {path}:{line_number}")
            rows.append(value)
    return rows


def phase_delta(record: dict, start: str, end: str) -> float | None:
    timestamps = record.get("pipeline_timestamps")
    if not isinstance(timestamps, dict):
        return None
    if timestamps.get(start) is None or timestamps.get(end) is None:
        return None
    value = (int(timestamps[end]) - int(timestamps[start])) / 1e6
    return value if value >= 0 else None


def summary(values: list[float], realtime_threshold_ms: float) -> dict:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {
            key: None
            for key in (
                "samples",
                "mean_ms",
                "median_ms",
                "std_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
                "fraction_within_realtime_threshold",
            )
        }
    array = np.asarray(values, dtype=float)
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.pstdev(values),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "fraction_within_realtime_threshold": float(
            np.mean(array <= realtime_threshold_ms)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.realtime_threshold_ms <= 0:
        raise ValueError("realtime-threshold-ms must be positive")
    paths = [resolve(path).resolve() for path in args.input]
    if args.input_root:
        paths.extend(sorted(resolve(args.input_root).resolve().glob("*/predictions*.jsonl")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No live prediction logs selected")
    output_dir = resolve(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    session_rows = []
    for path in paths:
        records = read_jsonl(path)
        session = path.parent.name + ("_" + path.stem if path.stem != "predictions" else "")
        measurements = {
            "intention_inference": [
                float(record["intention_inference_ms"])
                for record in records
                if record.get("intention_inference_ms") is not None
            ],
            "pose_inference": [
                float(record["pose_inference_ms"])
                for record in records
                if record.get("pose_inference_ms") is not None
            ],
            "feature_to_output": [
                value
                for record in records
                if (value := phase_delta(record, "feature_assembly_started_host_ns", "output_ready_host_ns")) is not None
            ],
            "gaze_callback_to_output": [
                value
                for record in records
                if (value := phase_delta(record, "gaze_callback_received_host_ns", "output_ready_host_ns")) is not None
            ],
        }
        for measurement, values in measurements.items():
            values_summary = summary(values, args.realtime_threshold_ms)
            session_rows.append(
                {
                    "session": session,
                    "source": portable(path),
                    "records": len(records),
                    "measurement": measurement,
                    **values_summary,
                }
            )
            samples.extend(
                {
                    "session": session,
                    "measurement": measurement,
                    "latency_ms": value,
                }
                for value in values
            )
    summary_frame = pd.DataFrame(session_rows)
    samples_frame = pd.DataFrame(samples)
    summary_frame.to_csv(output_dir / "live_latency_summary.csv", index=False)
    samples_frame.to_csv(output_dir / "live_latency_samples.csv", index=False)
    aggregate_by_measurement = {
        str(measurement): summary(
            group["latency_ms"].astype(float).tolist(),
            args.realtime_threshold_ms,
        )
        for measurement, group in samples_frame.groupby("measurement")
    }

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    intention = samples_frame.loc[samples_frame["measurement"] == "intention_inference"]
    if not intention.empty:
        sessions = list(intention["session"].unique())
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].boxplot(
            [intention.loc[intention["session"] == session, "latency_ms"] for session in sessions],
            tick_labels=[session.replace("_", " ") for session in sessions],
            showfliers=True,
        )
        axes[0].set_ylabel("Intention inference latency (ms)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=0.25)
        for session in sessions:
            values = np.sort(intention.loc[intention["session"] == session, "latency_ms"].to_numpy(float))
            axes[1].plot(values, np.arange(1, len(values) + 1) / len(values), label=session.replace("_", " "))
        axes[1].set_xlabel("Intention inference latency (ms)")
        axes[1].set_ylabel("Empirical CDF")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        figure.suptitle("Existing Mac live sessions (exploratory; not final tuned-model benchmark)")
        figure.tight_layout()
        figure.savefig(figures / "01_existing_mac_live_intention_latency.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "01_existing_mac_live_intention_latency.pdf", bbox_inches="tight")
        plt.close(figure)

    report = {
        "schema_version": 1,
        "platform": "Mac host used for recorded Aria live sessions",
        "sessions": [portable(path) for path in paths],
        "records": int(sum(len(read_jsonl(path)) for path in paths)),
        "scope": "exploratory_existing_sessions",
        "final_tuned_model_claimed": False,
        "realtime_threshold_ms": args.realtime_threshold_ms,
        "realtime_threshold_definition": "one 30-Hz frame interval",
        "capture_to_host_latency_available": False,
        "capture_to_host_limitation": (
            "Device and host clocks have no validated mapping; they are not subtracted."
        ),
        "aggregate_by_measurement": aggregate_by_measurement,
        "summary_csv": portable(output_dir / "live_latency_summary.csv"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Live sessions: {len(paths)}; samples: {len(samples_frame)}")
    print(f"Report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
