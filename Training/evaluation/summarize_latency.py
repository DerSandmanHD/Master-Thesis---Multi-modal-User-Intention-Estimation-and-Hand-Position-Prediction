#!/usr/bin/env python3
"""Aggregate cross-platform latency JSON files into tables and plots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aria_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    distributions = []
    unavailable = []
    checkpoint_hashes = set()
    fixture_hashes = set()
    protocols = set()
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        platform_name = path.stem
        if report.get("status") != "completed":
            unavailable.append(
                {
                    "platform": platform_name,
                    "status": report.get("status", "unknown"),
                    "reason": report.get("reason", ""),
                }
            )
            continue
        checkpoint_hashes.add(report["checkpoint_sha256"])
        fixture_hashes.add(report["fixture_sha256"])
        protocol = report["protocol"]
        protocols.add(
            (
                int(protocol["batch_size"]),
                int(protocol["warmup"]),
                int(protocol["repeats"]),
                bool(protocol["synchronize_before_and_after_each_measurement"]),
                float(protocol["realtime_threshold_ms"]),
            )
        )
        row = {
            "platform": platform_name,
            "device": report["hardware"]["device"],
            "hostname": report["hardware"]["hostname"],
            "processor": report["hardware"].get("processor", ""),
            "accelerator": report["hardware"].get("accelerator", {}).get("name", ""),
            "torch_version": report["hardware"]["torch_version"],
            "model_parameters": report["model"]["trainable_parameters"],
            "model_load_time_ms": report["model"]["load_time_ms"],
        }
        for section in ("model_forward", "offline_window"):
            for metric in (
                "mean_ms",
                "median_ms",
                "std_ms",
                "p95_ms",
                "p99_ms",
                "throughput_windows_per_second",
                "fraction_within_realtime_threshold",
            ):
                row[f"{section}_{metric}"] = report[section][metric]
        rows.append(row)
        for section, key in (
            ("model_forward", "raw_model_forward_ms"),
            ("offline_window", "raw_offline_window_ms"),
        ):
            distributions.extend(
                {
                    "platform": platform_name,
                    "measurement": section,
                    "latency_ms": float(value),
                }
                for value in report[key]
            )
    if len(checkpoint_hashes) > 1 or len(fixture_hashes) > 1:
        raise ValueError("Latency reports do not use one identical checkpoint/fixture")
    if len(protocols) > 1:
        raise ValueError("Latency reports do not use one identical timing protocol")
    frame = pd.DataFrame(rows)
    distribution_frame = pd.DataFrame(distributions)
    frame.to_csv(output_dir / "latency_summary.csv", index=False)
    distribution_frame.to_csv(output_dir / "latency_samples.csv", index=False)
    pd.DataFrame(unavailable).to_csv(output_dir / "unavailable_platforms.csv", index=False)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    if not frame.empty:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        x = np.arange(len(frame))
        labels = frame["platform"].str.replace("_", " ")
        for axis, section, title in (
            (axes[0], "model_forward", "Pure model forward"),
            (axes[1], "offline_window", "Offline window incl. transfer/postprocess"),
        ):
            axis.bar(x, frame[f"{section}_median_ms"], color="#4C78A8", label="median")
            axis.scatter(x, frame[f"{section}_p95_ms"], color="#E45756", label="p95", zorder=3)
            axis.scatter(x, frame[f"{section}_p99_ms"], color="#F58518", label="p99", zorder=3)
            axis.set_xticks(x, labels, rotation=25, ha="right")
            axis.set_ylabel("Latency (ms; batch size 1)")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
        protocol = next(iter(protocols))
        figure.suptitle(
            "Residual-v2 cross-platform latency "
            f"({protocol[1]} warm-up, {protocol[2]:,} measured)"
        )
        figure.tight_layout()
        figure.savefig(figures / "01_latency_median_p95_p99.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "01_latency_median_p95_p99.pdf", bbox_inches="tight")
        plt.close(figure)

    if not distribution_frame.empty:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for axis, measurement in zip(axes, ("model_forward", "offline_window")):
            subset = distribution_frame.loc[distribution_frame["measurement"] == measurement]
            for platform_name, group in subset.groupby("platform"):
                values = np.sort(group["latency_ms"].to_numpy(dtype=float))
                cdf = np.arange(1, len(values) + 1) / len(values)
                axis.plot(values, cdf, label=platform_name.replace("_", " "))
            axis.set_xlabel("Latency (ms)")
            axis.set_ylabel("Empirical CDF")
            axis.set_title(measurement.replace("_", " "))
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle("Latency distributions")
        figure.tight_layout()
        figure.savefig(figures / "02_latency_cdf.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "02_latency_cdf.pdf", bbox_inches="tight")
        plt.close(figure)

    summary = {
        "schema_version": 1,
        "completed_platforms": frame["platform"].tolist() if not frame.empty else [],
        "unavailable_platforms": unavailable,
        "identical_checkpoint_sha256": next(iter(checkpoint_hashes), None),
        "identical_fixture_sha256": next(iter(fixture_hashes), None),
        "protocol": (
            {
                "batch_size": next(iter(protocols))[0],
                "warmup": next(iter(protocols))[1],
                "repeats": next(iter(protocols))[2],
                "synchronized_per_measurement": next(iter(protocols))[3],
                "realtime_threshold_ms": next(iter(protocols))[4],
            }
            if protocols
            else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Latency reports: {len(frame)} completed, {len(unavailable)} unavailable; "
        f"output: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
