#!/usr/bin/env python3
"""Aggregate frozen CLIP RGB-to-embedding latency across platforms."""

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


SECTIONS = ("preprocessing_cpu", "encoder_forward", "rgb_to_embedding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    weights_hashes = set()
    frame_hashes = set()
    protocols = set()
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "completed":
                raise ValueError(report.get("reason", "benchmark incomplete"))
            weights_hashes.add(report["encoder"]["weights_sha256"])
            frame_hashes.add(report["source"]["decoded_rgb_sha256"])
            protocol = report["protocol"]
            protocols.add(
                (
                    int(protocol["batch_size"]),
                    int(protocol["warmup"]),
                    int(protocol["repeats"]),
                    float(protocol["visual_sampling_hz"]),
                    float(protocol["visual_update_budget_ms"]),
                )
            )
            row = {
                "platform": path.stem,
                "device": report["hardware"]["device"],
                "hostname": report["hardware"]["hostname"],
                "accelerator": report["hardware"].get("accelerator", {}).get("name", ""),
                "torch_version": report["hardware"]["torch_version"],
                "encoder_parameters": report["encoder"]["parameter_count"],
                "encoder_load_time_ms": report["encoder"]["load_time_ms"],
            }
            for section in SECTIONS:
                for metric in (
                    "mean_ms",
                    "median_ms",
                    "std_ms",
                    "p95_ms",
                    "p99_ms",
                    "fraction_within_realtime_threshold",
                ):
                    row[f"{section}_{metric}"] = report[section][metric]
            rows.append(row)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if len(weights_hashes) > 1:
        errors.append("Platforms used different CLIP weights")
    if len(frame_hashes) > 1:
        errors.append("Platforms used different decoded RGB frames")
    if len(protocols) > 1:
        errors.append("Platforms used different CLIP timing protocols")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "clip_latency_summary.csv", index=False)
    complete = not errors and not frame.empty
    if complete:
        figures = output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        labels = frame["platform"].str.replace("_", " ").tolist()
        x = np.arange(len(frame))
        titles = {
            "preprocessing_cpu": "CPU preprocessing",
            "encoder_forward": "Frozen CLIP encoder",
            "rgb_to_embedding": "RGB to normalized embedding",
        }
        for axis, section in zip(axes, SECTIONS):
            axis.bar(x, frame[f"{section}_median_ms"], color="#4C78A8", label="median")
            axis.scatter(x, frame[f"{section}_p95_ms"], color="#E45756", label="p95", zorder=3)
            axis.set_xticks(x, labels, rotation=25, ha="right")
            axis.set_ylabel("Latency (ms; batch 1)")
            axis.set_title(titles[section])
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
        protocol = next(iter(protocols))
        figure.suptitle(
            "Frozen CLIP ViT-B/32 latency "
            f"({protocol[1]} warm-up, {protocol[2]:,} measured; {protocol[3]:g} Hz)"
        )
        figure.tight_layout()
        figure.savefig(figures / "01_clip_latency.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "01_clip_latency.pdf", bbox_inches="tight")
        plt.close(figure)
    summary = {
        "schema_version": 1,
        "complete": complete,
        "platforms": frame["platform"].tolist() if not frame.empty else [],
        "identical_weights_sha256": next(iter(weights_hashes), None),
        "identical_decoded_rgb_sha256": next(iter(frame_hashes), None),
        "protocol": list(next(iter(protocols))) if len(protocols) == 1 else None,
        "errors": errors,
        "scope": (
            "CLIP cost is separate from the temporal residual transformer; "
            "cached-embedding training does not include this live RGB cost."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"CLIP latency complete: {complete}; platforms: {len(frame)}")
    print(f"Report: {output_dir}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
