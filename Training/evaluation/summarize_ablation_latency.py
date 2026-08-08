#!/usr/bin/env python3
"""Aggregate TCML latency measurements for the residual-v2 sensor ablations."""

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
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_gaze", "no_hands", "no_objects", "no_vio")
PLATFORMS = ("tcml_cpu", "tcml_cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="modality_ablation_v1")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = (
        PROJECT_ROOT
        / "Training/reports"
        / args.dataset_tag
        / args.experiment_tag
        / "latency"
    )
    rows = []
    errors = []
    protocols = set()
    source_windows = set()
    dataset_fingerprints: dict[str, set[str]] = {
        variant: set() for variant in VARIANTS
    }
    for variant in VARIANTS:
        for platform_name in PLATFORMS:
            path = output / variant / f"{platform_name}.json"
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("status") != "completed":
                    raise ValueError(report.get("reason", "measurement unavailable"))
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
                fixture = report["fixture_metadata"]
                source_windows.add(
                    (
                        fixture["sequence_id"],
                        fixture["participant"],
                        int(fixture["timestamp_ns"]),
                        int(fixture["source_index"]),
                    )
                )
                dataset_fingerprints[variant].add(
                    fixture["dataset_content_fingerprint"]
                )
                row = {
                    "variant": variant,
                    "platform": platform_name,
                    "device": report["hardware"]["device"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "fixture_sha256": report["fixture_sha256"],
                    "model_parameters": int(report["model"]["trainable_parameters"]),
                    "input_dim": int(report["model"]["input_dim"]),
                }
                for section in ("model_forward", "offline_window"):
                    for metric in (
                        "mean_ms",
                        "median_ms",
                        "std_ms",
                        "p95_ms",
                        "p99_ms",
                        "fraction_within_realtime_threshold",
                    ):
                        row[f"{section}_{metric}"] = float(report[section][metric])
                rows.append(row)
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"{variant}/{platform_name}: {type(exc).__name__}: {exc}"
                )
    if len(protocols) > 1:
        errors.append("Measurements do not share one timing protocol")
    if len(source_windows) > 1:
        errors.append("Fixtures do not represent the same dataset window")
    for variant, fingerprints in dataset_fingerprints.items():
        if len(fingerprints) != 1:
            errors.append(
                f"{variant} does not use one consistent dataset fingerprint"
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for platform_name, group in frame.groupby("platform"):
            baseline_rows = group.loc[group["variant"] == "full"]
            if len(baseline_rows) == 1:
                baseline = baseline_rows.iloc[0]
                mask = frame["platform"] == platform_name
                for metric in (
                    "model_forward_median_ms",
                    "model_forward_p95_ms",
                    "offline_window_median_ms",
                    "offline_window_p95_ms",
                    "model_parameters",
                ):
                    frame.loc[mask, f"delta_{metric}_vs_full"] = (
                        frame.loc[mask, metric] - baseline[metric]
                    )
    frame.to_csv(output / "ablation_latency_summary.csv", index=False)
    complete = not errors and len(frame) == len(VARIANTS) * len(PLATFORMS)
    if complete:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
        for axis, platform_name in zip(axes, PLATFORMS):
            group = frame.loc[frame["platform"] == platform_name].set_index("variant").loc[list(VARIANTS)]
            labels = [name.replace("_", " ") for name in VARIANTS]
            axis.bar(labels, group["model_forward_median_ms"], color=colors, label="median")
            axis.scatter(labels, group["model_forward_p95_ms"], color="black", marker="x", label="p95")
            axis.set_title(platform_name.replace("_", " ").upper())
            axis.set_ylabel("Model-forward latency (ms; batch 1)")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
        figure.suptitle("Residual-v2 modality-ablation latency on one matched test window")
        figure.tight_layout()
        figure.savefig(figures / "01_ablation_latency.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "01_ablation_latency.pdf", bbox_inches="tight")
        plt.close(figure)
    report = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "complete": complete,
        "variants": list(VARIANTS),
        "platforms": list(PLATFORMS),
        "same_source_window": len(source_windows) == 1,
        "source_window": (
            dict(
                zip(
                    ("sequence_id", "participant", "timestamp_ns", "source_index"),
                    next(iter(source_windows)),
                )
            )
            if len(source_windows) == 1
            else None
        ),
        "dataset_content_fingerprints_by_variant": {
            variant: next(iter(fingerprints), None)
            for variant, fingerprints in dataset_fingerprints.items()
        },
        "fingerprint_interpretation": (
            "Fingerprints are expected to differ across ablations because the "
            "feature schema changes; CPU and CUDA must agree within each variant."
        ),
        "protocol": list(next(iter(protocols))) if len(protocols) == 1 else None,
        "errors": errors,
    }
    (output / "latency_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Ablation latency complete: {complete}; report: {output}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
