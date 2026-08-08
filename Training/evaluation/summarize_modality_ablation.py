#!/usr/bin/env python3
"""Aggregate the preregistered n214 sensor-modality ablation study."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
ABLATIONS = ("no_gaze", "no_hands", "no_objects", "no_vio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="modality_ablation_v1")
    parser.add_argument("--baseline-experiment", default="benchmark_v2")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def test_row(variant: str, seed: int, run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    intention_test = metrics["test"]["best_intention"]
    pose_test = metrics["test"]["best_pose"]
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "test_intention_macro_f1": float(
            intention_test["intention"]["macro_f1"]
        ),
        "test_intention_accuracy": float(
            intention_test["intention"]["accuracy"]
        ),
        "test_receiving_hand_macro_f1": float(
            intention_test["receiving_hand"]["macro_f1_supported"]
        ),
        "test_pose_mae_cm": float(
            pose_test["pose_oracle"]["position_mae_cm"]
        ),
        "test_pose_end_to_end_mae_cm": float(
            pose_test["pose_end_to_end"]["position_mae_cm"]
        ),
        "test_pose_at_intention_checkpoint_mae_cm": float(
            intention_test["pose_oracle"]["position_mae_cm"]
        ),
        "test_pose_end_to_end_at_intention_checkpoint_mae_cm": float(
            intention_test["pose_end_to_end"]["position_mae_cm"]
        ),
        "test_intention_macro_f1_at_pose_checkpoint": float(
            pose_test["intention"]["macro_f1"]
        ),
        "trainable_parameters": int(metrics["trainable_parameters"]),
        "wall_seconds": float(metrics.get("runtime", {}).get("wall_seconds", np.nan)),
    }


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        row = {"variant": variant, "completed_seeds": int(len(group))}
        for metric in (
            "test_intention_macro_f1",
            "test_intention_accuracy",
            "test_receiving_hand_macro_f1",
            "test_pose_mae_cm",
            "test_pose_end_to_end_mae_cm",
            "test_pose_at_intention_checkpoint_mae_cm",
            "test_pose_end_to_end_at_intention_checkpoint_mae_cm",
            "test_intention_macro_f1_at_pose_checkpoint",
            "trainable_parameters",
            "wall_seconds",
        ):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {name: index for index, name in enumerate(("full", *ABLATIONS))}
    summary["order"] = summary["variant"].map(order)
    summary = summary.sort_values("order").drop(columns="order")
    baseline = summary.loc[summary["variant"] == "full"]
    if len(baseline) == 1:
        reference = baseline.iloc[0]
        summary["delta_intention_macro_f1_vs_full"] = (
            summary["test_intention_macro_f1_mean"]
            - reference["test_intention_macro_f1_mean"]
        )
        summary["delta_hand_macro_f1_vs_full"] = (
            summary["test_receiving_hand_macro_f1_mean"]
            - reference["test_receiving_hand_macro_f1_mean"]
        )
        summary["delta_pose_mae_cm_vs_full"] = (
            summary["test_pose_mae_cm_mean"]
            - reference["test_pose_mae_cm_mean"]
        )
        summary["parameter_change_vs_full"] = (
            summary["trainable_parameters_mean"]
            - reference["trainable_parameters_mean"]
        )
    return summary


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = summary["variant"].str.replace("_", " ").tolist()
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    specs = (
        ("test_intention_macro_f1", "Test intention macro-F1", False),
        ("test_receiving_hand_macro_f1", "Test receiving-hand macro-F1", False),
        (
            "test_pose_mae_cm",
            "Test pose MAE (best-pose checkpoint; cm; lower better)",
            True,
        ),
    )
    for axis, (metric, title, _) in zip(axes, specs):
        axis.bar(
            labels,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            capsize=4,
            color=colors,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Residual-v2 sensor ablation on frozen n214 (mean ± population SD; seeds 42/43/44)")
    figure.tight_layout()
    figure.savefig(figures / "01_sensor_ablation_metrics.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "01_sensor_ablation_metrics.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, summary["trainable_parameters_mean"], color=colors)
    axes[0].set_title("Trainable parameters")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, summary["wall_seconds_mean"], yerr=summary["wall_seconds_std"], capsize=4, color=colors)
    axes[1].set_title("Training wall time (s)")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].grid(axis="y", alpha=0.25)
    finite_wall_times = summary["wall_seconds_mean"].dropna()
    annotation_height = (
        float(finite_wall_times.max()) * 0.04 if not finite_wall_times.empty else 1.0
    )
    for index, value in enumerate(summary["wall_seconds_mean"]):
        if pd.isna(value):
            axes[1].text(
                index,
                annotation_height,
                "n/a\nlegacy run",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Ablation efficiency indicators")
    figure.tight_layout()
    figure.savefig(figures / "02_sensor_ablation_efficiency.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "02_sensor_ablation_efficiency.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    output_dir = PROJECT_ROOT / "Training/reports" / args.dataset_tag / args.experiment_tag
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for seed in SEEDS:
        run = root / args.baseline_experiment / "residual_v2" / f"{args.baseline_experiment}_residual_v2_seed{seed}"
        try:
            rows.append(test_row("full", seed, run))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"full seed {seed}: {exc}")
    for variant in ABLATIONS:
        for seed in SEEDS:
            run = root / args.experiment_tag / variant / f"{args.experiment_tag}_{variant}_seed{seed}"
            try:
                rows.append(test_row(variant, seed, run))
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{variant} seed {seed}: {exc}")
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "ablation_runs.csv", index=False)
    summary_frame = aggregate_runs(frame)
    summary_frame.to_csv(data_dir / "ablation_summary.csv", index=False)
    complete = not errors and len(summary_frame) == 5 and bool((summary_frame["completed_seeds"] == 3).all())
    if complete:
        save_plots(summary_frame, output_dir)
    report = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "baseline_experiment": args.baseline_experiment,
        "seeds": list(SEEDS),
        "test_evaluation_preregistered": True,
        "checkpoint_policy": {
            "intention_and_hand": "best_intention checkpoint selected on validation",
            "pose_mae": "best_pose checkpoint selected on validation",
            "deployment_pose_also_reported": "pose at best_intention checkpoint",
        },
        "complete": complete,
        "errors": errors,
        "variants": (
            summary_frame.astype(object)
            .where(pd.notna(summary_frame), None)
            .to_dict(orient="records")
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Ablation complete: {complete}; report: {output_dir}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
