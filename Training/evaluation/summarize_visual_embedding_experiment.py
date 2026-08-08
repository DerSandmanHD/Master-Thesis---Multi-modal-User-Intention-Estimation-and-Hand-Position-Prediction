#!/usr/bin/env python3
"""Compare validation-only CLIP variants against the frozen sensor baseline."""

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
VARIANTS = ("clip_only", "sensor_plus_clip", "sensor_plus_random")
F1_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="visual_embedding_screen_v1")
    parser.add_argument("--baseline-experiment", default="benchmark_v2")
    parser.add_argument("--baseline-model", default="residual_v2")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def validation_metrics(metrics: dict) -> dict:
    if "validation_by_checkpoint" in metrics:
        return metrics["validation_by_checkpoint"]["best_intention"]
    epoch = int(metrics["checkpoints"]["best_intention"]["epoch"])
    history_row = next(row for row in metrics["history"] if int(row["epoch"]) == epoch)
    return history_row["validation"]


def metric_row(
    *, variant: str, seed: int, run_dir: Path, validation_only: bool
) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if validation_only:
        if metrics.get("test_evaluation_skipped") is not True or "test" in metrics:
            raise ValueError(f"Visual screening leaked test metrics: {run_dir}")
    values = validation_metrics(metrics)
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "validation_intention_macro_f1": float(values["intention"]["macro_f1"]),
        "validation_receiving_hand_macro_f1": float(
            values["receiving_hand"]["macro_f1_supported"]
        ),
        "validation_pose_mae_cm": float(
            values["pose_oracle"]["position_mae_cm"]
        ),
        "trainable_parameters": int(metrics["trainable_parameters"]),
        "test_metrics_used": False,
    }


def save_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["variant"].str.replace("_", " ").tolist()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = (
        ("validation_intention_macro_f1", "Validation intention macro-F1", False),
        ("validation_receiving_hand_macro_f1", "Validation hand macro-F1", False),
        ("validation_pose_mae_cm", "Validation pose MAE (cm)", True),
    )
    for axis, (metric, title, lower_better) in zip(axes, specs):
        means = summary[f"{metric}_mean"]
        errors = summary[f"{metric}_std"]
        colors = ["#4C78A8" if name != "sensor_plus_random" else "#B0B0B0" for name in summary["variant"]]
        axis.bar(labels, means, yerr=errors, capsize=4, color=colors)
        axis.set_title(title + (" (lower better)" if lower_better else ""))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen CLIP embedding screening (validation only; seeds 42/43/44)")
    figure.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "01_visual_embedding_validation_comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "01_visual_embedding_validation_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    runs_root = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    output_dir = PROJECT_ROOT / "Training/reports" / args.dataset_tag / args.experiment_tag
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for seed in SEEDS:
        baseline_dir = (
            runs_root
            / args.baseline_experiment
            / args.baseline_model
            / f"{args.baseline_experiment}_{args.baseline_model}_seed{seed}"
        )
        try:
            rows.append(metric_row(variant="sensor_baseline", seed=seed, run_dir=baseline_dir, validation_only=False))
        except (FileNotFoundError, KeyError, StopIteration, TypeError, ValueError) as exc:
            missing.append(f"sensor_baseline seed {seed}: {exc}")
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = runs_root / args.experiment_tag / variant / f"{args.experiment_tag}_{variant}_seed{seed}"
            try:
                rows.append(metric_row(variant=variant, seed=seed, run_dir=run_dir, validation_only=True))
            except (FileNotFoundError, KeyError, StopIteration, TypeError, ValueError) as exc:
                missing.append(f"{variant} seed {seed}: {exc}")
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "validation_runs.csv", index=False)
    aggregates = []
    if not frame.empty:
        for variant, group in frame.groupby("variant", sort=False):
            row = {"variant": variant, "completed_seeds": int(len(group))}
            for metric in (
                "validation_intention_macro_f1",
                "validation_receiving_hand_macro_f1",
                "validation_pose_mae_cm",
                "trainable_parameters",
            ):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            aggregates.append(row)
    summary_frame = pd.DataFrame(aggregates)
    preferred_order = ["sensor_baseline", *VARIANTS]
    if not summary_frame.empty:
        summary_frame["order"] = summary_frame["variant"].map({name: i for i, name in enumerate(preferred_order)})
        summary_frame = summary_frame.sort_values("order").drop(columns="order")
    summary_frame.to_csv(data_dir / "validation_summary.csv", index=False)
    complete = not missing and len(summary_frame) == 4 and bool((summary_frame["completed_seeds"] == 3).all())

    selected = None
    selection_reason = None
    if complete:
        candidates = summary_frame.loc[
            summary_frame["variant"].isin(["sensor_baseline", "clip_only", "sensor_plus_clip"])
        ].copy()
        best_f1 = float(candidates["validation_intention_macro_f1_mean"].max())
        candidates = candidates.loc[
            candidates["validation_intention_macro_f1_mean"] >= best_f1 - F1_TOLERANCE
        ].sort_values(
            ["validation_pose_mae_cm_mean", "validation_receiving_hand_macro_f1_mean", "trainable_parameters_mean"],
            ascending=[True, False, True],
            kind="stable",
        )
        selected = str(candidates.iloc[0]["variant"])
        selection_reason = (
            "Highest mean validation intention F1 within 0.005 tolerance; then "
            "lowest pose MAE, highest hand F1, and lowest parameter count. "
            "The random-feature control is diagnostic and cannot be selected."
        )
        save_plot(summary_frame, output_dir)
    summary = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "selection_split": "validation",
        "test_metrics_used_for_selection": False,
        "seeds": list(SEEDS),
        "complete": complete,
        "missing_or_invalid": missing,
        "selected_variant_for_final_test": selected,
        "selection_rule": selection_reason,
        "random_control_selectable": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Visual screening complete: {complete}; selected: {selected}")
    print(f"Report: {output_dir}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
