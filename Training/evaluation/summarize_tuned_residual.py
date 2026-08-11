#!/usr/bin/env python3
"""Compare the final validation-selected residual-v2 config with the baseline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aria_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "aria_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from checkpoint_semantics import (
    DEFAULT_VALIDATION_SELECTION_RULE,
    mark_seed_aggregate,
    pose_selected_diagnostic,
    primary_result_row,
    select_primary_checkpoint_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--baseline-experiment", default="benchmark_v2")
    parser.add_argument("--tuned-experiment", default="residual_v2_tuned_v1")
    parser.add_argument(
        "--report-tag", default="residual_v2_tuned_v2_checkpoint_coherent"
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def validation_for_best_intention(metrics: dict) -> dict:
    if "validation_by_checkpoint" in metrics:
        return metrics["validation_by_checkpoint"]["best_intention"]
    epoch = int(metrics["checkpoints"]["best_intention"]["epoch"])
    return next(
        row["validation"]
        for row in metrics["history"]
        if int(row["epoch"]) == epoch
    )


def run_row(label: str, seed: int, run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    validation = validation_for_best_intention(metrics)
    primary = primary_result_row(
        metrics, checkpoint="best_intention", split="test"
    )
    diagnostic = pose_selected_diagnostic(metrics, split="test")
    return {
        "model": label,
        "seed": seed,
        "run_dir": str(run_dir),
        "validation_intention_macro_f1": float(validation["intention"]["macro_f1"]),
        "validation_receiving_hand_macro_f1": float(
            validation["receiving_hand"]["macro_f1_supported"]
        ),
        "validation_pose_mae_cm": float(
            validation["pose_oracle"]["position_mae_cm"]
        ),
        **primary,
        **diagnostic,
        "trainable_parameters": int(metrics["trainable_parameters"]),
        "test_evaluation_skipped": bool(metrics.get("test_evaluation_skipped", False)),
    }


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    output = PROJECT_ROOT / "Training/reports" / args.dataset_tag / args.report_tag
    data_dir = output / "data"
    figures = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for seed in SEEDS:
        paths = {
            "baseline": root / args.baseline_experiment / "residual_v2" / f"{args.baseline_experiment}_residual_v2_seed{seed}",
            "tuned": root / args.tuned_experiment / "residual_v2_tuned" / f"{args.tuned_experiment}_residual_v2_tuned_seed{seed}",
        }
        for label, path in paths.items():
            try:
                rows.append(run_row(label, seed, path))
            except (FileNotFoundError, KeyError, StopIteration, TypeError, ValueError) as exc:
                errors.append(f"{label} seed {seed}: {type(exc).__name__}: {exc}")
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "baseline_vs_tuned_runs.csv", index=False)
    aggregate_rows = []
    for label, group in frame.groupby("model", sort=False):
        row = {"model": label, "completed_seeds": len(group)}
        for metric in (
            "validation_intention_macro_f1",
            "validation_receiving_hand_macro_f1",
            "validation_pose_mae_cm",
            "test_intention_macro_f1",
            "test_intention_accuracy",
            "test_receiving_hand_macro_f1",
            "test_pose_mae_cm",
            "test_pose_orientation_error_deg",
            "test_pose_samples",
            "test_pose_end_to_end_mae_cm",
            "test_pose_end_to_end_orientation_error_deg",
            "test_pose_end_to_end_samples",
            "test_pose_target_coverage",
            "diagnostic_pose_selected_test_pose_mae_cm",
            "diagnostic_pose_selected_test_pose_orientation_error_deg",
            "diagnostic_pose_selected_test_pose_samples",
            "diagnostic_pose_selected_test_intention_macro_f1",
            "trainable_parameters",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        row["generalization_gap_f1"] = (
            row["validation_intention_macro_f1_mean"]
            - row["test_intention_macro_f1_mean"]
        )
        aggregate_rows.append(mark_seed_aggregate(row))
    aggregate = pd.DataFrame(aggregate_rows)
    if set(aggregate.get("model", [])) == {"baseline", "tuned"}:
        baseline = aggregate.loc[aggregate["model"] == "baseline"].iloc[0]
        for metric in (
            "test_intention_macro_f1_mean",
            "test_intention_accuracy_mean",
            "test_receiving_hand_macro_f1_mean",
            "test_pose_mae_cm_mean",
            "trainable_parameters_mean",
        ):
            aggregate[f"delta_{metric}_vs_baseline"] = aggregate[metric] - baseline[metric]
    aggregate.to_csv(data_dir / "baseline_vs_tuned_summary.csv", index=False)
    complete = not errors and len(frame) == 6
    if complete:
        labels = ["Baseline", "Tuned"]
        ordered = aggregate.set_index("model").loc[["baseline", "tuned"]]
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for axis, metric, title in (
            (axes[0], "test_intention_macro_f1", "Test intention macro-F1"),
            (axes[1], "test_receiving_hand_macro_f1", "Test hand macro-F1"),
            (
                axes[2],
                "test_pose_mae_cm",
                "Test pose error (same primary checkpoint; cm; lower better)",
            ),
        ):
            axis.bar(labels, ordered[f"{metric}_mean"], yerr=ordered[f"{metric}_std"], capsize=5, color=["#A0A0A0", "#4C78A8"])
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Residual-v2 baseline vs validation-selected tuning (mean ± population SD; seeds 42/43/44)")
        figure.tight_layout()
        figure.savefig(figures / "01_baseline_vs_tuned_test_metrics.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "01_baseline_vs_tuned_test_metrics.pdf", bbox_inches="tight")
        plt.close(figure)

        paired = frame.pivot(index="seed", columns="model", values="test_intention_macro_f1")
        figure, axis = plt.subplots(figsize=(7, 5))
        for seed, values in paired.iterrows():
            axis.plot(labels, [values["baseline"], values["tuned"]], marker="o", label=f"seed {seed}")
        axis.set_ylabel("Test intention macro-F1")
        axis.set_title("Paired seed comparison")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "02_paired_seed_intention_f1.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / "02_paired_seed_intention_f1.pdf", bbox_inches="tight")
        plt.close(figure)
    primary_results = []
    for label, group in frame.groupby("model", sort=False):
        selected = select_primary_checkpoint_row(group.to_dict(orient="records"))
        selected["model"] = label
        primary_results.append(selected)
    report = {
        "schema_version": 2,
        "dataset_tag": args.dataset_tag,
        "baseline_experiment": args.baseline_experiment,
        "tuned_experiment": args.tuned_experiment,
        "report_tag": args.report_tag,
        "seeds": list(SEEDS),
        "complete": complete,
        "errors": errors,
        "selection_used_test_metrics": False,
        "test_access": "after Stage-A and Stage-B validation selection",
        "checkpoint_policy": {
            "primary": "all primary metrics use one best_intention checkpoint",
            "selection_split": "validation",
            "seed_selection_rule": DEFAULT_VALIDATION_SELECTION_RULE,
            "pose_selected": "separately namespaced diagnostic only",
        },
        "primary_results": primary_results,
        "seed_aggregate_diagnostics": aggregate.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Tuned comparison complete: {complete}; report: {output}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
