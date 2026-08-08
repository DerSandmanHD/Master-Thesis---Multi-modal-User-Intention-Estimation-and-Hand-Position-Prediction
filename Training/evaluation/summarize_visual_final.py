#!/usr/bin/env python3
"""Report the final, validation-selected visual variant against the tuned sensor model."""

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
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--screen-experiment", default="visual_embedding_screen_v1")
    parser.add_argument("--baseline-experiment", default="residual_v2_tuned_v1")
    parser.add_argument("--baseline-model", default="residual_v2_tuned")
    parser.add_argument("--final-experiment", default="visual_embedding_final_v1")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def test_row(label: str, seed: int, run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("test_evaluation_skipped") is not False:
        raise ValueError(f"final test evaluation not declared: {run_dir}")
    intention_test = metrics["test"]["best_intention"]
    pose_test = metrics["test"]["best_pose"]
    return {
        "variant": label,
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
    }


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
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
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    result = pd.DataFrame(rows)
    if set(result.get("variant", [])) == {"sensor_baseline", "selected_visual"}:
        baseline = result.loc[result["variant"] == "sensor_baseline"].iloc[0]
        for metric in (
            "test_intention_macro_f1_mean",
            "test_intention_accuracy_mean",
            "test_receiving_hand_macro_f1_mean",
            "test_pose_mae_cm_mean",
            "test_pose_end_to_end_mae_cm_mean",
            "test_pose_at_intention_checkpoint_mae_cm_mean",
            "test_pose_end_to_end_at_intention_checkpoint_mae_cm_mean",
            "trainable_parameters_mean",
        ):
            result[f"delta_{metric}_vs_sensor"] = result[metric] - baseline[metric]
    return result


def save_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.set_index("variant").loc[
        ["sensor_baseline", "selected_visual"]
    ]
    labels = ["Tuned sensor", "Selected visual"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, metric, title in (
        (axes[0], "test_intention_macro_f1", "Test intention macro-F1"),
        (axes[1], "test_receiving_hand_macro_f1", "Test hand macro-F1"),
        (
            axes[2],
            "test_pose_mae_cm",
            "Test pose MAE (best-pose checkpoint; cm; lower better)",
        ),
    ):
        axis.bar(
            labels,
            ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"],
            capsize=5,
            color=["#4C78A8", "#59A14F"],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Final frozen visual comparison (mean ± population SD; seeds 42/43/44)"
    )
    figure.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "01_selected_visual_vs_sensor.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "01_selected_visual_vs_sensor.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    reports = PROJECT_ROOT / "Training/reports" / args.dataset_tag
    runs = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    output_dir = reports / args.final_experiment
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    screen = json.loads(
        (reports / args.screen_experiment / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    selected = screen.get("selected_variant_for_final_test")
    if not screen.get("complete") or selected not in {
        "sensor_baseline",
        "clip_only",
        "sensor_plus_clip",
    }:
        raise ValueError(f"Invalid or incomplete visual-screening selection: {selected!r}")

    rows = []
    errors = []
    for seed in SEEDS:
        baseline = (
            runs
            / args.baseline_experiment
            / args.baseline_model
            / f"{args.baseline_experiment}_{args.baseline_model}_seed{seed}"
        )
        try:
            rows.append(test_row("sensor_baseline", seed, baseline))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"sensor_baseline seed {seed}: {type(exc).__name__}: {exc}")
        if selected != "sensor_baseline":
            visual = (
                runs
                / args.final_experiment
                / selected
                / f"{args.final_experiment}_{selected}_seed{seed}"
            )
            try:
                rows.append(test_row("selected_visual", seed, visual))
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{selected} seed {seed}: {type(exc).__name__}: {exc}")

    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "final_visual_runs.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(data_dir / "final_visual_summary.csv", index=False)
    expected_rows = 3 if selected == "sensor_baseline" else 6
    complete = not errors and len(frame) == expected_rows
    if complete and selected != "sensor_baseline":
        save_plot(summary, output_dir)
    report = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "screen_experiment": args.screen_experiment,
        "baseline_experiment": args.baseline_experiment,
        "final_experiment": args.final_experiment,
        "validation_selected_variant": selected,
        "selection_used_test_metrics": False,
        "test_access": "one final evaluation after the visual variant was frozen on validation",
        "checkpoint_policy": {
            "intention_and_hand": "best_intention checkpoint selected on validation",
            "pose_mae": "best_pose checkpoint selected on validation",
            "deployment_pose_also_reported": "pose at best_intention checkpoint",
        },
        "new_visual_test_runs_required": selected != "sensor_baseline",
        "complete": complete,
        "errors": errors,
        "results": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Final visual comparison complete: {complete}; selected: {selected}")
    print(f"Report: {output_dir}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
