#!/usr/bin/env python3
"""Aggregate Stage-B validation runs and freeze the tuned configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
F1_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--stage-a-tag", default="residual_v2_hp_search_v1")
    parser.add_argument("--experiment-tag", default="residual_v2_hp_confirm_v1")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def nested(data: dict, *keys: str):
    value = data
    for key in keys:
        value = value[key]
    return value


def main() -> int:
    args = parse_args()
    stage_a_report = (
        PROJECT_ROOT
        / "Training/reports"
        / args.dataset_tag
        / args.stage_a_tag
    )
    stage_a = json.loads(
        (stage_a_report / "summary.json").read_text(encoding="utf-8")
    )
    trials = list(stage_a["selected_for_confirmation"])
    if len(trials) != 3:
        raise ValueError(f"Expected three Stage-B trials, got {trials}")
    runs_root = (
        PROJECT_ROOT
        / "Training/runs"
        / args.dataset_tag
        / args.experiment_tag
    )
    output_dir = (
        PROJECT_ROOT
        / "Training/reports"
        / args.dataset_tag
        / args.experiment_tag
    )
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in trials:
        for seed in SEEDS:
            run_dir = (
                runs_root
                / trial
                / f"{args.experiment_tag}_{trial}_seed{seed}"
            )
            row = {
                "trial_tag": trial,
                "seed": seed,
                "run_dir": str(run_dir),
                "status": "missing",
                "error": "",
            }
            try:
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(encoding="utf-8")
                )
                if metrics.get("test_evaluation_skipped") is not True:
                    raise ValueError("test evaluation was not disabled")
                if "test" in metrics:
                    raise ValueError("test metrics found in Stage B")
                validation = metrics["validation_by_checkpoint"]["best_intention"]
                row.update(
                    {
                        "status": "completed",
                        "validation_intention_macro_f1": float(
                            nested(validation, "intention", "macro_f1")
                        ),
                        "validation_receiving_hand_macro_f1": float(
                            nested(
                                validation,
                                "receiving_hand",
                                "macro_f1_supported",
                            )
                        ),
                        "validation_pose_mae_cm": float(
                            nested(validation, "pose_oracle", "position_mae_cm")
                        ),
                        "trainable_parameters": int(
                            metrics["trainable_parameters"]
                        ),
                        "wall_seconds": float(metrics["runtime"]["wall_seconds"]),
                    }
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                row["status"] = "error" if run_dir.exists() else "missing"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "confirmation_runs.csv", index=False)
    completed = frame.loc[frame["status"] == "completed"].copy()
    summary_rows = []
    for trial, group in completed.groupby("trial_tag"):
        row = {"trial_tag": trial, "completed_seeds": int(len(group))}
        for column in (
            "validation_intention_macro_f1",
            "validation_receiving_hand_macro_f1",
            "validation_pose_mae_cm",
            "trainable_parameters",
            "wall_seconds",
        ):
            values = group[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)
    aggregate = pd.DataFrame(summary_rows)
    complete = (
        len(aggregate) == 3
        and bool((aggregate["completed_seeds"] == len(SEEDS)).all())
    )
    if not aggregate.empty:
        best_f1 = float(
            aggregate["validation_intention_macro_f1_mean"].max()
        )
        aggregate["within_f1_tolerance"] = (
            aggregate["validation_intention_macro_f1_mean"]
            >= best_f1 - F1_TOLERANCE
        )
        aggregate = pd.concat(
            [
                aggregate.loc[aggregate["within_f1_tolerance"]].sort_values(
                    [
                        "validation_pose_mae_cm_mean",
                        "validation_receiving_hand_macro_f1_mean",
                        "trainable_parameters_mean",
                    ],
                    ascending=[True, False, True],
                    kind="stable",
                ),
                aggregate.loc[~aggregate["within_f1_tolerance"]].sort_values(
                    "validation_intention_macro_f1_mean",
                    ascending=False,
                    kind="stable",
                ),
            ],
            ignore_index=True,
        )
        aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))
    aggregate.to_csv(data_dir / "confirmation_summary.csv", index=False)
    winner = None if aggregate.empty else str(aggregate.iloc[0]["trial_tag"])
    summary = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "selection_split": "validation",
        "test_metrics_forbidden": True,
        "f1_tolerance": F1_TOLERANCE,
        "stage_a_selected_trials": trials,
        "seeds": list(SEEDS),
        "complete": complete,
        "selected_trial": winner,
        "selection_rule": (
            "maximize mean validation intention macro-F1; retain trials within "
            "0.005; minimize mean validation pose MAE; maximize mean hand F1; "
            "minimize parameters"
        ),
        "selected_metrics": (
            None if aggregate.empty else aggregate.iloc[0].to_dict()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if complete and winner is not None:
        source = (
            PROJECT_ROOT
            / "Training/configs/hyperparameter_search_v1"
            / f"{winner}.json"
        )
        selected_config = json.loads(source.read_text(encoding="utf-8"))
        selected_config["run_name"] = "residual_v2_tuned"
        selected_config["hyperparameter_selection"] = {
            "source_trial": winner,
            "stage_a_experiment": args.stage_a_tag,
            "confirmation_experiment": args.experiment_tag,
            "selection_split": "validation",
            "confirmation_seeds": list(SEEDS),
            "test_evaluation_used_for_selection": False,
            "selection_rule": summary["selection_rule"],
        }
        (output_dir / "selected_config.json").write_text(
            json.dumps(selected_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Confirmation complete: {complete}; selected: {winner}")
    print(f"Report: {output_dir}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
