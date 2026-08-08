#!/usr/bin/env python3
"""Validate, rank and visualize validation-only hyperparameter trials."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "aria_training_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "aria_training_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    "Training/configs/hyperparameter_search_v1/manifest.json"
)
PARAMETERS = (
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "batch_size",
    "orientation_loss_weight",
    "receiving_hand_loss_weight",
)
F1_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument(
        "--experiment-tag", default="residual_v2_hp_search_v1"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def nested(data: dict, *keys: str, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def expected_run_dir(
    runs_dir: Path, experiment_tag: str, trial_tag: str
) -> Path:
    return (
        runs_dir
        / trial_tag
        / f"{experiment_tag}_{trial_tag}_seed42"
    )


def validate_completed_run(
    run_dir: Path,
    *,
    dataset_tag: str,
    experiment_tag: str,
    expected_trial: dict,
) -> dict:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "data_metadata.json"
    for path in (metrics_path, config_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metrics.get("test_evaluation_skipped") is not True:
        raise ValueError("trial did not declare skipped test evaluation")
    if "test" in metrics:
        raise ValueError("validation-only trial unexpectedly contains test metrics")
    context = config.get("run_context", {})
    expected_context = {
        "dataset_tag": dataset_tag,
        "experiment_tag": experiment_tag,
        "model_tag": expected_trial["trial_tag"],
    }
    if context != expected_context:
        raise ValueError(
            f"run context mismatch: {context!r} != {expected_context!r}"
        )
    search = config.get("hyperparameter_search", {})
    if search.get("trial_index") != expected_trial["trial_index"]:
        raise ValueError("trial index mismatch")
    if search.get("selection_split") != "validation":
        raise ValueError("selection split is not validation")
    validation = metrics.get("validation_by_checkpoint", {}).get(
        "best_intention"
    )
    if not isinstance(validation, dict):
        raise ValueError("best-intention validation metrics missing")
    return {
        "config": config,
        "metrics": metrics,
        "metadata": metadata,
        "validation": validation,
    }


def row_for_trial(
    trial: dict,
    run_dir: Path,
    *,
    dataset_tag: str,
    experiment_tag: str,
) -> dict:
    row = {
        "trial_index": int(trial["trial_index"]),
        "trial_tag": trial["trial_tag"],
        "status": "missing",
        "error": "",
        "run_dir": str(run_dir),
        "config_sha256": trial["config_sha256"],
        **{name: trial[name] for name in PARAMETERS},
    }
    if not run_dir.exists():
        return row
    try:
        completed = validate_completed_run(
            run_dir,
            dataset_tag=dataset_tag,
            experiment_tag=experiment_tag,
            expected_trial=trial,
        )
        validation = completed["validation"]
        metrics = completed["metrics"]
        metadata = completed["metadata"]
        row.update(
            {
                "status": "completed",
                "validation_intention_macro_f1": float(
                    nested(validation, "intention", "macro_f1")
                ),
                "validation_intention_accuracy": float(
                    nested(validation, "intention", "accuracy")
                ),
                "validation_receiving_hand_macro_f1": float(
                    nested(
                        validation,
                        "receiving_hand",
                        "macro_f1_supported",
                        default=nested(
                            validation,
                            "receiving_hand",
                            "macro_f1",
                        ),
                    )
                ),
                "validation_pose_mae_cm": float(
                    nested(validation, "pose_oracle", "position_mae_cm")
                ),
                "validation_pose_end_to_end_mae_cm": float(
                    nested(
                        validation,
                        "pose_end_to_end",
                        "position_mae_cm",
                    )
                ),
                "best_intention_epoch": int(
                    nested(metrics, "checkpoints", "best_intention", "epoch")
                ),
                "trainable_parameters": int(
                    metrics["trainable_parameters"]
                ),
                "wall_seconds": float(
                    nested(metrics, "runtime", "wall_seconds", default=np.nan)
                ),
                "git_commit": nested(
                    metrics, "code_provenance", "commit", default=""
                ),
                "git_dirty": nested(
                    metrics, "code_provenance", "dirty", default=None
                ),
                "dataset_content_fingerprint": nested(
                    metadata,
                    "provenance",
                    "dataset_content_fingerprint",
                    default="",
                ),
            }
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def rank_completed(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.loc[frame["status"] == "completed"].copy()
    if completed.empty:
        return completed
    best_f1 = float(completed["validation_intention_macro_f1"].max())
    completed["within_f1_tolerance"] = (
        completed["validation_intention_macro_f1"] >= best_f1 - F1_TOLERANCE
    )
    candidates = completed.loc[completed["within_f1_tolerance"]].sort_values(
        [
            "validation_pose_mae_cm",
            "validation_receiving_hand_macro_f1",
            "trainable_parameters",
        ],
        ascending=[True, False, True],
        kind="stable",
    )
    remaining = completed.loc[~completed["within_f1_tolerance"]].sort_values(
        [
            "validation_intention_macro_f1",
            "validation_pose_mae_cm",
            "validation_receiving_hand_macro_f1",
            "trainable_parameters",
        ],
        ascending=[False, True, False, True],
        kind="stable",
    )
    ranked = pd.concat([candidates, remaining], ignore_index=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["selected_for_confirmation"] = ranked["rank"] <= 3
    return ranked


def save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_pareto(ranked: pd.DataFrame, figures_dir: Path) -> None:
    if ranked.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    sizes = 35 + 90 * (
        ranked["trainable_parameters"]
        / ranked["trainable_parameters"].max()
    )
    scatter = axis.scatter(
        ranked["validation_pose_mae_cm"],
        ranked["validation_intention_macro_f1"],
        c=ranked["validation_receiving_hand_macro_f1"],
        s=sizes,
        cmap="viridis",
        edgecolors=np.where(
            ranked["within_f1_tolerance"], "black", "none"
        ),
        linewidths=1.2,
        alpha=0.85,
    )
    for _, row in ranked.iterrows():
        axis.annotate(
            row["trial_tag"].replace("trial_", "t"),
            (row["validation_pose_mae_cm"], row["validation_intention_macro_f1"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(
        ranked["validation_intention_macro_f1"].max() - F1_TOLERANCE,
        color="#D62728",
        linestyle="--",
        linewidth=1,
        label=f"best F1 - {F1_TOLERANCE:.3f}",
    )
    axis.set_xlabel(
        "Validation pose MAE at best-intention checkpoint (cm; lower is better)"
    )
    axis.set_ylabel("Validation intention macro-F1")
    axis.set_title("Residual-v2 hyperparameter search: validation Pareto view")
    axis.legend(loc="lower left")
    figure.colorbar(scatter, ax=axis, label="Validation receiving-hand macro-F1")
    save_figure(figure, figures_dir, "01_validation_pareto")


def plot_parameter_effects(ranked: pd.DataFrame, figures_dir: Path) -> None:
    if ranked.empty:
        return
    figure, axes = plt.subplots(2, 5, figsize=(18, 8))
    for axis, parameter in zip(axes.flat, PARAMETERS):
        values = ranked[parameter].to_numpy(dtype=float)
        axis.scatter(
            values,
            ranked["validation_intention_macro_f1"],
            c=ranked["validation_pose_mae_cm"],
            cmap="plasma_r",
            s=35,
            alpha=0.8,
        )
        axis.set_title(parameter.replace("_", " "), fontsize=10)
        axis.set_ylabel("val intent macro-F1")
        if parameter == "learning_rate":
            axis.set_xscale("log")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Stage-A hyperparameters versus validation intention macro-F1\n"
        "point color encodes validation pose MAE",
        fontsize=14,
    )
    save_figure(figure, figures_dir, "02_hyperparameter_effects")


def plot_parallel_coordinates(ranked: pd.DataFrame, figures_dir: Path) -> None:
    if ranked.empty:
        return
    columns = list(PARAMETERS) + [
        "validation_intention_macro_f1",
        "validation_pose_mae_cm",
    ]
    values = ranked[columns].astype(float)
    normalized = values.copy()
    for column in columns:
        minimum = float(values[column].min())
        maximum = float(values[column].max())
        normalized[column] = (
            0.5
            if maximum == minimum
            else (values[column] - minimum) / (maximum - minimum)
        )
    figure, axis = plt.subplots(figsize=(16, 7))
    x = np.arange(len(columns))
    color_values = ranked["validation_intention_macro_f1"].to_numpy(dtype=float)
    norm = plt.Normalize(color_values.min(), color_values.max())
    cmap = plt.get_cmap("viridis")
    for row_index in range(len(normalized)):
        axis.plot(
            x,
            normalized.iloc[row_index].to_numpy(dtype=float),
            color=cmap(norm(color_values[row_index])),
            alpha=0.6,
            linewidth=1.2,
        )
    axis.set_xticks(x, [name.replace("_", "\n") for name in columns])
    axis.set_ylabel("Min-max normalized value within Stage A")
    axis.set_title("Hyperparameter search parallel coordinates")
    axis.grid(axis="x", alpha=0.25)
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axis,
        label="Validation intention macro-F1",
    )
    save_figure(figure, figures_dir, "03_parallel_coordinates")


def main() -> int:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs_dir = project_path(
        args.runs_dir
        or Path("Training/runs") / args.dataset_tag / args.experiment_tag
    )
    output_dir = project_path(
        args.output_dir
        or Path("Training/reports") / args.dataset_tag / args.experiment_tag
    )
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in manifest["trials"]:
        run_dir = expected_run_dir(
            runs_dir, args.experiment_tag, trial["trial_tag"]
        )
        rows.append(
            row_for_trial(
                trial,
                run_dir,
                dataset_tag=args.dataset_tag,
                experiment_tag=args.experiment_tag,
            )
        )
    frame = pd.DataFrame(rows).sort_values("trial_index")
    frame.to_csv(data_dir / "stage_a_trials.csv", index=False)
    ranked = rank_completed(frame)
    ranked.to_csv(data_dir / "stage_a_ranking.csv", index=False)

    status_counts = {
        str(key): int(value)
        for key, value in frame["status"].value_counts().items()
    }
    summary = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "manifest": str(manifest_path),
        "runs_dir": str(runs_dir),
        "selection_split": "validation",
        "test_metrics_forbidden": True,
        "f1_tolerance": F1_TOLERANCE,
        "pose_tiebreak_checkpoint": "best_intention",
        "expected_trials": len(frame),
        "status_counts": status_counts,
        "complete": status_counts.get("completed", 0) == len(frame),
        "selected_for_confirmation": (
            ranked.loc[
                ranked["selected_for_confirmation"], "trial_tag"
            ].tolist()
            if not ranked.empty
            else []
        ),
        "winner_if_search_stopped_now": (
            ranked.iloc[0].to_dict() if not ranked.empty else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    plot_pareto(ranked, figures_dir)
    plot_parameter_effects(ranked, figures_dir)
    plot_parallel_coordinates(ranked, figures_dir)

    print(f"Trial status: {status_counts}")
    print(f"Report: {output_dir}")
    if args.require_complete and not summary["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
