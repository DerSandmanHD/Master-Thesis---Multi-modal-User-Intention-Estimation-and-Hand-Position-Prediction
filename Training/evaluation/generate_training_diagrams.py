#!/usr/bin/env python3
"""Generate reproducible learning-curve diagrams from a seeded benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
TRAINING_ROOT = PROJECT_ROOT / "Training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from run_layout import (
    experiment_report_directory,
    validate_run_context,
    validate_tag,
)


DEFAULT_RUNS_DIR = PROJECT_ROOT / "Training" / "runs"
EXPECTED_SEEDS = (42, 43, 44)
MODEL_ORDER = (
    "hierarchical_gated_multimodal_transformer",
    "hierarchical_window_mlp",
    "hierarchical_gru",
    "hierarchical_residual_pose_transformer_v2",
)
MODEL_LABELS = {
    "hierarchical_gated_multimodal_transformer": "Transformer v1",
    "hierarchical_window_mlp": "MLP",
    "hierarchical_gru": "GRU",
    "hierarchical_residual_pose_transformer_v2": "Residual Transformer v2",
}
MODEL_COLORS = {
    "hierarchical_gated_multimodal_transformer": "#4C78A8",
    "hierarchical_window_mlp": "#F58518",
    "hierarchical_gru": "#54A24B",
    "hierarchical_residual_pose_transformer_v2": "#B279A2",
}
SEED_COLORS = {42: "#4C78A8", 43: "#F58518", 44: "#54A24B"}
LOSS_NAMES = (
    "total",
    "assistance",
    "assistance_type",
    "receiving_hand",
    "position",
    "orientation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="final_clean_v1")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=(
            "Directory containing the tagged benchmark run directories. "
            "Nested directories are searched recursively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated CSV, PNG and PDF files. Future structured "
            "runs default to Training/reports/<dataset>/<tag>/training_diagrams."
        ),
    )
    return parser.parse_args()


def nested_value(data: dict, *keys: str) -> float:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return float("nan")
        current = current[key]
    return float("nan") if current is None else float(current)


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def discover_runs(
    runs_dir: Path,
    tag: str = "final_clean_v1",
    dataset_tag: str | None = None,
) -> list[tuple[Path, int, dict]]:
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    discovered: list[tuple[Path, int, dict]] = []
    seen: set[tuple[str, int]] = set()
    run_pattern = re.compile(rf"^{re.escape(tag)}_(.+)_seed(\d+)$")
    for metrics_path in sorted(runs_dir.rglob(f"{tag}_*_seed*/metrics.json")):
        run_dir = metrics_path.parent
        match = run_pattern.fullmatch(run_dir.name)
        if match is None:
            continue
        config_path = run_dir / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            validate_run_context(
                config,
                dataset_tag=dataset_tag,
                experiment_tag=tag,
                source=str(run_dir),
            )
        elif dataset_tag is not None:
            raise FileNotFoundError(
                f"Structured run is missing config.json: {run_dir}"
            )
        seed = int(match.group(2))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_type = str(metrics.get("model_type", ""))
        key = (model_type, seed)
        if key in seen:
            raise ValueError(
                f"Duplicate final run for model/seed {key}: {run_dir}"
            )
        seen.add(key)
        discovered.append((run_dir, seed, metrics))

    expected = {
        (model_type, seed)
        for model_type in MODEL_ORDER
        for seed in EXPECTED_SEEDS
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise ValueError(
            "Final-run set is incomplete or unexpected. "
            f"Missing={missing}; unexpected={unexpected}"
        )
    if len(discovered) != 12:
        raise ValueError(f"Expected 12 final runs, found {len(discovered)}")
    return discovered


def history_rows(
    run_dir: Path,
    seed: int,
    metrics: dict,
) -> tuple[list[dict], dict]:
    model_type = str(metrics["model_type"])
    model_label = MODEL_LABELS[model_type]
    checkpoints = metrics["checkpoints"]
    best_intention_epoch = int(checkpoints["best_intention"]["epoch"])
    best_pose_epoch = int(checkpoints["best_pose"]["epoch"])
    rows: list[dict] = []

    for record in metrics["history"]:
        train = record["train"]
        validation = record["validation"]
        pose_key = (
            "pose_oracle"
            if model_type == "hierarchical_residual_pose_transformer_v2"
            else "pose"
        )
        row = {
            "run_name": run_dir.name,
            "model_type": model_type,
            "model": model_label,
            "seed": seed,
            "epoch": int(record["epoch"]),
            "best_intention_epoch": best_intention_epoch,
            "best_pose_epoch": best_pose_epoch,
            "train_intention_macro_f1": nested_value(
                train, "intention", "macro_f1"
            ),
            "validation_intention_macro_f1": nested_value(
                validation, "intention", "macro_f1"
            ),
            "train_assistance_macro_f1": nested_value(
                train, "assistance", "macro_f1"
            ),
            "validation_assistance_macro_f1": nested_value(
                validation, "assistance", "macro_f1"
            ),
            "validation_pose_position_mae_cm": nested_value(
                validation, pose_key, "position_mae_cm"
            ),
            "validation_pose_orientation_mean_deg": nested_value(
                validation, pose_key, "orientation_mean_deg"
            ),
        }
        for loss_name in LOSS_NAMES:
            row[f"train_loss_{loss_name}"] = nested_value(
                train, "loss", loss_name
            )
            row[f"validation_loss_{loss_name}"] = nested_value(
                validation, "loss", loss_name
            )
        rows.append(row)

    if model_type == "hierarchical_residual_pose_transformer_v2":
        intention_test = metrics["test"]["best_intention"]
        pose_test = metrics["test"]["best_pose"]["pose_oracle"]
    else:
        intention_test = metrics["test"]
        pose_test = metrics["test_by_checkpoint"]["best_pose"]["pose"]

    summary = {
        "run_name": run_dir.name,
        "model_type": model_type,
        "model": model_label,
        "seed": seed,
        "epochs_trained": len(metrics["history"]),
        "best_intention_epoch": best_intention_epoch,
        "best_validation_intention_macro_f1": float(
            checkpoints["best_intention"]["selection_value"]
        ),
        "best_pose_epoch": best_pose_epoch,
        "best_validation_pose_position_mae_cm": float(
            checkpoints["best_pose"]["selection_value"]
        ),
        "test_intention_macro_f1": nested_value(
            intention_test, "intention", "macro_f1"
        ),
        "test_pose_position_mae_cm": nested_value(
            pose_test, "position_mae_cm"
        ),
    }
    return rows, summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
        if figure._suptitle is not None
        else None
    )
    figure.savefig(figures_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def model_axes(
    ylabel: str,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    figure, raw_axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = {}
    for model_type, axis in zip(MODEL_ORDER, raw_axes.flat):
        axes[model_type] = axis
        axis.set_title(MODEL_LABELS[model_type])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.xaxis.get_major_locator().set_params(integer=True)
    return figure, axes


def plot_total_loss(history: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = model_axes("Total loss")
    for model_type in MODEL_ORDER:
        axis = axes[model_type]
        model_data = history[history["model_type"] == model_type]
        for seed in EXPECTED_SEEDS:
            run = model_data[model_data["seed"] == seed].sort_values("epoch")
            color = SEED_COLORS[seed]
            axis.plot(
                run["epoch"],
                run["train_loss_total"],
                color=color,
                linewidth=1.5,
                label=f"Train, seed {seed}",
            )
            axis.plot(
                run["epoch"],
                run["validation_loss_total"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"Validation, seed {seed}",
            )
        axis.legend(ncol=2)
    figure.suptitle("Training and validation total loss", fontsize=14)
    save_figure(
        figure,
        figures_dir,
        "01_total_loss_train_validation_by_model",
    )


def plot_validation_intention_f1(
    history: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figure, axes = model_axes("Validation intention macro-F1")
    for model_type in MODEL_ORDER:
        axis = axes[model_type]
        model_data = history[history["model_type"] == model_type]
        for seed in EXPECTED_SEEDS:
            run = model_data[model_data["seed"] == seed].sort_values("epoch")
            color = SEED_COLORS[seed]
            axis.plot(
                run["epoch"],
                run["validation_intention_macro_f1"],
                color=color,
                linewidth=1.7,
                label=f"Seed {seed}",
            )
            best_epoch = int(run["best_intention_epoch"].iloc[0])
            best_row = run[run["epoch"] == best_epoch]
            axis.scatter(
                best_row["epoch"],
                best_row["validation_intention_macro_f1"],
                color=color,
                marker="*",
                s=90,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_ylim(0.85, 0.95)
        axis.set_title(MODEL_LABELS[model_type], pad=10)
        axis.legend()
    figure.suptitle(
        "Validation intention macro-F1 (star: selected checkpoint)",
        fontsize=14,
    )
    save_figure(
        figure,
        figures_dir,
        "02_validation_intention_macro_f1_by_model",
    )


def plot_validation_pose_mae(
    history: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figure, axes = model_axes("Validation position MAE (cm)")
    for model_type in MODEL_ORDER:
        axis = axes[model_type]
        model_data = history[history["model_type"] == model_type]
        for seed in EXPECTED_SEEDS:
            run = model_data[model_data["seed"] == seed].sort_values("epoch")
            color = SEED_COLORS[seed]
            axis.plot(
                run["epoch"],
                run["validation_pose_position_mae_cm"],
                color=color,
                linewidth=1.7,
                label=f"Seed {seed}",
            )
            best_epoch = int(run["best_pose_epoch"].iloc[0])
            best_row = run[run["epoch"] == best_epoch]
            axis.scatter(
                best_row["epoch"],
                best_row["validation_pose_position_mae_cm"],
                color=color,
                marker="*",
                s=90,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.legend()
    figure.suptitle(
        "Validation pose position MAE (star: selected checkpoint)",
        fontsize=14,
    )
    save_figure(
        figure,
        figures_dir,
        "03_validation_pose_position_mae_by_model",
    )


def aggregate_history(history: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "train_loss_total",
        "validation_loss_total",
        "train_intention_macro_f1",
        "validation_intention_macro_f1",
        "validation_pose_position_mae_cm",
    )
    rows: list[dict] = []
    for model_type in MODEL_ORDER:
        model_data = history[history["model_type"] == model_type]
        common_last_epoch = min(
            int(model_data[model_data["seed"] == seed]["epoch"].max())
            for seed in EXPECTED_SEEDS
        )
        for epoch in range(1, common_last_epoch + 1):
            epoch_data = model_data[model_data["epoch"] == epoch]
            row = {
                "model_type": model_type,
                "model": MODEL_LABELS[model_type],
                "epoch": epoch,
                "seed_count": int(epoch_data["seed"].nunique()),
            }
            for metric in metrics:
                values = epoch_data[metric].dropna().to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_validation_f1(
    aggregate: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model_type in MODEL_ORDER:
        model_data = aggregate[
            aggregate["model_type"] == model_type
        ].sort_values("epoch")
        epochs = model_data["epoch"].to_numpy(dtype=float)
        mean = model_data[
            "validation_intention_macro_f1_mean"
        ].to_numpy(dtype=float)
        std = model_data[
            "validation_intention_macro_f1_std"
        ].to_numpy(dtype=float)
        color = MODEL_COLORS[model_type]
        axis.plot(
            epochs,
            mean,
            color=color,
            linewidth=2.0,
            label=MODEL_LABELS[model_type],
        )
        axis.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation intention macro-F1")
    axis.set_ylim(0.85, 0.95)
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.legend()
    axis.set_title(
        "Validation intention macro-F1, mean ± standard deviation\n"
        "(three seeds; only epochs available for all seeds of a model)"
    )
    save_figure(
        figure,
        figures_dir,
        "04_validation_intention_macro_f1_mean_std",
    )


def main() -> int:
    args = parse_args()
    tag = validate_tag(args.tag, "tag")
    dataset_tag = (
        validate_tag(args.dataset_tag, "dataset_tag")
        if args.dataset_tag
        else None
    )
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif dataset_tag:
        output_dir = (
            PROJECT_ROOT
            / experiment_report_directory(dataset_tag, tag)
            / "training_diagrams"
        )
    else:
        output_dir = PROJECT_ROOT / "Training" / "evaluation" / "generated"
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_records: list[dict] = []
    summaries: list[dict] = []
    for run_dir, seed, metrics in discover_runs(
        args.runs_dir,
        tag,
        dataset_tag,
    ):
        rows, summary = history_rows(run_dir, seed, metrics)
        history_records.extend(rows)
        summaries.append(summary)

    history = pd.DataFrame(history_records).sort_values(
        ["model_type", "seed", "epoch"]
    )
    summary = pd.DataFrame(summaries).sort_values(["model_type", "seed"])
    aggregate = aggregate_history(history)

    history.to_csv(data_dir / "training_history_by_seed.csv", index=False)
    aggregate.to_csv(data_dir / "training_history_mean_std.csv", index=False)
    summary.to_csv(data_dir / "run_summary.csv", index=False)

    configure_style()
    plot_total_loss(history, figures_dir)
    plot_validation_intention_f1(history, figures_dir)
    plot_validation_pose_mae(history, figures_dir)
    plot_mean_validation_f1(aggregate, figures_dir)

    manifest = {
        "tag": tag,
        "dataset_tag": dataset_tag,
        "runs_dir": display_path(args.runs_dir),
        "run_count": len(summary),
        "models": {
            label: sorted(
                summary.loc[summary["model"] == label, "seed"]
                .astype(int)
                .tolist()
            )
            for label in MODEL_LABELS.values()
        },
        "generated_data": sorted(path.name for path in data_dir.iterdir()),
        "generated_figures": sorted(path.name for path in figures_dir.iterdir()),
        "aggregation_note": (
            "Mean and standard deviation curves include only epochs available "
            "for all three seeds of the respective model."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Validated final runs: {len(summary)}")
    print(f"Evaluation output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
