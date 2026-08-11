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

from checkpoint_semantics import (
    mark_seed_aggregate,
    pose_selected_diagnostic,
    primary_result_row,
    select_primary_checkpoint_row,
)


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
FIGURE_CONTEXT = ""


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
    parser.add_argument(
        "--baseline-comparison-json",
        type=Path,
        default=None,
        help=(
            "Optional older compare_final_runs JSON used for a direct "
            "old-vs-new benchmark figure."
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

    primary = primary_result_row(
        metrics, checkpoint="best_intention", split="test"
    )
    validation_primary = primary_result_row(
        metrics, checkpoint="best_intention", split="validation"
    )
    diagnostic = pose_selected_diagnostic(metrics, split="test")

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
        "validation_intention_macro_f1": validation_primary[
            "validation_intention_macro_f1"
        ],
        "validation_receiving_hand_macro_f1": validation_primary[
            "validation_receiving_hand_macro_f1"
        ],
        "validation_pose_mae_cm": validation_primary[
            "validation_pose_mae_cm"
        ],
        **primary,
        **diagnostic,
    }
    # Compatibility aliases now point to the same primary checkpoint.
    summary["test_pose_position_mae_cm"] = summary["test_pose_mae_cm"]
    summary["test_pose_orientation_mean_deg"] = summary[
        "test_pose_orientation_error_deg"
    ]
    return rows, summary


def selected_intention_test(metrics: dict) -> dict:
    if metrics["model_type"] == "hierarchical_residual_pose_transformer_v2":
        return metrics["test"]["best_intention"]
    return metrics.get("test_by_checkpoint", {}).get(
        "best_intention",
        metrics["test"],
    )


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
    if FIGURE_CONTEXT:
        figure.text(
            0.5,
            0.005,
            FIGURE_CONTEXT,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )
    figure.tight_layout(
        rect=(0.0, 0.035, 1.0, 0.95)
        if figure._suptitle is not None
        else (0.0, 0.035, 1.0, 1.0)
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


def metric_mean_std(
    summary: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    standard_deviations = []
    for model_type in MODEL_ORDER:
        values = summary.loc[
            summary["model_type"] == model_type,
            metric,
        ].dropna().to_numpy(dtype=float)
        means.append(float(np.mean(values)))
        standard_deviations.append(float(np.std(values, ddof=0)))
    return np.asarray(means), np.asarray(standard_deviations)


def plot_metric_bars(
    summary: pd.DataFrame,
    figures_dir: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
    lower_is_better: bool = False,
) -> None:
    means, stds = metric_mean_std(summary, metric)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    labels = [MODEL_LABELS[model_type] for model_type in MODEL_ORDER]
    colors = [MODEL_COLORS[model_type] for model_type in MODEL_ORDER]
    positions = np.arange(len(labels))
    bars = axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.9,
    )
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if not lower_is_better:
        axis.set_ylim(max(0.0, float(np.nanmin(means - stds)) - 0.05), 1.0)
    for bar, mean, std in zip(bars, means, stds):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(figure, figures_dir, stem)


def plot_accuracy_vs_macro_f1(
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    accuracy, accuracy_std = metric_mean_std(
        summary,
        "test_intention_accuracy",
    )
    macro_f1, macro_f1_std = metric_mean_std(
        summary,
        "test_intention_macro_f1",
    )
    labels = [MODEL_LABELS[model_type] for model_type in MODEL_ORDER]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.7))
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        yerr=accuracy_std,
        capsize=4,
        label="Accuracy",
        color="#4C78A8",
    )
    axis.bar(
        positions + width / 2,
        macro_f1,
        width,
        yerr=macro_f1_std,
        capsize=4,
        label="Macro-F1",
        color="#F58518",
    )
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylim(0.75, 0.95)
    axis.set_ylabel("Test score")
    axis.set_title(
        "Test intention accuracy and macro-F1\n"
        "(macro-F1 is primary because the classes are imbalanced)"
    )
    axis.legend()
    save_figure(
        figure,
        figures_dir,
        "07_test_intention_accuracy_vs_macro_f1",
    )


def aggregate_confusion(
    discovered: list[tuple[Path, int, dict]],
    model_type: str,
    metric_name: str,
) -> tuple[list[str], np.ndarray]:
    matrices = []
    class_names: list[str] | None = None
    for _, _, metrics in discovered:
        if metrics["model_type"] != model_type:
            continue
        metric = selected_intention_test(metrics).get(metric_name)
        if not isinstance(metric, dict) or not metric.get("confusion_matrix"):
            continue
        names = [str(value) for value in metric["class_names"]]
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(
                f"Inconsistent {metric_name} class order for {model_type}"
            )
        matrices.append(np.asarray(metric["confusion_matrix"], dtype=float))
    if not matrices or class_names is None:
        return [], np.empty((0, 0), dtype=float)
    return class_names, np.sum(matrices, axis=0)


def normalized_confusion(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums > 0,
    )


def draw_confusion(
    axis: plt.Axes,
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    normalized = normalized_confusion(matrix)
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value * 100:.1f}%\n(n={int(matrix[row, column])})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "black",
            )
    return image


def plot_intention_confusions(
    discovered: list[tuple[Path, int, dict]],
    figures_dir: Path,
    data_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    records = []
    for model_type, axis in zip(MODEL_ORDER, axes.flat):
        names, matrix = aggregate_confusion(
            discovered,
            model_type,
            "intention",
        )
        draw_confusion(
            axis,
            matrix,
            names,
            MODEL_LABELS[model_type],
        )
        normalized = normalized_confusion(matrix)
        for row, ground_truth in enumerate(names):
            for column, prediction in enumerate(names):
                records.append(
                    {
                        "model_type": model_type,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "count_across_seeds": int(matrix[row, column]),
                        "row_fraction": float(normalized[row, column]),
                    }
                )
    figure.suptitle(
        "Test intention confusion matrices aggregated over three seeds",
        fontsize=14,
    )
    pd.DataFrame(records).to_csv(
        data_dir / "test_intention_confusion_matrices.csv",
        index=False,
    )
    save_figure(
        figure,
        figures_dir,
        "08_test_intention_confusion_matrices",
    )


def per_class_metric_rows(
    discovered: list[tuple[Path, int, dict]],
) -> pd.DataFrame:
    rows = []
    for run_dir, seed, metrics in discovered:
        metric = selected_intention_test(metrics)["intention"]
        confusion = np.asarray(metric["confusion_matrix"], dtype=float)
        true_positive = np.diag(confusion)
        predicted = confusion.sum(axis=0)
        actual = confusion.sum(axis=1)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            actual,
            out=np.zeros_like(true_positive),
            where=actual > 0,
        )
        for class_index, class_name in enumerate(metric["class_names"]):
            rows.append(
                {
                    "run_name": run_dir.name,
                    "model_type": metrics["model_type"],
                    "model": MODEL_LABELS[metrics["model_type"]],
                    "seed": seed,
                    "class_name": class_name,
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(metric["per_class_f1"][class_index]),
                    "support": int(metric["support"][class_index]),
                }
            )
    return pd.DataFrame(rows)


def plot_per_class_f1(
    discovered: list[tuple[Path, int, dict]],
    figures_dir: Path,
    data_dir: Path,
) -> None:
    rows = per_class_metric_rows(discovered)
    rows.to_csv(data_dir / "test_intention_per_class_metrics.csv", index=False)
    rows[[
        "run_name",
        "model_type",
        "model",
        "seed",
        "class_name",
        "f1",
        "support",
    ]].to_csv(data_dir / "test_intention_per_class_f1.csv", index=False)
    class_names = ["continue", "fetch", "handover"]
    positions = np.arange(len(MODEL_ORDER))
    width = 0.24
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for axis, metric_name in zip(axes, ("precision", "recall", "f1")):
        for class_index, (class_name, color) in enumerate(
            zip(class_names, colors)
        ):
            means = []
            stds = []
            for model_type in MODEL_ORDER:
                values = rows.loc[
                    (rows["model_type"] == model_type)
                    & (rows["class_name"] == class_name),
                    metric_name,
                ].to_numpy(dtype=float)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=0)))
            offset = (class_index - 1) * width
            axis.bar(
                positions + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=class_name,
                color=color,
            )
        axis.set_xticks(
            positions,
            [MODEL_LABELS[model_type] for model_type in MODEL_ORDER],
            rotation=18,
            ha="right",
        )
        axis.set_ylim(0.6, 1.0)
        axis.set_title(metric_name.capitalize())
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test score")
    axes[-1].legend(title="Intention class", loc="lower right")
    support = (
        rows.groupby("class_name", sort=False)["support"]
        .first()
        .reindex(class_names)
    )
    figure.suptitle(
        "Test intention metrics by class (mean ± SD over seeds)\n"
        + "; ".join(
            f"support {class_name}: n={int(value)} per seed"
            for class_name, value in support.items()
        ),
        fontsize=14,
    )
    save_figure(
        figure,
        figures_dir,
        "09_test_intention_per_class_f1",
    )


def plot_receiving_hand_confusion(
    discovered: list[tuple[Path, int, dict]],
    figures_dir: Path,
    data_dir: Path,
) -> None:
    model_type = "hierarchical_residual_pose_transformer_v2"
    names, matrix = aggregate_confusion(
        discovered,
        model_type,
        "receiving_hand",
    )
    if not names:
        return
    figure, axis = plt.subplots(figsize=(6.6, 5.8))
    image = draw_confusion(
        axis,
        matrix,
        names,
        "Residual v2 receiving-hand classification",
    )
    figure.colorbar(image, ax=axis, shrink=0.8)
    normalized = normalized_confusion(matrix)
    records = []
    for row, ground_truth in enumerate(names):
        for column, prediction in enumerate(names):
            records.append(
                {
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "count_across_seeds": int(matrix[row, column]),
                    "row_fraction": float(normalized[row, column]),
                }
            )
    pd.DataFrame(records).to_csv(
        data_dir / "test_receiving_hand_confusion_matrix.csv",
        index=False,
    )
    save_figure(
        figure,
        figures_dir,
        "10_test_receiving_hand_confusion_matrix",
    )


def plot_residual_overfitting(
    history: pd.DataFrame,
    figures_dir: Path,
) -> None:
    residual = history[
        history["model_type"]
        == "hierarchical_residual_pose_transformer_v2"
    ]
    common_last_epoch = min(
        int(residual.loc[residual["seed"] == seed, "epoch"].max())
        for seed in EXPECTED_SEEDS
    )
    epochs = np.arange(1, common_last_epoch + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for column, label, color in (
        ("train_intention_macro_f1", "Train", "#4C78A8"),
        ("validation_intention_macro_f1", "Validation", "#F58518"),
    ):
        means = []
        stds = []
        for epoch in epochs:
            values = residual.loc[residual["epoch"] == epoch, column].to_numpy(
                dtype=float
            )
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=0)))
        means_array = np.asarray(means)
        stds_array = np.asarray(stds)
        axes[0].plot(epochs, means_array, label=label, color=color, linewidth=2)
        axes[0].fill_between(
            epochs,
            means_array - stds_array,
            means_array + stds_array,
            color=color,
            alpha=0.18,
        )
    for column, label, color in (
        ("train_loss_total", "Train", "#4C78A8"),
        ("validation_loss_total", "Validation", "#F58518"),
    ):
        means = []
        stds = []
        for epoch in epochs:
            values = residual.loc[residual["epoch"] == epoch, column].to_numpy(
                dtype=float
            )
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=0)))
        means_array = np.asarray(means)
        stds_array = np.asarray(stds)
        axes[1].plot(epochs, means_array, label=label, color=color, linewidth=2)
        axes[1].fill_between(
            epochs,
            means_array - stds_array,
            means_array + stds_array,
            color=color,
            alpha=0.18,
        )
    axes[0].set_title("Residual v2 intention macro-F1")
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_title("Residual v2 total loss")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.xaxis.get_major_locator().set_params(integer=True)
        axis.legend()
    figure.suptitle(
        "Residual v2 generalization gap (mean ± SD over common epochs)",
        fontsize=14,
    )
    save_figure(
        figure,
        figures_dir,
        "11_residual_v2_generalization_gap",
    )


def plot_dataset_comparison(
    baseline_path: Path,
    current_summary: pd.DataFrame,
    current_tag: str,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    baseline = json.loads(
        baseline_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    rows = []
    for model_type in MODEL_ORDER:
        old_metrics = baseline["summary"][model_type]["metrics"]
        current = current_summary[
            current_summary["model_type"] == model_type
        ]
        for dataset_label, intent_mean, intent_std, pose_mean, pose_std in (
            (
                str(baseline.get("tag", "baseline")),
                float(old_metrics["intention_macro_f1"]["mean"]),
                float(old_metrics["intention_macro_f1"]["std"]),
                float(old_metrics["pose_checkpoint_oracle_mae_cm"]["mean"]),
                float(old_metrics["pose_checkpoint_oracle_mae_cm"]["std"]),
            ),
            (
                current_tag,
                float(current["test_intention_macro_f1"].mean()),
                float(current["test_intention_macro_f1"].std(ddof=0)),
                float(current["test_pose_position_mae_cm"].mean()),
                float(current["test_pose_position_mae_cm"].std(ddof=0)),
            ),
        ):
            rows.append(
                {
                    "dataset": dataset_label,
                    "model_type": model_type,
                    "model": MODEL_LABELS[model_type],
                    "intention_macro_f1_mean": intent_mean,
                    "intention_macro_f1_std": intent_std,
                    "pose_mae_cm_mean": pose_mean,
                    "pose_mae_cm_std": pose_std,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(data_dir / "dataset_comparison.csv", index=False)
    datasets = list(dict.fromkeys(comparison["dataset"].tolist()))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    colors = ("#9D9D9D", "#4C78A8")
    for dataset_index, (dataset, color) in enumerate(zip(datasets, colors)):
        subset = comparison[comparison["dataset"] == dataset].set_index(
            "model_type"
        ).loc[list(MODEL_ORDER)]
        offset = (dataset_index - 0.5) * width
        axes[0].bar(
            positions + offset,
            subset["intention_macro_f1_mean"],
            width,
            yerr=subset["intention_macro_f1_std"],
            capsize=3,
            label=dataset,
            color=color,
        )
        axes[1].bar(
            positions + offset,
            subset["pose_mae_cm_mean"],
            width,
            yerr=subset["pose_mae_cm_std"],
            capsize=3,
            label=dataset,
            color=color,
        )
    labels = [MODEL_LABELS[value] for value in MODEL_ORDER]
    axes[0].set_title("Test intention macro-F1")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.75, 0.9)
    axes[1].set_title("Test pose position MAE")
    axes[1].set_ylabel("Centimetres (lower is better)")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.legend()
    figure.suptitle("Old and expanded dataset benchmark", fontsize=14)
    save_figure(
        figure,
        figures_dir,
        "12_dataset_n156_vs_n214",
    )


def main() -> int:
    global FIGURE_CONTEXT
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
    discovered = discover_runs(
        args.runs_dir,
        tag,
        dataset_tag,
    )
    for run_dir, seed, metrics in discovered:
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
    benchmark_summary_records = []
    summary_metrics = (
        "test_intention_accuracy",
        "test_intention_macro_f1",
        "test_assistance_accuracy",
        "test_assistance_macro_f1",
        "test_assistance_type_accuracy",
        "test_assistance_type_macro_f1",
        "test_receiving_hand_accuracy",
        "test_receiving_hand_macro_f1",
        "test_pose_position_mae_cm",
        "test_pose_orientation_mean_deg",
    )
    for model_type in MODEL_ORDER:
        model_rows = summary[summary["model_type"] == model_type]
        record = {
            "model_type": model_type,
            "model": MODEL_LABELS[model_type],
            "seeds": ";".join(str(value) for value in EXPECTED_SEEDS),
        }
        for metric in summary_metrics:
            values = model_rows[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_mean"] = (
                float(np.mean(values)) if len(values) else np.nan
            )
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=0)) if len(values) else np.nan
            )
        benchmark_summary_records.append(mark_seed_aggregate(record))
    pd.DataFrame(benchmark_summary_records).to_csv(
        data_dir / "benchmark_test_summary_mean_std.csv",
        index=False,
    )
    primary_checkpoint_records = []
    for model_type in MODEL_ORDER:
        model_rows = summary.loc[summary["model_type"] == model_type]
        selected = select_primary_checkpoint_row(
            model_rows.to_dict(orient="records")
        )
        selected["model_type"] = model_type
        selected["model"] = MODEL_LABELS[model_type]
        primary_checkpoint_records.append(selected)
    pd.DataFrame(primary_checkpoint_records).to_csv(
        data_dir / "validation_selected_checkpoint_results.csv", index=False
    )

    configure_style()
    FIGURE_CONTEXT = (
        f"dataset={dataset_tag or 'legacy'}; experiment={tag}; "
        "participant-wise split; seeds=42,43,44; error bars=population SD"
    )
    plot_total_loss(history, figures_dir)
    plot_validation_intention_f1(history, figures_dir)
    plot_validation_pose_mae(history, figures_dir)
    plot_mean_validation_f1(aggregate, figures_dir)
    plot_metric_bars(
        summary,
        figures_dir,
        metric="test_intention_macro_f1",
        ylabel="Test intention macro-F1",
        title="Test intention macro-F1 (mean ± SD over seeds)",
        stem="05_test_intention_macro_f1_by_model",
    )
    plot_metric_bars(
        summary,
        figures_dir,
        metric="test_pose_position_mae_cm",
        ylabel="Test position MAE (cm)",
        title="Test pose position MAE (mean ± SD over seeds)",
        stem="06_test_pose_position_mae_by_model",
        lower_is_better=True,
    )
    plot_accuracy_vs_macro_f1(summary, figures_dir)
    plot_intention_confusions(discovered, figures_dir, data_dir)
    plot_per_class_f1(discovered, figures_dir, data_dir)
    plot_receiving_hand_confusion(discovered, figures_dir, data_dir)
    plot_residual_overfitting(history, figures_dir)
    if args.baseline_comparison_json is not None:
        plot_dataset_comparison(
            args.baseline_comparison_json,
            summary,
            tag,
            figures_dir,
            data_dir,
        )

    manifest = {
        "tag": tag,
        "dataset_tag": dataset_tag,
        "runs_dir": display_path(args.runs_dir),
        "run_count": len(summary),
        "primary_result_file": "data/validation_selected_checkpoint_results.csv",
        "seed_aggregate_file": "data/benchmark_test_summary_mean_std.csv",
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
        "baseline_comparison_json": (
            display_path(args.baseline_comparison_json)
            if args.baseline_comparison_json is not None
            else None
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
