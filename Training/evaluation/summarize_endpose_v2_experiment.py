#!/usr/bin/env python3
"""Compare improved endpose-v2 with endpose-v1 and the frozen t+1 baseline."""

from __future__ import annotations

import argparse
import json
import math
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
MODELS = ("t_plus_1_as_terminal", "terminal_endpose_v1", "terminal_endpose_v2")
LABELS = {
    "t_plus_1_as_terminal": "t+1 as terminal baseline",
    "terminal_endpose_v1": "Terminal endpose v1",
    "terminal_endpose_v2": "Improved dual-horizon endpose v2",
}
TIME_BINS = ("0-0.5s", "0.5-1s", "1-2s", "2-3s", ">=3s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="residual_v2_endpose_v2")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def metric_row(model: str, seed: int, report: dict, artifact: Path) -> dict:
    intention = report["test"]["best_intention"]
    pose = report["test"]["best_pose"]
    coverage = pose["pose_coverage"]
    target_windows = int(coverage.get("pose_targets", coverage["future_targets"]))
    handover_windows = int(intention["receiving_hand"]["samples"])
    auxiliary = pose.get("auxiliary_t_plus_1", {}).get("pose_oracle", {})
    return {
        "model": model,
        "model_label": LABELS[model],
        "seed": seed,
        "artifact": portable(artifact),
        "test_intention_macro_f1": float(intention["intention"]["macro_f1"]),
        "test_receiving_hand_macro_f1": float(
            intention["receiving_hand"]["macro_f1_supported"]
        ),
        "test_terminal_position_error_cm": float(
            pose["pose_oracle"]["position_mae_cm"]
        ),
        "test_terminal_orientation_error_deg": float(
            pose["pose_oracle"]["orientation_mean_deg"]
        ),
        "test_terminal_end_to_end_position_error_cm": float(
            pose["pose_end_to_end"]["position_mae_cm"]
        ),
        "test_target_windows": target_windows,
        "test_handover_windows": handover_windows,
        "test_target_window_coverage": target_windows / handover_windows,
        "auxiliary_t1_position_error_cm": (
            float(auxiliary["position_mae_cm"])
            if auxiliary.get("position_mae_cm") is not None
            else None
        ),
        "auxiliary_t1_orientation_error_deg": (
            float(auxiliary["orientation_mean_deg"])
            if auxiliary.get("orientation_mean_deg") is not None
            else None
        ),
        "trainable_parameters": int(report["trainable_parameters"]),
        "best_intention_epoch": int(report["checkpoints"]["best_intention"]["epoch"]),
        "best_pose_epoch": int(report["checkpoints"]["best_pose"]["epoch"]),
    }


def time_rows(model: str, seed: int, report: dict) -> list[dict]:
    groups = report["test"]["best_pose"]["pose_by_time_to_sequence_end"]
    return [
        {
            "model": model,
            "seed": seed,
            "time_to_sequence_end_bin": group,
            "samples": int(groups[group]["pose_oracle"]["samples"]),
            "position_error_cm": float(
                groups[group]["pose_oracle"]["position_mae_cm"]
            ),
            "orientation_error_deg": float(
                groups[group]["pose_oracle"]["orientation_mean_deg"]
            ),
        }
        for group in TIME_BINS
    ]


def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {*group_columns, "seed", "artifact", "model_label"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["completed_seeds"] = int(group["seed"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def load_latency(report_dir: Path, v1_report: Path) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    rows = []
    old = pd.read_csv(v1_report / "data/latency.csv")
    for _, row in old.iterrows():
        model = str(row["model"])
        if model == "terminal_endpose":
            model = "terminal_endpose_v1"
        rows.append({**row.to_dict(), "model": model})
    for device in ("cpu", "cuda"):
        path = report_dir / "latency/endpose_v2" / f"tcml_{device}.json"
        try:
            report = read(path)
            if report.get("status") != "completed":
                raise ValueError("latency benchmark not completed")
            rows.append(
                {
                    "model": "terminal_endpose_v2",
                    "device": device,
                    "model_forward_mean_ms": float(report["model_forward"]["mean_ms"]),
                    "model_forward_median_ms": float(
                        report["model_forward"]["median_ms"]
                    ),
                    "model_forward_p95_ms": float(report["model_forward"]["p95_ms"]),
                    "offline_window_mean_ms": float(report["offline_window"]["mean_ms"]),
                    "trainable_parameters": int(
                        report["model"]["trainable_parameters"]
                    ),
                    "report": portable(path),
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"latency {device}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), errors


def save(figure: plt.Figure, figures: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_comparison(summary: pd.DataFrame, figures: Path) -> None:
    indexed = summary.set_index("model").loc[list(MODELS)]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    specifications = (
        ("test_intention_macro_f1", "Intent macro-F1", False),
        ("test_receiving_hand_macro_f1", "Receiving-hand macro-F1", False),
        ("test_terminal_position_error_cm", "Terminal position error (cm)", True),
        (
            "test_terminal_orientation_error_deg",
            "Terminal orientation error (degrees)",
            True,
        ),
    )
    colors = ("#9C755F", "#4E79A7", "#59A14F")
    for axis, (metric, title, lower) in zip(axes.flat, specifications):
        values = indexed[f"{metric}_mean"].to_numpy()
        errors = indexed[f"{metric}_std"].to_numpy()
        axis.bar(range(3), values, yerr=errors, capsize=4, color=colors)
        axis.set_xticks(range(3), ["t+1", "Endpose v1", "Endpose v2"])
        axis.set_title(title + (" (lower is better)" if lower else ""))
        axis.grid(axis="y", alpha=0.25)
    save(figure, figures, "01_model_comparison")


def plot_time(curves: pd.DataFrame, figures: Path) -> None:
    indexed = curves.set_index(["model", "time_to_sequence_end_bin"])
    x = np.arange(len(TIME_BINS))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ("#9C755F", "#4E79A7", "#59A14F")
    for model, color in zip(MODELS, colors):
        subset = indexed.loc[model].loc[list(TIME_BINS)]
        axes[0].errorbar(
            x,
            subset["position_error_cm_mean"],
            yerr=subset["position_error_cm_std"],
            marker="o",
            capsize=3,
            label=LABELS[model],
            color=color,
        )
        axes[1].errorbar(
            x,
            subset["orientation_error_deg_mean"],
            yerr=subset["orientation_error_deg_std"],
            marker="o",
            capsize=3,
            label=LABELS[model],
            color=color,
        )
    for axis, ylabel in zip(
        axes, ("Position error (cm)", "Orientation error (degrees)")
    ):
        axis.set_xticks(x, TIME_BINS)
        axis.set_xlabel("Time remaining to sequence end")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(figure, figures, "02_error_vs_time_remaining")


def plot_latency(latency: pd.DataFrame, figures: Path) -> None:
    pivot = latency.pivot(index="model", columns="device", values="model_forward_mean_ms")
    pivot = pivot.loc[list(MODELS)]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    width = 0.35
    axis.bar(x - width / 2, pivot["cpu"], width, label="TCML CPU")
    axis.bar(x + width / 2, pivot["cuda"], width, label="TCML CUDA")
    axis.set_xticks(x, ["t+1", "Endpose v1", "Endpose v2"])
    axis.set_ylabel("Mean model-forward latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(figure, figures, "03_latency")


def safe_json(value):
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    latency: pd.DataFrame,
    audit: dict,
) -> None:
    indexed = summary.set_index("model")
    v1 = indexed.loc["terminal_endpose_v1"]
    v2 = indexed.loc["terminal_endpose_v2"]
    position_delta = (
        v2["test_terminal_position_error_cm_mean"]
        - v1["test_terminal_position_error_cm_mean"]
    )
    orientation_delta = (
        v2["test_terminal_orientation_error_deg_mean"]
        - v1["test_terminal_orientation_error_deg_mean"]
    )
    better = position_delta < 0
    lines = [
        "# Improved terminal end-pose experiment v2 (n214)",
        "",
        "Status: **complete**.",
        "",
        "Endpose-v2 keeps the exact robust terminal target and participant split from "
        "v1, but adds train-fitted position scaling, geodesic orientation loss, "
        "sequence/time-bin balancing, hand-specific residuals, and an auxiliary t+1 head. "
        "Hyperparameters and checkpoints were selected using validation only.",
        "",
        f"Target audit: **{audit['accepted_handover_sequences']}/{audit['handover_sequences']} "
        f"({audit['target_sequence_coverage']:.1%})** stable handover sequences.",
        "",
        "| Model | Intent F1 | Hand F1 | Position (cm) | Orientation (deg) | Coverage | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = indexed.loc[model]
        lines.append(
            f"| {LABELS[model]} | "
            f"{row['test_intention_macro_f1_mean']:.3f} ± {row['test_intention_macro_f1_std']:.3f} | "
            f"{row['test_receiving_hand_macro_f1_mean']:.3f} ± {row['test_receiving_hand_macro_f1_std']:.3f} | "
            f"{row['test_terminal_position_error_cm_mean']:.2f} ± {row['test_terminal_position_error_cm_std']:.2f} | "
            f"{row['test_terminal_orientation_error_deg_mean']:.2f} ± {row['test_terminal_orientation_error_deg_std']:.2f} | "
            f"{row['test_target_window_coverage_mean']:.1%} | "
            f"{int(round(row['trainable_parameters_mean'])):,} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"Endpose-v2 is **{'better' if better else 'not better'}** than endpose-v1 "
            f"on aggregate terminal position: change {position_delta:+.2f} cm; "
            f"orientation change {orientation_delta:+.2f}°. Negative changes are improvements.",
            "",
            "The full remaining-time curves are in `data/error_vs_time_remaining.csv` "
            "and `figures/02_error_vs_time_remaining.{png,pdf}`.",
            "",
            "## Latency",
            "",
            "| Model | TCML CPU mean (ms) | TCML CUDA mean (ms) |",
            "|---|---:|---:|",
        ]
    )
    latency_index = latency.set_index(["model", "device"])
    for model in MODELS:
        lines.append(
            f"| {LABELS[model]} | "
            f"{latency_index.loc[(model, 'cpu'), 'model_forward_mean_ms']:.3f} | "
            f"{latency_index.loc[(model, 'cuda'), 'model_forward_mean_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All test values are mean ± population standard deviation over seeds 42, 43 and 44.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    report = PROJECT_ROOT / "Training/reports" / args.dataset_tag / args.experiment_tag
    v1_report = (
        PROJECT_ROOT
        / "Training/reports"
        / args.dataset_tag
        / "residual_v2_endpose_v1"
    )
    data_dir = report / "data"
    figures = report / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    errors = []
    rows = []
    time_data = []
    for seed in SEEDS:
        paths = {
            "t_plus_1_as_terminal": v1_report
            / "baseline_terminal_eval"
            / f"seed{seed}.json",
            "terminal_endpose_v1": runs_root
            / "residual_v2_endpose_v1"
            / "residual_v2_endpose"
            / f"residual_v2_endpose_v1_residual_v2_endpose_seed{seed}"
            / "metrics.json",
            "terminal_endpose_v2": runs_root
            / args.experiment_tag
            / "residual_v2_endpose_v2"
            / f"{args.experiment_tag}_dual_horizon_endpose_v2_seed{seed}"
            / "metrics.json",
        }
        for model, path in paths.items():
            try:
                value = read(path)
                rows.append(metric_row(model, seed, value, path))
                time_data.extend(time_rows(model, seed, value))
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{model} seed {seed}: {type(exc).__name__}: {exc}")
    runs = pd.DataFrame(rows)
    times = pd.DataFrame(time_data)
    summary = aggregate(runs, ["model"])
    time_summary = aggregate(times, ["model", "time_to_sequence_end_bin"])
    runs.to_csv(data_dir / "model_runs.csv", index=False)
    summary.to_csv(data_dir / "model_summary.csv", index=False)
    times.to_csv(data_dir / "error_vs_time_remaining_by_seed.csv", index=False)
    time_summary.to_csv(data_dir / "error_vs_time_remaining.csv", index=False)
    latency, latency_errors = load_latency(report, v1_report)
    errors.extend(latency_errors)
    latency.to_csv(data_dir / "latency.csv", index=False)
    audit = read(report / "audit/endpose_target_audit.json")
    confirmation = read(
        PROJECT_ROOT
        / "Training/reports"
        / args.dataset_tag
        / "residual_v2_endpose_v2_hp_confirm"
        / "summary.json"
    )
    complete = (
        not errors
        and len(runs) == len(MODELS) * len(SEEDS)
        and set(summary.get("model", [])) == set(MODELS)
        and len(latency) == len(MODELS) * 2
        and audit.get("training_authorized_by_audit") is True
        and confirmation.get("complete") is True
        and confirmation.get("test_metrics_used") is False
    )
    if complete:
        plot_comparison(summary, figures)
        plot_time(time_summary, figures)
        plot_latency(latency, figures)
        write_markdown(report / "README.md", summary, time_summary, latency, audit)
    payload = {
        "schema_version": 1,
        "complete": complete,
        "errors": errors,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "seeds": list(SEEDS),
        "target_definition": "robust terminal receiving-hand pose from the latest stable 0.5-second segment after THIRD",
        "improvements": [
            "train-fitted normalized position loss",
            "geodesic quaternion loss",
            "sequence-balanced sampling",
            "sequence/time-bin-balanced terminal pose weighting",
            "hand-specific residuals",
            "auxiliary t+1 pose head",
        ],
        "selection": {
            "split": "validation",
            "test_metrics_used": False,
            "confirmation_summary": confirmation,
        },
        "target_audit": audit,
        "run_results": runs.to_dict(orient="records"),
        "aggregate_results": summary.to_dict(orient="records"),
        "latency": latency.to_dict(orient="records"),
    }
    (report / "comparison.json").write_text(
        json.dumps(safe_json(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    print(f"Endpose-v2 comparison complete={complete}; report={report}")
    for error in errors:
        print(f"ERROR: {error}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
