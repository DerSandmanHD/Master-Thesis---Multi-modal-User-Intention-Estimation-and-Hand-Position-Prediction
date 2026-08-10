#!/usr/bin/env python3
"""Create the final t+1-versus-terminal-endpose experiment report."""

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
MODEL_ORDER = ("t_plus_1_as_terminal", "terminal_endpose")
MODEL_LABELS = {
    "t_plus_1_as_terminal": "t+1 model\n(as terminal baseline)",
    "terminal_endpose": "Terminal endpose model",
}
TIME_GROUPS = ("0-0.5s", "0.5-1s", "1-2s", "2-3s", ">=3s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="residual_v2_endpose_v1")
    parser.add_argument("--t1-experiment", default="residual_v2_tuned_v1")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def endpose_run_path(root: Path, experiment: str, seed: int) -> Path:
    return (
        root
        / experiment
        / "residual_v2_endpose"
        / f"{experiment}_residual_v2_endpose_seed{seed}"
    )


def portable_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def result_row(model: str, seed: int, report: dict, run_or_report: Path) -> dict:
    test = report["test"]
    intention = test["best_intention"]
    pose = test["best_pose"]
    coverage = pose["pose_coverage"]
    handover_windows = int(intention["receiving_hand"]["samples"])
    target_windows = int(coverage.get("pose_targets", coverage["future_targets"]))
    return {
        "model": model,
        "model_label": MODEL_LABELS[model].replace("\n", " "),
        "seed": seed,
        "artifact": portable_artifact_path(run_or_report),
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
        "test_terminal_end_to_end_orientation_error_deg": float(
            pose["pose_end_to_end"]["orientation_mean_deg"]
        ),
        "test_last_observation_position_error_cm": float(
            pose["last_observation_oracle"]["position_mae_cm"]
        ),
        "test_target_windows": target_windows,
        "test_handover_windows": handover_windows,
        "test_target_window_coverage": (
            target_windows / handover_windows if handover_windows else None
        ),
        "trainable_parameters": int(report["trainable_parameters"]),
        "best_intention_epoch": int(
            report["checkpoints"]["best_intention"].get(
                "epoch", report["checkpoints"]["best_intention"].get("source_epoch")
            )
        ),
        "best_pose_epoch": int(
            report["checkpoints"]["best_pose"].get(
                "epoch", report["checkpoints"]["best_pose"].get("source_epoch")
            )
        ),
    }


def time_rows(model: str, seed: int, report: dict) -> list[dict]:
    groups = report["test"]["best_pose"]["pose_by_time_to_sequence_end"]
    rows = []
    for group in TIME_GROUPS:
        for evaluation in ("pose_oracle", "pose_end_to_end", "last_observation_oracle"):
            metrics = groups[group][evaluation]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "time_to_sequence_end_bin": group,
                    "evaluation": evaluation,
                    "samples": int(metrics["samples"]),
                    "position_error_cm": metric_value(metrics, "position_mae_cm"),
                    "orientation_error_deg": metric_value(
                        metrics, "orientation_mean_deg"
                    ),
                }
            )
    return rows


def native_t1_row(seed: int, report: dict) -> dict:
    native = report["source_native_t_plus_1_test"]
    intention = native["best_intention"]
    pose = native["best_pose"]
    return {
        "seed": seed,
        "target_definition": "receiving-hand pose at t+1 second",
        "test_intention_macro_f1": float(intention["intention"]["macro_f1"]),
        "test_receiving_hand_macro_f1": float(
            intention["receiving_hand"]["macro_f1_supported"]
        ),
        "native_t_plus_1_position_error_cm": float(
            pose["pose_oracle"]["position_mae_cm"]
        ),
        "native_t_plus_1_orientation_error_deg": float(
            pose["pose_oracle"]["orientation_mean_deg"]
        ),
    }


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "test_intention_macro_f1",
        "test_receiving_hand_macro_f1",
        "test_terminal_position_error_cm",
        "test_terminal_orientation_error_deg",
        "test_terminal_end_to_end_position_error_cm",
        "test_terminal_end_to_end_orientation_error_deg",
        "test_last_observation_position_error_cm",
        "test_target_window_coverage",
        "trainable_parameters",
    )
    rows = []
    for model in MODEL_ORDER:
        group = frame.loc[frame["model"] == model]
        if group.empty:
            continue
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model].replace("\n", " "),
            "completed_seeds": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_time(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, time_bin, evaluation), group in frame.groupby(
        ["model", "time_to_sequence_end_bin", "evaluation"], sort=False
    ):
        row = {
            "model": model,
            "time_to_sequence_end_bin": time_bin,
            "evaluation": evaluation,
            "completed_seeds": int(len(group)),
            "samples_per_seed_min": int(group["samples"].min()),
            "samples_per_seed_max": int(group["samples"].max()),
        }
        for metric in ("position_error_cm", "orientation_error_deg"):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=0)) if values.notna().any() else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_latency(report_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    for model, directory in (
        ("terminal_endpose", "endpose"),
        ("t_plus_1_as_terminal", "t_plus_1"),
    ):
        for device in ("cpu", "cuda"):
            path = report_dir / "latency" / directory / f"tcml_{device}.json"
            try:
                report = read_json(path)
                if report["status"] != "completed":
                    raise ValueError(report.get("reason", "latency unavailable"))
                rows.append(
                    {
                        "model": model,
                        "device": device,
                        "report": str(path),
                        "checkpoint_sha256": report["checkpoint_sha256"],
                        "trainable_parameters": report["model"][
                            "trainable_parameters"
                        ],
                        "model_forward_mean_ms": report["model_forward"]["mean_ms"],
                        "model_forward_p95_ms": report["model_forward"]["p95_ms"],
                        "offline_window_mean_ms": report["offline_window"]["mean_ms"],
                        "offline_window_p95_ms": report["offline_window"]["p95_ms"],
                    }
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{model} {device}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), errors


def plot_model_comparison(summary: pd.DataFrame, figures: Path) -> None:
    ordered = summary.set_index("model").loc[list(MODEL_ORDER)]
    panels = (
        ("test_intention_macro_f1", "Intent macro-F1", False),
        ("test_receiving_hand_macro_f1", "Receiving-hand macro-F1", False),
        ("test_terminal_position_error_cm", "Terminal position error (cm)", True),
        (
            "test_terminal_orientation_error_deg",
            "Terminal orientation error (deg)",
            True,
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    colors = ["#9C755F", "#4C78A8"]
    for axis, (metric, title, lower_better) in zip(axes, panels):
        values = ordered[f"{metric}_mean"].to_numpy(float)
        errors = ordered[f"{metric}_std"].to_numpy(float)
        axis.bar(labels, values, yerr=errors, capsize=5, color=colors)
        axis.set_title(title + ("\n(lower is better)" if lower_better else ""))
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(
        "Existing t+1 predictor versus terminal-endpose training\n"
        "common robust terminal targets; mean ± population SD, seeds 42/43/44"
    )
    figure.tight_layout()
    figure.savefig(figures / "01_terminal_model_comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "01_terminal_model_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_time_curve(summary: pd.DataFrame, figures: Path) -> None:
    selected = summary.loc[summary["evaluation"] == "pose_oracle"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(TIME_GROUPS))
    for model, color in zip(MODEL_ORDER, ("#9C755F", "#4C78A8")):
        group = selected.loc[selected["model"] == model].set_index(
            "time_to_sequence_end_bin"
        ).reindex(TIME_GROUPS)
        for axis, metric, ylabel in (
            (axes[0], "position_error_cm", "Position error (cm)"),
            (axes[1], "orientation_error_deg", "Orientation error (degrees)"),
        ):
            values = group[f"{metric}_mean"].to_numpy(float)
            errors = group[f"{metric}_std"].to_numpy(float)
            axis.errorbar(
                x,
                values,
                yerr=errors,
                marker="o",
                capsize=4,
                color=color,
                label=MODEL_LABELS[model].replace("\n", " "),
            )
            axis.set_xticks(x, TIME_GROUPS)
            axis.set_xlabel("Time remaining to sequence end")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Terminal-pose error versus remaining time (oracle receiving hand)")
    figure.tight_layout()
    figure.savefig(figures / "02_error_vs_time_remaining.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "02_error_vs_time_remaining.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_latency(frame: pd.DataFrame, figures: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, device in zip(axes, ("cpu", "cuda")):
        group = frame.loc[frame["device"] == device].set_index("model").loc[
            list(MODEL_ORDER)
        ]
        axis.bar(
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            group["model_forward_mean_ms"],
            color=["#9C755F", "#4C78A8"],
        )
        axis.set_title(f"TCML {device.upper()} forward latency")
        axis.set_ylabel("Mean latency (ms), batch size 1")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("Identical architecture; measured separately with 1000 repeats")
    figure.tight_layout()
    figure.savefig(figures / "03_latency_comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "03_latency_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_audit(audit: dict, figures: Path) -> None:
    splits = ("train", "validation", "test")
    accepted = [audit["splits"][name]["accepted_handover_sequences"] for name in splits]
    rejected = [audit["splits"][name]["rejected_handover_sequences"] for name in splits]
    figure, axis = plt.subplots(figsize=(7, 4.8))
    axis.bar(splits, accepted, label="Accepted", color="#59A14F")
    axis.bar(splits, rejected, bottom=accepted, label="Rejected", color="#E15759")
    axis.set_ylabel("Handover sequences")
    axis.set_title("Robust terminal-target audit by participant split")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "04_target_audit.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "04_target_audit.pdf", bbox_inches="tight")
    plt.close(figure)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    return value


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    time_summary: pd.DataFrame,
    audit: dict,
    latency: pd.DataFrame,
    native_summary: dict,
    complete: bool,
) -> None:
    indexed = summary.set_index("model")
    baseline = indexed.loc["t_plus_1_as_terminal"]
    terminal = indexed.loc["terminal_endpose"]
    oracle_time = time_summary.loc[
        time_summary["evaluation"] == "pose_oracle"
    ].set_index(["model", "time_to_sequence_end_bin"])
    baseline_far = oracle_time.loc[("t_plus_1_as_terminal", ">=3s")]
    terminal_far = oracle_time.loc[("terminal_endpose", ">=3s")]
    lines = [
        "# Terminal end-pose experiment (n214)",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        "The existing Residual-v2 model predicts the receiving-hand pose at **t+1 second**. "
        "The new, separate experiment predicts one **robust terminal handover pose**, formed "
        "from the latest stable 0.5-second receiving-hand segment after `THIRD`. Existing "
        "t+1 runs and checkpoints were not modified.",
        "",
        "## Target audit",
        "",
        f"Accepted terminal targets: **{audit['accepted_handover_sequences']}/"
        f"{audit['handover_sequences']} ({audit['target_sequence_coverage']:.1%})**. "
        "Rejected sequences remain in the intent/hand tasks but do not contribute pose loss.",
        "",
        "## Test comparison on the same terminal target",
        "",
        "| Model | Intent macro-F1 | Hand macro-F1 | Position error (cm) | Orientation error (deg) | Target coverage | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row = indexed.loc[model]
        lines.append(
            f"| {MODEL_LABELS[model].replace(chr(10), ' ')} | "
            f"{fmt(row['test_intention_macro_f1_mean'])} ± {fmt(row['test_intention_macro_f1_std'])} | "
            f"{fmt(row['test_receiving_hand_macro_f1_mean'])} ± {fmt(row['test_receiving_hand_macro_f1_std'])} | "
            f"{fmt(row['test_terminal_position_error_cm_mean'], 2)} ± {fmt(row['test_terminal_position_error_cm_std'], 2)} | "
            f"{fmt(row['test_terminal_orientation_error_deg_mean'], 2)} ± {fmt(row['test_terminal_orientation_error_deg_std'], 2)} | "
            f"{row['test_target_window_coverage_mean']:.1%} | "
            f"{int(round(row['trainable_parameters_mean'])):,} |"
        )
    lines.extend(
        [
            "",
            "Values are mean ± population standard deviation across seeds 42, 43 and 44. "
            "Intent and hand use the best-intention checkpoint; pose uses the best-pose "
            "checkpoint. Both checkpoints were selected exclusively on validation.",
            "",
            "## Result",
            "",
            "The dedicated terminal model did **not** improve the aggregate terminal-pose "
            "result. Relative to the existing t+1 checkpoint evaluated against the same "
            f"terminal target, its position error is "
            f"{terminal['test_terminal_position_error_cm_mean'] - baseline['test_terminal_position_error_cm_mean']:.2f} cm higher and its "
            f"orientation error is {terminal['test_terminal_orientation_error_deg_mean'] - baseline['test_terminal_orientation_error_deg_mean']:.2f}° higher. "
            f"Intent macro-F1 changes by {terminal['test_intention_macro_f1_mean'] - baseline['test_intention_macro_f1_mean']:+.3f} and "
            f"receiving-hand macro-F1 by {terminal['test_receiving_hand_macro_f1_mean'] - baseline['test_receiving_hand_macro_f1_mean']:+.3f}.",
            "",
            "The remaining-time analysis reveals a narrower benefit: at **>=3 seconds** "
            "before sequence end, the terminal model reaches "
            f"{terminal_far['position_error_cm_mean']:.2f} cm / "
            f"{terminal_far['orientation_error_deg_mean']:.2f}° versus "
            f"{baseline_far['position_error_cm_mean']:.2f} cm / "
            f"{baseline_far['orientation_error_deg_mean']:.2f}° for the t+1 baseline. "
            "The t+1 baseline is better in every bin from 0 to 3 seconds. Thus the terminal "
            "objective shows some long-horizon anticipation, but the overall hypothesis is "
            "not supported by this experiment.",
            "",
            "For context only, the original model's native t+1 position error was "
            f"{native_summary['native_t_plus_1_position_error_cm_mean']:.2f} ± "
            f"{native_summary['native_t_plus_1_position_error_cm_std']:.2f} cm. This native "
            "metric has a different target and must not be compared directly with terminal "
            "pose error; the table above re-evaluates that checkpoint on the shared terminal target.",
            "",
            "## Latency",
            "",
            "| Model | TCML CPU mean (ms) | TCML CUDA mean (ms) |",
            "|---|---:|---:|",
        ]
    )
    latency_index = latency.set_index(["model", "device"])
    for model in MODEL_ORDER:
        lines.append(
            f"| {MODEL_LABELS[model].replace(chr(10), ' ')} | "
            f"{latency_index.loc[(model, 'cpu'), 'model_forward_mean_ms']:.3f} | "
            f"{latency_index.loc[(model, 'cuda'), 'model_forward_mean_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Both models use the same Residual-v2 architecture; only the learned target and weights differ.",
            "",
            "## Files",
            "",
            "- `data/model_runs.csv`: all seed-level metrics",
            "- `data/model_summary.csv`: mean and standard deviation",
            "- `data/error_vs_time_remaining.csv`: terminal error by remaining-time bin",
            "- `comparison.json`: machine-readable complete report",
            "- `figures/`: matching PNG and PDF figures",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = PROJECT_ROOT / "Training/runs" / args.dataset_tag
    report_dir = (
        PROJECT_ROOT / "Training/reports" / args.dataset_tag / args.experiment_tag
    )
    data_dir = report_dir / "data"
    figures = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    audit_path = report_dir / "audit" / "endpose_target_audit.json"
    audit = read_json(audit_path)

    rows = []
    curve_rows = []
    native_rows = []
    errors = []
    for seed in SEEDS:
        new_path = endpose_run_path(runs_root, args.experiment_tag, seed)
        baseline_path = report_dir / "baseline_terminal_eval" / f"seed{seed}.json"
        try:
            new_report = read_json(new_path / "metrics.json")
            if new_report["pose_target_definition"]["mode"] != "terminal_endpose":
                raise ValueError("new run does not use terminal targets")
            rows.append(result_row("terminal_endpose", seed, new_report, new_path))
            curve_rows.extend(time_rows("terminal_endpose", seed, new_report))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"terminal_endpose seed {seed}: {type(exc).__name__}: {exc}")
        try:
            baseline_report = read_json(baseline_path)
            rows.append(
                result_row(
                    "t_plus_1_as_terminal", seed, baseline_report, baseline_path
                )
            )
            curve_rows.extend(
                time_rows("t_plus_1_as_terminal", seed, baseline_report)
            )
            native_rows.append(native_t1_row(seed, baseline_report))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"t_plus_1 seed {seed}: {type(exc).__name__}: {exc}")

    runs = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    native = pd.DataFrame(native_rows)
    runs.to_csv(data_dir / "model_runs.csv", index=False)
    curves.to_csv(data_dir / "error_vs_time_remaining_by_seed.csv", index=False)
    native.to_csv(data_dir / "native_t_plus_1_context.csv", index=False)
    summary = aggregate_runs(runs) if not runs.empty else pd.DataFrame()
    time_summary = aggregate_time(curves) if not curves.empty else pd.DataFrame()
    summary.to_csv(data_dir / "model_summary.csv", index=False)
    time_summary.to_csv(data_dir / "error_vs_time_remaining.csv", index=False)
    native_summary = {}
    if not native.empty:
        for metric in (
            "test_intention_macro_f1",
            "test_receiving_hand_macro_f1",
            "native_t_plus_1_position_error_cm",
            "native_t_plus_1_orientation_error_deg",
        ):
            values = pd.to_numeric(native[metric], errors="raise")
            native_summary[f"{metric}_mean"] = float(values.mean())
            native_summary[f"{metric}_std"] = float(values.std(ddof=0))

    latency, latency_errors = load_latency(report_dir)
    errors.extend(latency_errors)
    latency.to_csv(data_dir / "latency.csv", index=False)
    complete = (
        not errors
        and len(runs) == len(MODEL_ORDER) * len(SEEDS)
        and len(native) == len(SEEDS)
        and len(latency) == len(MODEL_ORDER) * 2
        and audit.get("training_authorized_by_audit") is True
        and set(summary.get("model", [])) == set(MODEL_ORDER)
    )
    if complete:
        plot_model_comparison(summary, figures)
        plot_time_curve(time_summary, figures)
        plot_latency(latency, figures)
        plot_audit(audit, figures)
        write_markdown(
            report_dir / "README.md",
            summary,
            time_summary,
            audit,
            latency,
            native_summary,
            complete,
        )

    report = {
        "schema_version": 1,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "complete": complete,
        "errors": errors,
        "seeds": list(SEEDS),
        "target_audit": audit,
        "checkpoint_policy": {
            "intention_and_hand": "best-intention checkpoint selected on validation",
            "pose": "best-pose checkpoint selected on validation",
            "test_used_for_selection": False,
        },
        "target_definitions": {
            "existing_t_plus_1": "receiving-hand pose at t+1 second",
            "new_terminal_endpose": "one robust pose from the latest stable 0.5-second receiving-hand segment after THIRD",
            "common_comparison_target": "robust terminal receiving-hand pose",
        },
        "run_results": runs.to_dict(orient="records"),
        "aggregate_results": summary.to_dict(orient="records"),
        "native_t_plus_1_context": native_summary,
        "latency": latency.to_dict(orient="records"),
        "generated_files": [
            "data/model_runs.csv",
            "data/model_summary.csv",
            "data/error_vs_time_remaining_by_seed.csv",
            "data/error_vs_time_remaining.csv",
            "data/native_t_plus_1_context.csv",
            "data/latency.csv",
            "figures/01_terminal_model_comparison.png",
            "figures/01_terminal_model_comparison.pdf",
            "figures/02_error_vs_time_remaining.png",
            "figures/02_error_vs_time_remaining.pdf",
            "figures/03_latency_comparison.png",
            "figures/03_latency_comparison.pdf",
            "figures/04_target_audit.png",
            "figures/04_target_audit.pdf",
            "README.md",
        ],
    }
    (report_dir / "comparison.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"End-pose comparison complete: {complete}; report: {report_dir}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
