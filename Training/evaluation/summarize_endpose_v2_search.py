#!/usr/bin/env python3
"""Summarize validation-only endpose-v2 search and confirmation stages."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_TARGET_VERSION = "terminal_endpose_unique_hand_capture_v2"
PARAMETERS = (
    "learning_rate",
    "pose_loss_weight",
    "orientation_loss_weight",
    "auxiliary_pose_loss_weight",
    "dropout",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "batch_size",
)
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "confirmation"), required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training/configs/endpose_v2_search/manifest.json"),
    )
    parser.add_argument("--stage-a-summary", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-tag", default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable(path: Path) -> str:
    resolved = resolve(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: dict, *keys: str):
    for key in keys:
        value = value[key]
    return value


def run_path(root: Path, experiment: str, trial: str, seed: int) -> Path:
    return root / trial / f"{experiment}_{trial}_seed{seed}"


def result_row(
    path: Path,
    trial: dict,
    *,
    seed: int,
    dataset_tag: str,
    experiment_tag: str,
) -> dict:
    row = {
        "trial_index": int(trial["trial_index"]),
        "trial_tag": trial["trial_tag"],
        "seed": seed,
        "status": "missing",
        "error": "",
        **{name: trial[name] for name in PARAMETERS},
    }
    if not path.exists():
        return row
    try:
        metrics = read(path / "metrics.json")
        config = read(path / "config.json")
        metadata = read(path / "data_metadata.json")
        if metrics.get("test_evaluation_skipped") is not True or "test" in metrics:
            raise ValueError("search run contains test evaluation")
        expected_context = {
            "dataset_tag": dataset_tag,
            "experiment_tag": experiment_tag,
            "model_tag": trial["trial_tag"],
        }
        if config.get("run_context") != expected_context:
            raise ValueError("run context mismatch")
        if metrics.get("model_type") != (
            "hierarchical_dual_horizon_residual_pose_transformer_v3"
        ):
            raise ValueError("wrong model type")
        target_definition = metadata.get("pose_target", {})
        if (
            target_definition.get("target_definition_version")
            != TERMINAL_TARGET_VERSION
        ):
            raise ValueError("run uses a stale terminal-target definition")
        pose = nested(metrics, "validation_by_checkpoint", "best_intention")
        row.update(
            {
                "status": "completed",
                "validation_terminal_position_error_cm": float(
                    nested(pose, "pose_end_to_end", "position_mae_cm")
                ),
                "validation_terminal_orientation_error_deg": float(
                    nested(pose, "pose_end_to_end", "orientation_mean_deg")
                ),
                "validation_auxiliary_t1_position_error_cm": float(
                    nested(
                        pose,
                        "auxiliary_t_plus_1",
                        "pose_end_to_end",
                        "position_mae_cm",
                    )
                ),
                "validation_intention_macro_f1": float(
                    nested(pose, "intention", "macro_f1")
                ),
                "validation_receiving_hand_macro_f1": float(
                    nested(pose, "receiving_hand", "macro_f1_supported")
                ),
                "best_intention_epoch": int(
                    nested(metrics, "checkpoints", "best_intention", "epoch")
                ),
                "metric_source_checkpoint": "best_intention",
                "pose_metric_semantics": "learned_end_to_end_predicted_receiving_hand_and_reference",
                "target_definition_version": TERMINAL_TARGET_VERSION,
                "trainable_parameters": int(metrics["trainable_parameters"]),
                "wall_seconds": float(nested(metrics, "runtime", "wall_seconds")),
                "dataset_content_fingerprint": nested(
                    metadata, "provenance", "dataset_content_fingerprint"
                ),
                "git_commit": metrics.get("code_provenance", {}).get("commit"),
                "git_dirty": metrics.get("code_provenance", {}).get("dirty"),
            }
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def rank(frame: pd.DataFrame, tolerance_cm: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    best = float(frame["validation_terminal_position_error_cm"].min())
    ranked = frame.copy()
    ranked["within_position_tolerance"] = (
        ranked["validation_terminal_position_error_cm"] <= best + tolerance_cm
    )
    preferred = ranked.loc[ranked["within_position_tolerance"]].sort_values(
        [
            "validation_terminal_orientation_error_deg",
            "validation_terminal_position_error_cm",
            "validation_intention_macro_f1",
            "validation_receiving_hand_macro_f1",
        ],
        ascending=[True, True, False, False],
        kind="stable",
    )
    remaining = ranked.loc[~ranked["within_position_tolerance"]].sort_values(
        [
            "validation_terminal_position_error_cm",
            "validation_terminal_orientation_error_deg",
            "validation_intention_macro_f1",
        ],
        ascending=[True, True, False],
        kind="stable",
    )
    result = pd.concat((preferred, remaining), ignore_index=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_search(frame: pd.DataFrame, figures: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    scatter = axis.scatter(
        frame["validation_terminal_position_error_cm"],
        frame["validation_terminal_orientation_error_deg"],
        c=frame["validation_intention_macro_f1"],
        s=65,
        cmap="viridis",
        edgecolors=np.where(frame["within_position_tolerance"], "black", "none"),
    )
    for _, row in frame.iterrows():
        axis.annotate(
            row["trial_tag"].replace("trial_", "t"),
            (
                row["validation_terminal_position_error_cm"],
                row["validation_terminal_orientation_error_deg"],
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Validation terminal position error (cm)")
    axis.set_ylabel("Validation terminal orientation error (degrees)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Validation intent macro-F1")
    save_figure(figure, figures, "01_validation_terminal_pose_search")


def main() -> int:
    args = parse_args()
    manifest = read(resolve(args.manifest))
    tolerance = float(manifest["position_tolerance_cm"])
    runs = resolve(
        args.runs_dir
        or Path("Training/runs") / args.dataset_tag / args.experiment_tag
    )
    output = resolve(
        args.output_dir
        or Path("Training/reports")
        / args.dataset_tag
        / (
            args.report_tag
            or f"{args.experiment_tag}_checkpoint_coherent_v2"
        )
    )
    data_dir = output / "data"
    figures = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if args.stage == "a":
        trials = manifest["trials"]
        seeds = (42,)
    else:
        if args.stage_a_summary is None:
            raise ValueError("confirmation requires --stage-a-summary")
        selected_tags = read(resolve(args.stage_a_summary))[
            "selected_for_confirmation"
        ]
        trial_by_tag = {trial["trial_tag"]: trial for trial in manifest["trials"]}
        trials = [trial_by_tag[tag] for tag in selected_tags]
        seeds = SEEDS

    rows = [
        result_row(
            run_path(runs, args.experiment_tag, trial["trial_tag"], seed),
            trial,
            seed=seed,
            dataset_tag=args.dataset_tag,
            experiment_tag=args.experiment_tag,
        )
        for trial in trials
        for seed in seeds
    ]
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(data_dir / "validation_runs.csv", index=False)
    completed = all_runs.loc[all_runs["status"] == "completed"].copy()
    errors = all_runs.loc[all_runs["status"] != "completed"].to_dict(
        orient="records"
    )

    if args.stage == "a":
        aggregates = completed
    else:
        metric_columns = [
            "validation_terminal_position_error_cm",
            "validation_terminal_orientation_error_deg",
            "validation_auxiliary_t1_position_error_cm",
            "validation_intention_macro_f1",
            "validation_receiving_hand_macro_f1",
            "trainable_parameters",
        ]
        aggregate_rows = []
        for trial in trials:
            group = completed.loc[completed["trial_tag"] == trial["trial_tag"]]
            row = {
                "trial_index": trial["trial_index"],
                "trial_tag": trial["trial_tag"],
                "completed_seeds": len(group),
                **{name: trial[name] for name in PARAMETERS},
            }
            for column in metric_columns:
                row[column] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=0))
            aggregate_rows.append(row)
        aggregates = pd.DataFrame(aggregate_rows)

    ranked = rank(aggregates, tolerance) if not aggregates.empty else aggregates
    if args.stage == "a" and not ranked.empty:
        ranked["selected_for_confirmation"] = ranked["rank"] <= 3
    ranked.to_csv(data_dir / "validation_ranking.csv", index=False)
    complete = not errors and len(completed) == len(trials) * len(seeds)
    selected_for_confirmation = (
        ranked.loc[ranked["rank"] <= 3, "trial_tag"].tolist()
        if args.stage == "a" and not ranked.empty
        else []
    )
    selected_trial = (
        str(ranked.iloc[0]["trial_tag"])
        if args.stage == "confirmation" and not ranked.empty
        else None
    )
    if not ranked.empty:
        plot_search(
            ranked,
            figures,
            "Endpose-v2 validation-only search"
            if args.stage == "a"
            else "Endpose-v2 three-seed validation confirmation",
        )

    if selected_trial is not None:
        winner = next(
            trial for trial in manifest["trials"] if trial["trial_tag"] == selected_trial
        )
        selected_config = read(resolve(Path(winner["config"])))
        selected_config["run_name"] = "dual_horizon_endpose_v2"
        selected_config["hyperparameter_confirmation"] = {
            "selected_trial": selected_trial,
            "selection_split": "validation",
            "seeds": list(SEEDS),
            "test_metrics_used": False,
            "selection_rule": manifest["selection_rule"],
            "source_summary": portable(args.stage_a_summary),
        }
        (output / "selected_config.json").write_text(
            json.dumps(selected_config, indent=2, ensure_ascii=False) + "\n"
        )

    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "complete": complete,
        "test_metrics_used": False,
        "selection_split": "validation",
        "metric_source_checkpoint": "best_intention",
        "pose_metric_semantics": "learned_end_to_end_predicted_receiving_hand_and_reference",
        "target_definition_version": TERMINAL_TARGET_VERSION,
        "position_tolerance_cm": tolerance,
        "selection_rule": manifest["selection_rule"],
        "completed_runs": len(completed),
        "expected_runs": len(trials) * len(seeds),
        "errors": errors,
        "selected_for_confirmation": selected_for_confirmation,
        "selected_trial": selected_trial,
        "ranking": ranked.replace({np.nan: None}).to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        f"Endpose-v2 {args.stage}: complete={complete}, "
        f"runs={len(completed)}/{len(trials) * len(seeds)}, "
        f"selected={selected_trial or selected_for_confirmation}"
    )
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
