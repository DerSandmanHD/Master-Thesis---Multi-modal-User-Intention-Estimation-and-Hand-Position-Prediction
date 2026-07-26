#!/usr/bin/env python3
"""Validate and aggregate the seeded final model-comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MODEL_TYPES = {
    "hierarchical_gated_multimodal_transformer",
    "hierarchical_window_mlp",
    "hierarchical_gru",
    "hierarchical_residual_pose_transformer_v2",
}
SUMMARY_METRICS = (
    "intention_macro_f1",
    "assistance_macro_f1",
    "assistance_type_macro_f1",
    "receiving_hand_macro_f1",
    "intent_checkpoint_pose_mae_cm",
    "intent_checkpoint_orientation_deg",
    "pose_checkpoint_oracle_mae_cm",
    "pose_checkpoint_oracle_orientation_deg",
    "pose_checkpoint_end_to_end_mae_cm",
    "pose_checkpoint_end_to_end_orientation_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="final_clean_v1")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Training/runs"),
    )
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required run artifact is missing: {path}") from exc


def standard_row(
    run_dir: Path,
    config: dict,
    metadata: dict,
    metrics: dict,
) -> dict:
    intention_test = metrics.get("test_by_checkpoint", {}).get(
        "best_intention", metrics["test"]
    )
    pose_test = metrics.get("test_by_checkpoint", {}).get("best_pose", intention_test)
    return {
        "run_dir": str(run_dir),
        "model_type": metrics["model_type"],
        "seed": int(config["training"]["seed"]),
        "trainable_parameters": int(metrics["trainable_parameters"]),
        "selected_sequences": int(
            metadata["split"]["dataset_filter"]["selected_sequences"]
        ),
        "sequence_fingerprint": metadata["split"]["dataset_filter"][
            "sequence_fingerprint"
        ],
        "train_windows": int(metadata["windows"]["train"]),
        "validation_windows": int(metadata["windows"]["validation"]),
        "test_windows": int(metadata["windows"]["test"]),
        "intention_macro_f1": intention_test["intention"]["macro_f1"],
        "assistance_macro_f1": intention_test["assistance"]["macro_f1"],
        "assistance_type_macro_f1": intention_test["assistance_type"]["macro_f1"],
        "receiving_hand_macro_f1": None,
        "intent_checkpoint_pose_mae_cm": intention_test["pose"]["position_mae_cm"],
        "intent_checkpoint_pose_samples": intention_test["pose"]["samples"],
        "intent_checkpoint_orientation_deg": intention_test["pose"][
            "orientation_mean_deg"
        ],
        "pose_checkpoint_oracle_mae_cm": pose_test["pose"]["position_mae_cm"],
        "pose_checkpoint_oracle_samples": pose_test["pose"]["samples"],
        "pose_checkpoint_oracle_orientation_deg": pose_test["pose"][
            "orientation_mean_deg"
        ],
        "pose_checkpoint_end_to_end_mae_cm": pose_test["pose"]["position_mae_cm"],
        "pose_checkpoint_end_to_end_samples": pose_test["pose"]["samples"],
        "pose_checkpoint_end_to_end_orientation_deg": pose_test["pose"][
            "orientation_mean_deg"
        ],
    }


def residual_row(
    run_dir: Path,
    config: dict,
    metadata: dict,
    metrics: dict,
) -> dict:
    intention_test = metrics["test"]["best_intention"]
    pose_test = metrics["test"]["best_pose"]
    return {
        "run_dir": str(run_dir),
        "model_type": metrics["model_type"],
        "seed": int(config["training"]["seed"]),
        "trainable_parameters": int(metrics["trainable_parameters"]),
        "selected_sequences": int(
            metadata["split"]["dataset_filter"]["selected_sequences"]
        ),
        "sequence_fingerprint": metadata["split"]["dataset_filter"][
            "sequence_fingerprint"
        ],
        "train_windows": int(metadata["windows"]["train"]),
        "validation_windows": int(metadata["windows"]["validation"]),
        "test_windows": int(metadata["windows"]["test"]),
        "intention_macro_f1": intention_test["intention"]["macro_f1"],
        "assistance_macro_f1": intention_test["assistance"]["macro_f1"],
        "assistance_type_macro_f1": intention_test["assistance_type"]["macro_f1"],
        "receiving_hand_macro_f1": intention_test["receiving_hand"]["macro_f1"],
        "intent_checkpoint_pose_mae_cm": intention_test["pose_end_to_end"][
            "position_mae_cm"
        ],
        "intent_checkpoint_pose_samples": intention_test["pose_end_to_end"]["samples"],
        "intent_checkpoint_orientation_deg": intention_test["pose_end_to_end"][
            "orientation_mean_deg"
        ],
        "pose_checkpoint_oracle_mae_cm": pose_test["pose_oracle"]["position_mae_cm"],
        "pose_checkpoint_oracle_samples": pose_test["pose_oracle"]["samples"],
        "pose_checkpoint_oracle_orientation_deg": pose_test["pose_oracle"][
            "orientation_mean_deg"
        ],
        "pose_checkpoint_end_to_end_mae_cm": pose_test["pose_end_to_end"][
            "position_mae_cm"
        ],
        "pose_checkpoint_end_to_end_samples": pose_test["pose_end_to_end"]["samples"],
        "pose_checkpoint_end_to_end_orientation_deg": pose_test["pose_end_to_end"][
            "orientation_mean_deg"
        ],
    }


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> int:
    args = parse_args()
    runs_root = project_path(args.runs_root)
    run_dirs = sorted(path for path in runs_root.glob(f"{args.tag}_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories match {runs_root / (args.tag + '_*')}"
        )

    rows = []
    reference_split = None
    reference_windows = None
    reference_fingerprint = None
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        metadata = read_json(run_dir / "data_metadata.json")
        metrics = read_json(run_dir / "metrics.json")
        dataset_filter = metadata["split"]["dataset_filter"]
        if not dataset_filter.get("enabled"):
            raise ValueError(f"Manifest filtering was disabled in {run_dir}")

        split_sequences = metadata["split"]["sequences"]
        windows = metadata["windows"]
        fingerprint = dataset_filter["sequence_fingerprint"]
        if reference_split is None:
            reference_split = split_sequences
            reference_windows = windows
            reference_fingerprint = fingerprint
        elif (
            split_sequences != reference_split
            or windows != reference_windows
            or fingerprint != reference_fingerprint
        ):
            raise ValueError(f"Dataset or split mismatch detected in {run_dir}")

        model_type = metrics["model_type"]
        if model_type == "hierarchical_residual_pose_transformer_v2":
            row = residual_row(run_dir, config, metadata, metrics)
        else:
            row = standard_row(run_dir, config, metadata, metrics)
        rows.append(row)

    expected_seeds = set(args.expected_seeds)
    run_counts = Counter(row["model_type"] for row in rows)
    missing_models = EXPECTED_MODEL_TYPES - set(run_counts)
    extra_models = set(run_counts) - EXPECTED_MODEL_TYPES
    if missing_models or extra_models:
        raise ValueError(
            f"Model set mismatch; missing={sorted(missing_models)}, "
            f"extra={sorted(extra_models)}"
        )
    for model_type in sorted(EXPECTED_MODEL_TYPES):
        model_rows = [row for row in rows if row["model_type"] == model_type]
        seeds = {row["seed"] for row in model_rows}
        if seeds != expected_seeds or len(model_rows) != len(expected_seeds):
            raise ValueError(
                f"Run/seed mismatch for {model_type}: runs={len(model_rows)}, "
                f"seeds={sorted(seeds)}, expected={sorted(expected_seeds)}"
            )

    summary = {}
    for model_type in sorted(EXPECTED_MODEL_TYPES):
        model_rows = [row for row in rows if row["model_type"] == model_type]
        summary[model_type] = {
            "runs": len(model_rows),
            "trainable_parameters": model_rows[0]["trainable_parameters"],
            "metrics": {
                metric: mean_std(
                    [
                        float(row[metric])
                        for row in model_rows
                        if row[metric] is not None
                    ]
                )
                for metric in SUMMARY_METRICS
                if any(row[metric] is not None for row in model_rows)
            },
        }

    report = {
        "tag": args.tag,
        "expected_seeds": sorted(expected_seeds),
        "sequence_fingerprint": reference_fingerprint,
        "split_sequences": reference_split,
        "windows": reference_windows,
        "runs": rows,
        "summary": summary,
    }
    output_json = project_path(
        args.output_json or Path("Training/reports") / f"{args.tag}_comparison.json"
    )
    output_csv = project_path(
        args.output_csv or Path("Training/reports") / f"{args.tag}_comparison.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Validated {len(rows)} runs on dataset {reference_fingerprint}")
    for model_type, values in summary.items():
        intention = values["metrics"]["intention_macro_f1"]
        pose = values["metrics"]["pose_checkpoint_end_to_end_mae_cm"]
        print(
            f"{model_type}: intent F1={intention['mean']:.4f}±{intention['std']:.4f}, "
            f"pose MAE={pose['mean']:.2f}±{pose['std']:.2f} cm"
        )
    print(f"JSON: {output_json}")
    print(f"CSV:  {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
