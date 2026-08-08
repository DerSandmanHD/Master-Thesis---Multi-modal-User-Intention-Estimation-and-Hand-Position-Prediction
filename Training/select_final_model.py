#!/usr/bin/env python3
"""Freeze one final architecture/seed checkpoint using validation metrics only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
F1_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--screen-experiment", default="visual_embedding_screen_v1")
    parser.add_argument("--sensor-experiment", default="residual_v2_tuned_v1")
    parser.add_argument("--sensor-model", default="residual_v2_tuned")
    parser.add_argument("--visual-experiment", default="visual_embedding_final_v1")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_metrics(metrics: dict) -> dict:
    if "validation_by_checkpoint" in metrics:
        return metrics["validation_by_checkpoint"]["best_intention"]
    epoch = int(metrics["checkpoints"]["best_intention"]["epoch"])
    return next(
        row["validation"]
        for row in metrics["history"]
        if int(row["epoch"]) == epoch
    )


def main() -> int:
    args = parse_args()
    reports = PROJECT_ROOT / "Training/reports" / args.dataset_tag
    screen = json.loads(
        (reports / args.screen_experiment / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    architecture = screen.get("selected_variant_for_final_test")
    if not screen.get("complete") or architecture not in {
        "sensor_baseline",
        "clip_only",
        "sensor_plus_clip",
    }:
        raise ValueError(f"Invalid visual-screen selection: {architecture!r}")
    if architecture == "sensor_baseline":
        experiment = args.sensor_experiment
        model = args.sensor_model
    else:
        experiment = args.visual_experiment
        model = architecture

    runs_root = PROJECT_ROOT / "Training/runs" / args.dataset_tag / experiment / model
    rows = []
    for seed in SEEDS:
        run_id = f"{experiment}_{model}_seed{seed}"
        run_dir = runs_root / run_id
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "best_intention_model.pt"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("test_evaluation_skipped") is not False or "test" not in metrics:
            raise ValueError(f"Final evaluation is incomplete: {run_dir}")
        values = validation_metrics(metrics)
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "validation_intention_macro_f1": float(
                    values["intention"]["macro_f1"]
                ),
                "validation_receiving_hand_macro_f1": float(
                    values["receiving_hand"]["macro_f1_supported"]
                ),
                "validation_pose_mae_cm": float(
                    values["pose_oracle"]["position_mae_cm"]
                ),
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "config_sha256": sha256_file(config_path),
            }
        )
    best_f1 = max(row["validation_intention_macro_f1"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["validation_intention_macro_f1"] >= best_f1 - F1_TOLERANCE
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_pose_mae_cm"],
            -row["validation_receiving_hand_macro_f1"],
            row["seed"],
        ),
    )
    output = (
        args.output.expanduser()
        if args.output
        else reports / "final_model_selection.json"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_tag": args.dataset_tag,
        "architecture_source": str(
            (reports / args.screen_experiment / "summary.json").relative_to(
                PROJECT_ROOT
            )
        ),
        "selected_architecture": architecture,
        "selected_experiment": experiment,
        "selected_model": model,
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_split": "validation",
        "test_metrics_read_for_selection": False,
        "selection_rule": (
            "retain seeds within 0.005 of best validation intention macro-F1; "
            "then minimize validation pose MAE; maximize validation hand F1; "
            "use lower seed only as deterministic final tie-break"
        ),
        "candidates": rows,
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Final checkpoint: {selected['checkpoint']}")
    print(f"SHA-256: {selected['checkpoint_sha256']}")
    print(f"Selection report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
