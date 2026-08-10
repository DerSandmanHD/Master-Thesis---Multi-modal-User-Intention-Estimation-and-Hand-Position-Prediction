#!/usr/bin/env python3
"""Validate the frozen end-pose experiment without training or test scoring."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import INTENTION_TO_ID, prepare_data, sha256_file
from model import HierarchicalResidualPoseTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_endpose_v1.json"),
    )
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-tag", default="dataset_v2_20260802_n214_5d136a34"
    )
    parser.add_argument("--experiment-tag", default="residual_v2_endpose_v1")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_constant_targets(bundle) -> list[str]:
    errors = []
    for split_name in SPLITS:
        dataset = getattr(bundle, split_name)
        for record in dataset.records:
            metadata = record.pose_target_metadata or {}
            if not metadata.get("eligible"):
                continue
            rows = np.flatnonzero(
                (record.intentions == INTENTION_TO_ID["handover"])
                & record.pose_valid
            )
            if not len(rows):
                errors.append(f"{record.sequence_id}: accepted but no target rows")
                continue
            unique = np.unique(record.pose_targets[rows], axis=0)
            if len(unique) != 1:
                errors.append(
                    f"{record.sequence_id}: terminal target differs across windows"
                )
    return errors


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config).resolve()
    audit_path = resolve(args.audit_report).resolve()
    output_path = resolve(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    errors = []
    config_hash = sha256_file(config_path)
    if audit.get("config_sha256") != config_hash:
        errors.append("audit was produced from a different config")
    if not audit.get("training_authorized_by_audit"):
        errors.append("target audit did not authorize training")
    if args.dataset_tag != config["source_hyperparameters"]["dataset_tag"]:
        errors.append("dataset tag differs from the frozen config contract")

    data_config = dict(config["data"])
    data_config["master_dir"] = str(
        resolve(Path(data_config["master_dir"])).resolve()
    )
    bundle = prepare_data(data_config, seed=int(config["training"]["seed"]))
    errors.extend(validate_constant_targets(bundle))
    if (
        bundle.provenance["source_content_fingerprint"]
        != audit.get("source_content_fingerprint")
    ):
        errors.append("source data changed after the target audit")

    expected_validation = sorted(config["data"]["validation_participants"])
    expected_test = sorted(config["data"]["test_participants"])
    if bundle.split_metadata["participants"]["validation"] != expected_validation:
        errors.append("validation participant split differs from config")
    if bundle.split_metadata["participants"]["test"] != expected_test:
        errors.append("test participant split differs from config")

    model = HierarchicalResidualPoseTransformer(
        input_dim=len(bundle.normalizer.output_feature_names),
        window_size=int(config["data"]["window_size"]),
        **config["model"],
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    batch = next(
        iter(
            DataLoader(
                bundle.train,
                batch_size=min(8, int(config["training"]["batch_size"])),
                shuffle=False,
                num_workers=0,
            )
        )
    )
    with torch.inference_mode():
        outputs = model(batch["features"], batch["hand_reference_pose"])
    expected_shapes = {
        "assistance_logits": [len(batch["features"]), 2],
        "assistance_type_logits": [len(batch["features"]), 2],
        "receiving_hand_logits": [len(batch["features"]), 2],
        "pose_candidates": [len(batch["features"]), 2, 7],
    }
    for key, expected in expected_shapes.items():
        if list(outputs[key].shape) != expected:
            errors.append(
                f"model output {key} has shape {list(outputs[key].shape)}, "
                f"expected {expected}"
            )
        if not bool(torch.isfinite(outputs[key]).all()):
            errors.append(f"model output {key} contains non-finite values")

    planned_runs = []
    for seed in SEEDS:
        relative = Path(
            f"Training/runs/{args.dataset_tag}/{args.experiment_tag}/"
            f"residual_v2_endpose/{args.experiment_tag}_residual_v2_endpose_seed{seed}"
        )
        absolute = PROJECT_ROOT / relative
        exists = absolute.exists()
        if exists:
            errors.append(f"planned run path already exists: {relative}")
        planned_runs.append(
            {"seed": seed, "path": str(relative), "already_exists": exists}
        )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "training_started": False,
        "test_metrics_computed": False,
        "config": str(config_path),
        "config_sha256": config_hash,
        "audit_report": str(audit_path),
        "audit_report_sha256": sha256_file(audit_path),
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "source_content_fingerprint": bundle.provenance[
            "source_content_fingerprint"
        ],
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "selected_sequences": bundle.split_metadata["dataset_filter"][
            "selected_sequences"
        ],
        "split_participants": bundle.split_metadata["participants"],
        "windows": {
            split_name: len(getattr(bundle, split_name)) for split_name in SPLITS
        },
        "terminal_target_windows": {
            split_name: getattr(bundle, split_name).residual_pose_count()
            for split_name in SPLITS
        },
        "pose_target_definition": bundle.split_metadata["pose_target"],
        "trainable_parameters": trainable_parameters,
        "model_forward_checked": True,
        "planned_runs": planned_runs,
        "selection_split": "validation",
        "checkpoint_selection": [
            "validation_intention_macro_f1",
            "validation_pose_oracle_position_mae_cm",
        ],
        "seeds": list(SEEDS),
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"End-pose dry-run: {report['status']} | "
        f"sequences={report['selected_sequences']} | "
        f"windows={report['windows']} | params={trainable_parameters:,}"
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
    print(f"Report: {output_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
