#!/usr/bin/env python3
"""Audit the improved dual-horizon endpose-v2 path without training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data, sha256_file
from endpose_v2 import residual_position_scale_m, wrap_endpose_v2_bundle
from train_residual import build_residual_model, residual_multitask_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_endpose_v2.json"),
    )
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config).resolve()
    audit_path = resolve(args.audit_report).resolve()
    output = resolve(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    errors = []
    if audit.get("config_sha256") != sha256_file(config_path):
        errors.append("audit was produced from a different base config")
    if audit.get("training_authorized_by_audit") is not True:
        errors.append("target audit did not authorize training")
    if args.dataset_tag != config["source_hyperparameters"]["dataset_tag"]:
        errors.append("dataset tag differs from frozen config")

    data_config = dict(config["data"])
    data_config["master_dir"] = str(resolve(Path(data_config["master_dir"])))
    bundle = wrap_endpose_v2_bundle(
        prepare_data(data_config, seed=int(config["training"]["seed"])),
        data_config,
    )
    if bundle.provenance["source_content_fingerprint"] != audit.get(
        "source_content_fingerprint"
    ):
        errors.append("source data differs from audit")

    training = dict(config["training"])
    minimum_scale = float(training["position_loss"]["minimum_scale_m"])
    training["resolved_position_scale_m"] = residual_position_scale_m(
        bundle.train,
        target_key="pose_target",
        valid_key="residual_pose_valid",
        minimum_scale_m=minimum_scale,
    )
    training["resolved_auxiliary_position_scale_m"] = residual_position_scale_m(
        bundle.train,
        target_key="auxiliary_pose_target",
        valid_key="auxiliary_residual_pose_valid",
        minimum_scale_m=minimum_scale,
    )
    model = build_residual_model(
        config["model_type"],
        input_dim=len(bundle.normalizer.output_feature_names),
        window_size=int(data_config["window_size"]),
        model_config=config["model"],
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    batch = next(
        iter(DataLoader(bundle.train, batch_size=8, shuffle=False, num_workers=0))
    )
    outputs = model(batch["features"], batch["hand_reference_pose"])
    expected_shapes = {
        "pose_candidates": [len(batch["features"]), 2, 7],
        "auxiliary_pose_candidates": [len(batch["features"]), 2, 7],
        "receiving_hand_logits": [len(batch["features"]), 2],
    }
    for key, shape in expected_shapes.items():
        if list(outputs[key].shape) != shape:
            errors.append(f"{key} shape {list(outputs[key].shape)} != {shape}")
        if not bool(torch.isfinite(outputs[key]).all()):
            errors.append(f"{key} contains non-finite values")
    loss, components = residual_multitask_loss(
        outputs,
        batch,
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        training,
    )
    if not bool(torch.isfinite(loss)) or not all(
        np.isfinite(value) for value in components.values()
    ):
        errors.append("dry-run loss is non-finite")

    sampling_mass = {}
    weights = bundle.train.sequence_sampling_weights().numpy()
    for weight, (record_index, _) in zip(weights, bundle.train.indices):
        sequence_id = bundle.train.records[record_index].sequence_id
        sampling_mass[sequence_id] = sampling_mass.get(sequence_id, 0.0) + float(weight)
    mass_values = list(sampling_mass.values())
    if max(mass_values) - min(mass_values) > 1e-6:
        errors.append("sequence-balanced sampling mass is not equal")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "training_started": False,
        "test_metrics_computed": False,
        "dataset_tag": args.dataset_tag,
        "experiment_tag": args.experiment_tag,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "audit_report": str(audit_path),
        "audit_report_sha256": sha256_file(audit_path),
        "selected_sequences": bundle.split_metadata["dataset_filter"][
            "selected_sequences"
        ],
        "sequence_fingerprint": bundle.split_metadata["dataset_filter"][
            "sequence_fingerprint"
        ],
        "source_content_fingerprint": bundle.provenance[
            "source_content_fingerprint"
        ],
        "split_participants": bundle.split_metadata["participants"],
        "windows": {name: len(getattr(bundle, name)) for name in SPLITS},
        "terminal_target_windows": {
            name: getattr(bundle, name).residual_pose_count() for name in SPLITS
        },
        "auxiliary_t1_target_windows": {
            name: getattr(bundle, name).auxiliary_pose_count() for name in SPLITS
        },
        "resolved_position_scale_m": training["resolved_position_scale_m"],
        "resolved_auxiliary_position_scale_m": training[
            "resolved_auxiliary_position_scale_m"
        ],
        "sequence_sampling_mass_min": min(mass_values),
        "sequence_sampling_mass_max": max(mass_values),
        "trainable_parameters": trainable_parameters,
        "model_forward_checked": True,
        "loss_forward_checked": True,
        "selection_split": "validation",
        "test_forbidden_during_search": True,
        "planned_search_trials": 12,
        "planned_confirmation_runs": 9,
        "planned_final_seeds": [42, 43, 44],
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Endpose-v2 dry-run: {report['status']}; "
        f"parameters={trainable_parameters:,}; "
        f"auxiliary targets={report['auxiliary_t1_target_windows']}"
    )
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Report: {output}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
