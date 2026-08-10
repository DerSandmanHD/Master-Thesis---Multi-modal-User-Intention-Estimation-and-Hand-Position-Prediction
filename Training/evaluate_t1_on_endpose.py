#!/usr/bin/env python3
"""Evaluate an existing t+1 Residual-v2 checkpoint against terminal targets."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from data import prepare_data, sha256_file
from model import HierarchicalResidualPoseTransformer
from train_residual import choose_device, class_weights, make_loader, run_epoch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_endpose_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    source_run = resolve(args.source_run_dir).resolve()
    target_config_path = resolve(args.target_config).resolve()
    output_path = resolve(args.output).resolve()
    source_config_path = source_run / "config.json"
    source_metrics_path = source_run / "metrics.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    seed = int(source_config["training"]["seed"])
    data_config = dict(target_config["data"])
    data_config["master_dir"] = str(
        resolve(Path(data_config["master_dir"])).resolve()
    )
    bundle = prepare_data(data_config, seed=seed)
    device = choose_device(args.device)
    evaluation_training_config = dict(source_config["training"])
    evaluation_training_config["num_workers"] = int(
        target_config["training"].get("num_workers", 0)
    )
    validation_loader = make_loader(
        bundle.validation,
        evaluation_training_config,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        bundle.test,
        evaluation_training_config,
        shuffle=False,
        device=device,
    )
    intention_counts = bundle.train.intention_counts()
    assistance_criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            [intention_counts[0], intention_counts[1] + intention_counts[2]],
            device,
        )
    )
    assistance_type_criterion = nn.CrossEntropyLoss(
        weight=class_weights(intention_counts[1:3], device)
    )
    receiving_hand_criterion = nn.CrossEntropyLoss(
        weight=class_weights(bundle.train.receiving_hand_counts(), device)
    )

    results = {"validation": {}, "test": {}}
    checkpoint_metadata = {}
    trainable_parameters = None
    for checkpoint_name, filename in (
        ("best_intention", "best_intention_model.pt"),
        ("best_pose", "best_pose_model.pt"),
    ):
        checkpoint_path = source_run / filename
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        input_dim = len(bundle.normalizer.output_feature_names)
        if int(checkpoint["input_dim"]) != input_dim:
            raise ValueError(
                f"Input dimension mismatch: checkpoint={checkpoint['input_dim']}, "
                f"terminal data={input_dim}"
            )
        model = HierarchicalResidualPoseTransformer(
            input_dim=input_dim,
            window_size=int(checkpoint["window_size"]),
            **checkpoint["model_config"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        results["validation"][checkpoint_name] = run_epoch(
            model,
            validation_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            receiving_hand_criterion,
            evaluation_training_config,
        )
        results["test"][checkpoint_name] = run_epoch(
            model,
            test_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            receiving_hand_criterion,
            evaluation_training_config,
        )
        checkpoint_metadata[checkpoint_name] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "source_epoch": int(checkpoint["epoch"]),
            "source_selection_metric": checkpoint["selection_metric"],
            "source_selection_value": float(checkpoint["selection_value"]),
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "existing_t_plus_1_model_used_as_terminal_pose_estimator",
        "source_model_target": "receiving-hand pose at t+1 second",
        "evaluation_target": "robust terminal receiving-hand pose",
        "source_run_dir": str(source_run),
        "source_config_sha256": sha256_file(source_config_path),
        "source_metrics_sha256": sha256_file(source_metrics_path),
        "target_config": str(target_config_path),
        "target_config_sha256": sha256_file(target_config_path),
        "seed": seed,
        "device": str(device),
        "trainable_parameters": trainable_parameters,
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "source_content_fingerprint": bundle.provenance[
            "source_content_fingerprint"
        ],
        "pose_target_definition": bundle.split_metadata["pose_target"],
        "split_participants": bundle.split_metadata["participants"],
        "checkpoint_selection_reused_without_test_reselection": True,
        "checkpoints": checkpoint_metadata,
        "validation_by_checkpoint": results["validation"],
        "test": results["test"],
        "source_native_t_plus_1_test": source_metrics.get("test"),
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pose = report["test"]["best_pose"]["pose_oracle"]
    print(
        f"t+1-as-terminal seed {seed}: position={pose['position_mae_cm']} cm, "
        f"orientation={pose['orientation_mean_deg']} deg"
    )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
