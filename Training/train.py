#!/usr/bin/env python3
"""Train and evaluate hierarchical absolute-pose backbone comparisons."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import INTENTION_NAMES, DataBundle, prepare_data, save_data_metadata
from metrics import classification_metrics, pose_metrics
from model import (
    HierarchicalGRU,
    HierarchicalGatedMultimodalTransformer,
    HierarchicalWindowMLP,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_TYPE = "hierarchical_gated_multimodal_transformer"
MODEL_TYPES = {
    DEFAULT_MODEL_TYPE: HierarchicalGatedMultimodalTransformer,
    "hierarchical_window_mlp": HierarchicalWindowMLP,
    "hierarchical_gru": HierarchicalGRU,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/hierarchical_baseline_v1.json"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--limit-sequences", type=int, default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(counts: list[int], device: torch.device) -> torch.Tensor:
    values = torch.tensor(counts, dtype=torch.float32, device=device)
    if torch.any(values == 0):
        return torch.ones_like(values)
    inverse = values.sum() / (len(values) * values)
    return inverse / inverse.mean()


def build_model(
    config: dict,
    *,
    input_dim: int,
    window_size: int,
) -> tuple[nn.Module, str]:
    model_type = str(config.get("model_type", DEFAULT_MODEL_TYPE))
    model_class = MODEL_TYPES.get(model_type)
    if model_class is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {sorted(MODEL_TYPES)}"
        )
    model = model_class(
        input_dim=input_dim,
        window_size=window_size,
        **config["model"],
    )
    return model, model_type


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    assistance_criterion: nn.Module,
    assistance_type_criterion: nn.Module,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    intentions = batch["intention"]
    assistance_target = (intentions != 0).long()
    assistance_loss = assistance_criterion(
        outputs["assistance_logits"], assistance_target
    )

    assistance_valid = assistance_target.bool()
    if bool(assistance_valid.any()):
        assistance_type_target = intentions[assistance_valid] - 1
        assistance_type_loss = assistance_type_criterion(
            outputs["assistance_type_logits"][assistance_valid],
            assistance_type_target,
        )
    else:
        assistance_type_loss = outputs["assistance_type_logits"].sum() * 0.0

    pose_valid = batch["pose_valid"] & (intentions == 2)
    if bool(pose_valid.any()):
        predicted_pose = outputs["pose"][pose_valid]
        target_pose = batch["pose_target"][pose_valid]
        position_loss = F.smooth_l1_loss(predicted_pose[:, :3], target_pose[:, :3])
        quaternion_similarity = torch.sum(
            predicted_pose[:, 3:7] * target_pose[:, 3:7], dim=-1
        ).abs()
        orientation_loss = (1.0 - quaternion_similarity).mean()
        pose_loss = (
            position_loss + float(config["orientation_loss_weight"]) * orientation_loss
        )
    else:
        position_loss = outputs["pose"].sum() * 0.0
        orientation_loss = outputs["pose"].sum() * 0.0
        pose_loss = outputs["pose"].sum() * 0.0

    total = (
        float(config["assistance_loss_weight"]) * assistance_loss
        + float(config["assistance_type_loss_weight"]) * assistance_type_loss
        + float(config["pose_loss_weight"]) * pose_loss
    )
    components = {
        "total": float(total.detach()),
        "assistance": float(assistance_loss.detach()),
        "assistance_type": float(assistance_type_loss.detach()),
        "position": float(position_loss.detach()),
        "orientation": float(orientation_loss.detach()),
    }
    return total, components


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    assistance_criterion: nn.Module,
    assistance_type_criterion: nn.Module,
    training_config: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[dict[str, float]] = []
    intention_predictions = []
    intention_targets = []
    assistance_predictions = []
    assistance_targets = []
    assistance_type_predictions = []
    assistance_type_targets = []
    pose_predictions = []
    pose_targets = []
    gate_values = []

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in loader:
            batch = move_batch(batch, device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"])
            loss, components = multitask_loss(
                outputs,
                batch,
                assistance_criterion,
                assistance_type_criterion,
                training_config,
            )
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["gradient_clip_norm"])
                )
                optimizer.step()
            losses.append(components)

            intentions = batch["intention"]
            assistance_target = (intentions != 0).long()
            assistance_prediction = outputs["assistance_logits"].argmax(dim=-1)
            assistance_type_prediction = outputs["assistance_type_logits"].argmax(
                dim=-1
            )
            intention_prediction = torch.zeros_like(intentions)
            predicted_assistance = assistance_prediction.bool()
            intention_prediction[predicted_assistance] = (
                assistance_type_prediction[predicted_assistance] + 1
            )

            intention_predictions.append(intention_prediction.cpu())
            intention_targets.append(intentions.cpu())
            assistance_predictions.append(assistance_prediction.cpu())
            assistance_targets.append(assistance_target.cpu())
            assistance_valid = assistance_target.bool()
            if bool(assistance_valid.any()):
                assistance_type_predictions.append(
                    assistance_type_prediction[assistance_valid].cpu()
                )
                assistance_type_targets.append((intentions[assistance_valid] - 1).cpu())

            pose_valid = batch["pose_valid"] & (intentions == 2)
            if bool(pose_valid.any()):
                pose_predictions.append(outputs["pose"][pose_valid].detach().cpu())
                pose_targets.append(batch["pose_target"][pose_valid].cpu())
            if "gate" in outputs:
                gate_values.append(outputs["gate"].detach().cpu())

    if not losses:
        raise RuntimeError("DataLoader produced no batches")
    averaged_losses = {
        key: sum(item[key] for item in losses) / len(losses) for key in losses[0]
    }
    intention = classification_metrics(
        torch.cat(intention_predictions),
        torch.cat(intention_targets),
        len(INTENTION_NAMES),
    )
    intention["class_names"] = INTENTION_NAMES
    assistance = classification_metrics(
        torch.cat(assistance_predictions), torch.cat(assistance_targets), 2
    )
    assistance["class_names"] = ["continue", "assistance"]
    if assistance_type_targets:
        assistance_type = classification_metrics(
            torch.cat(assistance_type_predictions),
            torch.cat(assistance_type_targets),
            2,
        )
        assistance_type["class_names"] = ["fetch", "handover"]
    else:
        assistance_type = {
            "accuracy": None,
            "macro_f1": None,
            "macro_f1_supported": None,
            "per_class_f1": [],
            "support": [],
            "samples": 0,
            "confusion_matrix": [],
            "class_names": ["fetch", "handover"],
        }
    if pose_targets:
        poses = pose_metrics(torch.cat(pose_predictions), torch.cat(pose_targets))
    else:
        poses = pose_metrics(torch.empty((0, 7)), torch.empty((0, 7)))
    mean_gate = None
    if gate_values:
        gate = torch.cat(gate_values).mean(dim=0).tolist()
        mean_gate = {"temporal": gate[0], "channel": gate[1]}
    return {
        "loss": averaged_losses,
        "intention": intention,
        "assistance": assistance,
        "assistance_type": assistance_type,
        "pose": poses,
        "mean_gate": mean_gate,
    }


def make_loader(
    dataset, config: dict, *, shuffle: bool, device: torch.device
) -> DataLoader:
    worker_count = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )


def train(args: argparse.Namespace) -> Path:
    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["master_dir"] = str(
        resolve_project_path(config["data"]["master_dir"])
    )
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if getattr(args, "seed", None) is not None:
        config["training"]["seed"] = args.seed
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = choose_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(config.get("run_name", "hierarchical_baseline"))
    run_dir = resolve_project_path(
        args.run_dir or f"Training/runs/{run_name}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    bundle: DataBundle = prepare_data(config["data"], seed, args.limit_sequences)
    save_data_metadata(bundle, run_dir / "data_metadata.json")
    print(
        f"Windows: train={len(bundle.train)}, validation={len(bundle.validation)}, "
        f"test={len(bundle.test)}"
    )
    print(f"Participant split: {bundle.split_metadata['participants']}")
    dataset_filter = bundle.split_metadata["dataset_filter"]
    print(
        "Dataset filter: "
        f"selected={dataset_filter['selected_sequences']}, "
        f"excluded_master_files={dataset_filter.get('excluded_master_files', 0)}, "
        f"fingerprint={dataset_filter['sequence_fingerprint']}"
    )
    print(
        "Discarded gap windows: "
        f"train={bundle.train.discarded_gap_windows}, "
        f"validation={bundle.validation.discarded_gap_windows}, "
        f"test={bundle.test.discarded_gap_windows}"
    )
    print(
        "Discarded unlabeled endpoints: "
        f"train={bundle.train.discarded_unlabeled_windows}, "
        f"validation={bundle.validation.discarded_unlabeled_windows}, "
        f"test={bundle.test.discarded_unlabeled_windows}"
    )

    training_config = config["training"]
    train_loader = make_loader(
        bundle.train, training_config, shuffle=True, device=device
    )
    validation_loader = make_loader(
        bundle.validation, training_config, shuffle=False, device=device
    )
    test_loader = make_loader(
        bundle.test, training_config, shuffle=False, device=device
    )

    model, model_type = build_model(
        config,
        input_dim=len(bundle.normalizer.output_feature_names),
        window_size=int(config["data"]["window_size"]),
    )
    model = model.to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Model: {model_type}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    intention_counts = bundle.train.intention_counts()
    assistance_criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            [intention_counts[0], intention_counts[1] + intention_counts[2]], device
        )
    )
    assistance_type_criterion = nn.CrossEntropyLoss(
        weight=class_weights(intention_counts[1:3], device)
    )

    best_score = -math.inf
    best_pose = math.inf
    epochs_without_improvement = 0
    history = []
    checkpoint_path = run_dir / "best_model.pt"
    pose_checkpoint_path = run_dir / "best_pose_model.pt"
    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            training_config,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            training_config,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        score = float(validation_metrics["intention"]["macro_f1"])
        pose_value = validation_metrics["pose"]["position_mae_cm"]
        pose_score = float(pose_value) if pose_value is not None else math.inf
        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']['total']:.4f} | "
            f"val intent F1={score:.4f} | "
            f"val assistance F1={validation_metrics['assistance']['macro_f1']:.4f} | "
            f"val fetch/handover F1={validation_metrics['assistance_type']['macro_f1']:.4f} | "
            f"val pose cm={validation_metrics['pose']['position_mae_cm']}"
        )
        improved = False
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config["model"],
                    "model_type": model_type,
                    "input_dim": len(bundle.normalizer.output_feature_names),
                    "window_size": int(config["data"]["window_size"]),
                    "trainable_parameters": trainable_parameters,
                    "epoch": epoch,
                    "selection_metric": "validation_intention_macro_f1",
                    "selection_value": score,
                    "validation_intention_macro_f1": score,
                },
                checkpoint_path,
            )
            improved = True
        if pose_score < best_pose:
            best_pose = pose_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config["model"],
                    "model_type": model_type,
                    "input_dim": len(bundle.normalizer.output_feature_names),
                    "window_size": int(config["data"]["window_size"]),
                    "trainable_parameters": trainable_parameters,
                    "epoch": epoch,
                    "selection_metric": "validation_pose_position_mae_cm",
                    "selection_value": pose_score,
                },
                pose_checkpoint_path,
            )
            improved = True
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if epochs_without_improvement >= int(
            training_config["early_stopping_patience"]
        ):
            print("Early stopping")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        assistance_criterion,
        assistance_type_criterion,
        training_config,
    )
    pose_checkpoint = torch.load(
        pose_checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(pose_checkpoint["model_state_dict"])
    pose_checkpoint_test_metrics = run_epoch(
        model,
        test_loader,
        device,
        assistance_criterion,
        assistance_type_criterion,
        training_config,
    )
    report = {
        "model_type": model_type,
        "trainable_parameters": trainable_parameters,
        "best_epoch": checkpoint["epoch"],
        "best_validation_intention_macro_f1": best_score,
        "best_validation_pose_position_mae_cm": best_pose,
        "test": test_metrics,
        "checkpoints": {
            "best_intention": {
                "path": str(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                "selection_metric": checkpoint["selection_metric"],
                "selection_value": float(checkpoint["selection_value"]),
            },
            "best_pose": {
                "path": str(pose_checkpoint_path),
                "epoch": int(pose_checkpoint["epoch"]),
                "selection_metric": pose_checkpoint["selection_metric"],
                "selection_value": float(pose_checkpoint["selection_value"]),
            },
        },
        "test_by_checkpoint": {
            "best_intention": test_metrics,
            "best_pose": pose_checkpoint_test_metrics,
        },
        "history": history,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Test intention macro F1: {test_metrics['intention']['macro_f1']:.4f}")
    print(f"Test assistance macro F1: {test_metrics['assistance']['macro_f1']:.4f}")
    print(
        "Test fetch/handover macro F1: "
        f"{test_metrics['assistance_type']['macro_f1']:.4f}"
    )
    print(
        "Best-pose checkpoint test position MAE: "
        f"{pose_checkpoint_test_metrics['pose']['position_mae_cm']} cm"
    )
    print(f"Metrics: {run_dir / 'metrics.json'}")
    return run_dir


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
