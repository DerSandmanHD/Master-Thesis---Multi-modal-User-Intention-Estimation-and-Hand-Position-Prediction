#!/usr/bin/env python3
"""Train and evaluate hierarchical absolute-pose backbone comparisons."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from artifact_freeze import (
    ARTIFACT_FREEZE_PROTOCOL,
    finalize_artifact_freeze,
    sha256_file,
    start_artifact_freeze,
)
from data import (
    INTENTION_NAMES,
    DataBundle,
    checkpoint_provenance,
    prepare_data,
    save_data_metadata,
)
from metrics import (
    POSITION_ERROR_DEFINITION,
    POSITION_RMS_ERROR_DEFINITION,
    classification_metrics,
    pose_metrics,
    sample_key_fingerprint,
)
from model import (
    HierarchicalGRU,
    HierarchicalGatedMultimodalTransformer,
    HierarchicalWindowMLP,
)
from run_layout import build_run_context, training_run_directory
from training_control import (
    available_validation_checkpoints,
    finite_diagnostic_improved,
    next_primary_patience,
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
        default=Path("Training/configs/models/transformer_v1.json"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Immutable dataset version used in the structured run path.",
    )
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help="Experiment group used in the structured run path.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Train/select checkpoints on train/validation without evaluating test.",
    )
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
    receiving_hand_criterion: nn.Module,
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

    receiving_hand = batch["receiving_hand"]
    hand_valid = (
        (intentions == 2) & (receiving_hand >= 0) & (receiving_hand < 2)
    )
    if bool(hand_valid.any()):
        receiving_hand_loss = receiving_hand_criterion(
            outputs["receiving_hand_logits"][hand_valid],
            receiving_hand[hand_valid],
        )
    else:
        receiving_hand_loss = outputs["receiving_hand_logits"].sum() * 0.0

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
        + float(config.get("receiving_hand_loss_weight", 1.0))
        * receiving_hand_loss
        + float(config["pose_loss_weight"]) * pose_loss
    )
    components = {
        "total": float(total.detach()),
        "assistance": float(assistance_loss.detach()),
        "assistance_type": float(assistance_type_loss.detach()),
        "receiving_hand": float(receiving_hand_loss.detach()),
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
    receiving_hand_criterion: nn.Module,
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
    hand_predictions = []
    hand_targets = []
    pose_predictions = []
    pose_targets = []
    fixed_pose_predictions = []
    fixed_pose_targets = []
    fixed_pose_sample_keys: list[str] = []
    pose_target_count = 0
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
                receiving_hand_criterion,
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

            receiving_hand = batch["receiving_hand"]
            hand_valid = (
                (intentions == 2)
                & (receiving_hand >= 0)
                & (receiving_hand < 2)
            )
            if bool(hand_valid.any()):
                hand_predictions.append(
                    outputs["receiving_hand_logits"]
                    .argmax(dim=-1)[hand_valid]
                    .cpu()
                )
                hand_targets.append(receiving_hand[hand_valid].cpu())

            pose_valid = batch["pose_valid"] & (intentions == 2)
            pose_target_count += int(pose_valid.sum().item())
            if bool(pose_valid.any()):
                pose_predictions.append(outputs["pose"][pose_valid].detach().cpu())
                pose_targets.append(batch["pose_target"][pose_valid].cpu())
            if "hand_reference_valid" in batch:
                fixed_pose_valid = (
                    pose_valid
                    & hand_valid
                    & batch["hand_reference_valid"].all(dim=1)
                )
                if bool(fixed_pose_valid.any()):
                    fixed_pose_predictions.append(
                        outputs["pose"][fixed_pose_valid].detach().cpu()
                    )
                    fixed_pose_targets.append(
                        batch["pose_target"][fixed_pose_valid].cpu()
                    )
                    valid_indices = fixed_pose_valid.nonzero(as_tuple=False).view(-1)
                    fixed_pose_sample_keys.extend(
                        f"{batch['sequence_id'][int(index)]}|"
                        f"{int(batch['timestamp_ns'][int(index)])}"
                        for index in valid_indices
                    )
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
    if hand_targets:
        receiving_hand_metrics = classification_metrics(
            torch.cat(hand_predictions), torch.cat(hand_targets), 2
        )
    else:
        receiving_hand_metrics = classification_metrics(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            2,
        )
    receiving_hand_metrics["class_names"] = ["left", "right"]
    if pose_targets:
        poses = pose_metrics(torch.cat(pose_predictions), torch.cat(pose_targets))
    else:
        poses = pose_metrics(torch.empty((0, 7)), torch.empty((0, 7)))
    if fixed_pose_targets:
        fixed_poses = pose_metrics(
            torch.cat(fixed_pose_predictions), torch.cat(fixed_pose_targets)
        )
    else:
        fixed_poses = pose_metrics(torch.empty((0, 7)), torch.empty((0, 7)))
    mean_gate = None
    if gate_values:
        gate = torch.cat(gate_values).mean(dim=0).tolist()
        mean_gate = {"temporal": gate[0], "channel": gate[1]}
    return {
        "loss": averaged_losses,
        "intention": intention,
        "assistance": assistance,
        "assistance_type": assistance_type,
        "receiving_hand": receiving_hand_metrics,
        "pose": poses,
        "pose_fixed_both_references": {
            **fixed_poses,
            "cohort_definition": (
                "pose_target_valid_and_both_hand_references_valid"
            ),
            "cohort_model_dependent": False,
            "coverage_denominator_pose_targets": pose_target_count,
            "sample_key_fingerprint": sample_key_fingerprint(
                fixed_pose_sample_keys
            ),
        },
        "pose_coverage": {
            "pose_targets": pose_target_count,
            "future_targets": pose_target_count,
            "predicted_pose_valid": int(poses["samples"]),
            "coverage": (
                float(poses["samples"] / pose_target_count)
                if pose_target_count
                else None
            ),
            "receiving_hand_context": (
                "legacy parallel single-pose baseline head; receiving-hand is "
                "predicted by a separate classification head"
            ),
        },
        "mean_gate": mean_gate,
    }


def make_loader(
    dataset, config: dict, *, shuffle: bool, device: torch.device
) -> DataLoader:
    worker_count = int(config["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )


def train(args: argparse.Namespace) -> Path:
    started_at = datetime.now().astimezone()
    started_monotonic = time.perf_counter()
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
    run_context = build_run_context(
        dataset_tag=getattr(args, "dataset_tag", None),
        experiment_tag=getattr(args, "experiment_tag", None),
        model_tag=run_name,
    )
    config["run_context"] = run_context
    skip_test_evaluation = bool(
        getattr(args, "skip_test_evaluation", False)
    )
    config["evaluation"] = {
        "validation_checkpoints": ["best_intention", "best_pose"],
        "test_evaluation_enabled": not skip_test_evaluation,
        "selection_split": "validation",
        "primary_checkpoint": "best_intention",
        "primary_checkpoint_rule": "maximize validation intention macro-F1",
        "pose_selected_checkpoint_role": "diagnostic_only",
    }
    run_dir = resolve_project_path(
        args.run_dir
        or training_run_directory(
            dataset_tag=run_context["dataset_tag"],
            experiment_tag=run_context["experiment_tag"],
            model_tag=run_context["model_tag"],
            seed=seed,
            timestamp=timestamp,
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    bundle: DataBundle = prepare_data(config["data"], seed, args.limit_sequences)
    save_data_metadata(bundle, run_dir / "data_metadata.json")
    artifact_manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=config_path,
        run_context=run_context,
        seed=seed,
        selection_policy={
            "selection_split": "validation",
            "primary_checkpoint": "best_intention",
            "primary_checkpoint_rule": "maximize validation intention macro-F1",
            "pose_selected_checkpoint": "best_pose",
            "pose_selected_checkpoint_role": "diagnostic_only",
            "pose_selected_checkpoint_rule": "minimize validation pose position error",
            "test_used_for_selection": False,
        },
        started_at=started_at.isoformat(),
    )
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
    test_loader = (
        None
        if skip_test_evaluation
        else make_loader(bundle.test, training_config, shuffle=False, device=device)
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
    receiving_hand_criterion = nn.CrossEntropyLoss(
        weight=class_weights(bundle.train.receiving_hand_counts(), device)
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
            receiving_hand_criterion,
            training_config,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            receiving_hand_criterion,
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
            "val pose mean Euclidean cm="
            f"{validation_metrics['pose']['position_mae_cm']}"
        )
        primary_improved = score > best_score
        if primary_improved:
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
                    "dataset_provenance": checkpoint_provenance(bundle),
                },
                checkpoint_path,
            )
        diagnostic_improved = finite_diagnostic_improved(pose_score, best_pose)
        if diagnostic_improved:
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
                    "selection_metric_definition": (
                        POSITION_ERROR_DEFINITION
                    ),
                    "dataset_provenance": checkpoint_provenance(bundle),
                },
                pose_checkpoint_path,
            )
        epochs_without_improvement = next_primary_patience(
            epochs_without_improvement,
            primary_improved=primary_improved,
            diagnostic_improved=diagnostic_improved,
        )
        if epochs_without_improvement >= int(
            training_config["early_stopping_patience"]
        ):
            print("Early stopping")
            break

    validation_results = {}
    test_results = {}
    checkpoint_metadata = {}
    loaded_checkpoints = {}
    checkpoints_to_evaluate, pose_diagnostic_status = (
        available_validation_checkpoints(checkpoint_path, pose_checkpoint_path)
    )
    for name, path in checkpoints_to_evaluate:
        checkpoint_value = torch.load(path, map_location=device, weights_only=True)
        loaded_checkpoints[name] = checkpoint_value
        model.load_state_dict(checkpoint_value["model_state_dict"])
        validation_results[name] = run_epoch(
            model,
            validation_loader,
            device,
            assistance_criterion,
            assistance_type_criterion,
            receiving_hand_criterion,
            training_config,
        )
        if test_loader is not None:
            test_results[name] = run_epoch(
                model,
                test_loader,
                device,
                assistance_criterion,
                assistance_type_criterion,
                receiving_hand_criterion,
                training_config,
            )
        checkpoint_metadata[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": int(checkpoint_value["epoch"]),
            "selection_metric": checkpoint_value["selection_metric"],
            "selection_value": float(checkpoint_value["selection_value"]),
            "selection_metric_definition": checkpoint_value.get(
                "selection_metric_definition"
            ),
        }
    checkpoint = loaded_checkpoints["best_intention"]
    report = {
        "model_type": model_type,
        "trainable_parameters": trainable_parameters,
        "best_epoch": checkpoint["epoch"],
        "best_validation_intention_macro_f1": best_score,
        "best_validation_pose_position_mae_cm": (
            best_pose if math.isfinite(best_pose) else None
        ),
        "best_validation_pose_mean_euclidean_error_cm": (
            best_pose if math.isfinite(best_pose) else None
        ),
        "pose_selected_diagnostic": pose_diagnostic_status,
        "legacy_pose_metric_alias": {
            "position_mae_cm": POSITION_ERROR_DEFINITION,
            "position_rmse_cm": POSITION_RMS_ERROR_DEFINITION,
        },
        "checkpoints": checkpoint_metadata,
        "validation_by_checkpoint": validation_results,
        "test_evaluation_skipped": skip_test_evaluation,
        "history": history,
        "run_context": run_context,
        "code_provenance": bundle.provenance.get("git", {}),
        "runtime": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": time.perf_counter() - started_monotonic,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "artifact_freeze": {
            "protocol": ARTIFACT_FREEZE_PROTOCOL,
            "manifest": str(artifact_manifest_path),
        },
    }
    if test_results:
        report["test"] = test_results["best_intention"]
        report["test_by_checkpoint"] = test_results
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frozen_checkpoints = {"best_intention": checkpoint_path}
    if pose_diagnostic_status["available"]:
        frozen_checkpoints["best_pose_diagnostic"] = pose_checkpoint_path
    finalize_artifact_freeze(
        artifact_manifest_path,
        checkpoint_paths=frozen_checkpoints,
        metrics_path=metrics_path,
        completed_at=report["runtime"]["completed_at"],
    )
    if test_results:
        test_metrics = test_results["best_intention"]
        print(f"Test intention macro F1: {test_metrics['intention']['macro_f1']:.4f}")
        print(f"Test assistance macro F1: {test_metrics['assistance']['macro_f1']:.4f}")
        print(
            "Test fetch/handover macro F1: "
            f"{test_metrics['assistance_type']['macro_f1']:.4f}"
        )
        if "best_pose" in test_results:
            print(
                "Diagnostic best-pose checkpoint test mean Euclidean position error="
                f"{test_results['best_pose']['pose']['position_mae_cm']} cm"
            )
        else:
            print(
                "Diagnostic best-pose checkpoint unavailable: "
                f"{pose_diagnostic_status['reason']}"
            )
    else:
        print("Test evaluation skipped by request")
    print(f"Metrics: {metrics_path}")
    print(f"Artifact freeze: {artifact_manifest_path}")
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
