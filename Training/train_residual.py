#!/usr/bin/env python3
"""Train hierarchical intention and residual receiving-hand pose model v2."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (
    INTENTION_NAMES,
    RECEIVING_HAND_NAMES,
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
)
from endpose_v2 import (
    DUAL_HORIZON_MODEL_TYPE,
    residual_position_scale_m,
    wrap_endpose_v2_bundle,
)
from model import (
    HierarchicalDualHorizonResidualPoseTransformer,
    HierarchicalResidualPoseTransformer,
)
from run_layout import build_run_context, training_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_GROUPS = ("0-25%", "25-50%", "50-75%", "75-100%")
TIME_TO_END_GROUPS = ("0-0.5s", "0.5-1s", "1-2s", "2-3s", ">=3s")
RESIDUAL_V2_MODEL_TYPE = "hierarchical_residual_pose_transformer_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_v2.json"),
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
        help=(
            "Do not construct a test DataLoader or compute test metrics. "
            "Validation metrics are still evaluated for both selected checkpoints."
        ),
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


def select_hand_pose(candidates: torch.Tensor, hand_ids: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 3 or candidates.shape[1:] != (2, 7):
        raise ValueError(
            f"Expected pose candidates [batch, 2, 7], got {candidates.shape}"
        )
    safe_ids = hand_ids.clamp(0, 1)
    batch_indices = torch.arange(len(candidates), device=candidates.device)
    return candidates[batch_indices, safe_ids]


def progress_masks(progress: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "0-25%": (progress >= 0.0) & (progress <= 0.25),
        "25-50%": (progress > 0.25) & (progress <= 0.5),
        "50-75%": (progress > 0.5) & (progress <= 0.75),
        "75-100%": progress > 0.75,
    }


def time_to_end_masks(seconds: torch.Tensor) -> dict[str, torch.Tensor]:
    finite = torch.isfinite(seconds) & (seconds >= 0.0)
    return {
        "0-0.5s": finite & (seconds <= 0.5),
        "0.5-1s": finite & (seconds > 0.5) & (seconds <= 1.0),
        "1-2s": finite & (seconds > 1.0) & (seconds <= 2.0),
        "2-3s": finite & (seconds > 2.0) & (seconds <= 3.0),
        ">=3s": finite & (seconds > 3.0),
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    safe = weights.to(dtype=values.dtype).clamp_min(0.0)
    denominator = safe.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (values * safe).sum() / denominator


def pose_regression_components(
    predicted_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    training_config: dict,
    position_scale_key: str,
    sample_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_config = dict(training_config.get("position_loss", {}))
    position_type = position_config.get("type", "smooth_l1_meters")
    if position_type == "normalized_smooth_l1":
        scale_values = training_config.get(position_scale_key)
        if scale_values is None:
            raise ValueError(f"Missing resolved loss scale: {position_scale_key}")
        scale = torch.as_tensor(
            scale_values,
            dtype=predicted_pose.dtype,
            device=predicted_pose.device,
        )
        normalized_error = (predicted_pose[:, :3] - target_pose[:, :3]) / scale
        per_axis = F.smooth_l1_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            reduction="none",
            beta=float(position_config.get("beta", 1.0)),
        )
    elif position_type == "smooth_l1_meters":
        per_axis = F.smooth_l1_loss(
            predicted_pose[:, :3],
            target_pose[:, :3],
            reduction="none",
            beta=float(position_config.get("beta_m", 1.0)),
        )
    else:
        raise ValueError(f"Unknown position loss type: {position_type}")
    position_loss = weighted_mean(per_axis.mean(dim=-1), sample_weights)

    quaternion_similarity = torch.sum(
        predicted_pose[:, 3:7] * target_pose[:, 3:7], dim=-1
    ).abs().clamp(0.0, 1.0)
    orientation_type = dict(training_config.get("orientation_loss", {})).get(
        "type", "cosine_distance"
    )
    if orientation_type == "cosine_distance":
        per_orientation = 1.0 - quaternion_similarity
    elif orientation_type == "geodesic_radians":
        # Clamp below one to avoid the singular derivative of acos at identity.
        per_orientation = 2.0 * torch.acos(
            quaternion_similarity.clamp(max=1.0 - 1e-7)
        )
    else:
        raise ValueError(f"Unknown orientation loss type: {orientation_type}")
    orientation_loss = weighted_mean(per_orientation, sample_weights)
    return position_loss, orientation_loss


def residual_multitask_loss(
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
        assistance_type_loss = assistance_type_criterion(
            outputs["assistance_type_logits"][assistance_valid],
            intentions[assistance_valid] - 1,
        )
    else:
        assistance_type_loss = outputs["assistance_type_logits"].sum() * 0.0

    receiving_hand = batch["receiving_hand"]
    hand_valid = (intentions == 2) & (receiving_hand >= 0) & (receiving_hand < 2)
    if bool(hand_valid.any()):
        receiving_hand_loss = receiving_hand_criterion(
            outputs["receiving_hand_logits"][hand_valid], receiving_hand[hand_valid]
        )
    else:
        receiving_hand_loss = outputs["receiving_hand_logits"].sum() * 0.0

    pose_valid = batch["residual_pose_valid"] & hand_valid
    if bool(pose_valid.any()):
        oracle_pose = select_hand_pose(outputs["pose_candidates"], receiving_hand)
        predicted_pose = oracle_pose[pose_valid]
        target_pose = batch["pose_target"][pose_valid]
        pose_weights = (
            batch["pose_sample_weight"][pose_valid]
            if "pose_sample_weight" in batch
            else None
        )
        position_loss, orientation_loss = pose_regression_components(
            predicted_pose,
            target_pose,
            training_config=config,
            position_scale_key="resolved_position_scale_m",
            sample_weights=pose_weights,
        )
        pose_loss = (
            position_loss + float(config["orientation_loss_weight"]) * orientation_loss
        )
    else:
        position_loss = outputs["position_delta"].sum() * 0.0
        orientation_loss = outputs["quaternion_delta"].sum() * 0.0
        pose_loss = outputs["position_delta"].sum() * 0.0

    auxiliary_pose_loss = outputs["position_delta"].sum() * 0.0
    auxiliary_position_loss = outputs["position_delta"].sum() * 0.0
    auxiliary_orientation_loss = outputs["quaternion_delta"].sum() * 0.0
    if "auxiliary_pose_candidates" in outputs:
        auxiliary_valid = batch["auxiliary_residual_pose_valid"] & hand_valid
        if bool(auxiliary_valid.any()):
            auxiliary_candidates = select_hand_pose(
                outputs["auxiliary_pose_candidates"], receiving_hand
            )
            auxiliary_weights = (
                batch["pose_sample_weight"][auxiliary_valid]
                if "pose_sample_weight" in batch
                else None
            )
            (
                auxiliary_position_loss,
                auxiliary_orientation_loss,
            ) = pose_regression_components(
                auxiliary_candidates[auxiliary_valid],
                batch["auxiliary_pose_target"][auxiliary_valid],
                training_config=config,
                position_scale_key="resolved_auxiliary_position_scale_m",
                sample_weights=auxiliary_weights,
            )
            auxiliary_pose_loss = auxiliary_position_loss + float(
                config.get(
                    "auxiliary_orientation_loss_weight",
                    config["orientation_loss_weight"],
                )
            ) * auxiliary_orientation_loss

    total = (
        float(config["assistance_loss_weight"]) * assistance_loss
        + float(config["assistance_type_loss_weight"]) * assistance_type_loss
        + float(config["receiving_hand_loss_weight"]) * receiving_hand_loss
        + float(config["pose_loss_weight"]) * pose_loss
        + float(config.get("auxiliary_pose_loss_weight", 0.0))
        * auxiliary_pose_loss
    )
    return total, {
        "total": float(total.detach()),
        "assistance": float(assistance_loss.detach()),
        "assistance_type": float(assistance_type_loss.detach()),
        "receiving_hand": float(receiving_hand_loss.detach()),
        "position": float(position_loss.detach()),
        "orientation": float(orientation_loss.detach()),
        "auxiliary_position": float(auxiliary_position_loss.detach()),
        "auxiliary_orientation": float(auxiliary_orientation_loss.detach()),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def empty_classification(class_names: list[str]) -> dict:
    return {
        "accuracy": None,
        "macro_f1": None,
        "macro_f1_supported": None,
        "per_class_f1": [],
        "support": [],
        "samples": 0,
        "confusion_matrix": [],
        "class_names": class_names,
    }


def append_pose(
    storage: dict[str, list[torch.Tensor]],
    name: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if bool(mask.any()):
        storage[f"{name}_predictions"].append(predictions[mask].detach().cpu())
        storage[f"{name}_targets"].append(targets[mask].detach().cpu())


def stored_pose_metrics(storage: dict[str, list[torch.Tensor]], name: str) -> dict:
    predictions = storage.get(f"{name}_predictions", [])
    targets = storage.get(f"{name}_targets", [])
    if not predictions:
        return pose_metrics(torch.empty((0, 7)), torch.empty((0, 7)))
    return pose_metrics(torch.cat(predictions), torch.cat(targets))


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
    pose_storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    auxiliary_pose_storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    progress_storage: dict[str, dict[str, list[torch.Tensor]]] = {
        group: defaultdict(list) for group in PROGRESS_GROUPS
    }
    time_to_end_storage: dict[str, dict[str, list[torch.Tensor]]] = {
        group: defaultdict(list) for group in TIME_TO_END_GROUPS
    }
    hand_pose_storage: dict[str, dict[str, list[torch.Tensor]]] = {
        hand: defaultdict(list) for hand in RECEIVING_HAND_NAMES
    }
    gate_values = []
    target_pose_count = 0
    oracle_reference_count = 0
    predicted_reference_count = 0
    auxiliary_target_pose_count = 0
    auxiliary_oracle_reference_count = 0
    auxiliary_predicted_reference_count = 0

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in loader:
            batch = move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"], batch["hand_reference_pose"])
            loss, components = residual_multitask_loss(
                outputs,
                batch,
                assistance_criterion,
                assistance_type_criterion,
                receiving_hand_criterion,
                training_config,
            )
            if optimizer is not None:
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
            known_hand = (
                (intentions == 2) & (receiving_hand >= 0) & (receiving_hand < 2)
            )
            predicted_hand = outputs["receiving_hand_logits"].argmax(dim=-1)
            if bool(known_hand.any()):
                hand_predictions.append(predicted_hand[known_hand].cpu())
                hand_targets.append(receiving_hand[known_hand].cpu())

            oracle_pose = select_hand_pose(outputs["pose_candidates"], receiving_hand)
            predicted_hand_pose = select_hand_pose(
                outputs["pose_candidates"], predicted_hand
            )
            oracle_reference = select_hand_pose(
                batch["hand_reference_pose"], receiving_hand
            )
            target_valid = batch["pose_valid"] & (intentions == 2) & known_hand
            oracle_valid = batch["residual_pose_valid"] & known_hand
            predicted_reference_valid = batch["hand_reference_valid"].gather(
                1, predicted_hand[:, None]
            )[:, 0]
            end_to_end_valid = target_valid & predicted_reference_valid
            target_pose_count += int(target_valid.sum())
            oracle_reference_count += int(oracle_valid.sum())
            predicted_reference_count += int(end_to_end_valid.sum())

            append_pose(
                pose_storage,
                "oracle",
                oracle_pose,
                batch["pose_target"],
                oracle_valid,
            )
            append_pose(
                pose_storage,
                "end_to_end",
                predicted_hand_pose,
                batch["pose_target"],
                end_to_end_valid,
            )
            append_pose(
                pose_storage,
                "last_observation",
                oracle_reference,
                batch["pose_target"],
                oracle_valid,
            )

            if "auxiliary_pose_candidates" in outputs:
                auxiliary_oracle_pose = select_hand_pose(
                    outputs["auxiliary_pose_candidates"], receiving_hand
                )
                auxiliary_predicted_hand_pose = select_hand_pose(
                    outputs["auxiliary_pose_candidates"], predicted_hand
                )
                auxiliary_target_valid = (
                    batch["auxiliary_pose_valid"] & (intentions == 2) & known_hand
                )
                auxiliary_oracle_valid = (
                    batch["auxiliary_residual_pose_valid"] & known_hand
                )
                auxiliary_end_to_end_valid = (
                    auxiliary_target_valid & predicted_reference_valid
                )
                auxiliary_target_pose_count += int(auxiliary_target_valid.sum())
                auxiliary_oracle_reference_count += int(
                    auxiliary_oracle_valid.sum()
                )
                auxiliary_predicted_reference_count += int(
                    auxiliary_end_to_end_valid.sum()
                )
                append_pose(
                    auxiliary_pose_storage,
                    "oracle",
                    auxiliary_oracle_pose,
                    batch["auxiliary_pose_target"],
                    auxiliary_oracle_valid,
                )
                append_pose(
                    auxiliary_pose_storage,
                    "end_to_end",
                    auxiliary_predicted_hand_pose,
                    batch["auxiliary_pose_target"],
                    auxiliary_end_to_end_valid,
                )
                append_pose(
                    auxiliary_pose_storage,
                    "last_observation",
                    oracle_reference,
                    batch["auxiliary_pose_target"],
                    auxiliary_oracle_valid,
                )

            for group, group_mask in progress_masks(batch["handover_progress"]).items():
                append_pose(
                    progress_storage[group],
                    "oracle",
                    oracle_pose,
                    batch["pose_target"],
                    oracle_valid & group_mask,
                )
                append_pose(
                    progress_storage[group],
                    "end_to_end",
                    predicted_hand_pose,
                    batch["pose_target"],
                    end_to_end_valid & group_mask,
                )
                append_pose(
                    progress_storage[group],
                    "last_observation",
                    oracle_reference,
                    batch["pose_target"],
                    oracle_valid & group_mask,
                )
            for group, group_mask in time_to_end_masks(
                batch["time_to_sequence_end_seconds"]
            ).items():
                append_pose(
                    time_to_end_storage[group],
                    "oracle",
                    oracle_pose,
                    batch["pose_target"],
                    oracle_valid & group_mask,
                )
                append_pose(
                    time_to_end_storage[group],
                    "end_to_end",
                    predicted_hand_pose,
                    batch["pose_target"],
                    end_to_end_valid & group_mask,
                )
                append_pose(
                    time_to_end_storage[group],
                    "last_observation",
                    oracle_reference,
                    batch["pose_target"],
                    oracle_valid & group_mask,
                )
            for hand_id, hand_name in enumerate(RECEIVING_HAND_NAMES):
                hand_mask = receiving_hand == hand_id
                append_pose(
                    hand_pose_storage[hand_name],
                    "oracle",
                    oracle_pose,
                    batch["pose_target"],
                    oracle_valid & hand_mask,
                )
                append_pose(
                    hand_pose_storage[hand_name],
                    "end_to_end",
                    predicted_hand_pose,
                    batch["pose_target"],
                    end_to_end_valid & hand_mask,
                )
                append_pose(
                    hand_pose_storage[hand_name],
                    "last_observation",
                    oracle_reference,
                    batch["pose_target"],
                    oracle_valid & hand_mask,
                )
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
        assistance_type = empty_classification(["fetch", "handover"])
    if hand_targets:
        receiving_hand_metrics = classification_metrics(
            torch.cat(hand_predictions), torch.cat(hand_targets), 2
        )
        receiving_hand_metrics["class_names"] = RECEIVING_HAND_NAMES
    else:
        receiving_hand_metrics = empty_classification(RECEIVING_HAND_NAMES)

    mean_gate = torch.cat(gate_values).mean(dim=0).tolist()
    return {
        "loss": averaged_losses,
        "intention": intention,
        "assistance": assistance,
        "assistance_type": assistance_type,
        "receiving_hand": receiving_hand_metrics,
        "pose_oracle": stored_pose_metrics(pose_storage, "oracle"),
        "pose_end_to_end": stored_pose_metrics(pose_storage, "end_to_end"),
        "last_observation_oracle": stored_pose_metrics(
            pose_storage, "last_observation"
        ),
        "pose_coverage": {
            "pose_targets": target_pose_count,
            "future_targets": target_pose_count,
            "oracle_reference_valid": oracle_reference_count,
            "predicted_reference_valid": predicted_reference_count,
        },
        "auxiliary_t_plus_1": {
            "pose_oracle": stored_pose_metrics(auxiliary_pose_storage, "oracle"),
            "pose_end_to_end": stored_pose_metrics(
                auxiliary_pose_storage, "end_to_end"
            ),
            "last_observation_oracle": stored_pose_metrics(
                auxiliary_pose_storage, "last_observation"
            ),
            "coverage": {
                "pose_targets": auxiliary_target_pose_count,
                "oracle_reference_valid": auxiliary_oracle_reference_count,
                "predicted_reference_valid": auxiliary_predicted_reference_count,
            },
        },
        "pose_by_handover_progress": {
            group: {
                "pose_oracle": stored_pose_metrics(progress_storage[group], "oracle"),
                "pose_end_to_end": stored_pose_metrics(
                    progress_storage[group], "end_to_end"
                ),
                "last_observation_oracle": stored_pose_metrics(
                    progress_storage[group], "last_observation"
                ),
            }
            for group in PROGRESS_GROUPS
        },
        "pose_by_time_to_sequence_end": {
            group: {
                "pose_oracle": stored_pose_metrics(
                    time_to_end_storage[group], "oracle"
                ),
                "pose_end_to_end": stored_pose_metrics(
                    time_to_end_storage[group], "end_to_end"
                ),
                "last_observation_oracle": stored_pose_metrics(
                    time_to_end_storage[group], "last_observation"
                ),
            }
            for group in TIME_TO_END_GROUPS
        },
        "pose_by_receiving_hand": {
            hand: {
                "pose_oracle": stored_pose_metrics(hand_pose_storage[hand], "oracle"),
                "pose_end_to_end": stored_pose_metrics(
                    hand_pose_storage[hand], "end_to_end"
                ),
                "last_observation_oracle": stored_pose_metrics(
                    hand_pose_storage[hand], "last_observation"
                ),
            }
            for hand in RECEIVING_HAND_NAMES
        },
        "mean_gate": {"temporal": mean_gate[0], "channel": mean_gate[1]},
    }


def make_loader(
    dataset, config: dict, *, shuffle: bool, device: torch.device
) -> DataLoader:
    worker_count = int(config["num_workers"])
    sampler = None
    sampling_mode = str(config.get("sampling_mode", "window_uniform"))
    if shuffle and sampling_mode == "sequence_balanced":
        if not hasattr(dataset, "sequence_sampling_weights"):
            raise ValueError(
                "sequence_balanced sampling requires an endpose-v2 dataset"
            )
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"]))
        sampler = WeightedRandomSampler(
            dataset.sequence_sampling_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    elif sampling_mode not in {"window_uniform", "sequence_balanced"}:
        raise ValueError(f"Unknown training sampling mode: {sampling_mode}")
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )


def build_residual_model(
    model_type: str,
    *,
    input_dim: int,
    window_size: int,
    model_config: dict,
) -> nn.Module:
    model_classes = {
        RESIDUAL_V2_MODEL_TYPE: HierarchicalResidualPoseTransformer,
        DUAL_HORIZON_MODEL_TYPE: HierarchicalDualHorizonResidualPoseTransformer,
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported residual model type: {model_type}")
    return model_classes[model_type](
        input_dim=input_dim,
        window_size=window_size,
        **model_config,
    )


def checkpoint_payload(
    model: nn.Module,
    config: dict,
    bundle: DataBundle,
    trainable_parameters: int,
    epoch: int,
    selection_metric: str,
    selection_value: float,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": config["model"],
        "model_type": config.get("model_type", RESIDUAL_V2_MODEL_TYPE),
        "input_dim": len(bundle.normalizer.output_feature_names),
        "window_size": int(config["data"]["window_size"]),
        "trainable_parameters": trainable_parameters,
        "epoch": epoch,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "dataset_provenance": checkpoint_provenance(bundle),
        "pose_target_definition": bundle.split_metadata.get("pose_target", {}),
        "pose_training_protocol": bundle.split_metadata.get(
            "endpose_v2_adapter"
        ),
    }
    if "position_mae_cm" in selection_metric:
        payload["selection_metric_definition"] = (
            POSITION_ERROR_DEFINITION
        )
    return payload


def train(args: argparse.Namespace) -> Path:
    started_at = datetime.now().astimezone()
    started_monotonic = time.perf_counter()
    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", RESIDUAL_V2_MODEL_TYPE))
    config["data"]["master_dir"] = str(
        resolve_project_path(config["data"]["master_dir"])
    )
    if not bool(config["data"].get("include_hand_references")):
        raise ValueError("Residual v2 requires data.include_hand_references=true")
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if getattr(args, "seed", None) is not None:
        config["training"]["seed"] = args.seed
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = choose_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(config.get("run_name", "hierarchical_residual_v2"))
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
    bundle = prepare_data(config["data"], seed, args.limit_sequences)
    if model_type == DUAL_HORIZON_MODEL_TYPE:
        bundle = wrap_endpose_v2_bundle(bundle, config["data"])
    training_config = config["training"]
    if dict(training_config.get("position_loss", {})).get("type") == (
        "normalized_smooth_l1"
    ):
        minimum_scale_m = float(
            training_config["position_loss"].get("minimum_scale_m", 0.02)
        )
        training_config["resolved_position_scale_m"] = residual_position_scale_m(
            bundle.train,
            target_key="pose_target",
            valid_key="residual_pose_valid",
            minimum_scale_m=minimum_scale_m,
        )
        if model_type == DUAL_HORIZON_MODEL_TYPE:
            training_config[
                "resolved_auxiliary_position_scale_m"
            ] = residual_position_scale_m(
                bundle.train,
                target_key="auxiliary_pose_target",
                valid_key="auxiliary_residual_pose_valid",
                minimum_scale_m=minimum_scale_m,
            )
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
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
        "Receiving-hand windows: "
        f"train={bundle.train.receiving_hand_counts()}, "
        f"validation={bundle.validation.receiving_hand_counts()}, "
        f"test={bundle.test.receiving_hand_counts()}"
    )
    print(
        "Residual pose targets: "
        f"train={bundle.train.residual_pose_count()}, "
        f"validation={bundle.validation.residual_pose_count()}, "
        f"test={bundle.test.residual_pose_count()}"
    )
    if model_type == DUAL_HORIZON_MODEL_TYPE:
        print(
            "Auxiliary t+1 pose targets: "
            f"train={bundle.train.auxiliary_pose_count()}, "
            f"validation={bundle.validation.auxiliary_pose_count()}, "
            f"test={bundle.test.auxiliary_pose_count()}"
        )
    print(f"Pose target: {bundle.split_metadata.get('pose_target', {})}")
    print(
        "Pose training: "
        f"sampling={training_config.get('sampling_mode', 'window_uniform')}, "
        f"position_loss={training_config.get('position_loss', {'type': 'smooth_l1_meters'})}, "
        f"orientation_loss={training_config.get('orientation_loss', {'type': 'cosine_distance'})}"
    )

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

    model = build_residual_model(
        model_type,
        input_dim=len(bundle.normalizer.output_feature_names),
        window_size=int(config["data"]["window_size"]),
        model_config=config["model"],
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
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

    best_intention = -math.inf
    best_pose = math.inf
    epochs_without_improvement = 0
    history = []
    intention_checkpoint = run_dir / "best_intention_model.pt"
    pose_checkpoint = run_dir / "best_pose_model.pt"
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
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        intention_score = float(validation_metrics["intention"]["macro_f1"])
        pose_value = validation_metrics["pose_oracle"]["position_mae_cm"]
        pose_score = float(pose_value) if pose_value is not None else math.inf
        improved = False
        if intention_score > best_intention:
            best_intention = intention_score
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    bundle,
                    trainable_parameters,
                    epoch,
                    "validation_intention_macro_f1",
                    intention_score,
                ),
                intention_checkpoint,
            )
            improved = True
        if pose_score < best_pose:
            best_pose = pose_score
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    bundle,
                    trainable_parameters,
                    epoch,
                    "validation_pose_oracle_position_mae_cm",
                    pose_score,
                ),
                pose_checkpoint,
            )
            improved = True
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']['total']:.4f} | "
            f"val intent F1={intention_score:.4f} | "
            "val hand supported F1="
            f"{validation_metrics['receiving_hand']['macro_f1_supported']:.4f} | "
            f"val pose oracle mean Euclidean={pose_value} cm | "
            "val pose end-to-end mean Euclidean="
            f"{validation_metrics['pose_end_to_end']['position_mae_cm']} cm | "
            "val early pose mean Euclidean="
            f"{validation_metrics['pose_by_handover_progress']['0-25%']['pose_oracle']['position_mae_cm']} cm"
        )
        if epochs_without_improvement >= int(
            training_config["early_stopping_patience"]
        ):
            print("Early stopping")
            break

    validation_results = {}
    test_results = {}
    checkpoint_metadata = {}
    for name, path in (
        ("best_intention", intention_checkpoint),
        ("best_pose", pose_checkpoint),
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
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
            "epoch": int(checkpoint["epoch"]),
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": float(checkpoint["selection_value"]),
            "selection_metric_definition": checkpoint.get(
                "selection_metric_definition"
            ),
        }

    report = {
        "model_type": model_type,
        "pose_target_definition": bundle.split_metadata.get("pose_target", {}),
        "pose_training_protocol": bundle.split_metadata.get(
            "endpose_v2_adapter"
        ),
        "trainable_parameters": trainable_parameters,
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
        "legacy_pose_metric_alias": {
            "position_mae_cm": POSITION_ERROR_DEFINITION,
            "position_rmse_cm": POSITION_RMS_ERROR_DEFINITION,
        },
    }
    if test_loader is not None:
        report["test"] = test_results
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, metrics in validation_results.items():
        print(
            f"{name} validation: "
            f"intent F1={metrics['intention']['macro_f1']:.4f}, "
            f"hand F1={metrics['receiving_hand']['macro_f1']:.4f}, "
            "pose oracle mean Euclidean="
            f"{metrics['pose_oracle']['position_mae_cm']} cm, "
            "pose end-to-end mean Euclidean="
            f"{metrics['pose_end_to_end']['position_mae_cm']} cm"
        )
    for name, metrics in test_results.items():
        print(
            f"{name} test: intent F1={metrics['intention']['macro_f1']:.4f}, "
            f"hand F1={metrics['receiving_hand']['macro_f1']:.4f}, "
            "pose oracle mean Euclidean="
            f"{metrics['pose_oracle']['position_mae_cm']} cm, "
            "pose end-to-end mean Euclidean="
            f"{metrics['pose_end_to_end']['position_mae_cm']} cm"
        )
    if skip_test_evaluation:
        print("Test evaluation skipped by request")
    print(f"Metrics: {metrics_path}")
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
