#!/usr/bin/env python3
"""Metrics for the three multi-task outputs."""

from __future__ import annotations

import math
import hashlib
import json

import torch


POSITION_MEAN_EUCLIDEAN_ERROR_KEY = (
    "position_mean_euclidean_error_cm"
)
POSITION_RMS_EUCLIDEAN_ERROR_KEY = (
    "position_root_mean_square_euclidean_error_cm"
)
LEGACY_POSITION_ERROR_KEY = "position_mae_cm"
POSITION_ERROR_DEFINITION = (
    "mean Euclidean norm of the 3D position error, in centimetres"
)
POSITION_RMS_ERROR_DEFINITION = (
    "root mean square of the Euclidean 3D position-error norm, "
    "in centimetres"
)


def sample_key_fingerprint(sample_keys: list[str]) -> str:
    """Stable identity for an unordered scientific evaluation cohort."""

    payload = json.dumps(sorted(str(key) for key in sample_keys), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for truth, prediction in zip(targets.long().view(-1), predictions.long().view(-1)):
        if 0 <= truth < num_classes and 0 <= prediction < num_classes:
            confusion[truth, prediction] += 1
    total = int(confusion.sum())
    accuracy = float(confusion.diag().sum() / total) if total else 0.0
    f1_values = []
    precision_values = []
    recall_values = []
    support = confusion.sum(dim=1)
    for index in range(num_classes):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - confusion[index, index])
        false_negative = float(confusion[index, :].sum() - confusion[index, index])
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        denominator = 2 * true_positive + false_positive + false_negative
        precision_values.append(
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall_values.append(
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / num_classes,
        "macro_f1_supported": (
            sum(value for value, count in zip(f1_values, support) if int(count) > 0)
            / max(1, int((support > 0).sum()))
        ),
        "per_class_precision": precision_values,
        "per_class_recall": recall_values,
        "per_class_f1": f1_values,
        "support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": total,
    }


def pose_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    if not len(predictions):
        return {
            "samples": 0,
            POSITION_MEAN_EUCLIDEAN_ERROR_KEY: None,
            POSITION_RMS_EUCLIDEAN_ERROR_KEY: None,
            # Compatibility alias for existing artifacts and consumers.
            "position_mae_cm": None,
            "position_rmse_cm": None,
            "position_median_cm": None,
            "orientation_mean_deg": None,
            "orientation_median_deg": None,
            "position_error_definition": POSITION_ERROR_DEFINITION,
            "position_rmse_definition": POSITION_RMS_ERROR_DEFINITION,
        }
    position_error = torch.linalg.vector_norm(predictions[:, :3] - targets[:, :3], dim=-1)
    predicted_quaternion = torch.nn.functional.normalize(predictions[:, 3:7], dim=-1)
    target_quaternion = torch.nn.functional.normalize(targets[:, 3:7], dim=-1)
    cosine = torch.sum(predicted_quaternion * target_quaternion, dim=-1).abs().clamp(0.0, 1.0)
    orientation_radians = 2.0 * torch.acos(cosine)
    mean_euclidean_error_cm = float(position_error.mean() * 100.0)
    rms_euclidean_error_cm = float(
        torch.sqrt(torch.mean(position_error.square())) * 100.0
    )
    return {
        "samples": len(predictions),
        POSITION_MEAN_EUCLIDEAN_ERROR_KEY: mean_euclidean_error_cm,
        # Do not reinterpret historical metrics.json files silently.
        LEGACY_POSITION_ERROR_KEY: mean_euclidean_error_cm,
        POSITION_RMS_EUCLIDEAN_ERROR_KEY: rms_euclidean_error_cm,
        "position_rmse_cm": rms_euclidean_error_cm,
        "position_median_cm": float(position_error.median() * 100.0),
        "orientation_mean_deg": float(orientation_radians.mean() * 180.0 / math.pi),
        "orientation_median_deg": float(
            orientation_radians.median() * 180.0 / math.pi
        ),
        "position_error_definition": POSITION_ERROR_DEFINITION,
        "position_rmse_definition": POSITION_RMS_ERROR_DEFINITION,
    }


def position_mean_euclidean_error_cm(metrics: dict) -> float | None:
    """Read the canonical value while remaining compatible with old runs."""

    if POSITION_MEAN_EUCLIDEAN_ERROR_KEY in metrics:
        return metrics[POSITION_MEAN_EUCLIDEAN_ERROR_KEY]
    return metrics.get(LEGACY_POSITION_ERROR_KEY)
