#!/usr/bin/env python3
"""Metrics for the three multi-task outputs."""

from __future__ import annotations

import math

import torch


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
    support = confusion.sum(dim=1)
    for index in range(num_classes):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - confusion[index, index])
        false_negative = float(confusion[index, :].sum() - confusion[index, index])
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / num_classes,
        "macro_f1_supported": (
            sum(value for value, count in zip(f1_values, support) if int(count) > 0)
            / max(1, int((support > 0).sum()))
        ),
        "per_class_f1": f1_values,
        "support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": total,
    }


def pose_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    if not len(predictions):
        return {
            "samples": 0,
            "position_mae_cm": None,
            "position_rmse_cm": None,
            "orientation_mean_deg": None,
        }
    position_error = torch.linalg.vector_norm(predictions[:, :3] - targets[:, :3], dim=-1)
    predicted_quaternion = torch.nn.functional.normalize(predictions[:, 3:7], dim=-1)
    target_quaternion = torch.nn.functional.normalize(targets[:, 3:7], dim=-1)
    cosine = torch.sum(predicted_quaternion * target_quaternion, dim=-1).abs().clamp(0.0, 1.0)
    orientation_radians = 2.0 * torch.acos(cosine)
    return {
        "samples": len(predictions),
        "position_mae_cm": float(position_error.mean() * 100.0),
        "position_rmse_cm": float(torch.sqrt(torch.mean(position_error.square())) * 100.0),
        "orientation_mean_deg": float(orientation_radians.mean() * 180.0 / math.pi),
    }
