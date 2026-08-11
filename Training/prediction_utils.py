"""Shared intention semantics for hierarchical and flat classifiers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def intention_head_mode(outputs: dict[str, torch.Tensor]) -> str:
    if "intention_logits" in outputs:
        if outputs["intention_logits"].ndim != 2 or outputs[
            "intention_logits"
        ].shape[-1] != 3:
            raise ValueError("Flat intention_logits must have shape [batch, 3]")
        return "flat"
    required = {"assistance_logits", "assistance_type_logits"}
    missing = sorted(required - set(outputs))
    if missing:
        raise ValueError(
            "Model outputs lack a supported intention head: " + ", ".join(missing)
        )
    return "hierarchical"


def intention_probabilities(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return normalized continue/fetch/handover probabilities."""

    if intention_head_mode(outputs) == "flat":
        return F.softmax(outputs["intention_logits"], dim=-1)
    assistance = F.softmax(outputs["assistance_logits"], dim=-1)
    assistance_type = F.softmax(outputs["assistance_type_logits"], dim=-1)
    return torch.stack(
        (
            assistance[:, 0],
            assistance[:, 1] * assistance_type[:, 0],
            assistance[:, 1] * assistance_type[:, 1],
        ),
        dim=-1,
    )


def intention_predictions(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the actual decision rule of the configured classifier head."""

    if intention_head_mode(outputs) == "flat":
        return outputs["intention_logits"].argmax(dim=-1)
    assistance = outputs["assistance_logits"].argmax(dim=-1)
    assistance_type = outputs["assistance_type_logits"].argmax(dim=-1)
    return torch.where(
        assistance == 0,
        torch.zeros_like(assistance_type),
        assistance_type + 1,
    )


def assistance_predictions(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if intention_head_mode(outputs) == "flat":
        return (intention_predictions(outputs) != 0).long()
    return outputs["assistance_logits"].argmax(dim=-1)


def assistance_type_predictions(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return fetch/handover IDs (0/1), evaluated on GT-assistance rows."""

    if intention_head_mode(outputs) == "flat":
        return outputs["intention_logits"][:, 1:3].argmax(dim=-1)
    return outputs["assistance_type_logits"].argmax(dim=-1)
