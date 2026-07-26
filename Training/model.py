#!/usr/bin/env python3
"""Hierarchical GTN-inspired model for multimodal Aria time series."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_window_input(
    x: torch.Tensor,
    *,
    input_dim: int,
    window_size: int,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, window, features], got {tuple(x.shape)}")
    if x.shape[1] != window_size or x.shape[2] != input_dim:
        raise ValueError(
            f"Expected window/features {window_size}/{input_dim}, "
            f"got {x.shape[1]}/{x.shape[2]}"
        )


def _absolute_pose_outputs(
    representation: torch.Tensor,
    assistance_head: nn.Module,
    assistance_type_head: nn.Module,
    pose_head: nn.Module,
) -> dict[str, torch.Tensor]:
    raw_pose = pose_head(representation)
    quaternion = F.normalize(raw_pose[:, 3:7], dim=-1, eps=1e-8)
    return {
        "assistance_logits": assistance_head(representation),
        "assistance_type_logits": assistance_type_head(representation),
        "pose": torch.cat((raw_pose[:, :3], quaternion), dim=-1),
    }


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions stored as x, y, z, w."""
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


class HierarchicalGatedMultimodalTransformer(nn.Module):
    """Fuse temporal and channel-wise representations for hierarchical intent.

    Input shape is [batch, window, features]. The feature dimension already
    includes observation-mask channels supplied by the data pipeline.

    The first classifier predicts continue versus assistance. The second is
    trained conditionally on assistance samples and predicts fetch versus
    handover. Future receiving-hand pose is an auxiliary handover-only target.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if input_dim <= 0 or window_size <= 1:
            raise ValueError(
                "input_dim must be positive and window_size must exceed one"
            )

        self.input_dim = input_dim
        self.window_size = window_size

        self.temporal_projection = nn.Linear(input_dim, d_model)
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_position = nn.Parameter(torch.zeros(1, window_size + 1, d_model))

        self.channel_projection = nn.Linear(window_size, d_model)
        self.channel_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.channel_identity = nn.Parameter(torch.zeros(1, input_dim + 1, d_model))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        channel_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.channel_encoder = nn.TransformerEncoder(
            channel_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.channel_norm = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        fused_dim = d_model * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.assistance_head = nn.Linear(fused_dim, 2)
        self.assistance_type_head = nn.Linear(fused_dim, 2)
        self.pose_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),
        )

        nn.init.normal_(self.temporal_cls, std=0.02)
        nn.init.normal_(self.channel_cls, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.channel_identity, std=0.02)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_window_input(
            x,
            input_dim=self.input_dim,
            window_size=self.window_size,
        )

        batch_size = x.shape[0]

        temporal = self.temporal_projection(x)
        temporal_cls = self.temporal_cls.expand(batch_size, -1, -1)
        temporal = torch.cat((temporal_cls, temporal), dim=1) + self.temporal_position
        temporal = self.temporal_norm(self.temporal_encoder(temporal)[:, 0])

        channel = self.channel_projection(x.transpose(1, 2))
        channel_cls = self.channel_cls.expand(batch_size, -1, -1)
        channel = torch.cat((channel_cls, channel), dim=1) + self.channel_identity
        channel = self.channel_norm(self.channel_encoder(channel)[:, 0])

        gate = F.softmax(self.gate(torch.cat((temporal, channel), dim=-1)), dim=-1)
        fused = torch.cat(
            (temporal * gate[:, 0:1], channel * gate[:, 1:2]),
            dim=-1,
        )
        fused = self.fusion_norm(fused)

        return fused, gate

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused, gate = self._encode(x)
        outputs = _absolute_pose_outputs(
            fused,
            self.assistance_head,
            self.assistance_type_head,
            self.pose_head,
        )
        outputs["gate"] = gate
        return outputs


class HierarchicalWindowMLP(nn.Module):
    """Feed-forward baseline over the same flattened observation window."""

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 128),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or window_size <= 1:
            raise ValueError(
                "input_dim must be positive and window_size must exceed one"
            )
        if not hidden_dims or any(int(value) <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain at least one positive value")

        self.input_dim = input_dim
        self.window_size = window_size
        dimensions = [input_dim * window_size, *(int(value) for value in hidden_dims)]
        layers: list[nn.Module] = []
        for source_dim, target_dim in zip(dimensions, dimensions[1:]):
            layers.extend(
                (
                    nn.Linear(source_dim, target_dim),
                    nn.LayerNorm(target_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.encoder = nn.Sequential(*layers)
        representation_dim = dimensions[-1]
        self.assistance_head = nn.Linear(representation_dim, 2)
        self.assistance_type_head = nn.Linear(representation_dim, 2)
        self.pose_head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, 7),
        )

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_window_input(
            x,
            input_dim=self.input_dim,
            window_size=self.window_size,
        )
        representation = self.encoder(x.flatten(start_dim=1))
        return _absolute_pose_outputs(
            representation,
            self.assistance_head,
            self.assistance_type_head,
            self.pose_head,
        )


class HierarchicalGRU(nn.Module):
    """Unidirectional recurrent baseline for online window prediction."""

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or window_size <= 1:
            raise ValueError(
                "input_dim must be positive and window_size must exceed one"
            )
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

        self.input_dim = input_dim
        self.window_size = window_size
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.assistance_head = nn.Linear(hidden_size, 2)
        self.assistance_type_head = nn.Linear(hidden_size, 2)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 7),
        )

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_window_input(
            x,
            input_dim=self.input_dim,
            window_size=self.window_size,
        )
        _, hidden = self.gru(x)
        representation = self.output_norm(hidden[-1])
        return _absolute_pose_outputs(
            representation,
            self.assistance_head,
            self.assistance_type_head,
            self.pose_head,
        )


class HierarchicalResidualPoseTransformer(HierarchicalGatedMultimodalTransformer):
    """Hierarchical intent model with learned hand selection and pose residual."""

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        fused_dim = d_model * 2
        self.receiving_hand_head = nn.Linear(fused_dim, 2)
        self.pose_head = nn.Sequential(
            nn.Linear(fused_dim + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),
        )
        final_layer = self.pose_head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        with torch.no_grad():
            final_layer.bias[6] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hand_reference_pose is None:
            raise ValueError("Residual model requires hand_reference_pose")
        if hand_reference_pose.ndim != 3 or hand_reference_pose.shape[1:] != (2, 7):
            raise ValueError(
                "Expected hand_reference_pose with shape [batch, 2, 7], got "
                f"{tuple(hand_reference_pose.shape)}"
            )
        if hand_reference_pose.shape[0] != x.shape[0]:
            raise ValueError("Feature and hand-reference batch sizes differ")

        fused, gate = self._encode(x)
        receiving_hand_logits = self.receiving_hand_head(fused)
        hand_probabilities = F.softmax(receiving_hand_logits, dim=-1)
        raw_residual = self.pose_head(torch.cat((fused, hand_probabilities), dim=-1))
        position_delta = raw_residual[:, :3]
        quaternion_delta = F.normalize(raw_residual[:, 3:7], dim=-1, eps=1e-8)

        candidate_positions = hand_reference_pose[:, :, :3] + position_delta[:, None, :]
        candidate_quaternions = quaternion_multiply(
            hand_reference_pose[:, :, 3:7],
            quaternion_delta[:, None, :].expand(-1, 2, -1),
        )
        candidate_quaternions = F.normalize(candidate_quaternions, dim=-1, eps=1e-8)
        pose_candidates = torch.cat(
            (candidate_positions, candidate_quaternions), dim=-1
        )
        return {
            "assistance_logits": self.assistance_head(fused),
            "assistance_type_logits": self.assistance_type_head(fused),
            "receiving_hand_logits": receiving_hand_logits,
            "position_delta": position_delta,
            "quaternion_delta": quaternion_delta,
            "pose_candidates": pose_candidates,
            "gate": gate,
        }
