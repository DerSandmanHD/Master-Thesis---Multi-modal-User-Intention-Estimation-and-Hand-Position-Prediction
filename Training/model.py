#!/usr/bin/env python3
"""GTN-inspired multi-task model for multimodal Aria time series."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedMultimodalTransformer(nn.Module):
    """Fuse temporal and channel-wise Transformer representations with a gate.

    Input shape is [batch, window, features]. The feature dimension already
    includes observation-mask channels supplied by the data pipeline.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        num_intentions: int = 3,
        num_objects: int = 9,
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
            raise ValueError("input_dim must be positive and window_size must exceed one")

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
        self.intention_head = nn.Linear(fused_dim, num_intentions)
        self.object_head = nn.Linear(fused_dim, num_objects)
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, window, features], got {tuple(x.shape)}")
        if x.shape[1] != self.window_size or x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected window/features {self.window_size}/{self.input_dim}, "
                f"got {x.shape[1]}/{x.shape[2]}"
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

        raw_pose = self.pose_head(fused)
        quaternion = F.normalize(raw_pose[:, 3:7], dim=-1, eps=1e-8)
        pose = torch.cat((raw_pose[:, :3], quaternion), dim=-1)
        return {
            "intention_logits": self.intention_head(fused),
            "object_logits": self.object_head(fused),
            "pose": pose,
            "gate": gate,
        }
