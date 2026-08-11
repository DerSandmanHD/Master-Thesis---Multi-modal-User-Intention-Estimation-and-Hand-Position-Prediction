#!/usr/bin/env python3
"""Hierarchical GTN-inspired model for multimodal Aria time series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


FUSION_MODES = (
    "temporal_channel_gated",
    "temporal_channel_simple",
    "temporal_only",
    "modality_gated",
)
INTENTION_HEAD_MODES = ("hierarchical", "flat")


def _index_list(value: object, *, field: str, input_dim: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of feature indices")
    indices = tuple(int(index) for index in value)
    if not indices:
        raise ValueError(f"{field} must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{field} contains duplicate feature indices")
    if any(index < 0 or index >= input_dim for index in indices):
        raise ValueError(
            f"{field} contains an index outside the input dimension {input_dim}"
        )
    return indices


def _resolve_model_modality_schema(
    modality_schema: object,
    *,
    input_dim: int,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate and reduce the persisted feature schema to model indices.

    The data pipeline stores groups below ``schema["groups"]`` and an optional
    ``active_modalities`` list defining their order.  A plain group mapping is
    accepted as well to keep unit tests and downstream tools lightweight.
    """
    if not isinstance(modality_schema, Mapping):
        raise ValueError("modality_gated fusion requires a modality_schema mapping")
    raw_groups = modality_schema.get("groups", modality_schema)
    if not isinstance(raw_groups, Mapping) or not raw_groups:
        raise ValueError("modality_schema.groups must be a non-empty mapping")

    raw_order = modality_schema.get("active_modalities")
    if raw_order is None:
        names = tuple(str(name) for name in raw_groups)
    else:
        if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, Sequence):
            raise ValueError("modality_schema.active_modalities must be a sequence")
        names = tuple(str(name) for name in raw_order)
        missing = [name for name in names if name not in raw_groups]
        if missing:
            raise ValueError(
                "active_modalities references missing groups: " + ", ".join(missing)
            )
    if not names or len(set(names)) != len(names):
        raise ValueError("modality names must be non-empty and unique")

    input_indices: list[tuple[int, ...]] = []
    availability_indices: list[tuple[int, ...]] = []
    for name in names:
        group = raw_groups[name]
        if not isinstance(group, Mapping):
            raise ValueError(f"modality group {name!r} must be a mapping")
        values = _index_list(
            group.get("input_indices"),
            field=f"modality_schema.groups.{name}.input_indices",
            input_dim=input_dim,
        )
        availability = _index_list(
            group.get("availability_indices"),
            field=f"modality_schema.groups.{name}.availability_indices",
            input_dim=input_dim,
        )
        if not set(availability).issubset(values):
            raise ValueError(
                f"availability_indices for modality {name!r} must also be input_indices"
            )
        input_indices.append(values)
        availability_indices.append(availability)
    return names, tuple(input_indices), tuple(availability_indices)


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
    receiving_hand_head: nn.Module,
    pose_head: nn.Module,
) -> dict[str, torch.Tensor]:
    raw_pose = pose_head(representation)
    quaternion = F.normalize(raw_pose[:, 3:7], dim=-1, eps=1e-8)
    return {
        "assistance_logits": assistance_head(representation),
        "assistance_type_logits": assistance_type_head(representation),
        "receiving_hand_logits": receiving_hand_head(representation),
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


def _configure_residual_intention_head(
    model: nn.Module,
    *,
    fused_dim: int,
    intention_head_mode: str,
) -> None:
    mode = str(intention_head_mode).strip().casefold()
    if mode not in INTENTION_HEAD_MODES:
        raise ValueError(
            f"Unknown intention_head_mode {intention_head_mode!r}; expected one of "
            + ", ".join(INTENTION_HEAD_MODES)
        )
    model.intention_head_mode = mode
    if mode == "flat":
        del model.assistance_head
        del model.assistance_type_head
        model.intention_head = nn.Linear(fused_dim, 3)


def _residual_intention_outputs(
    model: nn.Module,
    representation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.intention_head_mode == "flat":
        return {"intention_logits": model.intention_head(representation)}
    return {
        "assistance_logits": model.assistance_head(representation),
        "assistance_type_logits": model.assistance_type_head(representation),
    }


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
        fusion_mode: str = "temporal_channel_gated",
        modality_schema: Mapping[str, Any] | None = None,
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
        self.fusion_mode = str(fusion_mode).strip().casefold()
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"Unknown fusion_mode {fusion_mode!r}; expected one of "
                + ", ".join(FUSION_MODES)
            )

        if self.fusion_mode != "modality_gated":
            self.temporal_projection = nn.Linear(input_dim, d_model)
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_position = nn.Parameter(torch.zeros(1, window_size + 1, d_model))

        temporal_layer = nn.TransformerEncoderLayer(
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
        self.temporal_norm = nn.LayerNorm(d_model)

        uses_channel_encoder = self.fusion_mode in {
            "temporal_channel_gated",
            "temporal_channel_simple",
        }
        if uses_channel_encoder:
            # Keep names and shapes unchanged for strict loading of historical
            # temporal-channel checkpoints in the default mode.
            self.channel_projection = nn.Linear(window_size, d_model)
            self.channel_cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.channel_identity = nn.Parameter(
                torch.zeros(1, input_dim + 1, d_model)
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
            self.channel_encoder = nn.TransformerEncoder(
                channel_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.channel_norm = nn.LayerNorm(d_model)

        if self.fusion_mode == "temporal_channel_gated":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )

        self.modality_names: tuple[str, ...] = ()
        self._modality_input_indices: tuple[tuple[int, ...], ...] = ()
        self._modality_availability_indices: tuple[tuple[int, ...], ...] = ()
        if self.fusion_mode == "modality_gated":
            (
                self.modality_names,
                self._modality_input_indices,
                self._modality_availability_indices,
            ) = _resolve_model_modality_schema(
                modality_schema,
                input_dim=input_dim,
            )
            self.modality_encoders = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(len(indices), d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
                for indices in self._modality_input_indices
            )
            self.modality_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        fused_dim = d_model * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.assistance_head = nn.Linear(fused_dim, 2)
        self.assistance_type_head = nn.Linear(fused_dim, 2)
        self.receiving_hand_head = nn.Linear(fused_dim, 2)
        self.pose_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),
        )

        nn.init.normal_(self.temporal_cls, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        if uses_channel_encoder:
            nn.init.normal_(self.channel_cls, std=0.02)
            nn.init.normal_(self.channel_identity, std=0.02)

    def _encode_projected_temporal(self, temporal: torch.Tensor) -> torch.Tensor:
        batch_size = temporal.shape[0]
        temporal_cls = self.temporal_cls.expand(batch_size, -1, -1)
        temporal = torch.cat((temporal_cls, temporal), dim=1) + self.temporal_position
        return self.temporal_norm(self.temporal_encoder(temporal)[:, 0])

    def _encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_projected_temporal(self.temporal_projection(x))

    def _encode_channel(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        channel = self.channel_projection(x.transpose(1, 2))
        channel_cls = self.channel_cls.expand(batch_size, -1, -1)
        channel = torch.cat((channel_cls, channel), dim=1) + self.channel_identity
        return self.channel_norm(self.channel_encoder(channel)[:, 0])

    @staticmethod
    def _fixed_fusion_weights(
        reference: torch.Tensor,
        left: float,
        right: float,
    ) -> torch.Tensor:
        return reference.new_tensor((left, right)).expand(reference.shape[0], -1)

    def _encode_modalities(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_sequences: list[torch.Tensor] = []
        representations: list[torch.Tensor] = []
        availability: list[torch.Tensor] = []
        for encoder, input_indices, availability_indices in zip(
            self.modality_encoders,
            self._modality_input_indices,
            self._modality_availability_indices,
        ):
            value_index = torch.as_tensor(input_indices, device=x.device)
            mask_index = torch.as_tensor(availability_indices, device=x.device)
            values = torch.nan_to_num(
                x.index_select(2, value_index), nan=0.0, posinf=0.0, neginf=0.0
            )
            observed = torch.nan_to_num(
                x.index_select(2, mask_index),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ) > 0.5
            step_available = observed.any(dim=-1)
            step_weights = step_available.unsqueeze(-1).to(values.dtype)
            encoded = encoder(values * step_weights)
            encoded = encoded * step_weights
            representation = encoded.sum(dim=1) / step_weights.sum(dim=1).clamp_min(
                1.0
            )
            encoded_sequences.append(encoded)
            representations.append(representation)
            availability.append(step_available.any(dim=1))

        stacked = torch.stack(representations, dim=1)
        available = torch.stack(availability, dim=1)
        logits = self.modality_gate(stacked).squeeze(-1)

        # An explicit masked normalization avoids softmax(-inf, ..., -inf)
        # when a complete observation window contains no available modality.
        masked_logits = torch.where(
            available,
            logits,
            torch.full_like(logits, torch.finfo(logits.dtype).min),
        )
        max_logits = masked_logits.max(dim=1, keepdim=True).values
        max_logits = torch.where(
            available.any(dim=1, keepdim=True),
            max_logits,
            torch.zeros_like(max_logits),
        )
        unnormalized = (
            torch.exp(masked_logits - max_logits) * available.to(logits.dtype)
        )
        weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(logits.dtype).tiny
        )
        context = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        sequences = torch.stack(encoded_sequences, dim=2)
        fused_sequence = (
            sequences * weights[:, None, :, None]
        ).sum(dim=2)
        return fused_sequence, context, weights, available

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_window_input(
            x,
            input_dim=self.input_dim,
            window_size=self.window_size,
        )

        batch_size = x.shape[0]
        modality_weights = x.new_zeros((batch_size, 0))
        modality_available = torch.zeros(
            (batch_size, 0), dtype=torch.bool, device=x.device
        )

        if self.fusion_mode == "modality_gated":
            (
                modality_sequence,
                modality_context,
                modality_weights,
                modality_available,
            ) = self._encode_modalities(x)
            temporal = self._encode_projected_temporal(modality_sequence)
            fusion_weights = self._fixed_fusion_weights(temporal, 0.5, 0.5)
            fused = torch.cat((temporal * 0.5, modality_context * 0.5), dim=-1)
        else:
            temporal = self._encode_temporal(x)

        if self.fusion_mode == "temporal_channel_gated":
            channel = self._encode_channel(x)
            fusion_weights = F.softmax(
                self.gate(torch.cat((temporal, channel), dim=-1)), dim=-1
            )
            fused = torch.cat(
                (
                    temporal * fusion_weights[:, 0:1],
                    channel * fusion_weights[:, 1:2],
                ),
                dim=-1,
            )
        elif self.fusion_mode == "temporal_channel_simple":
            channel = self._encode_channel(x)
            fusion_weights = self._fixed_fusion_weights(temporal, 0.5, 0.5)
            fused = torch.cat((temporal * 0.5, channel * 0.5), dim=-1)
        elif self.fusion_mode == "temporal_only":
            fusion_weights = self._fixed_fusion_weights(temporal, 1.0, 0.0)
            fused = torch.cat((temporal, torch.zeros_like(temporal)), dim=-1)
        fused = self.fusion_norm(fused)
        return fused, {
            # ``gate`` remains the historical name used by existing analysis.
            "gate": fusion_weights,
            "fusion_weights": fusion_weights,
            "modality_weights": modality_weights,
            "modality_available": modality_available,
        }

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused, fusion_outputs = self._encode(x)
        outputs = _absolute_pose_outputs(
            fused,
            self.assistance_head,
            self.assistance_type_head,
            self.receiving_hand_head,
            self.pose_head,
        )
        outputs.update(fusion_outputs)
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
        self.receiving_hand_head = nn.Linear(representation_dim, 2)
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
            self.receiving_hand_head,
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
        self.receiving_hand_head = nn.Linear(hidden_size, 2)
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
            self.receiving_hand_head,
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
        fusion_mode: str = "temporal_channel_gated",
        modality_schema: Mapping[str, Any] | None = None,
        intention_head_mode: str = "hierarchical",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fusion_mode=fusion_mode,
            modality_schema=modality_schema,
        )
        fused_dim = d_model * 2
        _configure_residual_intention_head(
            self,
            fused_dim=fused_dim,
            intention_head_mode=intention_head_mode,
        )
        del self.pose_head
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

        fused, fusion_outputs = self._encode(x)
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
        outputs = {
            "receiving_hand_logits": receiving_hand_logits,
            "position_delta": position_delta,
            "quaternion_delta": quaternion_delta,
            "pose_candidates": pose_candidates,
        }
        outputs.update(_residual_intention_outputs(self, fused))
        outputs.update(fusion_outputs)
        return outputs


class HierarchicalDualHorizonResidualPoseTransformer(
    HierarchicalGatedMultimodalTransformer
):
    """Residual model with separate terminal and t+1 pose heads.

    Each pose head emits a distinct residual for the left and right hand.  The
    terminal head is the primary output; the t+1 head is an auxiliary training
    task that supplies a local-motion signal without changing the terminal
    target used for checkpoint selection and evaluation.
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
        fusion_mode: str = "temporal_channel_gated",
        modality_schema: Mapping[str, Any] | None = None,
        intention_head_mode: str = "hierarchical",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fusion_mode=fusion_mode,
            modality_schema=modality_schema,
        )
        fused_dim = d_model * 2
        _configure_residual_intention_head(
            self,
            fused_dim=fused_dim,
            intention_head_mode=intention_head_mode,
        )
        del self.pose_head
        self.receiving_hand_head = nn.Linear(fused_dim, 2)
        self.terminal_pose_head = self._make_pose_head(fused_dim, d_model, dropout)
        self.auxiliary_t1_pose_head = self._make_pose_head(
            fused_dim, d_model, dropout
        )

    @staticmethod
    def _make_pose_head(
        fused_dim: int, d_model: int, dropout: float
    ) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(fused_dim + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 14),
        )
        final_layer = head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        with torch.no_grad():
            final_layer.bias[6] = 1.0
            final_layer.bias[13] = 1.0
        return head

    @staticmethod
    def _pose_candidates(
        raw_residual: torch.Tensor,
        hand_reference_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = raw_residual.reshape(-1, 2, 7)
        position_delta = residual[:, :, :3]
        quaternion_delta = F.normalize(
            residual[:, :, 3:7], dim=-1, eps=1e-8
        )
        candidate_positions = hand_reference_pose[:, :, :3] + position_delta
        candidate_quaternions = quaternion_multiply(
            hand_reference_pose[:, :, 3:7], quaternion_delta
        )
        candidate_quaternions = F.normalize(
            candidate_quaternions, dim=-1, eps=1e-8
        )
        candidates = torch.cat(
            (candidate_positions, candidate_quaternions), dim=-1
        )
        return candidates, position_delta, quaternion_delta

    def forward(
        self,
        x: torch.Tensor,
        hand_reference_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hand_reference_pose is None:
            raise ValueError("Dual-horizon residual model requires hand_reference_pose")
        if hand_reference_pose.ndim != 3 or hand_reference_pose.shape[1:] != (2, 7):
            raise ValueError(
                "Expected hand_reference_pose with shape [batch, 2, 7], got "
                f"{tuple(hand_reference_pose.shape)}"
            )
        if hand_reference_pose.shape[0] != x.shape[0]:
            raise ValueError("Feature and hand-reference batch sizes differ")

        fused, fusion_outputs = self._encode(x)
        receiving_hand_logits = self.receiving_hand_head(fused)
        hand_probabilities = F.softmax(receiving_hand_logits, dim=-1)
        pose_input = torch.cat((fused, hand_probabilities), dim=-1)
        terminal = self._pose_candidates(
            self.terminal_pose_head(pose_input), hand_reference_pose
        )
        auxiliary = self._pose_candidates(
            self.auxiliary_t1_pose_head(pose_input), hand_reference_pose
        )
        outputs = {
            "receiving_hand_logits": receiving_hand_logits,
            "position_delta": terminal[1],
            "quaternion_delta": terminal[2],
            "pose_candidates": terminal[0],
            "auxiliary_position_delta": auxiliary[1],
            "auxiliary_quaternion_delta": auxiliary[2],
            "auxiliary_pose_candidates": auxiliary[0],
        }
        outputs.update(_residual_intention_outputs(self, fused))
        outputs.update(fusion_outputs)
        return outputs
