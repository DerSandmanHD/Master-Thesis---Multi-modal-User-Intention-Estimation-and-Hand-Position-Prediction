#!/usr/bin/env python3
"""Stateful causal inference engine shared by live sensor adapters."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from data import INTENTION_NAMES, RECEIVING_HAND_NAMES
from replay_stream_inference import (
    DeploymentArtifacts,
    TemporalDecisionFilter,
    hierarchical_intention_id,
    intention_id_from_probabilities,
    joint_intention_probabilities,
    load_artifacts,
    timed_forward,
)
from prediction_utils import intention_head_mode


class OnlineInferenceEngine:
    """Buffer normalized feature frames and emit causal residual-v2 predictions."""

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        device: str = "auto",
        step_size: int | None = None,
        smoothing_window: int = 3,
        minimum_confidence: float = 0.65,
        minimum_stable_predictions: int = 2,
        warm_up_models: bool = True,
    ) -> None:
        if smoothing_window <= 0 or minimum_stable_predictions <= 0:
            raise ValueError("Smoothing and stability window sizes must be positive")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between zero and one")
        self.artifacts: DeploymentArtifacts = load_artifacts(
            artifacts_dir, device, step_size
        )
        self.filter = TemporalDecisionFilter(
            smoothing_window=smoothing_window,
            minimum_confidence=minimum_confidence,
            minimum_stable_predictions=minimum_stable_predictions,
        )
        size = self.artifacts.window_size
        self.timestamps: deque[int] = deque(maxlen=size)
        self.features: deque[np.ndarray] = deque(maxlen=size)
        self.hand_poses: deque[np.ndarray] = deque(maxlen=size)
        self.hand_valid: deque[np.ndarray] = deque(maxlen=size)
        self.frames_since_prediction = 0
        self.last_timestamp_ns: int | None = None
        self.prediction_index = 0
        self.warmup_latency_ms: dict[str, float] | None = None
        if warm_up_models:
            self.warmup_latency_ms = self._warm_up_models()

    @property
    def feature_columns(self) -> list[str]:
        return self.artifacts.feature_columns

    @property
    def ready_frames(self) -> int:
        return len(self.features)

    @property
    def required_frames(self) -> int:
        return self.artifacts.window_size

    def reset(self) -> None:
        self.timestamps.clear()
        self.features.clear()
        self.hand_poses.clear()
        self.hand_valid.clear()
        self.filter.reset()
        self.frames_since_prediction = 0
        self.last_timestamp_ns = None

    def _warm_up_models(self) -> dict[str, float]:
        """Run one inference-only dummy forward for the deployment checkpoint."""

        features = torch.zeros(
            (
                1,
                self.artifacts.window_size,
                self.artifacts.model.input_dim,
            ),
            dtype=torch.float32,
            device=self.artifacts.device,
        )
        references = torch.zeros(
            (1, len(RECEIVING_HAND_NAMES), 7),
            dtype=torch.float32,
            device=self.artifacts.device,
        )
        references[:, :, 6] = 1.0
        _, model_ms = timed_forward(
            self.artifacts.model,
            features,
            references,
            self.artifacts.device,
        )
        return {"model": float(model_ms)}

    def _feature_array(self, values: Mapping[str, float | int | None]) -> np.ndarray:
        missing = [name for name in self.feature_columns if name not in values]
        if missing:
            raise ValueError(
                "Live frame lacks model feature keys: " + ", ".join(missing)
            )
        return np.asarray(
            [
                np.nan if values[name] is None else float(values[name])
                for name in self.feature_columns
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _hand_references(
        values: Mapping[str, float | int | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        valid = np.zeros(2, dtype=bool)
        robot_frame_valid = bool(values.get("robot_frame_valid", 0))
        for side_id, side in enumerate(RECEIVING_HAND_NAMES):
            columns = [
                *(f"{side}_wrist_robot_{axis}_m" for axis in "xyz"),
                *(f"{side}_wrist_robot_q{component}" for component in "xyzw"),
            ]
            raw = np.asarray(
                [
                    np.nan if values.get(column) is None else float(values[column])
                    for column in columns
                ],
                dtype=np.float32,
            )
            side_valid = bool(values.get(f"hand_{side}_valid", 0))
            quaternion_norm = float(np.linalg.norm(raw[3:7]))
            if (
                robot_frame_valid
                and side_valid
                and np.isfinite(raw).all()
                and quaternion_norm > 1e-6
            ):
                raw[3:7] /= quaternion_norm
                poses[side_id] = raw
                valid[side_id] = True
        return poses, valid

    def _latest_references(self) -> tuple[np.ndarray, np.ndarray]:
        references = np.zeros((2, 7), dtype=np.float32)
        references[:, 6] = 1.0
        validity = np.zeros(2, dtype=bool)
        poses = list(self.hand_poses)
        valid = list(self.hand_valid)
        for side in range(2):
            for index in range(len(valid) - 1, -1, -1):
                if bool(valid[index][side]):
                    references[side] = poses[index][side]
                    validity[side] = True
                    break
        return references, validity

    def push_frame(
        self,
        timestamp_ns: int,
        values: Mapping[str, float | int | None],
        *,
        pipeline_timestamps: Mapping[str, int] | None = None,
    ) -> dict | None:
        phases = dict(pipeline_timestamps or {})
        phases["engine_push_started_host_ns"] = time.monotonic_ns()
        timestamp_ns = int(timestamp_ns)
        if self.last_timestamp_ns is not None:
            if timestamp_ns <= self.last_timestamp_ns:
                raise ValueError("Live frame timestamps must be strictly increasing")
            if (
                timestamp_ns - self.last_timestamp_ns
                > self.artifacts.max_timestamp_gap_ns
            ):
                self.reset()
        self.last_timestamp_ns = timestamp_ns

        raw_features = self._feature_array(values)
        hand_poses, hand_valid = self._hand_references(values)
        self.timestamps.append(timestamp_ns)
        self.features.append(raw_features)
        self.hand_poses.append(hand_poses)
        self.hand_valid.append(hand_valid)
        self.frames_since_prediction += 1

        if len(self.features) < self.required_frames:
            return None
        if (
            self.prediction_index > 0
            and self.frames_since_prediction < self.artifacts.step_size
        ):
            return None

        raw_window = np.stack(self.features)
        observed_fraction = float(np.isfinite(raw_window).mean())
        if observed_fraction < self.artifacts.minimum_observed_fraction:
            self.filter.reset()
            self.frames_since_prediction = 0
            return None

        normalized = self.artifacts.normalizer.transform(raw_window)
        references, reference_valid = self._latest_references()
        feature_tensor = torch.from_numpy(normalized[None, ...]).to(
            self.artifacts.device
        )
        reference_tensor = torch.from_numpy(references[None, ...]).to(
            self.artifacts.device
        )
        inference_started_ns = time.monotonic_ns()
        phases["model_inference_started_host_ns"] = inference_started_ns
        # Compatibility timestamps: the single forward produces every task head.
        phases["intention_inference_started_host_ns"] = inference_started_ns
        outputs, model_ms = timed_forward(
            self.artifacts.model,
            feature_tensor,
            reference_tensor,
            self.artifacts.device,
        )
        inference_ended_ns = time.monotonic_ns()
        phases["model_inference_ended_host_ns"] = inference_ended_ns
        phases["intention_inference_ended_host_ns"] = inference_ended_ns
        probabilities = joint_intention_probabilities(outputs)
        raw_id = intention_id_from_probabilities(probabilities)
        decision_rule_raw_id = hierarchical_intention_id(outputs)
        hierarchical_output = (
            intention_head_mode(outputs) == "hierarchical"
        )
        phases["raw_decision_host_ns"] = time.monotonic_ns()
        stable_label, stable_confidence, smoothed = self.filter.update(probabilities)
        phases["stable_decision_host_ns"] = time.monotonic_ns()

        predicted_hand = None
        predicted_pose = None
        pose_reference_valid = None
        if stable_label == "handover":
            hand_id = int(
                outputs["receiving_hand_logits"].argmax(dim=-1).item()
            )
            predicted_hand = RECEIVING_HAND_NAMES[hand_id]
            pose_reference_valid = bool(reference_valid[hand_id])
            if pose_reference_valid:
                predicted_pose = (
                    outputs["pose_candidates"][0, hand_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        self.frames_since_prediction = 0
        self.prediction_index += 1
        phases["engine_prediction_ready_host_ns"] = time.monotonic_ns()
        modality_names = list(
            getattr(self.artifacts.model, "modality_names", ())
        )
        modality_weights = {
            name: float(outputs["modality_weights"][0, index])
            for index, name in enumerate(modality_names)
        }
        modality_available = {
            name: bool(outputs["modality_available"][0, index])
            for index, name in enumerate(modality_names)
        }
        return {
            "prediction_index": self.prediction_index,
            "timestamp_ns": timestamp_ns,
            "raw_intention": INTENTION_NAMES[raw_id],
            "raw_confidence": float(probabilities[raw_id]),
            "decision_rule_raw_intention": INTENTION_NAMES[
                decision_rule_raw_id
            ],
            "joint_probability_argmax_intention": INTENTION_NAMES[raw_id],
            "hierarchical_raw_intention": (
                INTENTION_NAMES[decision_rule_raw_id]
                if hierarchical_output
                else None
            ),

            "raw_p_continue": float(probabilities[0]),
            "raw_p_fetch": float(probabilities[1]),
            "raw_p_handover": float(probabilities[2]),

            "stable_intention": stable_label,
            "stable_confidence": stable_confidence,

            "p_continue": float(smoothed[0]),
            "p_fetch": float(smoothed[1]),
            "p_handover": float(smoothed[2]),

            "predicted_receiving_hand": predicted_hand,
            "predicted_hand_reference_valid": pose_reference_valid,
            "predicted_pose_robot": predicted_pose,
            "observed_fraction": observed_fraction,
            "model_inference_ms": model_ms,
            "intention_inference_ms": model_ms,
            "pose_inference_ms": None,
            "pose_reuses_primary_forward": True,
            "checkpoint_path": str(self.artifacts.checkpoint_path),
            "checkpoint_epoch": self.artifacts.checkpoint_epoch,
            "checkpoint_selection_metric": (
                self.artifacts.checkpoint_selection_metric
            ),
            "modality_weights": modality_weights,
            "modality_available": modality_available,
            "pipeline_timestamps": phases,
        }
