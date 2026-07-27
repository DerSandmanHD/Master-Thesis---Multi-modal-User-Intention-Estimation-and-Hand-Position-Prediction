#!/usr/bin/env python3
"""Replay a master CSV as a causal stream through the final residual-v2 model.

This is an inference-only bridge. It never sends commands to a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data import (
    INTENTION_NAMES,
    INTENTION_TO_ID,
    RECEIVING_HAND_NAMES,
    Normalizer,
)
from model import HierarchicalResidualPoseTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MODEL_TYPE = "hierarchical_residual_pose_transformer_v2"
POSE_COMPONENTS = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


@dataclass
class DeploymentArtifacts:
    intention_model: HierarchicalResidualPoseTransformer
    pose_model: HierarchicalResidualPoseTransformer
    normalizer: Normalizer
    feature_columns: list[str]
    window_size: int
    step_size: int
    minimum_observed_fraction: float
    max_timestamp_gap_ns: int
    future_horizon_seconds: float
    device: torch.device


@dataclass
class ReplayRecord:
    sequence_id: str
    participant: str
    timestamps_ns: np.ndarray
    features: np.ndarray
    intentions: np.ndarray
    pose_targets: np.ndarray
    pose_valid: np.ndarray
    hand_poses: np.ndarray
    hand_pose_valid: np.ndarray
    missing_features: list[str]
    pose_reference_schema_complete: bool


class TemporalDecisionFilter:
    """Smooth class probabilities and require repeated confident decisions."""

    def __init__(
        self,
        *,
        smoothing_window: int,
        minimum_confidence: float,
        minimum_stable_predictions: int,
    ) -> None:
        self.values: deque[np.ndarray] = deque(maxlen=smoothing_window)
        self.minimum_confidence = minimum_confidence
        self.minimum_stable_predictions = minimum_stable_predictions
        self.candidate: int | None = None
        self.candidate_count = 0

    def reset(self) -> None:
        self.values.clear()
        self.candidate = None
        self.candidate_count = 0

    def update(self, probabilities: np.ndarray) -> tuple[str, float, np.ndarray]:
        self.values.append(np.asarray(probabilities, dtype=np.float64))
        smoothed = np.mean(np.stack(self.values), axis=0)
        candidate = int(np.argmax(smoothed))
        if candidate == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = candidate
            self.candidate_count = 1
        confidence = float(smoothed[candidate])
        stable = (
            confidence >= self.minimum_confidence
            and self.candidate_count >= self.minimum_stable_predictions
        )
        return (
            INTENTION_NAMES[candidate] if stable else "uncertain",
            confidence,
            smoothed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("Training/final_clean_v1_residual_v2_seed44"),
        help="Directory with both checkpoints, config.json and data_metadata.json.",
    )
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Frames between predictions; defaults to the training stride.",
    )
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--minimum-confidence", type=float, default=0.65)
    parser.add_argument("--minimum-stable-predictions", type=int, default=2)
    parser.add_argument(
        "--print-mode", choices=("changes", "all", "none"), default="changes"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep according to source timestamps instead of processing immediately.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier when --realtime is active.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Stop early after this many predictions (useful for smoke tests).",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Diagnostic only: fill missing model inputs instead of rejecting them.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    return cwd_path if cwd_path.exists() else (PROJECT_ROOT / path).resolve()


def resolve_output_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_checkpoint_model(
    path: Path, device: torch.device
) -> tuple[HierarchicalResidualPoseTransformer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"{path.name} has model_type={checkpoint.get('model_type')!r}, "
            f"expected {EXPECTED_MODEL_TYPE!r}"
        )
    model = HierarchicalResidualPoseTransformer(
        input_dim=int(checkpoint["input_dim"]),
        window_size=int(checkpoint["window_size"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_artifacts(
    artifacts_dir: Path, requested_device: str, step_size: int | None
) -> DeploymentArtifacts:
    artifacts_dir = resolve_input_path(artifacts_dir)
    required = {
        "config": artifacts_dir / "config.json",
        "metadata": artifacts_dir / "data_metadata.json",
        "intention": artifacts_dir / "best_intention_model.pt",
        "pose": artifacts_dir / "best_pose_model.pt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment artifacts: " + ", ".join(missing))

    config = json.loads(required["config"].read_text(encoding="utf-8"))
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    normalizer_data = metadata["normalizer"]
    feature_columns = list(metadata["feature_columns"])
    if feature_columns != list(normalizer_data["feature_names"]):
        raise ValueError("Metadata feature_columns and normalizer feature_names differ")
    normalizer = Normalizer(
        mean=np.asarray(normalizer_data["mean"], dtype=np.float32),
        std=np.asarray(normalizer_data["std"], dtype=np.float32),
        feature_names=feature_columns,
    )
    if normalizer.output_feature_names != list(metadata["model_feature_columns"]):
        raise ValueError("Normalizer output feature order differs from metadata")

    device = choose_device(requested_device)
    intention_model, intention_checkpoint = load_checkpoint_model(
        required["intention"], device
    )
    pose_model, pose_checkpoint = load_checkpoint_model(required["pose"], device)
    if intention_checkpoint["selection_metric"] != "validation_intention_macro_f1":
        raise ValueError("Intention checkpoint was not selected by validation intent F1")
    if (
        pose_checkpoint["selection_metric"]
        != "validation_pose_oracle_position_mae_cm"
    ):
        raise ValueError("Pose checkpoint was not selected by validation pose MAE")

    window_size = int(config["data"]["window_size"])
    expected_input_dim = len(normalizer.output_feature_names)
    for name, checkpoint in (
        ("intention", intention_checkpoint),
        ("pose", pose_checkpoint),
    ):
        if int(checkpoint["window_size"]) != window_size:
            raise ValueError(f"{name} checkpoint and config window sizes differ")
        if int(checkpoint["input_dim"]) != expected_input_dim:
            raise ValueError(f"{name} checkpoint and metadata input dimensions differ")

    effective_step_size = (
        int(step_size) if step_size is not None else int(config["data"]["stride"])
    )
    if effective_step_size <= 0:
        raise ValueError("--step-size must be positive")
    max_gap_seconds = float(config["data"]["max_timestamp_gap_seconds"])
    return DeploymentArtifacts(
        intention_model=intention_model,
        pose_model=pose_model,
        normalizer=normalizer,
        feature_columns=feature_columns,
        window_size=window_size,
        step_size=effective_step_size,
        minimum_observed_fraction=float(
            config["data"]["minimum_observed_fraction"]
        ),
        max_timestamp_gap_ns=int(max_gap_seconds * 1e9),
        future_horizon_seconds=float(config["data"]["future_horizon_seconds"]),
        device=device,
    )


def replay_pose_columns(side: str) -> list[str]:
    return [
        *(f"{side}_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{side}_wrist_robot_q{component}" for component in "xyzw"),
    ]


def load_replay_record(
    path: Path,
    feature_columns: list[str],
    future_horizon_seconds: float,
) -> ReplayRecord:
    """Load inference inputs, treating labels and future targets as optional."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    available = set(header)
    required = {"timestamp_ns"}
    missing_required = sorted(required - available)
    if missing_required:
        raise ValueError(
            f"{path.name} lacks required replay columns: "
            + ", ".join(missing_required)
        )

    optional_core = {"sequence_id", "participant", "intent_label"}
    hand_schema = {
        "robot_frame_valid",
        "hand_left_valid",
        "hand_right_valid",
        *(column for side in RECEIVING_HAND_NAMES for column in replay_pose_columns(side)),
    }
    prefix = f"future_{future_horizon_seconds:g}s_"
    pose_columns = [
        *(f"{prefix}receiving_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{prefix}receiving_wrist_robot_q{component}" for component in "xyzw"),
    ]
    target_schema = {f"{prefix}receiving_wrist_valid", *pose_columns}
    use_columns = sorted(
        {
            "timestamp_ns",
            *(optional_core & available),
            *(hand_schema & available),
            *(target_schema & available),
            *(set(feature_columns) & available),
        }
    )
    frame = pd.read_csv(path, usecols=use_columns, low_memory=False)
    if frame.empty:
        raise ValueError(f"Empty master CSV: {path}")

    missing_features = [name for name in feature_columns if name not in frame]
    for column in missing_features:
        frame[column] = 0.0 if column.endswith(("_valid", "_interpolated")) else np.nan
    features = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
    )
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(np.int64)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Timestamps are not sorted in {path.name}")

    if "sequence_id" in frame:
        sequence_values = frame["sequence_id"].dropna().astype(str).unique()
        if len(sequence_values) != 1:
            raise ValueError(f"Inconsistent sequence_id values in {path.name}")
        sequence_id = str(sequence_values[0])
    else:
        sequence_id = path.stem.removesuffix("_master")
    if "participant" in frame:
        participant_values = frame["participant"].dropna().astype(str).unique()
        if len(participant_values) != 1:
            raise ValueError(f"Inconsistent participant values in {path.name}")
        participant = str(participant_values[0])
    else:
        participant = "unknown"

    intentions = np.full(len(frame), -2, dtype=np.int64)
    if "intent_label" in frame:
        labels = frame["intent_label"].map(INTENTION_TO_ID)
        known = labels.notna().to_numpy()
        intentions[known] = labels[known].to_numpy(np.int64)

    pose_targets = np.zeros((len(frame), 7), dtype=np.float32)
    pose_valid = np.zeros(len(frame), dtype=bool)
    if target_schema <= available:
        pose_targets = (
            frame[pose_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float32)
        )
        pose_valid = (
            pd.to_numeric(
                frame[f"{prefix}receiving_wrist_valid"], errors="coerce"
            )
            .fillna(0)
            .to_numpy()
            > 0
        )
        pose_valid &= np.isfinite(pose_targets).all(axis=1)
        norms = np.linalg.norm(pose_targets[:, 3:7], axis=1)
        pose_valid &= norms > 1e-6
        valid_rows = np.flatnonzero(pose_valid)
        if len(valid_rows):
            pose_targets[valid_rows, 3:7] /= norms[valid_rows, None]
        pose_targets[~pose_valid] = 0.0

    hand_poses = np.zeros((len(frame), 2, 7), dtype=np.float32)
    hand_poses[:, :, 6] = 1.0
    hand_pose_valid = np.zeros((len(frame), 2), dtype=bool)
    pose_reference_schema_complete = hand_schema <= available
    if pose_reference_schema_complete:
        robot_valid = (
            pd.to_numeric(frame["robot_frame_valid"], errors="coerce")
            .fillna(0)
            .to_numpy()
            > 0
        )
        for side_id, side in enumerate(RECEIVING_HAND_NAMES):
            current = (
                frame[replay_pose_columns(side)]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )
            explicit_valid = (
                pd.to_numeric(frame[f"hand_{side}_valid"], errors="coerce")
                .fillna(0)
                .to_numpy()
                > 0
            )
            current_valid = (
                explicit_valid & robot_valid & np.isfinite(current).all(axis=1)
            )
            norms = np.linalg.norm(current[:, 3:7], axis=1)
            current_valid &= norms > 1e-6
            valid_rows = np.flatnonzero(current_valid)
            if len(valid_rows):
                current[valid_rows, 3:7] /= norms[valid_rows, None]
                hand_poses[valid_rows, side_id] = current[valid_rows]
            hand_pose_valid[:, side_id] = current_valid

    return ReplayRecord(
        sequence_id=sequence_id,
        participant=participant,
        timestamps_ns=timestamps,
        features=features,
        intentions=intentions,
        pose_targets=pose_targets,
        pose_valid=pose_valid,
        hand_poses=hand_poses,
        hand_pose_valid=hand_pose_valid,
        missing_features=missing_features,
        pose_reference_schema_complete=pose_reference_schema_complete,
    )


def hand_references(
    hand_poses: np.ndarray,
    hand_pose_valid: np.ndarray,
    start: int,
    endpoint: int,
) -> tuple[np.ndarray, np.ndarray]:
    references = np.zeros((2, 7), dtype=np.float32)
    references[:, 6] = 1.0
    validity = np.zeros(2, dtype=bool)
    for side in range(2):
        valid_rows = np.flatnonzero(hand_pose_valid[start : endpoint + 1, side])
        if len(valid_rows):
            references[side] = hand_poses[start + int(valid_rows[-1]), side]
            validity[side] = True
    return references, validity


def joint_intention_probabilities(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    assistance = F.softmax(outputs["assistance_logits"], dim=-1)[0]
    assistance_type = F.softmax(outputs["assistance_type_logits"], dim=-1)[0]
    probabilities = torch.stack(
        (
            assistance[0],
            assistance[1] * assistance_type[0],
            assistance[1] * assistance_type[1],
        )
    )
    return probabilities.detach().cpu().numpy()


def hierarchical_intention_id(outputs: dict[str, torch.Tensor]) -> int:
    assistance = int(outputs["assistance_logits"].argmax(dim=-1).item())
    if assistance == 0:
        return 0
    return int(outputs["assistance_type_logits"].argmax(dim=-1).item()) + 1


def quaternion_error_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = prediction / max(float(np.linalg.norm(prediction)), 1e-8)
    expected = target / max(float(np.linalg.norm(target)), 1e-8)
    cosine = float(np.clip(abs(np.dot(predicted, expected)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def timed_forward(
    model: HierarchicalResidualPoseTransformer,
    features: torch.Tensor,
    references: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(features, references)
    synchronize(device)
    return outputs, (time.perf_counter() - started) * 1000.0


def maybe_wait_for_replay(
    timestamp_ns: int,
    *,
    first_timestamp_ns: int,
    wall_start: float,
    speed: float,
) -> None:
    target_elapsed = (timestamp_ns - first_timestamp_ns) / 1e9 / speed
    remaining = target_elapsed - (time.perf_counter() - wall_start)
    if remaining > 0:
        time.sleep(remaining)


def replay(args: argparse.Namespace) -> list[dict]:
    if args.smoothing_window <= 0 or args.minimum_stable_predictions <= 0:
        raise ValueError("Smoothing and stability window sizes must be positive")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("--minimum-confidence must be between zero and one")
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.max_predictions is not None and args.max_predictions <= 0:
        raise ValueError("--max-predictions must be positive")

    artifacts = load_artifacts(args.artifacts_dir, args.device, args.step_size)
    master_csv = resolve_input_path(args.master_csv)
    record = load_replay_record(
        master_csv,
        artifacts.feature_columns,
        artifacts.future_horizon_seconds,
    )
    if record.missing_features:
        if not args.allow_missing_features:
            raise ValueError(
                "Replay input does not match the final training schema; missing "
                "model features: "
                + ", ".join(record.missing_features)
                + ". Copy a current final master CSV. Use "
                "--allow-missing-features only for diagnostic tests."
            )
        print(
            "WARNING: replay input lacks "
            f"{len(record.missing_features)}/{len(artifacts.feature_columns)} "
            "model features: "
            + ", ".join(record.missing_features)
        )
    if not record.pose_reference_schema_complete:
        print(
            "WARNING: pose output is disabled because the replay input lacks the "
            "complete robot-frame hand-reference schema"
        )
    normalized_features = artifacts.normalizer.transform(record.features)
    if normalized_features.shape[1] != artifacts.intention_model.input_dim:
        raise ValueError("Normalized replay input dimension differs from checkpoint")

    decision_filter = TemporalDecisionFilter(
        smoothing_window=args.smoothing_window,
        minimum_confidence=args.minimum_confidence,
        minimum_stable_predictions=args.minimum_stable_predictions,
    )
    first_timestamp_ns = int(record.timestamps_ns[artifacts.window_size - 1])
    wall_start = time.perf_counter()
    previous_printed_label: str | None = None
    rows: list[dict] = []

    for endpoint in range(
        artifacts.window_size - 1,
        len(record.timestamps_ns),
        artifacts.step_size,
    ):
        start = endpoint - artifacts.window_size + 1
        timestamps = record.timestamps_ns[start : endpoint + 1]
        if np.any(np.diff(timestamps) > artifacts.max_timestamp_gap_ns):
            decision_filter.reset()
            continue
        observed_fraction = float(
            np.isfinite(record.features[start : endpoint + 1]).mean()
        )
        if observed_fraction < artifacts.minimum_observed_fraction:
            decision_filter.reset()
            continue

        timestamp_ns = int(record.timestamps_ns[endpoint])
        if args.realtime:
            maybe_wait_for_replay(
                timestamp_ns,
                first_timestamp_ns=first_timestamp_ns,
                wall_start=wall_start,
                speed=args.speed,
            )

        references, reference_valid = hand_references(
            record.hand_poses, record.hand_pose_valid, start, endpoint
        )
        feature_tensor = torch.from_numpy(
            normalized_features[start : endpoint + 1][None, ...]
        ).to(artifacts.device)
        reference_tensor = torch.from_numpy(references[None, ...]).to(artifacts.device)
        intention_outputs, intention_ms = timed_forward(
            artifacts.intention_model,
            feature_tensor,
            reference_tensor,
            artifacts.device,
        )
        probabilities = joint_intention_probabilities(intention_outputs)
        raw_intention_id = hierarchical_intention_id(intention_outputs)
        stable_label, stable_confidence, smoothed = decision_filter.update(
            probabilities
        )

        predicted_hand: str | None = None
        predicted_pose: np.ndarray | None = None
        pose_ms: float | None = None
        pose_reference_valid: bool | None = None
        if stable_label == "handover":
            pose_outputs, pose_ms = timed_forward(
                artifacts.pose_model,
                feature_tensor,
                reference_tensor,
                artifacts.device,
            )
            hand_id = int(pose_outputs["receiving_hand_logits"].argmax(dim=-1).item())
            predicted_hand = RECEIVING_HAND_NAMES[hand_id]
            pose_reference_valid = bool(reference_valid[hand_id])
            if pose_reference_valid:
                predicted_pose = (
                    pose_outputs["pose_candidates"][0, hand_id].detach().cpu().numpy()
                )

        target_id = int(record.intentions[endpoint])
        if target_id >= 0:
            target_label = INTENTION_NAMES[target_id]
        elif target_id == -1:
            target_label = "transition"
        else:
            target_label = "unavailable"
        target_pose_valid = bool(record.pose_valid[endpoint] and target_id == 2)
        position_error_cm: float | None = None
        orientation_error_deg: float | None = None
        if predicted_pose is not None and target_pose_valid:
            target_pose = record.pose_targets[endpoint]
            position_error_cm = float(
                np.linalg.norm(predicted_pose[:3] - target_pose[:3]) * 100.0
            )
            orientation_error_deg = quaternion_error_deg(
                predicted_pose[3:7], target_pose[3:7]
            )

        row = {
            "sequence_id": record.sequence_id,
            "participant": record.participant,
            "endpoint_row": endpoint,
            "timestamp_ns": timestamp_ns,
            "elapsed_seconds": (timestamp_ns - int(record.timestamps_ns[0])) / 1e9,
            "target_intention": target_label,
            "raw_intention": INTENTION_NAMES[raw_intention_id],
            "raw_confidence": float(probabilities[raw_intention_id]),
            "stable_intention": stable_label,
            "stable_confidence": stable_confidence,
            "p_continue": float(smoothed[0]),
            "p_fetch": float(smoothed[1]),
            "p_handover": float(smoothed[2]),
            "predicted_receiving_hand": predicted_hand,
            "predicted_hand_reference_valid": pose_reference_valid,
            "intention_inference_ms": intention_ms,
            "pose_inference_ms": pose_ms,
            "target_pose_valid": target_pose_valid,
            "pose_position_error_cm": position_error_cm,
            "pose_orientation_error_deg": orientation_error_deg,
        }
        for index, component in enumerate(POSE_COMPONENTS):
            row[f"predicted_pose_{component}"] = (
                float(predicted_pose[index]) if predicted_pose is not None else None
            )
        rows.append(row)

        should_print = args.print_mode == "all" or (
            args.print_mode == "changes"
            and (
                stable_label != previous_printed_label
                or stable_label == "handover"
            )
        )
        if should_print:
            pose_text = ""
            if predicted_pose is not None:
                pose_text = (
                    f" | hand={predicted_hand} | xyz="
                    f"({predicted_pose[0]:+.3f}, {predicted_pose[1]:+.3f}, "
                    f"{predicted_pose[2]:+.3f}) m"
                )
            elif stable_label == "handover":
                pose_text = f" | hand={predicted_hand} | pose=no valid reference"
            print(
                f"[{row['elapsed_seconds']:7.2f}s] "
                f"raw={row['raw_intention']} ({row['raw_confidence']:.3f}) | "
                f"stable={stable_label} ({stable_confidence:.3f}) | "
                f"truth={target_label} | intent={intention_ms:.2f} ms"
                f"{pose_text}"
            )
            previous_printed_label = stable_label

        if args.max_predictions is not None and len(rows) >= args.max_predictions:
            break

    if not rows:
        raise RuntimeError("Replay produced no valid prediction windows")
    return rows


def write_rows(rows: list[dict], output_path: Path) -> None:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {output_path}")


def print_summary(rows: list[dict]) -> None:
    labeled = [
        row
        for row in rows
        if row["target_intention"] not in {"transition", "unavailable"}
    ]
    correct = sum(
        row["raw_intention"] == row["target_intention"] for row in labeled
    )
    intention_times = [float(row["intention_inference_ms"]) for row in rows]
    pose_times = [
        float(row["pose_inference_ms"])
        for row in rows
        if row["pose_inference_ms"] is not None
    ]
    accuracy = f"{correct / len(labeled):.4f}" if labeled else "n/a"
    print(
        f"Replay complete: predictions={len(rows)}, labeled={len(labeled)}, "
        f"raw accuracy={accuracy}"
    )
    print(
        "Intention inference: "
        f"mean={np.mean(intention_times):.2f} ms, "
        f"p95={np.percentile(intention_times, 95):.2f} ms"
    )
    if pose_times:
        print(
            "Pose inference: "
            f"mean={np.mean(pose_times):.2f} ms, "
            f"p95={np.percentile(pose_times, 95):.2f} ms"
        )
    pose_evaluations = [
        row for row in rows if row["pose_position_error_cm"] is not None
    ]
    if pose_evaluations:
        position_errors = [
            float(row["pose_position_error_cm"]) for row in pose_evaluations
        ]
        orientation_errors = [
            float(row["pose_orientation_error_deg"]) for row in pose_evaluations
        ]
        print(
            f"Pose replay evaluation: samples={len(pose_evaluations)}, "
            f"position MAE={np.mean(position_errors):.2f} cm, "
            f"orientation mean={np.mean(orientation_errors):.2f} deg"
        )


def main() -> int:
    args = parse_args()
    try:
        rows = replay(args)
        print_summary(rows)
        if args.output_csv is not None:
            write_rows(rows, args.output_csv)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
