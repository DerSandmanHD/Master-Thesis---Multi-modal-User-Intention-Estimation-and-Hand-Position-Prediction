#!/usr/bin/env python3
"""Participant-safe windowed datasets built from per-sequence master CSVs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


INTENTION_TO_ID = {"transition": -1, "continue": 0, "fetch": 1, "handover": 2}
INTENTION_NAMES = ["continue", "fetch", "handover"]


def _candidate_features() -> list[str]:
    columns = [
        "gaze_valid",
        "gaze_yaw_rad",
        "gaze_pitch_rad",
        "gaze_depth_m",
        "gaze_origin_robot_x_m",
        "gaze_origin_robot_y_m",
        "gaze_origin_robot_z_m",
        "gaze_direction_robot_x",
        "gaze_direction_robot_y",
        "gaze_direction_robot_z",
        "hand_left_tracking_confidence",
        "hand_right_tracking_confidence",
        "hand_left_valid",
        "hand_right_valid",
    ]
    for side in ("left", "right"):
        columns.extend(f"{side}_wrist_robot_{axis}_m" for axis in "xyz")
        columns.extend(f"{side}_wrist_robot_q{component}" for component in "xyzw")
    columns.extend(
        [
            "slam_device_linear_velocity_x_device",
            "slam_device_linear_velocity_y_device",
            "slam_device_linear_velocity_z_device",
            "slam_angular_velocity_x_device",
            "slam_angular_velocity_y_device",
            "slam_angular_velocity_z_device",
            "slam_quality_score",
            "apriltag_0_valid",
        ]
    )
    for marker_id in range(6, 15):
        columns.extend(f"aruco_{marker_id}_robot_{axis}_m" for axis in "xyz")
        columns.extend(
            [
                f"aruco_{marker_id}_gaze_angle_rad",
                f"aruco_{marker_id}_gaze_distance_m",
                f"aruco_{marker_id}_valid",
            ]
        )
    return columns


def select_feature_columns(columns: Iterable[str], profile: str) -> list[str]:
    if profile != "multimodal_robot_frame_v1":
        raise ValueError(f"Unknown feature profile: {profile}")
    available = set(columns)
    selected = [column for column in _candidate_features() if column in available]
    if len(selected) < 20:
        raise ValueError(
            "Master CSV exposes too few v1 input features. Rebuild it with the current "
            f"build_master_dataset.py (found {len(selected)})."
        )
    return selected


def horizon_prefix(seconds: float) -> str:
    return f"future_{seconds:g}s_"


@dataclass
class SequenceRecord:
    sequence_id: str
    participant: str
    timestamps_ns: np.ndarray
    features: np.ndarray
    intentions: np.ndarray
    pose_targets: np.ndarray
    pose_valid: np.ndarray


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    def transform(self, values: np.ndarray) -> np.ndarray:
        observed = np.isfinite(values)
        normalized = (values - self.mean) / self.std
        normalized = np.where(observed, normalized, 0.0).astype(np.float32)
        return np.concatenate((normalized, observed.astype(np.float32)), axis=1)

    @property
    def output_feature_names(self) -> list[str]:
        return self.feature_names + [f"{name}__observed" for name in self.feature_names]

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "output_feature_names": self.output_feature_names,
        }


def fit_normalizer(records: list[SequenceRecord], feature_names: list[str]) -> Normalizer:
    if not records:
        raise ValueError("Cannot fit normalization without training sequences")
    feature_count = len(feature_names)
    sums = np.zeros(feature_count, dtype=np.float64)
    square_sums = np.zeros(feature_count, dtype=np.float64)
    counts = np.zeros(feature_count, dtype=np.int64)
    for record in records:
        finite = np.isfinite(record.features)
        safe = np.where(finite, record.features, 0.0).astype(np.float64)
        sums += safe.sum(axis=0)
        square_sums += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    mean = sums / safe_counts
    variance = np.maximum(square_sums / safe_counts - np.square(mean), 0.0)
    std = np.sqrt(variance)
    mean[counts == 0] = 0.0
    std[(counts == 0) | (std < 1e-6)] = 1.0
    return Normalizer(mean.astype(np.float32), std.astype(np.float32), feature_names)


def load_record(
    path: Path,
    feature_columns: list[str],
    *,
    future_horizon_seconds: float,
) -> SequenceRecord:
    prefix = horizon_prefix(future_horizon_seconds)
    pose_columns = [
        *(f"{prefix}receiving_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{prefix}receiving_wrist_robot_q{component}" for component in "xyzw"),
    ]
    required_core = [
        "sequence_id",
        "participant",
        "timestamp_ns",
        "intent_label",
        f"{prefix}receiving_wrist_valid",
        *pose_columns,
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_core) - set(header))
    if missing:
        raise ValueError(
            f"{path.name} is not compatible with the training schema; missing: "
            f"{', '.join(missing)}. Rebuild master datasets after semantic annotation."
        )
    available_features = [column for column in feature_columns if column in header]
    frame = pd.read_csv(path, usecols=[*required_core, *available_features])
    for column in feature_columns:
        if column not in frame:
            frame[column] = 0.0 if column.endswith("_valid") else np.nan
    if frame.empty:
        raise ValueError(f"Empty master CSV: {path}")
    sequence_values = frame["sequence_id"].dropna().astype(str).unique()
    participant_values = frame["participant"].dropna().astype(str).unique()
    if len(sequence_values) != 1 or len(participant_values) != 1:
        raise ValueError(f"Inconsistent sequence/participant columns in {path}")

    labels = frame["intent_label"].map(INTENTION_TO_ID)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), "intent_label"].astype(str).unique())
        raise ValueError(f"Unknown intention labels in {path.name}: {unknown}")

    poses = frame[pose_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    explicit_pose_valid = (
        pd.to_numeric(frame[f"{prefix}receiving_wrist_valid"], errors="coerce")
        .fillna(0)
        .to_numpy()
        > 0
    )
    pose_valid = explicit_pose_valid & np.isfinite(poses).all(axis=1)
    quaternion_norm = np.linalg.norm(poses[:, 3:7], axis=1)
    pose_valid &= quaternion_norm > 1e-6
    valid_indices = np.flatnonzero(pose_valid)
    if len(valid_indices):
        poses[valid_indices, 3:7] /= quaternion_norm[valid_indices, None]
    poses[~pose_valid] = 0.0

    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(np.int64)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Timestamps are not sorted in {path.name}")
    return SequenceRecord(
        sequence_id=sequence_values[0],
        participant=participant_values[0],
        timestamps_ns=timestamps,
        features=features,
        intentions=labels.to_numpy(np.int64),
        pose_targets=poses,
        pose_valid=pose_valid,
    )


def split_records(
    records: list[SequenceRecord],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    validation_participants: list[str],
    test_participants: list[str],
) -> tuple[dict[str, list[SequenceRecord]], dict]:
    participants = sorted({record.participant for record in records})
    if len(participants) < 3:
        raise ValueError("At least three participants are required for leakage-safe train/val/test splits")

    explicit = bool(validation_participants or test_participants)
    if explicit:
        validation = set(validation_participants)
        test = set(test_participants)
        unknown = (validation | test) - set(participants)
        if unknown:
            raise ValueError(f"Configured split participants not found: {sorted(unknown)}")
        if validation & test:
            raise ValueError("Validation and test participant sets overlap")
        if not validation or not test:
            raise ValueError("Explicit splits require both validation and test participants")
    else:
        shuffled = participants.copy()
        random.Random(seed).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        test_count = max(1, round(len(shuffled) * test_fraction))
        while validation_count + test_count >= len(shuffled):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("Not enough participants for requested split fractions")
        validation = set(shuffled[:validation_count])
        test = set(shuffled[validation_count : validation_count + test_count])

    train = set(participants) - validation - test
    split = {
        "train": [record for record in records if record.participant in train],
        "validation": [record for record in records if record.participant in validation],
        "test": [record for record in records if record.participant in test],
    }
    if any(not values for values in split.values()):
        raise ValueError("One of train/validation/test contains no sequences")
    metadata = {
        "strategy": "explicit_participants" if explicit else "seeded_participant_group_split",
        "seed": seed,
        "participants": {
            name: sorted({record.participant for record in values})
            for name, values in split.items()
        },
        "sequences": {
            name: sorted(record.sequence_id for record in values)
            for name, values in split.items()
        },
    }
    return split, metadata


class WindowDataset(Dataset):
    def __init__(
        self,
        records: list[SequenceRecord],
        *,
        window_size: int,
        stride: int,
        pose_intent_ids: list[int],
        minimum_observed_fraction: float,
        max_timestamp_gap_seconds: float,
    ) -> None:
        self.records = records
        self.window_size = window_size
        self.indices: list[tuple[int, int]] = []
        self.discarded_gap_windows = 0
        self.discarded_observation_windows = 0
        self.discarded_unlabeled_windows = 0
        max_timestamp_gap_ns = int(max_timestamp_gap_seconds * 1e9)
        if max_timestamp_gap_ns <= 0:
            raise ValueError("max_timestamp_gap_seconds must be greater than zero")
        pose_intents = set(pose_intent_ids)
        if pose_intents != {INTENTION_TO_ID["handover"]}:
            raise ValueError(
                "Hierarchical baseline requires pose_intent_ids=[2] so pose loss "
                "is restricted to handover"
            )
        for record_index, record in enumerate(records):
            raw_feature_count = record.features.shape[1] // 2
            for endpoint in range(window_size - 1, len(record.features), stride):
                start = endpoint - window_size + 1
                if record.intentions[endpoint] < 0:
                    self.discarded_unlabeled_windows += 1
                    continue
                if np.any(
                    np.diff(record.timestamps_ns[start : endpoint + 1])
                    > max_timestamp_gap_ns
                ):
                    self.discarded_gap_windows += 1
                    continue
                observed_fraction = float(
                    record.features[
                        start : endpoint + 1, raw_feature_count:
                    ].mean()
                )
                if observed_fraction < minimum_observed_fraction:
                    self.discarded_observation_windows += 1
                    continue
                self.indices.append((record_index, endpoint))
            record.pose_valid &= np.isin(record.intentions, list(pose_intents))
        if not self.indices:
            raise ValueError("No valid windows were created; inspect window size and data coverage")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        record_index, endpoint = self.indices[index]
        record = self.records[record_index]
        start = endpoint - self.window_size + 1
        return {
            "features": torch.from_numpy(record.features[start : endpoint + 1]),
            "intention": torch.tensor(record.intentions[endpoint], dtype=torch.long),
            "pose_target": torch.from_numpy(record.pose_targets[endpoint]),
            "pose_valid": torch.tensor(record.pose_valid[endpoint], dtype=torch.bool),
            "sequence_id": record.sequence_id,
            "participant": record.participant,
            "timestamp_ns": torch.tensor(record.timestamps_ns[endpoint], dtype=torch.long),
        }

    def intention_counts(self) -> list[int]:
        counts = np.zeros(len(INTENTION_NAMES), dtype=np.int64)
        for record_index, endpoint in self.indices:
            counts[self.records[record_index].intentions[endpoint]] += 1
        return counts.tolist()


@dataclass
class DataBundle:
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    normalizer: Normalizer
    feature_columns: list[str]
    split_metadata: dict


def prepare_data(data_config: dict, seed: int, limit_sequences: int | None = None) -> DataBundle:
    master_dir = Path(data_config["master_dir"]).expanduser().resolve()
    files = sorted(master_dir.glob("*_master.csv"))
    if limit_sequences is not None:
        files = files[: max(0, limit_sequences)]
    if not files:
        raise FileNotFoundError(f"No *_master.csv files found in {master_dir}")

    first_header = pd.read_csv(files[0], nrows=0).columns.tolist()
    feature_columns = select_feature_columns(first_header, data_config["feature_profile"])
    records = [
        load_record(
            path,
            feature_columns,
            future_horizon_seconds=float(data_config["future_horizon_seconds"]),
        )
        for path in files
    ]
    split, split_metadata = split_records(
        records,
        seed=seed,
        validation_fraction=float(data_config["validation_fraction"]),
        test_fraction=float(data_config["test_fraction"]),
        validation_participants=list(data_config.get("validation_participants", [])),
        test_participants=list(data_config.get("test_participants", [])),
    )
    normalizer = fit_normalizer(split["train"], feature_columns)
    normalized_split = {
        name: [replace(record, features=normalizer.transform(record.features)) for record in values]
        for name, values in split.items()
    }
    dataset_args = {
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "pose_intent_ids": [int(value) for value in data_config["pose_intent_ids"]],
        "minimum_observed_fraction": float(data_config["minimum_observed_fraction"]),
        "max_timestamp_gap_seconds": float(data_config["max_timestamp_gap_seconds"]),
    }
    return DataBundle(
        train=WindowDataset(normalized_split["train"], **dataset_args),
        validation=WindowDataset(normalized_split["validation"], **dataset_args),
        test=WindowDataset(normalized_split["test"], **dataset_args),
        normalizer=normalizer,
        feature_columns=feature_columns,
        split_metadata=split_metadata,
    )


def save_data_metadata(bundle: DataBundle, path: Path) -> None:
    data = {
        "label_mapping": INTENTION_TO_ID,
        "transition_policy": "context_only_never_window_target",
        "feature_columns": bundle.feature_columns,
        "model_feature_columns": bundle.normalizer.output_feature_names,
        "normalizer": bundle.normalizer.to_dict(),
        "split": bundle.split_metadata,
        "windows": {
            "train": len(bundle.train),
            "validation": len(bundle.validation),
            "test": len(bundle.test),
        },
        "discarded_windows": {
            name: {
                "timestamp_gap": dataset.discarded_gap_windows,
                "low_observation": dataset.discarded_observation_windows,
                "unlabeled_endpoint": dataset.discarded_unlabeled_windows,
            }
            for name, dataset in (
                ("train", bundle.train),
                ("validation", bundle.validation),
                ("test", bundle.test),
            )
        },
        "intention_counts": {
            name: dict(zip(INTENTION_NAMES, dataset.intention_counts()))
            for name, dataset in (
                ("train", bundle.train),
                ("validation", bundle.validation),
                ("test", bundle.test),
            )
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
