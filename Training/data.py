#!/usr/bin/env python3
"""Participant-safe windowed datasets built from per-sequence master CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


INTENTION_TO_ID = {"transition": -1, "continue": 0, "fetch": 1, "handover": 2}
INTENTION_NAMES = ["continue", "fetch", "handover"]
RECEIVING_HAND_TO_ID = {"left": 0, "right": 1}
RECEIVING_HAND_NAMES = ["left", "right"]
SUPPORTED_MODALITY_ABLATIONS = ("gaze", "hands", "objects", "vio")
TRAINING_DATA_BUILDER_VERSION = "training_data_pipeline_v3_modality_ablation"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_command(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_provenance() -> dict:
    commit = _git_command("rev-parse", "HEAD")
    status = _git_command("status", "--porcelain", "--untracked-files=all")
    status_lines = [] if not status else status.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status_lines) if status is not None else None,
        "changed_paths": [
            line[3:] if len(line) > 3 else line
            for line in status_lines
        ],
        "status_available": status is not None,
    }


def runtime_provenance() -> dict:
    container_variables = (
        "APPTAINER_CONTAINER",
        "APPTAINER_NAME",
        "CONTAINER_IMAGE_DIGEST",
        "SINGULARITY_CONTAINER",
        "SINGULARITY_NAME",
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
    )
    container = {
        name: os.environ[name]
        for name in container_variables
        if os.environ.get(name)
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "container_environment": container,
        "container_digest_available": bool(
            container.get("CONTAINER_IMAGE_DIGEST")
        ),
    }


def build_dataset_provenance(
    files: list[Path],
    *,
    master_dir: Path,
    feature_profile: str,
    feature_columns: list[str],
    feature_ablation: dict,
    future_horizon_seconds: float,
    filter_metadata: dict,
) -> tuple[dict, str | None]:
    master_files = []
    for path in files:
        master_files.append(
            {
                "sequence_id": sequence_id_from_master_path(path),
                "file_name": path.name,
                "relative_path": str(path.relative_to(master_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest_snapshot = None
    manifest = None
    manifest_path_value = filter_metadata.get("manifest_path")
    if manifest_path_value:
        manifest_path = Path(manifest_path_value)
        manifest_snapshot = manifest_path.read_text(encoding="utf-8")
        manifest = {
            "source_path": str(manifest_path),
            "sha256": hashlib.sha256(
                manifest_snapshot.encode("utf-8")
            ).hexdigest(),
            "snapshot_file": "dataset_manifest_snapshot.csv",
        }

    builder_files = {}
    for relative in (
        Path("Code/build_master_dataset.py"),
        Path("Code/dataset_qa.py"),
        Path("Training/data.py"),
        Path("singularity/aria.recipe"),
    ):
        path = PROJECT_ROOT / relative
        if path.is_file():
            builder_files[str(relative)] = sha256_file(path)

    schema_payload = {
        "feature_profile": feature_profile,
        "feature_columns": feature_columns,
        "feature_ablation": feature_ablation,
        "future_horizon_seconds": float(future_horizon_seconds),
        "builder_version": TRAINING_DATA_BUILDER_VERSION,
    }
    schema_fingerprint = hashlib.sha256(
        json.dumps(
            schema_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    content_payload = {
        "master_files": [
            {
                "sequence_id": item["sequence_id"],
                "sha256": item["sha256"],
            }
            for item in master_files
        ],
        "manifest_sha256": manifest["sha256"] if manifest else None,
        "schema_fingerprint": schema_fingerprint,
    }
    content_fingerprint = hashlib.sha256(
        json.dumps(
            content_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        {
            "builder_version": TRAINING_DATA_BUILDER_VERSION,
            "dataset_content_fingerprint": content_fingerprint,
            "schema": {
                **schema_payload,
                "fingerprint": schema_fingerprint,
            },
            "master_files": master_files,
            "manifest": manifest,
            "builder_file_sha256": builder_files,
            "git": git_provenance(),
            "runtime": runtime_provenance(),
        },
        manifest_snapshot,
    )


def checkpoint_provenance(bundle: "DataBundle") -> dict:
    schema = bundle.provenance.get("schema", {})
    git = bundle.provenance.get("git", {})
    return {
        "dataset_content_fingerprint": bundle.provenance.get(
            "dataset_content_fingerprint"
        ),
        "schema_fingerprint": schema.get("fingerprint"),
        "builder_version": bundle.provenance.get("builder_version"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }


def canonical_participant(value: str) -> str:
    """Return a stable display/grouping name independent of input casing."""
    participant = str(value).strip()
    if not participant:
        raise ValueError("Participant name must not be empty")
    return participant.casefold().capitalize()


def truthy(value: object) -> bool:
    return str(value).strip().casefold() == "true"


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
            "robot_frame_valid",
            "robot_anchor_interpolated",
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


def normalize_excluded_modalities(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError(
            "data.ablation_exclude_modalities must be a list containing "
            "gaze, hands, objects, or vio"
        )
    normalized: list[str] = []
    for value in values:
        modality = str(value).strip().casefold()
        if modality not in SUPPORTED_MODALITY_ABLATIONS:
            raise ValueError(
                f"Unknown ablation modality {value!r}; expected one of "
                f"{', '.join(SUPPORTED_MODALITY_ABLATIONS)}"
            )
        if modality not in normalized:
            normalized.append(modality)
    return normalized


def feature_modalities(column: str) -> set[str]:
    modalities: set[str] = set()
    if column.startswith("gaze_"):
        modalities.add("gaze")
    if column.startswith(("hand_", "left_wrist_", "right_wrist_")):
        modalities.add("hands")
    if column.startswith("aruco_"):
        modalities.add("objects")
        if "_gaze_" in column:
            modalities.add("gaze")
    if column.startswith("slam_") or column in {
        "robot_frame_valid",
        "robot_anchor_interpolated",
    }:
        modalities.add("vio")
    return modalities


def select_feature_columns(
    columns: Iterable[str],
    profile: str,
    excluded_modalities: object = None,
) -> list[str]:
    if profile != "multimodal_robot_frame_v1":
        raise ValueError(f"Unknown feature profile: {profile}")
    excluded = set(normalize_excluded_modalities(excluded_modalities))
    available = set(columns)
    selected = [
        column
        for column in _candidate_features()
        if column in available and not (feature_modalities(column) & excluded)
    ]
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
    receiving_hand_ids: np.ndarray | None = None
    hand_poses: np.ndarray | None = None
    hand_pose_valid: np.ndarray | None = None


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


def fit_normalizer(
    records: list[SequenceRecord], feature_names: list[str]
) -> Normalizer:
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
    include_hand_references: bool = False,
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
    if include_hand_references:
        required_core.extend(
            [
                "receiving_hand",
                "robot_frame_valid",
                "hand_left_valid",
                "hand_right_valid",
                *(
                    f"{side}_wrist_robot_{axis}_m"
                    for side in ("left", "right")
                    for axis in "xyz"
                ),
                *(
                    f"{side}_wrist_robot_q{component}"
                    for side in ("left", "right")
                    for component in "xyzw"
                ),
            ]
        )
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_core) - set(header))
    if missing:
        raise ValueError(
            f"{path.name} is not compatible with the training schema; missing: "
            f"{', '.join(missing)}. Rebuild master datasets after semantic annotation."
        )
    available_features = [column for column in feature_columns if column in header]
    use_columns = list(dict.fromkeys([*required_core, *available_features]))
    frame = pd.read_csv(path, usecols=use_columns)
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

    poses = (
        frame[pose_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    )
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

    receiving_hand_ids = None
    hand_poses = None
    hand_pose_valid = None
    if include_hand_references:
        receiving_hand_ids = (
            frame["receiving_hand"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(RECEIVING_HAND_TO_ID)
            .fillna(-1)
            .to_numpy(np.int64)
        )
        hand_poses = np.zeros((len(frame), 2, 7), dtype=np.float32)
        hand_poses[:, :, 6] = 1.0
        hand_pose_valid = np.zeros((len(frame), 2), dtype=bool)
        robot_valid = (
            pd.to_numeric(frame["robot_frame_valid"], errors="coerce")
            .fillna(0)
            .to_numpy()
            > 0
        )
        for side_id, side in enumerate(RECEIVING_HAND_NAMES):
            current_columns = [
                *(f"{side}_wrist_robot_{axis}_m" for axis in "xyz"),
                *(f"{side}_wrist_robot_q{component}" for component in "xyzw"),
            ]
            current = (
                frame[current_columns]
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
            current_quaternion_norm = np.linalg.norm(current[:, 3:7], axis=1)
            current_valid &= current_quaternion_norm > 1e-6
            current_valid_indices = np.flatnonzero(current_valid)
            if len(current_valid_indices):
                current[current_valid_indices, 3:7] /= current_quaternion_norm[
                    current_valid_indices, None
                ]
                hand_poses[current_valid_indices, side_id] = current[
                    current_valid_indices
                ]
            hand_pose_valid[:, side_id] = current_valid

    features = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
    )
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(np.int64)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Timestamps are not sorted in {path.name}")
    return SequenceRecord(
        sequence_id=sequence_values[0],
        participant=canonical_participant(participant_values[0]),
        timestamps_ns=timestamps,
        features=features,
        intentions=labels.to_numpy(np.int64),
        pose_targets=poses,
        pose_valid=pose_valid,
        receiving_hand_ids=receiving_hand_ids,
        hand_poses=hand_poses,
        hand_pose_valid=hand_pose_valid,
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
    records = [
        replace(record, participant=canonical_participant(record.participant))
        for record in records
    ]
    participants = sorted({record.participant for record in records})
    if len(participants) < 3:
        raise ValueError(
            "At least three participants are required for leakage-safe train/val/test splits"
        )

    explicit = bool(validation_participants or test_participants)
    if explicit:
        validation = {canonical_participant(value) for value in validation_participants}
        test = {canonical_participant(value) for value in test_participants}
        unknown = (validation | test) - set(participants)
        if unknown:
            raise ValueError(
                f"Configured split participants not found: {sorted(unknown)}"
            )
        if validation & test:
            raise ValueError("Validation and test participant sets overlap")
        if not validation or not test:
            raise ValueError(
                "Explicit splits require both validation and test participants"
            )
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
                raise ValueError(
                    "Not enough participants for requested split fractions"
                )
        validation = set(shuffled[:validation_count])
        test = set(shuffled[validation_count : validation_count + test_count])

    train = set(participants) - validation - test
    split = {
        "train": [record for record in records if record.participant in train],
        "validation": [
            record for record in records if record.participant in validation
        ],
        "test": [record for record in records if record.participant in test],
    }
    if any(not values for values in split.values()):
        raise ValueError("One of train/validation/test contains no sequences")
    metadata = {
        "strategy": "explicit_participants"
        if explicit
        else "seeded_participant_group_split",
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
        include_hand_references: bool = False,
    ) -> None:
        self.records = records
        self.window_size = window_size
        self.indices: list[tuple[int, int]] = []
        self.discarded_gap_windows = 0
        self.discarded_observation_windows = 0
        self.discarded_unlabeled_windows = 0
        self.include_hand_references = include_hand_references
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
                    record.features[start : endpoint + 1, raw_feature_count:].mean()
                )
                if observed_fraction < minimum_observed_fraction:
                    self.discarded_observation_windows += 1
                    continue
                self.indices.append((record_index, endpoint))
            record.pose_valid &= np.isin(record.intentions, list(pose_intents))
        if not self.indices:
            raise ValueError(
                "No valid windows were created; inspect window size and data coverage"
            )

        self.hand_reference_poses: list[np.ndarray] = []
        self.hand_reference_valid: list[np.ndarray] = []
        self.hand_reference_age_seconds: list[np.ndarray] = []
        if include_hand_references:
            for record_index, endpoint in self.indices:
                record = records[record_index]
                if (
                    record.receiving_hand_ids is None
                    or record.hand_poses is None
                    or record.hand_pose_valid is None
                ):
                    raise ValueError(
                        f"Hand-reference data was not loaded for {record.sequence_id}"
                    )
                start = endpoint - window_size + 1
                references = np.zeros((2, 7), dtype=np.float32)
                references[:, 6] = 1.0
                validity = np.zeros(2, dtype=bool)
                ages = np.full(2, np.nan, dtype=np.float32)
                for side_id in range(2):
                    valid_rows = np.flatnonzero(
                        record.hand_pose_valid[start : endpoint + 1, side_id]
                    )
                    if len(valid_rows):
                        reference_row = start + int(valid_rows[-1])
                        references[side_id] = record.hand_poses[reference_row, side_id]
                        validity[side_id] = True
                        ages[side_id] = (
                            int(record.timestamps_ns[endpoint])
                            - int(record.timestamps_ns[reference_row])
                        ) / 1e9
                self.hand_reference_poses.append(references)
                self.hand_reference_valid.append(validity)
                self.hand_reference_age_seconds.append(ages)

        self.handover_progress = np.full(len(self.indices), -1.0, dtype=np.float32)
        handover_indices: dict[int, list[tuple[int, int]]] = {}
        for dataset_index, (record_index, endpoint) in enumerate(self.indices):
            if (
                int(records[record_index].intentions[endpoint])
                == INTENTION_TO_ID["handover"]
            ):
                handover_indices.setdefault(record_index, []).append(
                    (dataset_index, endpoint)
                )
        for values in handover_indices.values():
            values.sort(key=lambda item: item[1])
            for progress_index, (dataset_index, _) in enumerate(values):
                self.handover_progress[dataset_index] = progress_index / max(
                    1, len(values) - 1
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        record_index, endpoint = self.indices[index]
        record = self.records[record_index]
        start = endpoint - self.window_size + 1
        item = {
            "features": torch.from_numpy(record.features[start : endpoint + 1]),
            "intention": torch.tensor(record.intentions[endpoint], dtype=torch.long),
            "pose_target": torch.from_numpy(record.pose_targets[endpoint]),
            "pose_valid": torch.tensor(record.pose_valid[endpoint], dtype=torch.bool),
            "sequence_id": record.sequence_id,
            "participant": record.participant,
            "timestamp_ns": torch.tensor(
                record.timestamps_ns[endpoint], dtype=torch.long
            ),
            "handover_progress": torch.tensor(
                self.handover_progress[index], dtype=torch.float32
            ),
        }
        if self.include_hand_references:
            assert record.receiving_hand_ids is not None
            receiving_hand = int(record.receiving_hand_ids[endpoint])
            reference_valid = self.hand_reference_valid[index]
            residual_pose_valid = (
                bool(record.pose_valid[endpoint])
                and receiving_hand in (0, 1)
                and bool(reference_valid[receiving_hand])
            )
            item.update(
                {
                    "receiving_hand": torch.tensor(receiving_hand, dtype=torch.long),
                    "hand_reference_pose": torch.from_numpy(
                        self.hand_reference_poses[index]
                    ),
                    "hand_reference_valid": torch.from_numpy(reference_valid),
                    "hand_reference_age_seconds": torch.from_numpy(
                        self.hand_reference_age_seconds[index]
                    ),
                    "residual_pose_valid": torch.tensor(
                        residual_pose_valid, dtype=torch.bool
                    ),
                }
            )
        return item

    def intention_counts(self) -> list[int]:
        counts = np.zeros(len(INTENTION_NAMES), dtype=np.int64)
        for record_index, endpoint in self.indices:
            counts[self.records[record_index].intentions[endpoint]] += 1
        return counts.tolist()

    def receiving_hand_counts(self) -> list[int]:
        counts = np.zeros(len(RECEIVING_HAND_NAMES), dtype=np.int64)
        if not self.include_hand_references:
            return counts.tolist()
        for record_index, endpoint in self.indices:
            record = self.records[record_index]
            assert record.receiving_hand_ids is not None
            hand_id = int(record.receiving_hand_ids[endpoint])
            if int(record.intentions[endpoint]) == INTENTION_TO_ID[
                "handover"
            ] and hand_id in (0, 1):
                counts[hand_id] += 1
        return counts.tolist()

    def residual_pose_count(self) -> int:
        if not self.include_hand_references:
            return 0
        return sum(
            bool(self[index]["residual_pose_valid"]) for index in range(len(self))
        )


@dataclass
class DataBundle:
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    normalizer: Normalizer
    feature_columns: list[str]
    split_metadata: dict
    provenance: dict
    manifest_snapshot: str | None


def sequence_id_from_master_path(path: Path) -> str:
    suffix = "_master.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a master CSV path: {path}")
    return path.name[: -len(suffix)]


def manifest_filtered_master_files(
    files: list[Path],
    master_dir: Path,
    filter_config: dict | None,
) -> tuple[list[Path], dict]:
    file_by_sequence = {sequence_id_from_master_path(path): path for path in files}
    if len(file_by_sequence) != len(files):
        raise ValueError("Duplicate master dataset sequence IDs")
    if not filter_config:
        selected_ids = sorted(file_by_sequence)
        return files, {
            "enabled": False,
            "selected_sequences": len(selected_ids),
            "sequence_fingerprint": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        }

    manifest_value = filter_config.get("path", "dataset_manifest.csv")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = master_dir.parent / manifest_path
    manifest_path = manifest_path.resolve()
    allowed_statuses = {
        str(value).strip() for value in filter_config.get("allowed_statuses", ["valid"])
    }
    allowed_actions = {
        str(value).strip()
        for value in filter_config.get(
            "allowed_next_actions", ["ready_for_master_merge"]
        )
    }
    strict = bool(filter_config.get("strict", True))

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "sequence_id",
        "include_in_training",
        "status",
        "next_action",
        "master_csv_exists",
    }
    available_columns = set(rows[0]) if rows else set()
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {', '.join(missing_columns)}"
        )

    manifest_ids: set[str] = set()
    eligible_ids: set[str] = set()
    rejected_reasons: Counter[str] = Counter()
    for row in rows:
        sequence_id = row["sequence_id"].strip()
        if not sequence_id:
            raise ValueError(f"Manifest {manifest_path} contains an empty sequence_id")
        if sequence_id in manifest_ids:
            raise ValueError(f"Manifest contains duplicate sequence_id: {sequence_id}")
        manifest_ids.add(sequence_id)

        reasons = []
        if not truthy(row["include_in_training"]):
            reasons.append("excluded_from_training")
        if row["status"].strip() not in allowed_statuses:
            reasons.append(f"status:{row['status'].strip() or 'empty'}")
        if row["next_action"].strip() not in allowed_actions:
            reasons.append(f"next_action:{row['next_action'].strip() or 'empty'}")
        if not truthy(row["master_csv_exists"]):
            reasons.append("manifest_master_csv_missing")
        if reasons:
            rejected_reasons.update(reasons)
        else:
            eligible_ids.add(sequence_id)

    unlisted_master_ids = sorted(set(file_by_sequence) - manifest_ids)
    missing_master_ids = sorted(eligible_ids - set(file_by_sequence))
    if strict and unlisted_master_ids:
        preview = ", ".join(unlisted_master_ids[:10])
        raise ValueError(
            f"{len(unlisted_master_ids)} master CSVs are absent from {manifest_path}: "
            f"{preview}"
        )
    if strict and missing_master_ids:
        preview = ", ".join(missing_master_ids[:10])
        raise ValueError(
            f"{len(missing_master_ids)} eligible manifest rows have no master CSV: "
            f"{preview}"
        )

    selected_ids = sorted(eligible_ids & set(file_by_sequence))
    if not selected_ids:
        raise ValueError(
            f"Manifest filter selected no master datasets from {manifest_path}"
        )
    selected_files = [file_by_sequence[sequence_id] for sequence_id in selected_ids]
    fingerprint = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    return selected_files, {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "allowed_statuses": sorted(allowed_statuses),
        "allowed_next_actions": sorted(allowed_actions),
        "strict": strict,
        "manifest_rows": len(rows),
        "master_files_found": len(files),
        "selected_sequences": len(selected_ids),
        "excluded_master_files": len(files) - len(selected_files),
        "unlisted_master_sequence_ids": unlisted_master_ids,
        "eligible_without_master_sequence_ids": missing_master_ids,
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "sequence_ids": selected_ids,
        "sequence_fingerprint": fingerprint,
    }


def prepare_data(
    data_config: dict, seed: int, limit_sequences: int | None = None
) -> DataBundle:
    master_dir = Path(data_config["master_dir"]).expanduser().resolve()
    files = sorted(master_dir.glob("*_master.csv"))
    files, filter_metadata = manifest_filtered_master_files(
        files,
        master_dir,
        data_config.get("manifest_filter"),
    )
    if limit_sequences is not None:
        files = files[: max(0, limit_sequences)]
        selected_ids = [sequence_id_from_master_path(path) for path in files]
        filter_metadata = {
            **filter_metadata,
            "limit_sequences": int(limit_sequences),
            "selected_sequences": len(selected_ids),
            "sequence_ids": selected_ids,
            "sequence_fingerprint": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        }
    if not files:
        raise FileNotFoundError(f"No *_master.csv files found in {master_dir}")

    first_header = pd.read_csv(files[0], nrows=0).columns.tolist()
    excluded_modalities = normalize_excluded_modalities(
        data_config.get("ablation_exclude_modalities")
    )
    full_feature_columns = select_feature_columns(
        first_header, data_config["feature_profile"]
    )
    feature_columns = select_feature_columns(
        first_header,
        data_config["feature_profile"],
        excluded_modalities,
    )
    feature_ablation = {
        "excluded_modalities": excluded_modalities,
        "excluded_feature_columns": [
            column for column in full_feature_columns if column not in feature_columns
        ],
        "full_raw_feature_count": len(full_feature_columns),
        "retained_raw_feature_count": len(feature_columns),
        "retained_model_feature_count_with_masks": len(feature_columns) * 2,
    }
    provenance, manifest_snapshot = build_dataset_provenance(
        files,
        master_dir=master_dir,
        feature_profile=str(data_config["feature_profile"]),
        feature_columns=feature_columns,
        feature_ablation=feature_ablation,
        future_horizon_seconds=float(
            data_config["future_horizon_seconds"]
        ),
        filter_metadata=filter_metadata,
    )
    filter_metadata = {
        **filter_metadata,
        "dataset_content_fingerprint": provenance[
            "dataset_content_fingerprint"
        ],
    }
    include_hand_references = bool(data_config.get("include_hand_references", False))
    records = [
        load_record(
            path,
            feature_columns,
            future_horizon_seconds=float(data_config["future_horizon_seconds"]),
            include_hand_references=include_hand_references,
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
    split_metadata["feature_ablation"] = feature_ablation
    split_metadata["dataset_filter"] = filter_metadata
    normalizer = fit_normalizer(split["train"], feature_columns)
    normalized_split = {
        name: [
            replace(record, features=normalizer.transform(record.features))
            for record in values
        ]
        for name, values in split.items()
    }

    def make_dataset(records_for_split: list[SequenceRecord]) -> WindowDataset:
        return WindowDataset(
            records_for_split,
            window_size=int(data_config["window_size"]),
            stride=int(data_config["stride"]),
            pose_intent_ids=[int(value) for value in data_config["pose_intent_ids"]],
            minimum_observed_fraction=float(data_config["minimum_observed_fraction"]),
            max_timestamp_gap_seconds=float(data_config["max_timestamp_gap_seconds"]),
            include_hand_references=include_hand_references,
        )

    return DataBundle(
        train=make_dataset(normalized_split["train"]),
        validation=make_dataset(normalized_split["validation"]),
        test=make_dataset(normalized_split["test"]),
        normalizer=normalizer,
        feature_columns=feature_columns,
        split_metadata=split_metadata,
        provenance=provenance,
        manifest_snapshot=manifest_snapshot,
    )


def save_data_metadata(bundle: DataBundle, path: Path) -> None:
    data = {
        "label_mapping": INTENTION_TO_ID,
        "transition_policy": "context_only_never_window_target",
        "feature_ablation": bundle.split_metadata.get("feature_ablation", {}),
        "feature_columns": bundle.feature_columns,
        "model_feature_columns": bundle.normalizer.output_feature_names,
        "normalizer": bundle.normalizer.to_dict(),
        "split": bundle.split_metadata,
        "provenance": bundle.provenance,
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
        "receiving_hand_counts": {
            name: dict(zip(RECEIVING_HAND_NAMES, dataset.receiving_hand_counts()))
            for name, dataset in (
                ("train", bundle.train),
                ("validation", bundle.validation),
                ("test", bundle.test),
            )
        },
        "residual_pose_counts": {
            name: dataset.residual_pose_count()
            for name, dataset in (
                ("train", bundle.train),
                ("validation", bundle.validation),
                ("test", bundle.test),
            )
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (path.parent / "dataset_provenance.json").write_text(
        json.dumps(bundle.provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if bundle.manifest_snapshot is not None:
        (path.parent / "dataset_manifest_snapshot.csv").write_text(
            bundle.manifest_snapshot,
            encoding="utf-8",
        )
