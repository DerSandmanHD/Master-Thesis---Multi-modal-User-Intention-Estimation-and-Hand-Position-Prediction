from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "Training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data import (  # noqa: E402
    DataBundle,
    Normalizer,
    _candidate_features,
    build_dataset_provenance,
    feature_modalities,
    save_data_metadata,
    select_feature_columns,
)
from modality_schema import (  # noqa: E402
    MODALITY_NAMES,
    MODALITY_SCHEMA_VERSION,
    feature_dependencies,
    fusion_modality,
    resolve_modality_schema,
)


def _assert_disjoint_complete(schema: dict) -> None:
    indices = [
        index
        for group in schema["groups"].values()
        for index in group["input_indices"]
    ]
    assert len(indices) == len(set(indices))
    assert sorted(indices) == list(range(schema["model_feature_count"]))


def test_exact_sensor_groups_and_mask_dimensions() -> None:
    raw = _candidate_features()
    schema = resolve_modality_schema(raw)

    assert len(raw) == 92
    assert schema["version"] == MODALITY_SCHEMA_VERSION
    assert schema["modality_names"] == list(MODALITY_NAMES)
    assert schema["active_modalities"] == ["gaze", "hands", "objects", "vio"]
    assert schema["raw_feature_count"] == 92
    assert schema["model_feature_count"] == 184
    assert {
        name: group["raw_feature_count"]
        for name, group in schema["groups"].items()
    } == {"gaze": 10, "hands": 18, "objects": 54, "vio": 10}
    assert {
        name: len(group["availability_indices"])
        for name, group in schema["groups"].items()
    } == {"gaze": 9, "hands": 14, "objects": 45, "vio": 7}
    assert all(
        name.endswith("__observed")
        for group in schema["groups"].values()
        for name in group["availability_mask_feature_names"]
    )
    assert "gaze_valid" not in schema["groups"]["gaze"][
        "availability_feature_names"
    ]
    assert "hand_left_tracking_confidence" not in schema["groups"]["hands"][
        "availability_feature_names"
    ]
    assert "aruco_6_valid" not in schema["groups"]["objects"][
        "availability_feature_names"
    ]
    assert "robot_frame_valid" not in schema["groups"]["vio"][
        "availability_feature_names"
    ]
    _assert_disjoint_complete(schema)


@pytest.mark.parametrize(
    ("prefix", "visual_dimensions"),
    (("clip_pca", 32), ("random_visual", 7)),
)
def test_visual_group_is_dynamic_and_dimension_safe(
    prefix: str, visual_dimensions: int
) -> None:
    raw = [
        *_candidate_features(),
        *(f"{prefix}_{index:03d}" for index in range(visual_dimensions)),
    ]
    schema = resolve_modality_schema(raw)

    assert schema["active_modalities"] == list(MODALITY_NAMES)
    assert schema["raw_feature_count"] == 92 + visual_dimensions
    assert schema["model_feature_count"] == 2 * (92 + visual_dimensions)
    visual = schema["groups"]["clip"]
    assert visual["raw_feature_count"] == visual_dimensions
    assert visual["model_feature_count"] == visual_dimensions * 2
    assert len(visual["availability_indices"]) == visual_dimensions
    assert max(visual["input_indices"]) < schema["model_feature_count"]
    _assert_disjoint_complete(schema)


def test_clip_only_schema_has_one_complete_group() -> None:
    raw = [f"clip_pca_{index:03d}" for index in range(32)]
    schema = resolve_modality_schema(raw)

    assert schema["active_modalities"] == ["clip"]
    assert schema["raw_feature_count"] == 32
    assert schema["model_feature_count"] == 64
    assert schema["groups"]["clip"]["input_indices"] == list(range(64))
    _assert_disjoint_complete(schema)


def test_dependencies_are_distinct_from_unique_fusion_ownership() -> None:
    relation = "aruco_6_gaze_angle_rad"
    assert feature_dependencies(relation) == {"gaze", "objects"}
    assert feature_dependencies(f"{relation}__observed") == {"gaze", "objects"}
    assert fusion_modality(relation) == "objects"
    assert feature_dependencies("aruco_6_robot_x_m") == {"objects"}
    assert feature_dependencies("apriltag_0_valid") == {"vio"}
    assert fusion_modality("clip_pca_000") == "clip"
    assert fusion_modality("random_visual_000") == "clip"


def test_data_ablation_delegates_to_dependencies_and_removes_apriltag() -> None:
    raw = _candidate_features()
    assert feature_modalities("aruco_10_gaze_distance_m") == {
        "gaze",
        "objects",
    }
    assert feature_modalities("apriltag_0_valid") == {"vio"}

    no_vio = select_feature_columns(
        raw, "multimodal_robot_frame_v1", excluded_modalities=["vio"]
    )
    assert len(no_vio) == 82
    assert "apriltag_0_valid" not in no_vio
    assert all("vio" not in feature_dependencies(name) for name in no_vio)

    no_gaze = select_feature_columns(
        raw, "multimodal_robot_frame_v1", excluded_modalities=["gaze"]
    )
    assert len(no_gaze) == 64
    assert not any("_gaze_" in name for name in no_gaze)


@pytest.mark.parametrize(
    "column",
    (
        "intent_label",
        "receiving_hand",
        "target_object_id",
        "future_1s_receiving_wrist_robot_x_m",
        "future_target_timestamp_ns",
        "participant",
        "gaze_label",
        "clip_label",
        "aruco_15_valid",
    ),
)
def test_unknown_label_metadata_and_target_columns_fail_closed(column: str) -> None:
    with pytest.raises(ValueError, match="refuse to assign"):
        feature_dependencies(column)
    with pytest.raises(ValueError, match="refuse to assign"):
        resolve_modality_schema([column])


def test_resolver_requires_one_matching_mask_per_raw_feature() -> None:
    raw = ["gaze_valid", "gaze_yaw_rad"]
    complete = [
        "gaze_valid",
        "gaze_yaw_rad",
        "gaze_valid__observed",
        "gaze_yaw_rad__observed",
    ]
    schema = resolve_modality_schema(raw, reversed(complete))
    assert sorted(schema["groups"]["gaze"]["input_indices"]) == list(range(4))

    with pytest.raises(ValueError, match="matching __observed"):
        resolve_modality_schema(raw, complete[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_modality_schema(raw, [*complete, complete[-1]])
    with pytest.raises(ValueError, match="must not contain observation-mask"):
        resolve_modality_schema(["gaze_valid__observed"])


def test_schema_fingerprint_is_deterministic_and_order_sensitive() -> None:
    raw = _candidate_features()
    first = resolve_modality_schema(raw)
    second = resolve_modality_schema(list(raw))
    reordered = resolve_modality_schema(list(reversed(raw)))

    assert len(first["fingerprint"]) == 64
    assert first["fingerprint"] == second["fingerprint"]
    assert first["fingerprint"] != reordered["fingerprint"]


def test_provenance_and_data_metadata_store_the_same_schema(tmp_path: Path) -> None:
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    master_path = master_dir / "P1_0_master.csv"
    master_path.write_text("placeholder\n", encoding="utf-8")
    raw = _candidate_features()
    provenance, manifest_snapshot = build_dataset_provenance(
        [master_path],
        master_dir=master_dir,
        feature_profile="multimodal_robot_frame_v1",
        feature_columns=raw,
        feature_ablation={},
        future_horizon_seconds=1.0,
        filter_metadata={},
    )
    schema = provenance["schema"]["modality_schema"]
    assert schema == resolve_modality_schema(raw)
    builder_paths = {
        Path(value).as_posix() for value in provenance["builder_file_sha256"]
    }
    assert "Training/modality_schema.py" in builder_paths

    class EmptyDataset:
        discarded_gap_windows = 0
        discarded_observation_windows = 0
        discarded_unlabeled_windows = 0

        def __len__(self) -> int:
            return 0

        @staticmethod
        def intention_counts() -> list[int]:
            return [0, 0, 0]

        @staticmethod
        def receiving_hand_counts() -> list[int]:
            return [0, 0]

        @staticmethod
        def residual_pose_count() -> int:
            return 0

        @staticmethod
        def pose_target_sequence_audit() -> dict:
            return {}

    empty = EmptyDataset()
    bundle = DataBundle(
        train=empty,  # type: ignore[arg-type]
        validation=empty,  # type: ignore[arg-type]
        test=empty,  # type: ignore[arg-type]
        normalizer=Normalizer(
            mean=np.zeros(len(raw), dtype=np.float32),
            std=np.ones(len(raw), dtype=np.float32),
            feature_names=raw,
        ),
        feature_columns=raw,
        split_metadata={"modality_schema": schema},
        provenance=provenance,
        manifest_snapshot=manifest_snapshot,
    )
    metadata_path = tmp_path / "run" / "data_metadata.json"
    save_data_metadata(bundle, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["modality_schema"] == schema
    assert metadata["split"]["modality_schema"] == schema
    assert metadata["provenance"]["schema"]["modality_schema"] == schema
