#!/usr/bin/env python3
"""Central semantic modality schema for training input features.

Feature dependencies and fusion ownership intentionally answer different
questions.  A derived object--gaze feature depends on both source modalities
for a strict ablation, but it must belong to exactly one group when a model
learns modality-wise fusion weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable


MODALITY_SCHEMA_VERSION = "semantic_modality_schema_v1"
MODALITY_NAMES = ("gaze", "hands", "objects", "vio", "clip")
OBSERVED_SUFFIX = "__observed"

GAZE_FEATURES = (
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
)
HAND_FEATURES = (
    "hand_left_tracking_confidence",
    "hand_right_tracking_confidence",
    "hand_left_valid",
    "hand_right_valid",
    *(f"{side}_wrist_robot_{axis}_m" for side in ("left", "right") for axis in "xyz"),
    *(
        f"{side}_wrist_robot_q{component}"
        for side in ("left", "right")
        for component in "xyzw"
    ),
)
VIO_FEATURES = (
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
)
OBJECT_FEATURES = tuple(
    f"aruco_{marker_id}_{suffix}"
    for marker_id in range(6, 15)
    for suffix in (
        "robot_x_m",
        "robot_y_m",
        "robot_z_m",
        "gaze_angle_rad",
        "gaze_distance_m",
        "valid",
    )
)

_GAZE_FEATURE_SET = frozenset(GAZE_FEATURES)
_HAND_FEATURE_SET = frozenset(HAND_FEATURES)
_VIO_FEATURE_SET = frozenset(VIO_FEATURES)
_OBJECT_FEATURE_SET = frozenset(OBJECT_FEATURES)
_VISUAL_FEATURE_PATTERN = re.compile(r"^(?:clip_pca|random_visual)_\d+$")

_GAZE_AVAILABILITY_FEATURES = _GAZE_FEATURE_SET - {"gaze_valid"}
_HAND_AVAILABILITY_FEATURES = frozenset(
    name
    for name in HAND_FEATURES
    if name.startswith(("left_wrist_", "right_wrist_"))
)
_OBJECT_AVAILABILITY_FEATURES = frozenset(
    name for name in OBJECT_FEATURES if not name.endswith("_valid")
)
_VIO_AVAILABILITY_FEATURES = frozenset(
    name for name in VIO_FEATURES if name.startswith("slam_")
)


def _base_feature_name(column: str) -> str:
    if not isinstance(column, str) or not column or column != column.strip():
        raise ValueError(f"Invalid feature name: {column!r}")
    if column.endswith(OBSERVED_SUFFIX):
        return column[: -len(OBSERVED_SUFFIX)]
    return column


def _known_raw_modality(column: str) -> str:
    name = _base_feature_name(column)
    if name in _GAZE_FEATURE_SET:
        return "gaze"
    if name in _HAND_FEATURE_SET:
        return "hands"
    if name in _OBJECT_FEATURE_SET:
        return "objects"
    if name in _VIO_FEATURE_SET:
        return "vio"
    if _VISUAL_FEATURE_PATTERN.fullmatch(name):
        return "clip"
    raise ValueError(
        f"Feature {column!r} is not part of {MODALITY_SCHEMA_VERSION}; "
        "refuse to assign unknown, label, metadata, or target columns"
    )


def feature_dependencies(column: str) -> set[str]:
    """Return source modalities required to construct a feature.

    Dependencies may overlap.  In particular, object--gaze relation features
    are removed by either a strict gaze or a strict object ablation.
    """

    name = _base_feature_name(column)
    modality = _known_raw_modality(name)
    if modality == "objects" and (
        name.endswith("_gaze_angle_rad") or name.endswith("_gaze_distance_m")
    ):
        return {"gaze", "objects"}
    return {modality}


def fusion_modality(column: str) -> str:
    """Return the single semantic owner used for learned modality fusion."""

    return _known_raw_modality(column)


def is_availability_feature(column: str) -> bool:
    """Whether observation of this content channel makes a modality available.

    Validity/status columns are model inputs, but their own presence does not
    imply that the underlying sensor measurement is present.  This prevents a
    finite ``*_valid=0`` flag from activating an otherwise missing modality.
    """

    name = _base_feature_name(column)
    modality = _known_raw_modality(name)
    if modality == "gaze":
        return name in _GAZE_AVAILABILITY_FEATURES
    if modality == "hands":
        return name in _HAND_AVAILABILITY_FEATURES
    if modality == "objects":
        return name in _OBJECT_AVAILABILITY_FEATURES
    if modality == "vio":
        return name in _VIO_AVAILABILITY_FEATURES
    return modality == "clip"


def _feature_list(values: Iterable[str], *, description: str) -> list[str]:
    if isinstance(values, str):
        raise ValueError(f"{description} must be an iterable of feature names")
    result = list(values)
    if not result:
        raise ValueError(f"{description} must not be empty")
    invalid = [value for value in result if not isinstance(value, str)]
    if invalid:
        raise ValueError(f"{description} contains non-string values: {invalid!r}")
    duplicates = sorted(name for name, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"{description} contains duplicate names: {duplicates}")
    return result


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_modality_schema(
    raw_feature_columns: Iterable[str],
    model_feature_columns: Iterable[str] | None = None,
) -> dict:
    """Resolve semantic groups against the exact final model feature order.

    The current normalizer emits one normalized value and one ``__observed``
    channel for every raw feature.  The resolver validates that contract and
    stores indices rather than relying on a hard-coded half-width.  Therefore
    sensor-only, visual-only, sensor-plus-visual, and ablated schemas all use
    the same code path.
    """

    raw_names = _feature_list(
        raw_feature_columns, description="raw_feature_columns"
    )
    observed_raw_names = [
        name for name in raw_names if name.endswith(OBSERVED_SUFFIX)
    ]
    if observed_raw_names:
        raise ValueError(
            "raw_feature_columns must not contain observation-mask channels: "
            f"{observed_raw_names}"
        )
    owners = {name: fusion_modality(name) for name in raw_names}

    expected_model_names = [
        *raw_names,
        *(f"{name}{OBSERVED_SUFFIX}" for name in raw_names),
    ]
    if model_feature_columns is None:
        model_names = expected_model_names
    else:
        model_names = _feature_list(
            model_feature_columns, description="model_feature_columns"
        )
        missing = sorted(set(expected_model_names) - set(model_names))
        unexpected = sorted(set(model_names) - set(expected_model_names))
        if missing or unexpected or len(model_names) != len(expected_model_names):
            raise ValueError(
                "model_feature_columns must contain exactly one normalized value "
                "and matching __observed channel for every raw feature; "
                f"missing={missing}, unexpected={unexpected}"
            )

    model_index = {name: index for index, name in enumerate(model_names)}
    groups: dict[str, dict] = {}
    for modality in MODALITY_NAMES:
        group_raw_names = [name for name in raw_names if owners[name] == modality]
        if not group_raw_names:
            continue
        observed_names = [f"{name}{OBSERVED_SUFFIX}" for name in group_raw_names]
        value_indices = [model_index[name] for name in group_raw_names]
        observed_indices = [model_index[name] for name in observed_names]
        availability_names = [
            name for name in group_raw_names if is_availability_feature(name)
        ]
        availability_mask_names = [
            f"{name}{OBSERVED_SUFFIX}" for name in availability_names
        ]
        groups[modality] = {
            "raw_feature_names": group_raw_names,
            "model_feature_names": [*group_raw_names, *observed_names],
            "value_indices": value_indices,
            "observed_indices": observed_indices,
            "input_indices": [*value_indices, *observed_indices],
            "availability_feature_names": availability_names,
            "availability_mask_feature_names": availability_mask_names,
            "availability_indices": [
                model_index[name] for name in availability_mask_names
            ],
            "raw_feature_count": len(group_raw_names),
            "model_feature_count": len(group_raw_names) * 2,
        }

    grouped_indices = [
        index
        for group in groups.values()
        for index in group["input_indices"]
    ]
    if len(grouped_indices) != len(set(grouped_indices)):
        raise RuntimeError("Resolved modality groups overlap")
    if sorted(grouped_indices) != list(range(len(model_names))):
        raise RuntimeError("Resolved modality groups do not cover the model schema")
    if any(not group["availability_indices"] for group in groups.values()):
        empty = [
            name
            for name, group in groups.items()
            if not group["availability_indices"]
        ]
        raise ValueError(f"Modalities have no content-based availability signals: {empty}")

    payload = {
        "version": MODALITY_SCHEMA_VERSION,
        "modality_names": list(MODALITY_NAMES),
        "active_modalities": list(groups),
        "raw_feature_columns": raw_names,
        "model_feature_columns": model_names,
        "raw_feature_count": len(raw_names),
        "model_feature_count": len(model_names),
        "groups": groups,
    }
    return {**payload, "fingerprint": _fingerprint(payload)}

