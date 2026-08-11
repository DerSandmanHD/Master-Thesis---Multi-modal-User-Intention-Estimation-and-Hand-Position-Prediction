#!/usr/bin/env python3
"""Grouped metrics and cluster-bootstrap intervals for window predictions.

The exported rows are overlapping observation windows.  Window-level metrics
are therefore descriptive, not independent-subject estimates.  This module
also reports metrics within each sequence and participant, equal-weighted
summaries of those groups, and uncertainty intervals obtained by resampling
whole participant or sequence clusters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


INTENTION_NAMES = ("continue", "fetch", "handover")
HAND_NAMES = ("left", "right")
INTENTION_NAME_TO_ID = {name: index for index, name in enumerate(INTENTION_NAMES)}
HAND_NAME_TO_ID = {name: index for index, name in enumerate(HAND_NAMES)}
PRIMARY_POSE_METHODS = (
    "learned_oracle_hand",
    "persistence",
    "constant_velocity",
)

SEQUENCE_AGGREGATION_DEFINITION = (
    "Compute metrics over all windows within each sequence, then summarize the "
    "per-sequence scalar metrics with an unweighted mean/median so every "
    "sequence has equal influence. No single sequence label is created because "
    "a sequence may legitimately contain multiple intention phases."
)
PARTICIPANT_AGGREGATION_DEFINITION = (
    "First compute metrics within every sequence and average sequence metrics "
    "equally inside each participant. Then average participant estimates "
    "equally. Thus neither long sequences nor participants with more windows "
    "receive extra weight."
)


@dataclass(frozen=True)
class PoseMethod:
    """One prediction/baseline represented by precomputed per-window errors."""

    name: str
    source_prefix: str
    position_error_column: str
    orientation_error_column: str | None


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _truth_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes", "y"})
    )


def _categorical_ids(
    values: pd.Series,
    *,
    names: Mapping[str, int],
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.fillna("").astype(str).str.strip().str.casefold().map(names)
    return numeric.where(numeric.notna(), text).astype("Int64")


def _hand_ids(values: pd.Series) -> pd.Series:
    return _categorical_ids(values, names=HAND_NAME_TO_ID)


def _normalized_hand(values: pd.Series) -> pd.Series:
    ids = _hand_ids(values)
    result = pd.Series("", index=values.index, dtype="object")
    for hand_id, name in enumerate(HAND_NAMES):
        result.loc[ids == hand_id] = name
    return result


def discover_pose_methods(frame: pd.DataFrame) -> list[PoseMethod]:
    """Discover learned and baseline methods from ``*_position_error_cm``."""

    canonical_names = {
        "predicted": "learned_end_to_end",
        "learned_end_to_end": "learned_end_to_end",
        "oracle": "learned_oracle_hand",
        "learned_oracle": "learned_oracle_hand",
        "transformer": "learned_model",
        "pose": "learned_model",
        "persistence": "persistence",
        "last_observation": "persistence_legacy",
        "constant_velocity": "constant_velocity",
    }
    priority = {
        prefix: index
        for index, prefix in enumerate(
            (
                "predicted",
                "learned_end_to_end",
                "oracle",
                "learned_oracle",
                "transformer",
                "pose",
                "persistence",
                "last_observation",
                "constant_velocity",
            )
        )
    }
    suffix = "_position_error_cm"
    prefixes = sorted(
        (
            column[: -len(suffix)]
            for column in frame.columns
            if column.endswith(suffix) and len(column) > len(suffix)
        ),
        key=lambda value: (priority.get(value, len(priority)), value),
    )
    methods: list[PoseMethod] = []
    used_names: set[str] = set()
    for prefix in prefixes:
        name = canonical_names.get(prefix, prefix)
        if name in used_names:
            continue
        orientation = f"{prefix}_orientation_error_deg"
        methods.append(
            PoseMethod(
                name=name,
                source_prefix=prefix,
                position_error_column=f"{prefix}{suffix}",
                orientation_error_column=(
                    orientation if orientation in frame.columns else None
                ),
            )
        )
        used_names.add(name)
    return methods


def prepare_prediction_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize current hierarchical/flat export schemas and validate groups."""

    if frame.empty:
        raise ValueError("Prediction CSV contains no rows")
    required_groups = {"participant", "sequence_id"}
    missing_groups = required_groups - set(frame.columns)
    if missing_groups:
        raise ValueError(f"Prediction CSV misses columns: {sorted(missing_groups)}")

    target_column = _first_column(
        frame,
        ("target_intention_id", "ground_truth_intention_id", "target_intention"),
    )
    prediction_column = _first_column(
        frame,
        (
            "predicted_intention_id",
            "prediction_intention_id",
            "predicted_intention",
        ),
    )
    if target_column is None or prediction_column is None:
        raise ValueError("Prediction CSV must contain target and predicted intentions")

    normalized = frame.copy()
    normalized["participant"] = (
        normalized["participant"].fillna("").astype(str).str.strip()
    )
    normalized["sequence_id"] = (
        normalized["sequence_id"].fillna("").astype(str).str.strip()
    )
    if normalized["participant"].eq("").any() or normalized["sequence_id"].eq("").any():
        raise ValueError("participant and sequence_id must be non-empty on every row")
    participant_counts = normalized.groupby("sequence_id")["participant"].nunique()
    inconsistent = participant_counts.loc[participant_counts != 1]
    if len(inconsistent):
        raise ValueError(
            "Each sequence must belong to exactly one participant: "
            + ", ".join(inconsistent.index.astype(str))
        )
    if "sample_key" in normalized and normalized["sample_key"].notna().any():
        keys = normalized["sample_key"].dropna().astype(str)
        duplicates = keys.loc[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"Duplicate sample_key values: {duplicates[:5]}")

    normalized["_target_intention_id"] = _categorical_ids(
        normalized[target_column], names=INTENTION_NAME_TO_ID
    )
    normalized["_predicted_intention_id"] = _categorical_ids(
        normalized[prediction_column], names=INTENTION_NAME_TO_ID
    )

    target_assistance_column = _first_column(
        normalized,
        ("target_assistance_id", "ground_truth_assistance_id"),
    )
    predicted_assistance_column = _first_column(
        normalized,
        ("predicted_assistance_id", "prediction_assistance_id"),
    )
    if target_assistance_column and predicted_assistance_column:
        normalized["_target_assistance_id"] = _categorical_ids(
            normalized[target_assistance_column],
            names={"continue": 0, "no_assistance": 0, "assistance": 1},
        )
        normalized["_predicted_assistance_id"] = _categorical_ids(
            normalized[predicted_assistance_column],
            names={"continue": 0, "no_assistance": 0, "assistance": 1},
        )
        assistance_source = "explicit_exported_head_decisions"
    else:
        normalized["_target_assistance_id"] = (
            normalized["_target_intention_id"] > 0
        ).where(normalized["_target_intention_id"].between(0, 2)).astype("Int64")
        normalized["_predicted_assistance_id"] = (
            normalized["_predicted_intention_id"] > 0
        ).where(normalized["_predicted_intention_id"].between(0, 2)).astype("Int64")
        assistance_source = "derived_from_final_three_class_decision"

    target_type_column = _first_column(
        normalized,
        ("target_assistance_type_id", "ground_truth_assistance_type_id"),
    )
    predicted_type_column = _first_column(
        normalized,
        ("predicted_assistance_type_id", "prediction_assistance_type_id"),
    )
    if target_type_column and predicted_type_column:
        normalized["_target_assistance_type_id"] = _categorical_ids(
            normalized[target_type_column], names={"fetch": 0, "handover": 1}
        )
        normalized["_predicted_assistance_type_id"] = _categorical_ids(
            normalized[predicted_type_column], names={"fetch": 0, "handover": 1}
        )
        assistance_type_source = "explicit_exported_head_decisions"
    else:
        normalized["_target_assistance_type_id"] = (
            normalized["_target_intention_id"] - 1
        ).where(normalized["_target_intention_id"].isin((1, 2))).astype("Int64")
        conditional_probability_columns = next(
            (
                pair
                for pair in (
                    (
                        "fetch_given_assistance_probability",
                        "handover_given_assistance_probability",
                    ),
                    (
                        "fetch_probability_given_assistance",
                        "handover_probability_given_assistance",
                    ),
                )
                if set(pair).issubset(normalized.columns)
            ),
            None,
        )
        if conditional_probability_columns is not None:
            fetch_column, handover_column = conditional_probability_columns
            fetch = pd.to_numeric(
                normalized[fetch_column], errors="coerce"
            )
            handover = pd.to_numeric(
                normalized[handover_column], errors="coerce"
            )
            normalized["_predicted_assistance_type_id"] = (
                (handover > fetch).astype(int).where(fetch.notna() & handover.notna())
            ).astype("Int64")
            assistance_type_source = "derived_from_exported_conditional_probabilities"
        else:
            normalized["_predicted_assistance_type_id"] = (
                normalized["_predicted_intention_id"] - 1
            ).where(normalized["_predicted_intention_id"].isin((1, 2))).astype(
                "Int64"
            )
            assistance_type_source = "derived_from_final_three_class_decision"

    target_hand_column = _first_column(
        normalized,
        (
            "target_receiving_hand",
            "ground_truth_receiving_hand",
            "sequence_receiving_hand",
            "receiving_hand",
        ),
    )
    predicted_hand_column = _first_column(
        normalized,
        ("predicted_receiving_hand", "prediction_receiving_hand"),
    )
    if target_hand_column:
        normalized["_target_hand"] = _normalized_hand(
            normalized[target_hand_column]
        )
    else:
        normalized["_target_hand"] = ""
    if predicted_hand_column:
        normalized["_predicted_hand"] = _normalized_hand(
            normalized[predicted_hand_column]
        )
    else:
        normalized["_predicted_hand"] = ""

    context_hand_column = _first_column(
        normalized, ("sequence_receiving_hand", "receiving_hand")
    )
    if context_hand_column:
        normalized["_receiving_hand_context"] = _normalized_hand(
            normalized[context_hand_column]
        )
    else:
        normalized["_receiving_hand_context"] = normalized["_target_hand"]
    # Receiving hand is normally constant within a recorded sequence. Infer
    # this context for non-handover rows without inventing a value for sequences
    # that contain conflicting hand annotations.
    for _, indices in normalized.groupby("sequence_id", sort=False).groups.items():
        direct_values = set(normalized.loc[indices, "_receiving_hand_context"])
        target_values = set(normalized.loc[indices, "_target_hand"])
        values = sorted((direct_values | target_values) & set(HAND_NAMES))
        if len(values) > 1:
            sequence_id = str(normalized.loc[indices, "sequence_id"].iloc[0])
            raise ValueError(
                "Conflicting receiving-hand annotations within sequence "
                f"{sequence_id}: {values}"
            )
        if len(values) == 1:
            normalized.loc[indices, "_receiving_hand_context"] = values[0]

    if "pose_valid" in normalized:
        normalized["_pose_target_valid"] = _truth_values(normalized["pose_valid"])
    else:
        error_columns = [
            method.position_error_column for method in discover_pose_methods(normalized)
        ]
        if error_columns:
            normalized["_pose_target_valid"] = normalized[error_columns].apply(
                pd.to_numeric, errors="coerce"
            ).notna().any(axis=1)
        else:
            normalized["_pose_target_valid"] = False

    return normalized, {
        "target_intention_column": target_column,
        "predicted_intention_column": prediction_column,
        "target_receiving_hand_column": target_hand_column,
        "predicted_receiving_hand_column": predicted_hand_column,
        "receiving_hand_context_column": context_hand_column,
        "target_assistance_column": target_assistance_column,
        "predicted_assistance_column": predicted_assistance_column,
        "assistance_metric_source": assistance_source,
        "target_assistance_type_column": target_type_column,
        "predicted_assistance_type_column": predicted_type_column,
        "assistance_type_metric_source": assistance_type_source,
        "head_neutral": True,
        "head_neutral_definition": (
            "Evaluation consumes exported three-class target/prediction IDs, so "
            "the same logic applies to hierarchical and flat classifier heads."
        ),
    }


def classification_metrics(
    target: pd.Series,
    prediction: pd.Series,
    *,
    class_names: tuple[str, ...],
    denominator: int | None = None,
) -> dict:
    target_values = pd.to_numeric(target, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    prediction_values = pd.to_numeric(prediction, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    class_count = len(class_names)
    valid = (
        np.isfinite(target_values)
        & np.isfinite(prediction_values)
        & (target_values >= 0)
        & (target_values < class_count)
        & (prediction_values >= 0)
        & (prediction_values < class_count)
    )
    true = target_values[valid].astype(np.int64)
    predicted = prediction_values[valid].astype(np.int64)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    if len(true):
        np.add.at(confusion, (true, predicted), 1)

    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(class_count, dtype=float),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) > 0,
    )
    samples = int(len(true))
    denominator_value = int(len(target) if denominator is None else denominator)
    supported = support > 0
    per_class = [
        {
            "class_id": index,
            "class_name": class_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "predicted": int(predicted_count[index]),
        }
        for index in range(class_count)
    ]
    return {
        "samples": samples,
        "denominator": denominator_value,
        "coverage": (
            float(samples / denominator_value) if denominator_value else None
        ),
        "accuracy": float((true == predicted).mean()) if samples else None,
        "macro_f1": float(f1.mean()) if samples else None,
        "macro_f1_supported": (
            float(f1[supported].mean()) if supported.any() else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def pose_metrics(
    frame: pd.DataFrame,
    method: PoseMethod,
    *,
    coverage_denominator: int,
) -> dict:
    position = pd.to_numeric(
        frame[method.position_error_column], errors="coerce"
    ).to_numpy(dtype=float)
    position = position[np.isfinite(position)]
    orientation = np.empty(0, dtype=float)
    if method.orientation_error_column:
        orientation = pd.to_numeric(
            frame[method.orientation_error_column], errors="coerce"
        ).to_numpy(dtype=float)
        orientation = orientation[np.isfinite(orientation)]
    return {
        "source_prefix": method.source_prefix,
        "position_samples": int(len(position)),
        "orientation_samples": int(len(orientation)),
        "coverage_denominator_pose_targets": int(coverage_denominator),
        "coverage": (
            float(len(position) / coverage_denominator)
            if coverage_denominator
            else None
        ),
        "position_mean_cm": float(position.mean()) if len(position) else None,
        "position_median_cm": float(np.median(position)) if len(position) else None,
        "position_rmse_cm": (
            float(np.sqrt(np.mean(np.square(position)))) if len(position) else None
        ),
        "orientation_mean_deg": (
            float(orientation.mean()) if len(orientation) else None
        ),
        "orientation_median_deg": (
            float(np.median(orientation)) if len(orientation) else None
        ),
    }


def paired_pose_comparisons(
    frame: pd.DataFrame,
    pose_methods: list[PoseMethod],
) -> dict[str, dict]:
    """Compare learned and baseline errors on their exact shared windows."""

    learned = [method for method in pose_methods if method.name.startswith("learned_")]
    baselines = [
        method
        for method in pose_methods
        if method.name in {"persistence", "persistence_legacy", "constant_velocity"}
    ]
    denominator = int(frame["_pose_target_valid"].sum())
    comparisons: dict[str, dict] = {}
    for learned_method in learned:
        for baseline in baselines:
            learned_error = pd.to_numeric(
                frame[learned_method.position_error_column], errors="coerce"
            ).to_numpy(dtype=float)
            baseline_error = pd.to_numeric(
                frame[baseline.position_error_column], errors="coerce"
            ).to_numpy(dtype=float)
            shared = (
                frame["_pose_target_valid"].to_numpy(dtype=bool)
                & np.isfinite(learned_error)
                & np.isfinite(baseline_error)
            )
            difference = learned_error[shared] - baseline_error[shared]
            key = f"{learned_method.name}_minus_{baseline.name}"
            values = {
                "learned_method": learned_method.name,
                "baseline_method": baseline.name,
                "difference_definition": (
                    "learned position error minus baseline position error on the "
                    "same windows; negative values favor the learned method"
                ),
                "shared_samples": int(shared.sum()),
                "coverage_denominator_pose_targets": denominator,
                "coverage": float(shared.sum() / denominator) if denominator else None,
                "position_mean_difference_cm": (
                    float(difference.mean()) if len(difference) else None
                ),
                "position_median_difference_cm": (
                    float(np.median(difference)) if len(difference) else None
                ),
                "learned_win_fraction": (
                    float((difference < 0).mean()) if len(difference) else None
                ),
            }
            if (
                learned_method.orientation_error_column
                and baseline.orientation_error_column
            ):
                learned_orientation = pd.to_numeric(
                    frame[learned_method.orientation_error_column], errors="coerce"
                ).to_numpy(dtype=float)
                baseline_orientation = pd.to_numeric(
                    frame[baseline.orientation_error_column], errors="coerce"
                ).to_numpy(dtype=float)
                orientation_shared = (
                    frame["_pose_target_valid"].to_numpy(dtype=bool)
                    & np.isfinite(learned_orientation)
                    & np.isfinite(baseline_orientation)
                )
                orientation_difference = (
                    learned_orientation[orientation_shared]
                    - baseline_orientation[orientation_shared]
                )
                values.update(
                    {
                        "orientation_shared_samples": int(
                            orientation_shared.sum()
                        ),
                        "orientation_mean_difference_deg": (
                            float(orientation_difference.mean())
                            if len(orientation_difference)
                            else None
                        ),
                        "orientation_median_difference_deg": (
                            float(np.median(orientation_difference))
                            if len(orientation_difference)
                            else None
                        ),
                    }
                )
            comparisons[key] = values
    return comparisons


def summarize_windows(
    frame: pd.DataFrame,
    pose_methods: list[PoseMethod],
) -> dict:
    """Return metrics pooled over the supplied rows (normally windows)."""

    target = frame["_target_intention_id"]
    prediction = frame["_predicted_intention_id"]
    valid_target = target.between(0, 2)
    intention = classification_metrics(
        target,
        prediction,
        class_names=INTENTION_NAMES,
        denominator=len(frame),
    )
    assistance_target = frame["_target_assistance_id"]
    assistance_prediction = frame["_predicted_assistance_id"]
    assistance_valid_target = assistance_target.between(0, 1)
    assistance = classification_metrics(
        assistance_target,
        assistance_prediction,
        class_names=("continue", "assistance"),
        denominator=int(assistance_valid_target.sum()),
    )
    assistance_type_target = frame["_target_assistance_type_id"]
    assistance_type_prediction = frame["_predicted_assistance_type_id"]
    assistance_rows = assistance_type_target.between(0, 1)
    assistance_type = classification_metrics(
        assistance_type_target.loc[assistance_rows],
        assistance_type_prediction.loc[assistance_rows],
        class_names=("fetch", "handover"),
        denominator=int(assistance_rows.sum()),
    )

    handover_rows = valid_target & target.eq(2)
    hand_target = frame.loc[handover_rows, "_target_hand"].map(HAND_NAME_TO_ID)
    hand_prediction = frame.loc[handover_rows, "_predicted_hand"].map(
        HAND_NAME_TO_ID
    )
    receiving_hand = classification_metrics(
        hand_target,
        hand_prediction,
        class_names=HAND_NAMES,
        denominator=int(handover_rows.sum()),
    )

    pose_target_rows = frame["_pose_target_valid"]
    pose_denominator = int(pose_target_rows.sum())
    pose_frame = frame.loc[pose_target_rows]
    pose = {
        method.name: pose_metrics(
            pose_frame,
            method,
            coverage_denominator=pose_denominator,
        )
        for method in pose_methods
    }
    fair_common = None
    methods_by_name = {method.name: method for method in pose_methods}
    missing_primary = [
        name for name in PRIMARY_POSE_METHODS if name not in methods_by_name
    ]
    if not missing_primary:
        common_rows = pose_target_rows.copy()
        for name in PRIMARY_POSE_METHODS:
            method = methods_by_name[name]
            position = pd.to_numeric(
                frame[method.position_error_column], errors="coerce"
            ).to_numpy(dtype=float)
            common_rows &= pd.Series(
                np.isfinite(position), index=frame.index
            )
            if method.orientation_error_column is not None:
                orientation = pd.to_numeric(
                    frame[method.orientation_error_column], errors="coerce"
                ).to_numpy(dtype=float)
                common_rows &= pd.Series(
                    np.isfinite(orientation), index=frame.index
                )
        if "fair_common" in frame:
            declared = pose_target_rows & _truth_values(frame["fair_common"])
            if not declared.equals(common_rows):
                raise ValueError(
                    "fair_common must equal the recomputed intersection of valid "
                    "targets and all three primary pose methods"
                )
        common_frame = frame.loc[common_rows]
        common_keys = (
            common_frame["sample_key"].astype(str).tolist()
            if "sample_key" in common_frame
            else common_frame.index.astype(str).tolist()
        )
        method_key_fingerprints = {}
        for name in PRIMARY_POSE_METHODS:
            method = methods_by_name[name]
            finite = pd.Series(
                np.isfinite(
                    pd.to_numeric(
                        common_frame[method.position_error_column],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                ),
                index=common_frame.index,
            )
            if not finite.all():
                raise ValueError(
                    f"fair_common contains unavailable {name} pose predictions"
                )
            keys = [key for key, valid in zip(common_keys, finite) if valid]
            method_key_fingerprints[name] = hashlib.sha256(
                "\n".join(keys).encode("utf-8")
            ).hexdigest()
        if len(set(method_key_fingerprints.values())) > 1:
            raise ValueError("fair_common methods do not use identical sample keys")
        fair_common = {
            "comparison_role": (
                "Primary paired t+1 comparison with ground-truth receiving-hand "
                "context. learned_end_to_end is intentionally excluded and remains "
                "a predicted-hand diagnostic."
            ),
            "required_methods": list(PRIMARY_POSE_METHODS),
            "shared_samples": int(common_rows.sum()),
            "coverage_denominator_pose_targets": pose_denominator,
            "coverage": (
                float(common_rows.sum() / pose_denominator)
                if pose_denominator
                else None
            ),
            "methods": {
                name: pose_metrics(
                    common_frame,
                    methods_by_name[name],
                    coverage_denominator=pose_denominator,
                )
                for name in PRIMARY_POSE_METHODS
            },
            "method_sample_key_fingerprints": method_key_fingerprints,
        }
    elif "fair_common" in frame and _truth_values(frame["fair_common"]).any():
        raise ValueError(
            "fair_common is declared but primary pose methods are missing: "
            + ", ".join(missing_primary)
        )
    return {
        "windows": int(len(frame)),
        "classification": {
            "intention": intention,
            "assistance": assistance,
            "assistance_type_given_ground_truth_assistance": assistance_type,
            "receiving_hand_given_ground_truth_handover": receiving_hand,
        },
        "pose_target_denominator": pose_denominator,
        "pose": pose,
        "pose_fair_common": fair_common,
        "paired_pose_comparisons": paired_pose_comparisons(frame, pose_methods),
    }


def _finite_group_values(
    groups: Mapping[str, dict],
    path: tuple[str, ...],
) -> np.ndarray:
    values: list[float] = []
    for metrics in groups.values():
        value: object = metrics
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _distribution(values: np.ndarray) -> dict:
    return {
        "groups_with_metric": int(len(values)),
        "unweighted_mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
    }


def equal_weight_group_summary(
    groups: Mapping[str, dict],
    pose_methods: list[PoseMethod],
) -> dict:
    classification = {}
    for task in (
        "intention",
        "assistance",
        "assistance_type_given_ground_truth_assistance",
        "receiving_hand_given_ground_truth_handover",
    ):
        classification[task] = {
            metric: _distribution(
                _finite_group_values(groups, ("classification", task, metric))
            )
            for metric in ("accuracy", "macro_f1", "macro_f1_supported")
        }
    pose = {
        method.name: {
            metric: _distribution(
                _finite_group_values(groups, ("pose", method.name, metric))
            )
            for metric in (
                "position_mean_cm",
                "position_median_cm",
                "orientation_mean_deg",
                "coverage",
            )
        }
        for method in pose_methods
    }
    fair_common = {
        name: {
            metric: _distribution(
                _finite_group_values(
                    groups,
                    ("pose_fair_common", "methods", name, metric),
                )
            )
            for metric in (
                "position_mean_cm",
                "position_median_cm",
                "orientation_mean_deg",
                "coverage",
            )
        }
        for name in PRIMARY_POSE_METHODS
    }
    return {
        "classification": classification,
        "pose": pose,
        "pose_fair_common": fair_common,
    }


def equal_weight_point_estimate(
    groups: Mapping[str, dict],
    pose_methods: list[PoseMethod],
) -> dict:
    """Collapse group metrics without weighting by their number of windows."""

    classification = {}
    for task in (
        "intention",
        "assistance",
        "assistance_type_given_ground_truth_assistance",
        "receiving_hand_given_ground_truth_handover",
    ):
        task_values = {}
        for metric in ("accuracy", "macro_f1", "macro_f1_supported", "coverage"):
            values = _finite_group_values(
                groups, ("classification", task, metric)
            )
            task_values[metric] = float(values.mean()) if len(values) else None
            task_values[f"{metric}_groups"] = int(len(values))
        task_values["samples"] = int(
            sum(
                int(metrics["classification"][task].get("samples", 0))
                for metrics in groups.values()
            )
        )
        task_values["denominator"] = int(
            sum(
                int(metrics["classification"][task].get("denominator", 0))
                for metrics in groups.values()
            )
        )
        classification[task] = task_values
    pose = {}
    for method in pose_methods:
        method_values = {}
        for metric in (
            "position_mean_cm",
            "position_median_cm",
            "position_rmse_cm",
            "orientation_mean_deg",
            "orientation_median_deg",
            "coverage",
        ):
            values = _finite_group_values(groups, ("pose", method.name, metric))
            method_values[metric] = float(values.mean()) if len(values) else None
            method_values[f"{metric}_groups"] = int(len(values))
        method_values["position_samples"] = int(
            sum(
                int(metrics["pose"][method.name].get("position_samples", 0))
                for metrics in groups.values()
            )
        )
        method_values["orientation_samples"] = int(
            sum(
                int(metrics["pose"][method.name].get("orientation_samples", 0))
                for metrics in groups.values()
            )
        )
        method_values["coverage_denominator_pose_targets"] = int(
            sum(
                int(
                    metrics["pose"][method.name].get(
                        "coverage_denominator_pose_targets", 0
                    )
                )
                for metrics in groups.values()
            )
        )
        pose[method.name] = method_values
    fair_common_methods = {}
    for name in PRIMARY_POSE_METHODS:
        method_values = {}
        for metric in (
            "position_mean_cm",
            "position_median_cm",
            "position_rmse_cm",
            "orientation_mean_deg",
            "orientation_median_deg",
            "coverage",
        ):
            values = _finite_group_values(
                groups,
                ("pose_fair_common", "methods", name, metric),
            )
            method_values[metric] = float(values.mean()) if len(values) else None
            method_values[f"{metric}_groups"] = int(len(values))
        method_values["position_samples"] = int(
            sum(
                int(
                    (metrics.get("pose_fair_common") or {})
                    .get("methods", {})
                    .get(name, {})
                    .get("position_samples", 0)
                )
                for metrics in groups.values()
            )
        )
        fair_common_methods[name] = method_values
    fair_common = {
        "required_methods": list(PRIMARY_POSE_METHODS),
        "shared_samples": int(
            sum(
                int((metrics.get("pose_fair_common") or {}).get("shared_samples", 0))
                for metrics in groups.values()
            )
        ),
        "coverage_denominator_pose_targets": int(
            sum(
                int(
                    (metrics.get("pose_fair_common") or {}).get(
                        "coverage_denominator_pose_targets", 0
                    )
                )
                for metrics in groups.values()
            )
        ),
        "methods": fair_common_methods,
    }
    paired = {}
    paired_names = sorted(
        {
            name
            for metrics in groups.values()
            for name in metrics.get("paired_pose_comparisons", {})
        }
    )
    for name in paired_names:
        values = {}
        for metric in (
            "position_mean_difference_cm",
            "position_median_difference_cm",
            "learned_win_fraction",
            "orientation_mean_difference_deg",
            "orientation_median_difference_deg",
            "coverage",
        ):
            samples = _finite_group_values(
                groups, ("paired_pose_comparisons", name, metric)
            )
            values[metric] = float(samples.mean()) if len(samples) else None
            values[f"{metric}_groups"] = int(len(samples))
        values["shared_samples"] = int(
            sum(
                int(
                    metrics.get("paired_pose_comparisons", {})
                    .get(name, {})
                    .get("shared_samples", 0)
                )
                for metrics in groups.values()
            )
        )
        paired[name] = values
    return {
        "windows": int(sum(int(metrics.get("windows", 0)) for metrics in groups.values())),
        "classification": classification,
        "pose": pose,
        "pose_fair_common": fair_common,
        "paired_pose_comparisons": paired,
    }


def summarize_group_level(
    frame: pd.DataFrame,
    *,
    group_column: str,
    pose_methods: list[PoseMethod],
    definition: str,
) -> dict:
    groups = {
        str(name): summarize_windows(group, pose_methods)
        for name, group in frame.groupby(group_column, sort=True)
    }
    return {
        "group_column": group_column,
        "group_count": len(groups),
        "aggregation_definition": definition,
        "groups": groups,
        "point_estimate": equal_weight_point_estimate(groups, pose_methods),
        "equal_weighted_group_summary": equal_weight_group_summary(
            groups, pose_methods
        ),
    }


def summarize_participant_level(
    frame: pd.DataFrame,
    *,
    pose_methods: list[PoseMethod],
) -> dict:
    participants: dict[str, dict] = {}
    participant_metrics: dict[str, dict] = {}
    for participant_name, participant_frame in frame.groupby(
        "participant", sort=True
    ):
        sequence_groups = {
            str(sequence): summarize_windows(sequence_frame, pose_methods)
            for sequence, sequence_frame in participant_frame.groupby(
                "sequence_id", sort=True
            )
        }
        metrics = equal_weight_point_estimate(sequence_groups, pose_methods)
        participant_metrics[str(participant_name)] = metrics
        participants[str(participant_name)] = {
            "windows": int(len(participant_frame)),
            "sequence_count": len(sequence_groups),
            "aggregation_definition": (
                "Unweighted mean of this participant's per-sequence scalar metrics."
            ),
            "metrics": metrics,
            "sequence_metrics": sequence_groups,
            "pooled_window_diagnostic": summarize_windows(
                participant_frame, pose_methods
            ),
        }
    return {
        "group_column": "participant",
        "group_count": len(participants),
        "aggregation_definition": PARTICIPANT_AGGREGATION_DEFINITION,
        "groups": participants,
        "point_estimate": equal_weight_point_estimate(
            participant_metrics, pose_methods
        ),
        "equal_weighted_participant_summary": equal_weight_group_summary(
            participant_metrics, pose_methods
        ),
    }


def _bootstrap_statistic_values(
    frame: pd.DataFrame,
    pose_methods: list[PoseMethod],
    *,
    estimand: str,
) -> dict[str, float]:
    if estimand == "participant_balanced":
        summary = summarize_participant_level(
            frame, pose_methods=pose_methods
        )["point_estimate"]
    elif estimand == "sequence_balanced":
        summary = summarize_group_level(
            frame,
            group_column="sequence_id",
            pose_methods=pose_methods,
            definition=SEQUENCE_AGGREGATION_DEFINITION,
        )["point_estimate"]
    else:
        raise ValueError(f"Unknown bootstrap estimand: {estimand}")
    values: dict[str, float] = {}
    for task in (
        "intention",
        "assistance",
        "assistance_type_given_ground_truth_assistance",
        "receiving_hand_given_ground_truth_handover",
    ):
        metrics = summary["classification"][task]
        for metric in ("accuracy", "macro_f1", "macro_f1_supported", "coverage"):
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                values[f"classification.{task}.{metric}"] = float(value)
    for method in pose_methods:
        metrics = summary["pose"][method.name]
        for metric in (
            "position_mean_cm",
            "position_median_cm",
            "position_rmse_cm",
            "orientation_mean_deg",
            "orientation_median_deg",
            "coverage",
        ):
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                values[f"pose.{method.name}.native.{metric}"] = float(value)
    for name, metrics in summary.get("pose_fair_common", {}).get(
        "methods", {}
    ).items():
        for metric in (
            "position_mean_cm",
            "position_median_cm",
            "position_rmse_cm",
            "orientation_mean_deg",
            "orientation_median_deg",
            "coverage",
        ):
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                values[f"pose.{name}.fair_common.{metric}"] = float(value)
    for comparison, metrics in summary.get("paired_pose_comparisons", {}).items():
        for metric in (
            "position_mean_difference_cm",
            "position_median_difference_cm",
            "orientation_mean_difference_deg",
            "orientation_median_difference_deg",
            "learned_win_fraction",
            "coverage",
        ):
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                values[f"paired_pose.{comparison}.{metric}"] = float(value)
    return values


def cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    pose_methods: list[PoseMethod],
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict:
    """Resample complete clusters and return percentile intervals.

    All rows belonging to a sampled participant/sequence are retained. A
    cluster drawn twice is included twice. These intervals quantify sensitivity
    to the observed cluster sample; they are intentionally distinct from the
    standard deviation across independently trained random seeds.
    """

    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    clusters = {
        str(name): np.asarray(indices, dtype=np.int64)
        for name, indices in frame.groupby(cluster_column, sort=True).indices.items()
    }
    if not clusters:
        raise ValueError(f"No clusters found in {cluster_column}")
    names = tuple(clusters)
    estimand = (
        "participant_balanced"
        if cluster_column == "participant"
        else "sequence_balanced"
    )
    original = _bootstrap_statistic_values(
        frame, pose_methods, estimand=estimand
    )
    contributing_clusters = {key: 0 for key in original}
    for indices in clusters.values():
        cluster_values = _bootstrap_statistic_values(
            frame.iloc[indices].copy(),
            pose_methods,
            estimand=estimand,
        )
        for key in contributing_clusters:
            if key in cluster_values:
                contributing_clusters[key] += 1
    if len(clusters) < 2:
        return {
            "cluster_unit": cluster_column.removeprefix("_"),
            "cluster_count": len(clusters),
            "iterations_requested": int(iterations),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
            "interval_method": "not_estimable",
            "estimand": estimand,
            "status": "insufficient_independent_clusters",
            "resampling_definition": None,
            "small_cluster_warning": True,
            "small_cluster_interpretation": (
                "Fewer than two clusters cannot support a cluster-bootstrap "
                "interval. Point estimates are retained without CI bounds."
            ),
            "interpretation": (
                "No confidence interval is reported; training-seed variability "
                "is a separate quantity."
            ),
            "metrics": {
                key: {
                    "estimate": estimate,
                    "lower": None,
                    "upper": None,
                    "valid_replicates": 0,
                    "metric_contributing_clusters": contributing_clusters[key],
                    "status": "insufficient_metric_contributing_clusters",
                }
                for key, estimate in original.items()
            },
        }
    replicates: dict[str, list[float]] = {key: [] for key in original}
    rng = np.random.default_rng(int(seed))
    for _ in range(iterations):
        selected = rng.choice(names, size=len(names), replace=True)
        pieces = []
        for draw_index, name in enumerate(selected):
            piece = frame.iloc[clusters[str(name)]].copy()
            if cluster_column == "participant":
                piece["participant"] = f"draw_{draw_index}"
                piece["sequence_id"] = (
                    f"draw_{draw_index}|" + piece["sequence_id"].astype(str)
                )
            else:
                piece["sequence_id"] = f"draw_{draw_index}"
            pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=True)
        values = _bootstrap_statistic_values(
            sampled, pose_methods, estimand=estimand
        )
        for key in replicates:
            value = values.get(key)
            if value is not None and np.isfinite(value):
                replicates[key].append(float(value))

    alpha = (1.0 - confidence_level) / 2.0
    intervals = {}
    for key, estimate in original.items():
        values = np.asarray(replicates[key], dtype=float)
        enough_clusters = contributing_clusters[key] >= 2
        enough_replicates = len(values) >= max(2, int(np.ceil(iterations * 0.8)))
        estimable = enough_clusters and enough_replicates
        intervals[key] = {
            "estimate": estimate,
            "lower": (
                float(np.quantile(values, alpha)) if estimable else None
            ),
            "upper": (
                float(np.quantile(values, 1.0 - alpha)) if estimable else None
            ),
            "valid_replicates": int(len(values)),
            "valid_replicate_fraction": float(len(values) / iterations),
            "metric_contributing_clusters": contributing_clusters[key],
            "status": (
                "estimable"
                if estimable
                else (
                    "insufficient_metric_contributing_clusters"
                    if not enough_clusters
                    else "insufficient_valid_replicates"
                )
            ),
        }
    return {
        "cluster_unit": cluster_column.removeprefix("_"),
        "cluster_count": len(clusters),
        "iterations_requested": int(iterations),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "interval_method": "percentile_cluster_bootstrap",
        "estimand": estimand,
        "status": (
            "exploratory_discrete_small_cluster_sample"
            if len(clusters) < 10
            else "estimable"
        ),
        "resampling_definition": (
            f"Sample {len(clusters)} complete {cluster_column.removeprefix('_')} "
            "clusters with replacement per replicate and retain every window "
            "from each sampled cluster."
        ),
        "small_cluster_warning": (
            len(clusters) < 10
        ),
        "small_cluster_interpretation": (
            "With fewer than 10 clusters, especially n=3 participants, the "
            "percentile distribution is highly discrete and the interval is "
            "exploratory rather than a stable population-level interval."
            if len(clusters) < 10
            else None
        ),
        "interpretation": (
            "Cluster-resampling interval conditional on the observed independent "
            "groups; it is not uncertainty from overlapping windows and is not "
            "standard deviation across training seeds."
        ),
        "metrics": intervals,
    }


def build_grouped_evaluation(
    raw_frame: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
    include_sequence_bootstrap: bool = False,
) -> dict:
    frame, columns = prepare_prediction_frame(raw_frame)
    pose_methods = discover_pose_methods(frame)
    window = summarize_windows(frame, pose_methods)
    sequence = summarize_group_level(
        frame,
        group_column="sequence_id",
        pose_methods=pose_methods,
        definition=SEQUENCE_AGGREGATION_DEFINITION,
    )
    participant = summarize_participant_level(
        frame,
        pose_methods=pose_methods,
    )
    per_hand = {}
    for hand in HAND_NAMES:
        hand_frame = frame.loc[frame["_receiving_hand_context"] == hand]
        window_summary = summarize_windows(hand_frame, pose_methods)
        sequence_summary = summarize_group_level(
            hand_frame,
            group_column="sequence_id",
            pose_methods=pose_methods,
            definition=SEQUENCE_AGGREGATION_DEFINITION,
        )
        participant_summary = summarize_participant_level(
            hand_frame,
            pose_methods=pose_methods,
        )
        hand_bootstrap = (
            cluster_bootstrap_intervals(
                hand_frame,
                cluster_column="participant",
                pose_methods=pose_methods,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
            if len(hand_frame)
            else None
        )
        per_hand[hand] = {
            **window_summary,
            "window_level_role": "pooled_window_diagnostic",
            "sequence_balanced": sequence_summary["point_estimate"],
            "participant_balanced": participant_summary["point_estimate"],
            "participant_cluster_bootstrap": hand_bootstrap,
        }
    bootstraps = {
        "participant_cluster": cluster_bootstrap_intervals(
            frame,
            cluster_column="participant",
            pose_methods=pose_methods,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    }
    if include_sequence_bootstrap:
        bootstraps["sequence_cluster"] = cluster_bootstrap_intervals(
            frame,
            cluster_column="sequence_id",
            pose_methods=pose_methods,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    return {
        "schema_version": "grouped_prediction_evaluation_v1",
        "evaluation_unit_warning": (
            "Windows overlap and must not be interpreted as independent people."
        ),
        "input_columns": columns,
        "pose_methods": [
            {
                "name": method.name,
                "source_prefix": method.source_prefix,
                "position_error_column": method.position_error_column,
                "orientation_error_column": method.orientation_error_column,
                "reporting_role": (
                    "predicted_receiving_hand_diagnostic"
                    if method.name == "learned_end_to_end"
                    else (
                        "primary_ground_truth_hand_pose_comparison"
                        if method.name
                        in {
                            "learned_oracle_hand",
                            "persistence",
                            "constant_velocity",
                        }
                        else "additional_diagnostic"
                    )
                ),
            }
            for method in pose_methods
        ],
        "window_level": window,
        "sequence_level": sequence,
        "participant_level": participant,
        "per_receiving_hand": {
            "context_definition": (
                "Use the target receiving hand on a row; if exactly one hand is "
                "annotated anywhere in a sequence, propagate it only as a grouping "
                "context to the other rows in that sequence. Conflicting sequences "
                "are not imputed."
            ),
            "groups": per_hand,
            "primary_aggregation": (
                "participant-balanced metrics with participant-cluster bootstrap; "
                "pooled windows are descriptive only"
            ),
        },
        "cluster_bootstrap": bootstraps,
        "training_seed_variability": {
            "included": False,
            "reason": (
                "A per-window prediction export represents one executable "
                "checkpoint. Across-training-seed standard deviation must be "
                "computed from independent runs and reported separately from "
                "cluster-bootstrap confidence intervals."
            ),
        },
    }


def report_table_rows(report: Mapping[str, object]) -> list[dict]:
    """Create a compact long-form CSV table from the nested JSON report."""

    rows: list[dict] = []

    def append_scope(level: str, group: str, summary: Mapping[str, object]) -> None:
        classification = summary["classification"]
        assert isinstance(classification, Mapping)
        for task, raw_metrics in classification.items():
            metrics = raw_metrics
            assert isinstance(metrics, Mapping)
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "domain": "classification",
                    "method": task,
                    "samples": metrics.get("samples"),
                    "denominator": metrics.get("denominator"),
                    "coverage": metrics.get("coverage"),
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "macro_f1_supported": metrics.get("macro_f1_supported"),
                }
            )
        pose = summary["pose"]
        assert isinstance(pose, Mapping)
        for method, raw_metrics in pose.items():
            metrics = raw_metrics
            assert isinstance(metrics, Mapping)
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "domain": "pose",
                    "method": method,
                    "samples": metrics.get("position_samples"),
                    "denominator": metrics.get(
                        "coverage_denominator_pose_targets"
                    ),
                    "coverage": metrics.get("coverage"),
                    "position_mean_cm": metrics.get("position_mean_cm"),
                    "position_median_cm": metrics.get("position_median_cm"),
                    "position_rmse_cm": metrics.get("position_rmse_cm"),
                    "orientation_mean_deg": metrics.get("orientation_mean_deg"),
                    "orientation_median_deg": metrics.get(
                        "orientation_median_deg"
                    ),
                }
            )
        fair_common = summary.get("pose_fair_common")
        if isinstance(fair_common, Mapping):
            fair_methods = fair_common.get("methods", fair_common)
            if isinstance(fair_methods, Mapping):
                for method, raw_metrics in fair_methods.items():
                    if not isinstance(raw_metrics, Mapping):
                        continue
                    rows.append(
                        {
                            "level": level,
                            "group": group,
                            "domain": "pose_fair_common",
                            "method": method,
                            "samples": raw_metrics.get("position_samples"),
                            "denominator": fair_common.get(
                                "coverage_denominator_pose_targets"
                            ),
                            "coverage": raw_metrics.get("coverage"),
                            "position_mean_cm": raw_metrics.get(
                                "position_mean_cm"
                            ),
                            "position_median_cm": raw_metrics.get(
                                "position_median_cm"
                            ),
                            "position_rmse_cm": raw_metrics.get(
                                "position_rmse_cm"
                            ),
                            "orientation_mean_deg": raw_metrics.get(
                                "orientation_mean_deg"
                            ),
                            "orientation_median_deg": raw_metrics.get(
                                "orientation_median_deg"
                            ),
                        }
                    )
        paired = summary.get("paired_pose_comparisons", {})
        assert isinstance(paired, Mapping)
        for comparison, raw_metrics in paired.items():
            metrics = raw_metrics
            assert isinstance(metrics, Mapping)
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "domain": "paired_pose_difference",
                    "method": comparison,
                    "samples": metrics.get("shared_samples"),
                    "denominator": metrics.get(
                        "coverage_denominator_pose_targets"
                    ),
                    "coverage": metrics.get("coverage"),
                    "position_mean_difference_cm": metrics.get(
                        "position_mean_difference_cm"
                    ),
                    "position_median_difference_cm": metrics.get(
                        "position_median_difference_cm"
                    ),
                    "orientation_mean_difference_deg": metrics.get(
                        "orientation_mean_difference_deg"
                    ),
                    "learned_win_fraction": metrics.get("learned_win_fraction"),
                }
            )

    window = report["window_level"]
    assert isinstance(window, Mapping)
    append_scope("window", "all", window)
    sequence_level = report["sequence_level"]
    assert isinstance(sequence_level, Mapping)
    sequence_groups = sequence_level["groups"]
    assert isinstance(sequence_groups, Mapping)
    for group, metrics in sequence_groups.items():
        assert isinstance(metrics, Mapping)
        append_scope("sequence", str(group), metrics)
    participant_level = report["participant_level"]
    assert isinstance(participant_level, Mapping)
    participant_groups = participant_level["groups"]
    assert isinstance(participant_groups, Mapping)
    for group, participant in participant_groups.items():
        assert isinstance(participant, Mapping)
        metrics = participant["metrics"]
        assert isinstance(metrics, Mapping)
        append_scope("participant", str(group), metrics)
    hand_level = report["per_receiving_hand"]
    assert isinstance(hand_level, Mapping)
    hand_groups = hand_level["groups"]
    assert isinstance(hand_groups, Mapping)
    for hand, metrics in hand_groups.items():
        assert isinstance(metrics, Mapping)
        append_scope("receiving_hand", str(hand), metrics)
        for aggregation in ("sequence_balanced", "participant_balanced"):
            balanced = metrics.get(aggregation)
            if isinstance(balanced, Mapping):
                append_scope(
                    f"receiving_hand_{aggregation}",
                    str(hand),
                    balanced,
                )

    bootstraps = report["cluster_bootstrap"]
    assert isinstance(bootstraps, Mapping)
    for bootstrap_name, bootstrap in bootstraps.items():
        assert isinstance(bootstrap, Mapping)
        metrics = bootstrap["metrics"]
        assert isinstance(metrics, Mapping)
        for metric, values in metrics.items():
            assert isinstance(values, Mapping)
            rows.append(
                {
                    "level": "cluster_bootstrap",
                    "group": bootstrap_name,
                    "domain": "confidence_interval",
                    "method": metric,
                    "estimate": values.get("estimate"),
                    "ci_lower": values.get("lower"),
                    "ci_upper": values.get("upper"),
                    "samples": values.get("valid_replicates"),
                    "status": values.get("status"),
                    "metric_contributing_clusters": values.get(
                        "metric_contributing_clusters"
                    ),
                    "valid_replicate_fraction": values.get(
                        "valid_replicate_fraction"
                    ),
                }
            )
    return rows
