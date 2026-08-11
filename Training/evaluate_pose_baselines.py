#!/usr/bin/env python3
"""Evaluate pure, timestamp-aware t+1 receiving-wrist pose baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import INTENTION_TO_ID, WindowDataset, prepare_data
from pose_baselines import (
    POSE_COMPONENTS,
    BaselineEstimate,
    ObservationSeries,
    constant_velocity_pose as estimate_constant_velocity_pose,
    extract_hand_observations,
    normalize_quaternion,
    observation_pose,
    persistence_pose,
    pose_columns,
    pose_matches,
    pose_metric_summary,
    resolve_target_timing,
    sample_key_fingerprint,
    single_pose_errors,
    truthy,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAXIMUM_OBSERVATION_AGE_SECONDS = 0.25
DEFAULT_VELOCITY_LOOKBACK_SECONDS = 0.5
DEFAULT_MINIMUM_VELOCITY_FIT_SPAN_SECONDS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_v2.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("Training/reports/t_plus_1_pose_baselines.json"),
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=Path("Training/reports/t_plus_1_pose_baselines.csv"),
    )
    parser.add_argument(
        "--maximum-observation-age-seconds",
        type=float,
        default=DEFAULT_MAXIMUM_OBSERVATION_AGE_SECONDS,
        help="Reject a latest wrist capture older than this at the window endpoint.",
    )
    parser.add_argument(
        "--velocity-lookback-seconds",
        type=float,
        default=DEFAULT_VELOCITY_LOOKBACK_SECONDS,
        help="History duration on real hand-capture timestamps used for velocity.",
    )
    parser.add_argument(
        "--minimum-velocity-fit-span-seconds",
        type=float,
        default=DEFAULT_MINIMUM_VELOCITY_FIT_SPAN_SECONDS,
        help="Minimum time span between unique captures in a velocity fit.",
    )
    parser.add_argument("--limit-sequences", type=int, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean_pose(targets: np.ndarray) -> np.ndarray:
    """Return arithmetic position and sign-invariant quaternion mean.

    Retained for backwards-compatible imports by historical export scripts.  It
    is not a fallback for either primary baseline.
    """

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values):
        raise ValueError("At least one valid 7D training pose is required")
    if not np.isfinite(values).all():
        raise ValueError("Training pose targets contain non-finite values")

    quaternions = values[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1e-6):
        raise ValueError("Training pose targets contain invalid quaternions")
    quaternions = quaternions / norms[:, None]
    accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, -1]
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return np.concatenate((values[:, :3].mean(axis=0), quaternion)).astype(
        np.float32
    )


def window_observations(
    frame: pd.DataFrame,
    start: int,
    endpoint: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning deduplicated real capture timestamps.

    Current masters expose ``hand_timestamp_ns``.  Legacy synthetic fixtures
    without it retain their former master-timestamp behaviour only for scripts
    importing this wrapper; the baseline evaluator itself requires the real
    source timestamp.
    """

    source_column = (
        "hand_timestamp_ns" if "hand_timestamp_ns" in frame else "timestamp_ns"
    )
    observations = extract_hand_observations(
        frame,
        start,
        endpoint,
        side,
        source_timestamp_column=source_column,
        require_causal=True,
    )
    return (
        observations.row_indices,
        observations.capture_timestamps_ns,
        observations.poses,
    )


def constant_velocity_pose(
    timestamps_ns: np.ndarray,
    poses: np.ndarray,
    desired_timestamp_ns: int,
    lookback_seconds: float,
) -> tuple[np.ndarray | None, int, float | None]:
    """Compatibility wrapper for the former tuple-returning public function."""

    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float32)
    if not len(timestamps_ns):
        return None, 0, None
    observations = ObservationSeries(
        row_indices=np.arange(len(timestamps_ns), dtype=np.int64),
        capture_timestamps_ns=timestamps_ns,
        poses=poses,
    )
    endpoint_timestamp_ns = int(timestamps_ns[-1])
    estimate = estimate_constant_velocity_pose(
        observations,
        endpoint_timestamp_ns,
        int(desired_timestamp_ns),
        maximum_age_seconds=max(lookback_seconds, 1e-6),
        lookback_seconds=lookback_seconds,
        minimum_fit_span_seconds=1e-6,
    )
    return estimate.pose, estimate.fit_samples, estimate.estimated_speed_m_s


def metric_dict(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict:
    """Compatibility metric helper with the richer corrected pose semantics."""

    return pose_metric_summary(
        predictions,
        targets,
        coverage_denominator=len(targets),
    )


def required_observation_columns() -> list[str]:
    columns = [
        "timestamp_ns",
        "receiving_hand",
        "robot_frame_valid",
        "hand_left_valid",
        "hand_right_valid",
    ]
    for side in ("left", "right"):
        columns.extend(pose_columns(side))
    return columns


def load_observation_frame(
    path: Path,
    expected_timestamps: np.ndarray,
    *,
    require_source_timestamps: bool = False,
    require_target_timestamps: bool = False,
) -> pd.DataFrame:
    """Load pose columns while keeping legacy importers compatible."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = set(required_observation_columns())
    if require_source_timestamps:
        required.add("hand_timestamp_ns")
    if require_target_timestamps:
        required.add("future_target_timestamp_ns")
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(
            f"{path.name} lacks pose-baseline columns: {', '.join(missing)}. "
            "Rebuild it with the current master builder."
        )
    optional = [
        column
        for column in ("hand_timestamp_ns", "future_target_timestamp_ns")
        if column in header
    ]
    use_columns = list(dict.fromkeys([*required_observation_columns(), *optional]))
    frame = pd.read_csv(path, usecols=use_columns, low_memory=False)
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(
        np.int64
    )
    if len(timestamps) != len(expected_timestamps) or not np.array_equal(
        timestamps, expected_timestamps
    ):
        raise ValueError(f"Row alignment mismatch in {path.name}")
    return frame


def valid_targets(dataset: WindowDataset) -> np.ndarray:
    targets = [
        dataset.records[record_index].pose_targets[endpoint]
        for record_index, endpoint in dataset.indices
        if bool(dataset.records[record_index].pose_valid[endpoint])
    ]
    return np.asarray(targets, dtype=np.float32)


def unavailable(reason: str) -> BaselineEstimate:
    return BaselineEstimate(None, False, reason)


def alignment_summary(values: list[float]) -> dict:
    if not values:
        return {
            "samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
            "maximum_absolute_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
        "maximum_absolute_ms": float(np.abs(array).max()),
    }


def add_estimate_to_detail(
    detail: dict,
    name: str,
    estimate: BaselineEstimate,
    target: np.ndarray,
) -> None:
    detail[f"{name}_available"] = estimate.available
    detail[f"{name}_reason"] = estimate.reason
    detail[f"{name}_latest_capture_timestamp_ns"] = (
        estimate.latest_capture_timestamp_ns
    )
    detail[f"{name}_observation_age_seconds"] = estimate.observation_age_seconds
    detail[f"{name}_fit_samples"] = estimate.fit_samples
    detail[f"{name}_fit_span_seconds"] = estimate.fit_span_seconds
    detail[f"{name}_estimated_speed_m_s"] = estimate.estimated_speed_m_s
    for component in POSE_COMPONENTS:
        detail[f"{name}_{component}"] = None
    detail[f"{name}_position_error_cm"] = None
    detail[f"{name}_orientation_error_deg"] = None
    if estimate.available:
        assert estimate.pose is not None
        for component, value in zip(POSE_COMPONENTS, estimate.pose):
            detail[f"{name}_{component}"] = float(value)
        position_error, orientation_error = single_pose_errors(
            estimate.pose, target
        )
        detail[f"{name}_position_error_cm"] = position_error
        detail[f"{name}_orientation_error_deg"] = orientation_error


def evaluate_split(
    split_name: str,
    dataset: WindowDataset,
    master_dir: Path,
    horizon_seconds: float,
    velocity_lookback_seconds: float,
    maximum_observation_age_seconds: float,
    minimum_velocity_fit_span_seconds: float,
) -> tuple[dict, list[dict]]:
    handover_id = INTENTION_TO_ID["handover"]
    endpoints_by_record: dict[int, list[int]] = defaultdict(list)
    handover_windows = 0
    for record_index, endpoint in dataset.indices:
        record = dataset.records[record_index]
        if int(record.intentions[endpoint]) == handover_id:
            handover_windows += 1
            if bool(record.pose_valid[endpoint]):
                endpoints_by_record[record_index].append(endpoint)

    method_names = ("persistence", "constant_velocity")
    native_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    native_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    native_keys: dict[str, list[str]] = defaultdict(list)
    common_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    common_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    reasons: dict[str, Counter] = defaultdict(Counter)
    common_keys: list[str] = []
    details: list[dict] = []
    valid_target_count = sum(len(values) for values in endpoints_by_record.values())
    master_alignment_errors: list[float] = []
    actual_capture_errors: list[float] = []

    for record_index, endpoints in sorted(endpoints_by_record.items()):
        record = dataset.records[record_index]
        frame = load_observation_frame(
            master_dir / f"{record.sequence_id}_master.csv",
            record.timestamps_ns,
            require_source_timestamps=True,
            require_target_timestamps=True,
        )
        for endpoint in endpoints:
            target = record.pose_targets[endpoint].astype(np.float32)
            endpoint_timestamp_ns = int(record.timestamps_ns[endpoint])
            side = str(frame.iloc[endpoint]["receiving_hand"]).strip().lower()
            sample_key = (
                f"{split_name}|{record.sequence_id}|{endpoint_timestamp_ns}"
            )
            start = endpoint - dataset.window_size + 1

            timing = None
            observations = ObservationSeries.empty()
            if side in {"left", "right"}:
                timing = resolve_target_timing(
                    frame,
                    endpoint,
                    horizon_seconds=horizon_seconds,
                )
                target_row_pose = observation_pose(
                    frame, timing.target_row_index, side
                )
                if target_row_pose is None or not pose_matches(
                    target_row_pose, target
                ):
                    raise ValueError(
                        "Future target does not match the receiving-hand pose at "
                        f"its source row: {sample_key}"
                    )
                observations = extract_hand_observations(
                    frame,
                    start,
                    endpoint,
                    side,
                    source_timestamp_column="hand_timestamp_ns",
                    require_causal=True,
                )
                persistence = persistence_pose(
                    observations,
                    endpoint_timestamp_ns,
                    maximum_age_seconds=maximum_observation_age_seconds,
                )
                velocity = estimate_constant_velocity_pose(
                    observations,
                    endpoint_timestamp_ns,
                    timing.prediction_horizon_timestamp_ns,
                    maximum_age_seconds=maximum_observation_age_seconds,
                    lookback_seconds=velocity_lookback_seconds,
                    minimum_fit_span_seconds=minimum_velocity_fit_span_seconds,
                )
                master_alignment_errors.append(timing.master_alignment_error_ms)
                actual_capture_errors.append(timing.actual_capture_error_ms)
            else:
                persistence = unavailable("unknown_receiving_hand")
                velocity = unavailable("unknown_receiving_hand")

            estimates = {
                "persistence": persistence,
                "constant_velocity": velocity,
            }
            for name, estimate in estimates.items():
                reasons[name][estimate.reason] += 1
                if estimate.available:
                    assert estimate.pose is not None
                    native_predictions[name].append(estimate.pose)
                    native_targets[name].append(target)
                    native_keys[name].append(sample_key)

            fair_common = all(
                estimates[name].available for name in method_names
            )
            if fair_common:
                common_keys.append(sample_key)
                for name in method_names:
                    assert estimates[name].pose is not None
                    common_predictions[name].append(estimates[name].pose)
                    common_targets[name].append(target)

            detail = {
                "sample_key": sample_key,
                "split": split_name,
                "participant": record.participant,
                "sequence_id": record.sequence_id,
                "endpoint_row": endpoint,
                "endpoint_timestamp_ns": endpoint_timestamp_ns,
                "receiving_hand": side,
                "valid_future_pose_target": True,
                "fair_common": fair_common,
                "unique_observation_captures": int(len(observations.poses)),
                "duplicate_aligned_captures_removed": (
                    observations.duplicate_captures_removed
                ),
                "noncausal_captures_removed": (
                    observations.noncausal_captures_removed
                ),
                "nominal_target_timestamp_ns": (
                    timing.nominal_timestamp_ns if timing else None
                ),
                "aligned_target_master_timestamp_ns": (
                    timing.aligned_master_timestamp_ns if timing else None
                ),
                "actual_target_hand_capture_timestamp_ns": (
                    timing.actual_hand_capture_timestamp_ns if timing else None
                ),
                "target_master_alignment_error_ms": (
                    timing.master_alignment_error_ms if timing else None
                ),
                "target_actual_capture_error_ms": (
                    timing.actual_capture_error_ms if timing else None
                ),
            }
            for component, value in zip(POSE_COMPONENTS, target):
                detail[f"target_{component}"] = float(value)
            for name, estimate in estimates.items():
                add_estimate_to_detail(detail, name, estimate, target)
            details.append(detail)

    split_report = {
        "handover_windows": handover_windows,
        "valid_pose_targets": valid_target_count,
        "valid_target_coverage": (
            float(valid_target_count / handover_windows)
            if handover_windows
            else None
        ),
        "coverage_denominator_definition": (
            "handover windows with a finite, valid t+1 receiving-wrist target"
        ),
        "fair_common": {
            "samples": len(common_keys),
            "coverage_denominator": valid_target_count,
            "coverage": (
                float(len(common_keys) / valid_target_count)
                if valid_target_count
                else None
            ),
            "sample_key_fingerprint": sample_key_fingerprint(common_keys),
            "definition": (
                "intersection of valid t+1 targets for which pure persistence "
                "and pure constant velocity are both available"
            ),
        },
        "target_timing": {
            "nominal_definition": "endpoint timestamp_ns + future horizon",
            "aligned_master_error": alignment_summary(master_alignment_errors),
            "actual_hand_capture_error": alignment_summary(actual_capture_errors),
        },
        "baselines": {},
    }
    for name in method_names:
        split_report["baselines"][name] = {
            "native_metrics": pose_metric_summary(
                native_predictions[name],
                native_targets[name],
                coverage_denominator=valid_target_count,
            ),
            "fair_common_metrics": pose_metric_summary(
                common_predictions[name],
                common_targets[name],
                coverage_denominator=valid_target_count,
            ),
            "availability_reasons": dict(sorted(reasons[name].items())),
            "native_sample_key_fingerprint": sample_key_fingerprint(
                native_keys[name]
            ),
        }
    return split_report, details


def evaluate(
    config: dict,
    *,
    config_path: Path,
    report_path: Path,
    details_path: Path,
    velocity_lookback_seconds: float = DEFAULT_VELOCITY_LOOKBACK_SECONDS,
    maximum_observation_age_seconds: float = (
        DEFAULT_MAXIMUM_OBSERVATION_AGE_SECONDS
    ),
    minimum_velocity_fit_span_seconds: float = (
        DEFAULT_MINIMUM_VELOCITY_FIT_SPAN_SECONDS
    ),
    limit_sequences: int | None = None,
) -> dict:
    for name, value in (
        ("velocity_lookback_seconds", velocity_lookback_seconds),
        ("maximum_observation_age_seconds", maximum_observation_age_seconds),
        (
            "minimum_velocity_fit_span_seconds",
            minimum_velocity_fit_span_seconds,
        ),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
    if minimum_velocity_fit_span_seconds > velocity_lookback_seconds:
        raise ValueError(
            "minimum_velocity_fit_span_seconds cannot exceed the lookback"
        )

    data_config = dict(config["data"])
    if data_config.get("pose_target"):
        raise ValueError(
            "This evaluator is only for PRIMARY future-offset t+1 targets, "
            "not terminal/endpose targets"
        )
    horizon_seconds = float(data_config["future_horizon_seconds"])
    master_dir = project_path(Path(data_config["master_dir"]))
    data_config["master_dir"] = str(master_dir)
    seed = int(config["training"]["seed"])
    bundle = prepare_data(data_config, seed, limit_sequences)

    split_reports = {}
    detail_rows = []
    for split_name, dataset in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        split_report, split_details = evaluate_split(
            split_name,
            dataset,
            master_dir,
            horizon_seconds,
            velocity_lookback_seconds,
            maximum_observation_age_seconds,
            minimum_velocity_fit_span_seconds,
        )
        split_reports[split_name] = split_report
        detail_rows.extend(split_details)

    report = {
        "schema_version": 2,
        "task": {
            "name": "primary_t_plus_1_future_receiving_wrist_pose",
            "pose_target_mode": "future_offset",
            "future_horizon_seconds": horizon_seconds,
            "coordinate_frame": "static robot-marker frame",
        },
        "config": str(config_path),
        "master_dir": str(master_dir),
        "seed": seed,
        "oracle_receiving_hand": True,
        "receiving_hand_context": (
            "ground-truth receiving hand is shared by both pose baselines"
        ),
        "timestamp_policy": {
            "observation_time": (
                "real hand_timestamp_ns; repeated merge-asof captures are deduplicated"
            ),
            "causality": "captures after the master endpoint are excluded",
            "target_measurement_time": (
                "hand_timestamp_ns at the master row referenced by "
                "future_target_timestamp_ns"
            ),
            "constant_velocity_prediction_horizon": (
                "endpoint timestamp_ns + future_horizon_seconds; future target "
                "capture jitter is never exposed to the baseline"
            ),
        },
        "baseline_policy": {
            "fallbacks": "none",
            "maximum_observation_age_seconds": maximum_observation_age_seconds,
            "velocity_lookback_seconds": velocity_lookback_seconds,
            "minimum_velocity_fit_span_seconds": (
                minimum_velocity_fit_span_seconds
            ),
            "constant_velocity_orientation": (
                "zero angular velocity; hold latest valid normalized quaternion"
            ),
            "primary_comparison": "fair_common_metrics",
        },
        "split_participants": bundle.split_metadata["participants"],
        "splits": split_reports,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False)
    return report


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    report_path = project_path(args.report_out)
    details_path = project_path(args.details_out)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = evaluate(
            config,
            config_path=config_path,
            report_path=report_path,
            details_path=details_path,
            velocity_lookback_seconds=args.velocity_lookback_seconds,
            maximum_observation_age_seconds=(
                args.maximum_observation_age_seconds
            ),
            minimum_velocity_fit_span_seconds=(
                args.minimum_velocity_fit_span_seconds
            ),
            limit_sequences=args.limit_sequences,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2

    for split_name in ("train", "validation", "test"):
        split = report["splits"][split_name]
        print(
            f"{split_name}: handover={split['handover_windows']}, "
            f"valid targets={split['valid_pose_targets']}, "
            f"fair common={split['fair_common']['samples']}"
        )
        for name, result in split["baselines"].items():
            metrics = result["fair_common_metrics"]
            print(
                f"  {name}: mean={metrics['position_mean_euclidean_error_cm']} cm, "
                f"median={metrics['position_median_euclidean_error_cm']} cm, "
                f"orientation={metrics['orientation_mean_deg']} deg, "
                f"n={metrics['samples']}, coverage={metrics['coverage']}"
            )
    print(f"Report:  {report_path}")
    print(f"Details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
