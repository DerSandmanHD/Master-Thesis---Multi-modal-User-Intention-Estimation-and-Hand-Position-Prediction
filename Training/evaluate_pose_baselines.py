#!/usr/bin/env python3
"""Evaluate leakage-safe naive baselines for future receiving-hand pose."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import INTENTION_TO_ID, WindowDataset, prepare_data
from metrics import pose_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/transformer_v1.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("Training/reports/pose_baselines.json"),
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=Path("Training/reports/pose_baselines.csv"),
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=None,
        help="Optional metrics.json whose test pose result is copied into the report.",
    )
    parser.add_argument(
        "--velocity-lookback-seconds",
        type=float,
        default=0.5,
        help="History duration used to fit the constant linear velocity baseline.",
    )
    parser.add_argument("--limit-sequences", type=int, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def truthy(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def normalize_quaternion(values: np.ndarray) -> np.ndarray | None:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-6:
        return None
    return quaternion / norm


def mean_pose(targets: np.ndarray) -> np.ndarray:
    """Return arithmetic position and sign-invariant quaternion mean."""
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
    return np.concatenate((values[:, :3].mean(axis=0), quaternion)).astype(np.float32)


def pose_columns(side: str) -> list[str]:
    return [
        *(f"{side}_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{side}_wrist_robot_q{component}" for component in "xyzw"),
    ]


def observation_pose(frame: pd.DataFrame, row_index: int, side: str) -> np.ndarray | None:
    row = frame.iloc[row_index]
    if not truthy(row[f"hand_{side}_valid"]) or not truthy(row["robot_frame_valid"]):
        return None
    values = pd.to_numeric(row[pose_columns(side)], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    quaternion = normalize_quaternion(values[3:7])
    if quaternion is None:
        return None
    return np.concatenate((values[:3], quaternion)).astype(np.float32)


def window_observations(
    frame: pd.DataFrame,
    start: int,
    endpoint: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = []
    timestamps = []
    poses = []
    for row_index in range(start, endpoint + 1):
        pose = observation_pose(frame, row_index, side)
        if pose is not None:
            indices.append(row_index)
            timestamps.append(int(frame.iloc[row_index]["timestamp_ns"]))
            poses.append(pose)
    if not poses:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 7), dtype=np.float32),
        )
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(poses, dtype=np.float32),
    )


def constant_velocity_pose(
    timestamps_ns: np.ndarray,
    poses: np.ndarray,
    desired_timestamp_ns: int,
    lookback_seconds: float,
) -> tuple[np.ndarray | None, int, float | None]:
    if len(poses) < 2:
        return None, len(poses), None
    latest_timestamp = int(timestamps_ns[-1])
    earliest_timestamp = latest_timestamp - int(lookback_seconds * 1e9)
    selected = timestamps_ns >= earliest_timestamp
    times = (timestamps_ns[selected] - latest_timestamp).astype(np.float64) / 1e9
    positions = poses[selected, :3].astype(np.float64)
    if len(times) < 2 or float(np.ptp(times)) <= 1e-6:
        return None, len(times), None

    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return None, len(times), None
    velocity = np.sum(centered[:, None] * positions, axis=0) / denominator
    delta_seconds = (desired_timestamp_ns - latest_timestamp) / 1e9
    predicted_position = poses[-1, :3] + velocity * delta_seconds
    prediction = np.concatenate((predicted_position, poses[-1, 3:7])).astype(np.float32)
    return prediction, len(times), float(np.linalg.norm(velocity))


def metric_dict(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    if not predictions:
        return pose_metrics(torch.empty((0, 7)), torch.empty((0, 7)))
    return pose_metrics(
        torch.from_numpy(np.asarray(predictions, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
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


def load_observation_frame(path: Path, expected_timestamps: np.ndarray) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_observation_columns()) - set(header))
    if missing:
        raise ValueError(
            f"{path.name} lacks pose-baseline columns: {', '.join(missing)}. "
            "Rebuild it with the current master builder."
        )
    frame = pd.read_csv(path, usecols=required_observation_columns(), low_memory=False)
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(np.int64)
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


def evaluate_split(
    split_name: str,
    dataset: WindowDataset,
    master_dir: Path,
    training_mean: np.ndarray,
    horizon_seconds: float,
    velocity_lookback_seconds: float,
) -> tuple[dict, list[dict]]:
    handover_id = INTENTION_TO_ID["handover"]
    endpoints_by_record: dict[int, list[int]] = defaultdict(list)
    for record_index, endpoint in dataset.indices:
        record = dataset.records[record_index]
        if int(record.intentions[endpoint]) == handover_id and bool(record.pose_valid[endpoint]):
            endpoints_by_record[record_index].append(endpoint)

    predictions: dict[str, list[np.ndarray]] = {
        "training_mean": [],
        "last_observation": [],
        "constant_velocity": [],
    }
    native_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    native_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    targets: list[np.ndarray] = []
    sources: dict[str, Counter] = defaultdict(Counter)
    details = []

    for record_index, endpoints in sorted(endpoints_by_record.items()):
        record = dataset.records[record_index]
        frame = load_observation_frame(
            master_dir / f"{record.sequence_id}_master.csv", record.timestamps_ns
        )
        for endpoint in endpoints:
            target = record.pose_targets[endpoint].astype(np.float32)
            targets.append(target)
            side = str(frame.iloc[endpoint]["receiving_hand"]).strip().lower()
            start = endpoint - dataset.window_size + 1
            if side in {"left", "right"}:
                _, observed_timestamps, observed_poses = window_observations(
                    frame, start, endpoint, side
                )
            else:
                observed_timestamps = np.empty(0, dtype=np.int64)
                observed_poses = np.empty((0, 7), dtype=np.float32)

            if len(observed_poses):
                last_prediction = observed_poses[-1].copy()
                last_source = "last_observation"
                last_age_seconds = (
                    int(record.timestamps_ns[endpoint]) - int(observed_timestamps[-1])
                ) / 1e9
                native_predictions["last_observation"].append(last_prediction)
                native_targets["last_observation"].append(target)
            else:
                last_prediction = training_mean.copy()
                last_source = "training_mean_fallback"
                last_age_seconds = None

            desired_timestamp_ns = int(record.timestamps_ns[endpoint]) + int(
                horizon_seconds * 1e9
            )
            velocity_prediction, velocity_samples, speed_m_s = constant_velocity_pose(
                observed_timestamps,
                observed_poses,
                desired_timestamp_ns,
                velocity_lookback_seconds,
            )
            if velocity_prediction is not None:
                velocity_source = "constant_velocity"
                native_predictions["constant_velocity"].append(velocity_prediction)
                native_targets["constant_velocity"].append(target)
            elif len(observed_poses):
                velocity_prediction = last_prediction.copy()
                velocity_source = "last_observation_fallback"
            else:
                velocity_prediction = training_mean.copy()
                velocity_source = "training_mean_fallback"

            predictions["training_mean"].append(training_mean.copy())
            predictions["last_observation"].append(last_prediction)
            predictions["constant_velocity"].append(velocity_prediction)
            sources["training_mean"]["training_mean"] += 1
            sources["last_observation"][last_source] += 1
            sources["constant_velocity"][velocity_source] += 1

            detail = {
                "split": split_name,
                "participant": record.participant,
                "sequence_id": record.sequence_id,
                "endpoint_row": endpoint,
                "endpoint_timestamp_ns": int(record.timestamps_ns[endpoint]),
                "receiving_hand": side,
                "last_observation_age_seconds": last_age_seconds,
                "velocity_fit_samples": velocity_samples,
                "estimated_speed_m_s": speed_m_s,
                "last_observation_source": last_source,
                "constant_velocity_source": velocity_source,
            }
            for component, value in zip(POSE_COMPONENTS, target):
                detail[f"target_{component}"] = float(value)
            for baseline_name, prediction in (
                ("training_mean", training_mean),
                ("last_observation", last_prediction),
                ("constant_velocity", velocity_prediction),
            ):
                for component, value in zip(POSE_COMPONENTS, prediction):
                    detail[f"{baseline_name}_{component}"] = float(value)
            details.append(detail)

    split_report = {
        "valid_pose_targets": len(targets),
        "baselines": {},
    }
    for name, values in predictions.items():
        split_report["baselines"][name] = {
            "metrics": metric_dict(values, targets),
            "prediction_sources": dict(sorted(sources[name].items())),
            "native_metrics": (
                metric_dict(native_predictions[name], native_targets[name])
                if name in native_predictions
                else metric_dict(values, targets)
            ),
        }
    return split_report, details


def evaluate(
    config: dict,
    *,
    config_path: Path,
    report_path: Path,
    details_path: Path,
    velocity_lookback_seconds: float,
    model_metrics_path: Path | None = None,
    limit_sequences: int | None = None,
) -> dict:
    if velocity_lookback_seconds <= 0.0:
        raise ValueError("velocity-lookback-seconds must be greater than zero")
    data_config = dict(config["data"])
    master_dir = project_path(Path(data_config["master_dir"]))
    data_config["master_dir"] = str(master_dir)
    seed = int(config["training"]["seed"])
    bundle = prepare_data(data_config, seed, limit_sequences)

    train_targets = valid_targets(bundle.train)
    training_mean = mean_pose(train_targets)
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
            training_mean,
            float(data_config["future_horizon_seconds"]),
            velocity_lookback_seconds,
        )
        split_reports[split_name] = split_report
        detail_rows.extend(split_details)

    report = {
        "config": str(config_path),
        "master_dir": str(master_dir),
        "seed": seed,
        "future_horizon_seconds": float(data_config["future_horizon_seconds"]),
        "velocity_lookback_seconds": velocity_lookback_seconds,
        "oracle_receiving_hand": True,
        "fallback_policy": {
            "last_observation": "training_mean",
            "constant_velocity": "last_observation_then_training_mean",
        },
        "training_target_count_for_mean": int(len(train_targets)),
        "training_mean_pose": training_mean.tolist(),
        "split_participants": bundle.split_metadata["participants"],
        "splits": split_reports,
    }
    if model_metrics_path is not None:
        model_report = json.loads(model_metrics_path.read_text(encoding="utf-8"))
        report["transformer_reference"] = {
            "metrics_path": str(model_metrics_path),
            "test_pose": model_report["test"]["pose"],
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
    model_metrics_path = (
        project_path(args.model_metrics) if args.model_metrics is not None else None
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = evaluate(
            config,
            config_path=config_path,
            report_path=report_path,
            details_path=details_path,
            velocity_lookback_seconds=args.velocity_lookback_seconds,
            model_metrics_path=model_metrics_path,
            limit_sequences=args.limit_sequences,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2

    for split_name in ("train", "validation", "test"):
        split = report["splits"][split_name]
        print(f"{split_name}: valid pose targets={split['valid_pose_targets']}")
        for name, result in split["baselines"].items():
            metrics = result["metrics"]
            print(
                f"  {name}: position mean Euclidean error="
                f"{metrics['position_mae_cm']:.2f} cm, "
                f"RMSE={metrics['position_rmse_cm']:.2f} cm, "
                f"orientation={metrics['orientation_mean_deg']:.2f} deg, "
                f"sources={result['prediction_sources']}"
            )
    print(f"Report:  {report_path}")
    print(f"Details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
