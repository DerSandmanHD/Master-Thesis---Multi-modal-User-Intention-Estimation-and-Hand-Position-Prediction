#!/usr/bin/env python3
"""Export window-level predictions and grouped errors from a trained run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import INTENTION_NAMES, INTENTION_TO_ID, WindowDataset, prepare_data
from evaluate_pose_baselines import (
    POSE_COMPONENTS,
    load_observation_frame,
    mean_pose,
    valid_targets,
    window_observations,
)
from metrics import (
    classification_metrics,
    pose_metrics,
    position_mean_euclidean_error_cm,
)
from model import HierarchicalGatedMultimodalTransformer
from run_discovery import resolve_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing config.json, metrics.json and best_model.pt.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("Training/runs"),
        help="Recursive search root when --run-dir is only a run basename.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Allow export even when recomputed metrics differ from metrics.json.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pose_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    position_cm = float(np.linalg.norm(prediction[:3] - target[:3]) * 100.0)
    predicted_quaternion = prediction[3:7]
    target_quaternion = target[3:7]
    predicted_norm = float(np.linalg.norm(predicted_quaternion))
    target_norm = float(np.linalg.norm(target_quaternion))
    if predicted_norm <= 1e-6 or target_norm <= 1e-6:
        raise ValueError("Cannot score an invalid quaternion")
    cosine = abs(
        float(np.dot(predicted_quaternion, target_quaternion))
        / (predicted_norm * target_norm)
    )
    orientation_deg = math.degrees(2.0 * math.acos(np.clip(cosine, 0.0, 1.0)))
    return position_cm, orientation_deg


def handover_metadata(dataset: WindowDataset) -> dict[tuple[int, int], dict]:
    endpoints: dict[int, list[int]] = defaultdict(list)
    handover_id = INTENTION_TO_ID["handover"]
    for record_index, endpoint in dataset.indices:
        record = dataset.records[record_index]
        if int(record.intentions[endpoint]) == handover_id:
            endpoints[record_index].append(endpoint)

    metadata = {}
    for record_index, values in endpoints.items():
        values = sorted(values)
        record = dataset.records[record_index]
        first_timestamp = int(record.timestamps_ns[values[0]])
        last_timestamp = int(record.timestamps_ns[values[-1]])
        for index, endpoint in enumerate(values):
            timestamp = int(record.timestamps_ns[endpoint])
            metadata[(record_index, endpoint)] = {
                "handover_window_index": index,
                "handover_window_count": len(values),
                "handover_progress": index / max(1, len(values) - 1),
                "handover_elapsed_seconds": (timestamp - first_timestamp) / 1e9,
                "handover_remaining_seconds": (last_timestamp - timestamp) / 1e9,
            }
    return metadata


def progress_group(progress: float | None) -> str | None:
    if progress is None:
        return None
    if progress <= 0.25:
        return "0-25%"
    if progress <= 0.5:
        return "25-50%"
    if progress <= 0.75:
        return "50-75%"
    return "75-100%"


def metric_pose_arrays(rows: list[dict], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    valid = [row for row in rows if row["pose_valid"]]
    if not valid:
        return torch.empty((0, 7)), torch.empty((0, 7))
    predictions = np.asarray(
        [[row[f"{prefix}_{component}"] for component in POSE_COMPONENTS] for row in valid],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[row[f"target_{component}"] for component in POSE_COMPONENTS] for row in valid],
        dtype=np.float32,
    )
    return torch.from_numpy(predictions), torch.from_numpy(targets)


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "windows": 0,
            "intention_accuracy": None,
            "valid_pose_targets": 0,
            "transformer_pose": pose_metrics(torch.empty((0, 7)), torch.empty((0, 7))),
            "last_observation_pose": pose_metrics(
                torch.empty((0, 7)), torch.empty((0, 7))
            ),
            "transformer_position_wins": 0,
            "last_observation_position_wins": 0,
            "position_ties": 0,
        }
    intention_accuracy = sum(
        row["predicted_intention_id"] == row["target_intention_id"] for row in rows
    ) / len(rows)
    transformer_predictions, targets = metric_pose_arrays(rows, "transformer")
    last_predictions, _ = metric_pose_arrays(rows, "last_observation")
    valid = [row for row in rows if row["pose_valid"]]
    transformer_wins = sum(
        row["transformer_position_error_cm"] < row["last_observation_position_error_cm"]
        for row in valid
    )
    last_wins = sum(
        row["last_observation_position_error_cm"] < row["transformer_position_error_cm"]
        for row in valid
    )
    return {
        "windows": len(rows),
        "intention_accuracy": intention_accuracy,
        "valid_pose_targets": len(valid),
        "transformer_pose": pose_metrics(transformer_predictions, targets),
        "last_observation_pose": pose_metrics(last_predictions, targets),
        "transformer_position_wins": transformer_wins,
        "last_observation_position_wins": last_wins,
        "position_ties": len(valid) - transformer_wins - last_wins,
    }


def grouped_summary(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            groups[str(value)].append(row)
    return {name: summarize_rows(values) for name, values in sorted(groups.items())}


def assert_close(name: str, actual: float | int | None, expected: float | int | None) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(f"Reference mismatch for {name}: {actual} != {expected}")
        return
    if isinstance(actual, int) and isinstance(expected, int):
        matches = actual == expected
    else:
        matches = math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-5)
    if not matches:
        raise ValueError(f"Reference mismatch for {name}: {actual} != {expected}")


def verify_reference(summary: dict, rows: list[dict], reference: dict) -> None:
    target_ids = torch.tensor([row["target_intention_id"] for row in rows])
    predicted_ids = torch.tensor([row["predicted_intention_id"] for row in rows])
    intention = classification_metrics(predicted_ids, target_ids, len(INTENTION_NAMES))
    expected_intention = reference["test"]["intention"]
    assert_close("intention.samples", intention["samples"], expected_intention["samples"])
    assert_close("intention.accuracy", intention["accuracy"], expected_intention["accuracy"])
    assert_close("intention.macro_f1", intention["macro_f1"], expected_intention["macro_f1"])

    actual_pose = summary["overall"]["transformer_pose"]
    expected_pose = reference["test"]["pose"]
    for key in ("samples", "position_mae_cm", "position_rmse_cm", "orientation_mean_deg"):
        assert_close(f"pose.{key}", actual_pose[key], expected_pose[key])


def export_predictions(
    run_dir: Path,
    *,
    output_csv: Path,
    report_path: Path,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int = 0,
    verify_metrics: bool = True,
) -> dict:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    data_config["master_dir"] = str(project_path(Path(data_config["master_dir"])))
    bundle = prepare_data(data_config, int(config["training"]["seed"]))
    dataset = bundle.test

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_input_dim = len(bundle.normalizer.output_feature_names)
    if int(checkpoint["input_dim"]) != expected_input_dim:
        raise ValueError(
            f"Checkpoint input_dim={checkpoint['input_dim']} but data exposes "
            f"{expected_input_dim} features"
        )
    if int(checkpoint["window_size"]) != int(data_config["window_size"]):
        raise ValueError("Checkpoint and data config use different window sizes")
    model = HierarchicalGatedMultimodalTransformer(
        input_dim=expected_input_dim,
        window_size=int(checkpoint["window_size"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    training_mean = mean_pose(valid_targets(bundle.train))
    master_dir = Path(data_config["master_dir"])
    frame_cache: dict[str, pd.DataFrame] = {}
    progress = handover_metadata(dataset)
    rows = []
    dataset_offset = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            outputs = model(features)
            assistance_probabilities = F.softmax(outputs["assistance_logits"], dim=-1).cpu()
            type_probabilities = F.softmax(outputs["assistance_type_logits"], dim=-1).cpu()
            assistance_predictions = assistance_probabilities.argmax(dim=-1)
            type_predictions = type_probabilities.argmax(dim=-1)
            intention_predictions = torch.zeros_like(assistance_predictions)
            assistance_mask = assistance_predictions.bool()
            intention_predictions[assistance_mask] = type_predictions[assistance_mask] + 1
            predicted_poses = outputs["pose"].cpu().numpy()
            gates = outputs["gate"].cpu().numpy()
            batch_size_actual = len(batch["intention"])

            for batch_index in range(batch_size_actual):
                dataset_index = dataset_offset + batch_index
                record_index, endpoint = dataset.indices[dataset_index]
                record = dataset.records[record_index]
                timestamp_ns = int(batch["timestamp_ns"][batch_index])
                if timestamp_ns != int(record.timestamps_ns[endpoint]):
                    raise ValueError(
                        f"DataLoader order mismatch at dataset index {dataset_index}"
                    )
                sequence_id = record.sequence_id
                if sequence_id not in frame_cache:
                    frame_cache[sequence_id] = load_observation_frame(
                        master_dir / f"{sequence_id}_master.csv", record.timestamps_ns
                    )
                frame = frame_cache[sequence_id]
                side = str(frame.iloc[endpoint]["receiving_hand"]).strip().lower()
                target_intention = int(batch["intention"][batch_index])
                predicted_intention = int(intention_predictions[batch_index])
                target_pose = batch["pose_target"][batch_index].numpy().astype(np.float32)
                transformer_pose = predicted_poses[batch_index].astype(np.float32)
                is_pose_valid = bool(batch["pose_valid"][batch_index]) and target_intention == 2

                row = {
                    "split": "test",
                    "dataset_index": dataset_index,
                    "participant": record.participant,
                    "sequence_id": sequence_id,
                    "endpoint_row": endpoint,
                    "endpoint_timestamp_ns": timestamp_ns,
                    "receiving_hand": side,
                    "target_intention_id": target_intention,
                    "target_intention": INTENTION_NAMES[target_intention],
                    "predicted_intention_id": predicted_intention,
                    "predicted_intention": INTENTION_NAMES[predicted_intention],
                    "intention_correct": target_intention == predicted_intention,
                    "assistance_probability": float(assistance_probabilities[batch_index, 1]),
                    "fetch_probability_given_assistance": float(
                        type_probabilities[batch_index, 0]
                    ),
                    "handover_probability_given_assistance": float(
                        type_probabilities[batch_index, 1]
                    ),
                    "gate_temporal": float(gates[batch_index, 0]),
                    "gate_channel": float(gates[batch_index, 1]),
                    "pose_valid": is_pose_valid,
                }
                handover = progress.get((record_index, endpoint), {})
                row.update(handover)
                row["handover_progress_group"] = progress_group(
                    handover.get("handover_progress")
                )
                for component, value in zip(POSE_COMPONENTS, transformer_pose):
                    row[f"transformer_{component}"] = float(value)

                if is_pose_valid:
                    for component, value in zip(POSE_COMPONENTS, target_pose):
                        row[f"target_{component}"] = float(value)
                    transformer_position, transformer_orientation = pose_error(
                        transformer_pose, target_pose
                    )
                    row["transformer_position_error_cm"] = transformer_position
                    row["transformer_orientation_error_deg"] = transformer_orientation

                    start = endpoint - dataset.window_size + 1
                    if side in {"left", "right"}:
                        _, observed_timestamps, observed_poses = window_observations(
                            frame, start, endpoint, side
                        )
                    else:
                        observed_timestamps = np.empty(0, dtype=np.int64)
                        observed_poses = np.empty((0, 7), dtype=np.float32)
                    if len(observed_poses):
                        last_pose = observed_poses[-1]
                        last_source = "last_observation"
                        last_age = (timestamp_ns - int(observed_timestamps[-1])) / 1e9
                    else:
                        last_pose = training_mean
                        last_source = "training_mean_fallback"
                        last_age = None
                    last_position, last_orientation = pose_error(last_pose, target_pose)
                    row["last_observation_source"] = last_source
                    row["last_observation_age_seconds"] = last_age
                    row["last_observation_position_error_cm"] = last_position
                    row["last_observation_orientation_error_deg"] = last_orientation
                    for component, value in zip(POSE_COMPONENTS, last_pose):
                        row[f"last_observation_{component}"] = float(value)
                else:
                    for component in POSE_COMPONENTS:
                        row[f"target_{component}"] = None
                        row[f"last_observation_{component}"] = None
                    row["transformer_position_error_cm"] = None
                    row["transformer_orientation_error_deg"] = None
                    row["last_observation_source"] = None
                    row["last_observation_age_seconds"] = None
                    row["last_observation_position_error_cm"] = None
                    row["last_observation_orientation_error_deg"] = None
                rows.append(row)
            dataset_offset += batch_size_actual

    if dataset_offset != len(dataset):
        raise RuntimeError(f"Exported {dataset_offset} of {len(dataset)} test windows")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "oracle_last_observation": True,
        "overall": summarize_rows(rows),
        "participants": grouped_summary(rows, "participant"),
        "receiving_hands": grouped_summary(rows, "receiving_hand"),
        "handover_progress": grouped_summary(
            [row for row in rows if row["pose_valid"]], "handover_progress_group"
        ),
        "sequences": grouped_summary(rows, "sequence_id"),
    }

    reference_status = "not_checked"
    if metrics_path.exists() and verify_metrics:
        reference = json.loads(metrics_path.read_text(encoding="utf-8"))
        verify_reference(summary, rows, reference)
        reference_status = "matched"
    elif verify_metrics:
        raise FileNotFoundError(f"Reference metrics not found: {metrics_path}")
    summary["reference_metrics"] = {
        "path": str(metrics_path),
        "status": reference_status,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        print("ERROR: ValueError: batch-size must be greater than zero")
        return 2
    if args.num_workers < 0:
        print("ERROR: ValueError: num-workers cannot be negative")
        return 2
    device = choose_device(args.device)
    try:
        run_dir = resolve_run_directory(
            project_path(args.run_dir),
            runs_root=project_path(args.runs_root),
            required_artifacts=(
                "config.json",
                "metrics.json",
                "best_model.pt",
            ),
        )
        output_csv = (
            project_path(args.output_csv)
            if args.output_csv
            else run_dir / "test_predictions.csv"
        )
        report_path = (
            project_path(args.report_out)
            if args.report_out
            else run_dir / "test_prediction_analysis.json"
        )
        summary = export_predictions(
            run_dir,
            output_csv=output_csv,
            report_path=report_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            verify_metrics=not args.skip_reference_check,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2

    overall = summary["overall"]
    transformer = overall["transformer_pose"]
    last = overall["last_observation_pose"]
    print(f"Device: {device}")
    print(f"Test windows: {overall['windows']}")
    print(f"Valid pose targets: {overall['valid_pose_targets']}")
    print(
        "Transformer: "
        "mean Euclidean error="
        f"{position_mean_euclidean_error_cm(transformer):.2f} cm, "
        f"RMSE={transformer['position_rmse_cm']:.2f} cm, "
        f"orientation={transformer['orientation_mean_deg']:.2f} deg"
    )
    print(
        "Last observation: "
        "mean Euclidean error="
        f"{position_mean_euclidean_error_cm(last):.2f} cm, "
        f"RMSE={last['position_rmse_cm']:.2f} cm, "
        f"orientation={last['orientation_mean_deg']:.2f} deg"
    )
    print(
        "Position wins: "
        f"transformer={overall['transformer_position_wins']}, "
        f"last_observation={overall['last_observation_position_wins']}, "
        f"ties={overall['position_ties']}"
    )
    print(f"Reference metrics: {summary['reference_metrics']['status']}")
    print(f"Predictions: {output_csv}")
    print(f"Analysis:    {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
