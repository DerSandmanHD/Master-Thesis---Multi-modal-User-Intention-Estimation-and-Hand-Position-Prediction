#!/usr/bin/env python3
"""Export window-level residual-v2 probabilities, hands, and future poses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from artifact_freeze import (
    MANIFEST_NAME,
    canonical_json_hash,
    sha256_file,
    validate_artifact_freeze,
)
from data import INTENTION_NAMES, RECEIVING_HAND_NAMES, prepare_data
from prediction_utils import (
    assistance_predictions,
    assistance_type_predictions,
    intention_predictions,
    intention_probabilities,
)
from pose_baselines import (
    BaselineEstimate,
    constant_velocity_pose,
    extract_hand_observations,
    persistence_pose,
    pose_columns,
    pose_metric_summary,
    resolve_target_timing,
    sample_key_fingerprint,
    single_pose_errors,
)
from train_residual import RESIDUAL_V2_MODEL_TYPE, build_residual_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_COMPONENTS = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
DEFAULT_MAXIMUM_OBSERVATION_AGE_SECONDS = 0.25
DEFAULT_VELOCITY_LOOKBACK_SECONDS = 0.5
DEFAULT_MINIMUM_VELOCITY_FIT_SPAN_SECONDS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_intention_model.pt")
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=None,
        help=(
            "Override data.master_dir from the saved run config. This keeps "
            "cluster-trained runs exportable after they are copied elsewhere."
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--final-test-report",
        type=Path,
        default=None,
        help=(
            "Required for split=test. Hash-bound authorization emitted by "
            "evaluate_frozen_run.py for this exact checkpoint."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--maximum-observation-age-seconds",
        type=float,
        default=DEFAULT_MAXIMUM_OBSERVATION_AGE_SECONDS,
    )
    parser.add_argument(
        "--velocity-lookback-seconds",
        type=float,
        default=DEFAULT_VELOCITY_LOOKBACK_SECONDS,
    )
    parser.add_argument(
        "--minimum-velocity-fit-span-seconds",
        type=float,
        default=DEFAULT_MINIMUM_VELOCITY_FIT_SPAN_SECONDS,
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
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


def sequence_context(
    master_dir: Path,
    sequence_id: str,
    expected_timestamps_ns: np.ndarray,
) -> pd.DataFrame:
    path = master_dir / f"{sequence_id}_master.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {
        "timestamp_ns",
        "hand_timestamp_ns",
        "future_target_timestamp_ns",
        "receiving_hand",
        "robot_frame_valid",
        "hand_left_valid",
        "hand_right_valid",
        *(column for side in ("left", "right") for column in pose_columns(side)),
    }
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(
            f"{path.name} lacks corrected t+1 evaluation columns: "
            f"{', '.join(missing)}"
        )
    optional = [
        column
        for column in (
            "time_since_start_s",
            "future_target_hand_timestamp_ns",
            "target_object_id",
            "target_object_known",
        )
        if column in header
    ]
    frame = pd.read_csv(path, usecols=[*sorted(required), *optional])
    timestamps = frame["timestamp_ns"].to_numpy(np.int64)
    if not np.array_equal(timestamps, expected_timestamps_ns):
        raise ValueError(f"Master row alignment differs for {sequence_id}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Master timeline is invalid for {sequence_id}")
    if "time_since_start_s" not in frame:
        frame["time_since_start_s"] = (timestamps - timestamps[0]) / 1e9
    elapsed = pd.to_numeric(
        frame["time_since_start_s"], errors="raise"
    ).to_numpy(np.float64)
    if np.any(np.diff(elapsed) < 0):
        raise ValueError(f"Master elapsed time is invalid for {sequence_id}")
    return frame


def pose_values(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = [f"{prefix}_{component}" for component in POSE_COMPONENTS]
    return frame[columns].to_numpy(np.float32)


def validate_final_test_binding(
    path: Path,
    *,
    run_dir: Path,
    checkpoint_hash: str,
    artifact_manifest_fingerprint: str,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    stored_fingerprint = report.get("report_fingerprint")
    if stored_fingerprint != canonical_json_hash(
        {**report, "report_fingerprint": None}
    ):
        raise ValueError("Final-test authorization report fingerprint mismatch")
    if (
        report.get("schema_version") != 2
        or report.get("evaluation_protocol")
        != "validation_frozen_checkpoint_single_test_v2"
        or report.get("split") != "test"
    ):
        raise ValueError("Unsupported final-test authorization report")
    if report.get("test_used_for_model_or_checkpoint_selection") is not False:
        raise ValueError("Final-test report violates test-set discipline")
    if not isinstance(report.get("matrix_authorization"), dict):
        raise ValueError("Final-test report has no matrix authorization")
    source_run = Path(str(report.get("source_run", ""))).expanduser().resolve()
    if source_run != run_dir.resolve():
        raise ValueError("Final-test report belongs to another run")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("name") != "best_intention":
        raise ValueError("Final-test report is not the main best_intention checkpoint")
    if str(checkpoint.get("sha256", "")).lower() != checkpoint_hash.lower():
        raise ValueError("Final-test report belongs to another checkpoint")
    if report.get("source_artifact_manifest_fingerprint") != (
        artifact_manifest_fingerprint
    ):
        raise ValueError("Final-test report belongs to another artifact freeze")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "report_fingerprint": stored_fingerprint,
        "evaluation_protocol": report["evaluation_protocol"],
        "matrix_authorization": report["matrix_authorization"],
    }


def main() -> int:
    args = parse_args()
    for name in (
        "maximum_observation_age_seconds",
        "velocity_lookback_seconds",
        "minimum_velocity_fit_span_seconds",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    if args.minimum_velocity_fit_span_seconds > args.velocity_lookback_seconds:
        raise ValueError("Minimum velocity fit span cannot exceed its lookback")
    run_dir = resolve(args.run_dir).resolve()
    output_csv = resolve(args.output_csv).resolve()
    report_path = (
        resolve(args.report_out).resolve()
        if args.report_out
        else output_csv.with_suffix(".json")
    )
    if output_csv.exists() or report_path.exists():
        raise FileExistsError(
            "Prediction CSV/report already exists; refusing to overwrite a "
            "checkpoint-bound artifact"
        )
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / args.checkpoint
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze_path = run_dir / MANIFEST_NAME
    freeze = validate_artifact_freeze(freeze_path)
    data_config = dict(config["data"])
    if data_config.get("pose_target"):
        raise ValueError(
            "This export is for PRIMARY t+1 future-offset models; use the "
            "terminal/endpose evaluator for a terminal target"
        )
    master_dir = (
        resolve(args.master_dir).resolve()
        if args.master_dir is not None
        else Path(data_config["master_dir"]).expanduser()
    )
    if not master_dir.is_absolute():
        master_dir = PROJECT_ROOT / master_dir
    data_config["master_dir"] = str(master_dir)
    bundle = prepare_data(data_config, seed=int(config["training"]["seed"]))
    dataset = getattr(bundle, args.split)
    frozen_eligibility = freeze.get("dataset", {}).get("window_eligibility", {})
    expected_endpoint_fingerprint = frozen_eligibility.get(
        "endpoint_fingerprints", {}
    ).get(args.split)
    expected_endpoint_count = frozen_eligibility.get("endpoint_counts", {}).get(
        args.split
    )
    if (
        dataset.endpoint_fingerprint() != expected_endpoint_fingerprint
        or len(dataset) != expected_endpoint_count
    ):
        raise ValueError("Export split differs from the frozen endpoint set")
    selected_sequences = set(args.sequence)
    if args.split == "test" and selected_sequences:
        raise ValueError(
            "Test prediction exports must cover the complete frozen split; "
            "--sequence is diagnostic-only for train/validation"
        )
    if selected_sequences:
        available = {record.sequence_id for record in dataset.records}
        missing = sorted(selected_sequences - available)
        if missing:
            raise ValueError(f"Sequences are not in {args.split}: {missing}")
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_hash = sha256_file(checkpoint_path)
    frozen_checkpoint = (
        freeze.get("output_artifacts", {})
        .get("checkpoints", {})
        .get("best_intention")
    )
    if args.checkpoint == "best_intention_model.pt" and (
        not isinstance(frozen_checkpoint, dict)
        or frozen_checkpoint.get("sha256") != checkpoint_hash
    ):
        raise ValueError("Prediction checkpoint differs from artifact freeze")
    final_test_binding = None
    if args.split == "test":
        if args.final_test_report is None:
            raise ValueError("--final-test-report is required for split=test")
        final_test_binding = validate_final_test_binding(
            resolve(args.final_test_report).resolve(),
            run_dir=run_dir,
            checkpoint_hash=checkpoint_hash,
            artifact_manifest_fingerprint=freeze["manifest_fingerprint"],
        )
    checkpoint_selection_metric = str(checkpoint.get("selection_metric", ""))
    if not checkpoint_selection_metric.startswith("validation_"):
        raise ValueError(
            "Learned-model evaluation requires a validation-selected checkpoint"
        )
    input_dim = len(bundle.normalizer.output_feature_names)
    if int(checkpoint["input_dim"]) != input_dim:
        raise ValueError("Checkpoint and data-loader input dimensions differ")
    current_schema_fingerprint = bundle.provenance["schema"]["fingerprint"]
    checkpoint_schema_fingerprint = checkpoint.get("feature_schema_fingerprint")
    if (
        checkpoint_schema_fingerprint is not None
        and checkpoint_schema_fingerprint != current_schema_fingerprint
    ):
        raise ValueError("Checkpoint and evaluation feature schemas differ")
    current_modality_fingerprint = bundle.split_metadata["modality_schema"][
        "fingerprint"
    ]
    checkpoint_modality_fingerprint = checkpoint.get(
        "modality_schema_fingerprint"
    )
    if (
        checkpoint_modality_fingerprint is not None
        and checkpoint_modality_fingerprint != current_modality_fingerprint
    ):
        raise ValueError("Checkpoint and evaluation modality schemas differ")
    checkpoint_dataset = checkpoint.get("dataset_provenance", {}).get(
        "dataset_content_fingerprint"
    )
    if (
        checkpoint_dataset is not None
        and checkpoint_dataset
        != bundle.provenance["dataset_content_fingerprint"]
    ):
        raise ValueError("Checkpoint and evaluation dataset fingerprints differ")
    model = build_residual_model(
        str(checkpoint.get("model_type", RESIDUAL_V2_MODEL_TYPE)),
        input_dim=input_dim,
        window_size=int(checkpoint["window_size"]),
        model_config=checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    context_cache = {
        record.sequence_id: sequence_context(
            master_dir, record.sequence_id, record.timestamps_ns
        )
        for record in dataset.records
        if not selected_sequences or record.sequence_id in selected_sequences
    }
    rows = []
    dataset_offset = 0
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            hand_reference = batch["hand_reference_pose"].to(device, non_blocking=True)
            outputs = model(features, hand_reference)
            hand_probabilities = F.softmax(outputs["receiving_hand_logits"], dim=-1)
            class_probabilities = intention_probabilities(outputs)
            predicted_intention = intention_predictions(outputs)
            predicted_assistance = assistance_predictions(outputs)
            predicted_assistance_type = assistance_type_predictions(outputs)
            predicted_hand = hand_probabilities.argmax(dim=-1)
            batch_indices = torch.arange(len(features), device=device)
            predicted_pose = outputs["pose_candidates"][batch_indices, predicted_hand]
            oracle_hand = batch["receiving_hand"].to(device).clamp(0, 1)
            oracle_pose = outputs["pose_candidates"][batch_indices, oracle_hand]

            for batch_index in range(len(features)):
                dataset_index = dataset_offset + batch_index
                record_index, endpoint = dataset.indices[dataset_index]
                record = dataset.records[record_index]
                sequence_id = record.sequence_id
                if selected_sequences and sequence_id not in selected_sequences:
                    continue
                target_intention_id = int(batch["intention"][batch_index])
                prediction_id = int(predicted_intention[batch_index])
                prediction_assistance_id = int(
                    predicted_assistance[batch_index]
                )
                prediction_assistance_type_id = int(
                    predicted_assistance_type[batch_index]
                )
                gt_hand_id = int(batch["receiving_hand"][batch_index])
                pred_hand_id = int(predicted_hand[batch_index])
                target_pose_valid = bool(batch["pose_valid"][batch_index])
                learned_oracle_available = bool(
                    batch["residual_pose_valid"][batch_index]
                )
                predicted_reference_valid = bool(
                    batch["hand_reference_valid"][batch_index, pred_hand_id]
                )
                learned_end_to_end_available = (
                    target_pose_valid and predicted_reference_valid
                )
                target_pose = batch["pose_target"][batch_index].numpy()
                predicted_pose_np = predicted_pose[batch_index].cpu().numpy()
                oracle_pose_np = oracle_pose[batch_index].cpu().numpy()
                endpoint_timestamp_ns = int(record.timestamps_ns[endpoint])
                sample_key = f"{args.split}|{sequence_id}|{endpoint_timestamp_ns}"
                context = context_cache[sequence_id]
                side = (
                    RECEIVING_HAND_NAMES[gt_hand_id]
                    if gt_hand_id in (0, 1)
                    else ""
                )
                target_object_value = (
                    pd.to_numeric(
                        pd.Series([context.iloc[endpoint]["target_object_id"]]),
                        errors="coerce",
                    ).iloc[0]
                    if "target_object_id" in context
                    else np.nan
                )
                target_object_id = (
                    int(target_object_value)
                    if pd.notna(target_object_value) and int(target_object_value) >= 0
                    else None
                )
                persistence = BaselineEstimate(
                    None, False, "invalid_future_pose_target"
                )
                velocity = BaselineEstimate(
                    None, False, "invalid_future_pose_target"
                )
                target_timing = None
                if target_pose_valid and side:
                    target_timing = resolve_target_timing(
                        context,
                        endpoint,
                        horizon_seconds=float(data_config["future_horizon_seconds"]),
                    )
                    observations = extract_hand_observations(
                        context,
                        endpoint - dataset.window_size + 1,
                        endpoint,
                        side,
                        source_timestamp_column="hand_timestamp_ns",
                        require_causal=True,
                    )
                    persistence = persistence_pose(
                        observations,
                        endpoint_timestamp_ns,
                        maximum_age_seconds=(
                            args.maximum_observation_age_seconds
                        ),
                    )
                    velocity = constant_velocity_pose(
                        observations,
                        endpoint_timestamp_ns,
                        target_timing.prediction_horizon_timestamp_ns,
                        maximum_age_seconds=(
                            args.maximum_observation_age_seconds
                        ),
                        lookback_seconds=args.velocity_lookback_seconds,
                        minimum_fit_span_seconds=(
                            args.minimum_velocity_fit_span_seconds
                        ),
                    )
                fair_common = bool(
                    target_pose_valid
                    and learned_oracle_available
                    and persistence.available
                    and velocity.available
                )
                row = {
                    "sample_key": sample_key,
                    "split": args.split,
                    "dataset_index": dataset_index,
                    "sequence_id": sequence_id,
                    "participant": record.participant,
                    "sequence_receiving_hand": side,
                    "target_object_id": target_object_id,
                    "endpoint_row": endpoint,
                    "endpoint_timestamp_ns": endpoint_timestamp_ns,
                    "video_time_s": float(context.iloc[endpoint]["time_since_start_s"]),
                    "target_intention_id": target_intention_id,
                    "target_intention": INTENTION_NAMES[target_intention_id],
                    "predicted_intention_id": prediction_id,
                    "predicted_intention": INTENTION_NAMES[prediction_id],
                    "intention_correct": target_intention_id == prediction_id,
                    "target_assistance_id": int(target_intention_id != 0),
                    "predicted_assistance_id": prediction_assistance_id,
                    "target_assistance_type_id": (
                        target_intention_id - 1
                        if target_intention_id in (1, 2)
                        else None
                    ),
                    "predicted_assistance_type_id": (
                        prediction_assistance_type_id
                    ),
                    "predicted_assistance_type": (
                        ("fetch", "handover")[prediction_assistance_type_id]
                    ),
                    "continue_probability": float(class_probabilities[batch_index, 0]),
                    "fetch_probability": float(class_probabilities[batch_index, 1]),
                    "handover_probability": float(class_probabilities[batch_index, 2]),
                    "assistance_probability": float(
                        class_probabilities[batch_index, 1:].sum()
                    ),
                    "fetch_given_assistance_probability": float(
                        class_probabilities[batch_index, 1]
                        / class_probabilities[batch_index, 1:].sum().clamp_min(1e-12)
                    ),
                    "handover_given_assistance_probability": float(
                        class_probabilities[batch_index, 2]
                        / class_probabilities[batch_index, 1:].sum().clamp_min(1e-12)
                    ),
                    "target_receiving_hand": (
                        RECEIVING_HAND_NAMES[gt_hand_id]
                        if target_intention_id == 2 and gt_hand_id in (0, 1)
                        else ""
                    ),
                    "predicted_receiving_hand": RECEIVING_HAND_NAMES[pred_hand_id],
                    "predicted_receiving_hand_probability": float(
                        hand_probabilities[batch_index, pred_hand_id]
                    ),
                    "left_hand_probability": float(hand_probabilities[batch_index, 0]),
                    "right_hand_probability": float(hand_probabilities[batch_index, 1]),
                    "pose_valid": target_pose_valid,
                    "learned_oracle_available": learned_oracle_available,
                    "learned_end_to_end_available": learned_end_to_end_available,
                    "predicted_hand_reference_valid": predicted_reference_valid,
                    "fair_common": fair_common,
                    "nominal_target_timestamp_ns": (
                        target_timing.nominal_timestamp_ns
                        if target_timing is not None
                        else None
                    ),
                    "aligned_target_master_timestamp_ns": (
                        target_timing.aligned_master_timestamp_ns
                        if target_timing is not None
                        else None
                    ),
                    "actual_target_hand_capture_timestamp_ns": (
                        target_timing.actual_hand_capture_timestamp_ns
                        if target_timing is not None
                        else None
                    ),
                    "target_actual_capture_error_ms": (
                        target_timing.actual_capture_error_ms
                        if target_timing is not None
                        else None
                    ),
                    "persistence_available": persistence.available,
                    "persistence_reason": persistence.reason,
                    "persistence_observation_age_seconds": (
                        persistence.observation_age_seconds
                    ),
                    "constant_velocity_available": velocity.available,
                    "constant_velocity_reason": velocity.reason,
                    "constant_velocity_observation_age_seconds": (
                        velocity.observation_age_seconds
                    ),
                    "constant_velocity_fit_samples": velocity.fit_samples,
                    "constant_velocity_fit_span_seconds": velocity.fit_span_seconds,
                    "constant_velocity_estimated_speed_m_s": (
                        velocity.estimated_speed_m_s
                    ),
                    "fusion_mode": str(
                        getattr(model, "fusion_mode", "temporal_channel_gated")
                    ),
                    "fusion_temporal_weight": float(
                        outputs["fusion_weights"][batch_index, 0]
                    ),
                    "fusion_channel_weight": (
                        float(outputs["fusion_weights"][batch_index, 1])
                        if getattr(model, "fusion_mode", "")
                        != "modality_gated"
                        else None
                    ),
                    "fusion_modality_context_weight": (
                        float(outputs["fusion_weights"][batch_index, 1])
                        if getattr(model, "fusion_mode", "")
                        == "modality_gated"
                        else None
                    ),
                    "gate_temporal": float(
                        outputs["fusion_weights"][batch_index, 0]
                    ),
                    "gate_channel": (
                        float(outputs["fusion_weights"][batch_index, 1])
                        if getattr(model, "fusion_mode", "")
                        != "modality_gated"
                        else None
                    ),
                }
                for modality_index, modality_name in enumerate(
                    getattr(model, "modality_names", ())
                ):
                    row[f"modality_{modality_name}_weight"] = float(
                        outputs["modality_weights"][
                            batch_index, modality_index
                        ]
                    )
                    row[f"modality_{modality_name}_available"] = bool(
                        outputs["modality_available"][
                            batch_index, modality_index
                        ]
                    )
                for component, value in zip(POSE_COMPONENTS, predicted_pose_np):
                    row[f"predicted_{component}"] = float(value)
                    row[f"learned_end_to_end_{component}"] = float(value)
                for component, value in zip(POSE_COMPONENTS, oracle_pose_np):
                    row[f"oracle_{component}"] = float(value)
                    row[f"learned_oracle_{component}"] = float(value)
                for baseline_name, estimate in (
                    ("persistence", persistence),
                    ("constant_velocity", velocity),
                ):
                    for component in POSE_COMPONENTS:
                        row[f"{baseline_name}_{component}"] = None
                    if estimate.available:
                        assert estimate.pose is not None
                        for component, value in zip(POSE_COMPONENTS, estimate.pose):
                            row[f"{baseline_name}_{component}"] = float(value)
                if target_pose_valid:
                    for component, value in zip(POSE_COMPONENTS, target_pose):
                        row[f"target_{component}"] = float(value)
                    learned_end_to_end_error = (
                        single_pose_errors(predicted_pose_np, target_pose)
                        if learned_end_to_end_available
                        else (None, None)
                    )
                    learned_oracle_error = (
                        single_pose_errors(oracle_pose_np, target_pose)
                        if learned_oracle_available
                        else (None, None)
                    )
                    row["predicted_position_error_cm"] = learned_end_to_end_error[0]
                    row["predicted_orientation_error_deg"] = (
                        learned_end_to_end_error[1]
                    )
                    row["oracle_position_error_cm"] = learned_oracle_error[0]
                    row["oracle_orientation_error_deg"] = learned_oracle_error[1]
                    for baseline_name, estimate in (
                        ("persistence", persistence),
                        ("constant_velocity", velocity),
                    ):
                        error = (
                            single_pose_errors(estimate.pose, target_pose)
                            if estimate.available and estimate.pose is not None
                            else (None, None)
                        )
                        row[f"{baseline_name}_position_error_cm"] = error[0]
                        row[f"{baseline_name}_orientation_error_deg"] = error[1]
                else:
                    for component in POSE_COMPONENTS:
                        row[f"target_{component}"] = None
                    row["predicted_position_error_cm"] = None
                    row["predicted_orientation_error_deg"] = None
                    row["oracle_position_error_cm"] = None
                    row["oracle_orientation_error_deg"] = None
                    row["persistence_position_error_cm"] = None
                    row["persistence_orientation_error_deg"] = None
                    row["constant_velocity_position_error_cm"] = None
                    row["constant_velocity_orientation_error_deg"] = None
                rows.append(row)
            dataset_offset += len(features)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    sequence_summary = {}
    for sequence_id, group in frame.groupby("sequence_id"):
        valid_pose = group.loc[group["pose_valid"]]
        learned_pose = valid_pose.loc[valid_pose["learned_oracle_available"]]
        sequence_summary[sequence_id] = {
            "windows": int(len(group)),
            "intention_accuracy": float(group["intention_correct"].mean()),
            "pose_windows": int(len(valid_pose)),
            "learned_oracle_pose_mean_error_cm": (
                float(learned_pose["oracle_position_error_cm"].mean())
                if len(learned_pose)
                else None
            ),
            "first_video_time_s": float(group["video_time_s"].min()),
            "last_video_time_s": float(group["video_time_s"].max()),
        }

    valid_target_count = int(frame["pose_valid"].sum())
    fair_frame = frame.loc[frame["fair_common"]].copy()
    fair_keys = fair_frame["sample_key"].astype(str).tolist()
    methods = {
        "persistence": ("persistence", "persistence_available"),
        "constant_velocity": (
            "constant_velocity",
            "constant_velocity_available",
        ),
        "learned_model_oracle_hand": (
            "learned_oracle",
            "learned_oracle_available",
        ),
    }
    pose_comparison = {}
    for method_name, (prefix, availability_column) in methods.items():
        native = frame.loc[frame[availability_column] & frame["pose_valid"]]
        pose_comparison[method_name] = {
            "native_metrics": pose_metric_summary(
                pose_values(native, prefix),
                pose_values(native, "target"),
                coverage_denominator=valid_target_count,
            ),
            "fair_common_metrics": pose_metric_summary(
                pose_values(fair_frame, prefix),
                pose_values(fair_frame, "target"),
                coverage_denominator=valid_target_count,
            ),
            "native_sample_key_fingerprint": sample_key_fingerprint(
                native["sample_key"].astype(str).tolist()
            ),
        }
    end_to_end = frame.loc[
        frame["learned_end_to_end_available"] & frame["pose_valid"]
    ]
    future_pose_loss_enabled = (
        float(config["training"].get("pose_loss_weight", 0.0)) > 0.0
    )
    full_split_export = not selected_sequences and len(rows) == len(dataset)
    result_role = (
        "primary_validation_selected_checkpoint"
        if (
            checkpoint_selection_metric == "validation_intention_macro_f1"
            and future_pose_loss_enabled
            and full_split_export
            and (args.split != "test" or final_test_binding is not None)
        )
        else "oracle_pose_selected_diagnostic"
    )
    visual_provenance = bundle.provenance.get("schema", {}).get(
        "visual_features", {"enabled": False}
    )
    exported_endpoint_payload = "\n".join(
        f"{row['sequence_id']}:{int(row['endpoint_timestamp_ns'])}"
        for row in rows
    )
    report = {
        "schema_version": 3,
        "report_fingerprint": None,
        "task": {
            "name": "primary_t_plus_1_future_receiving_wrist_pose",
            "future_horizon_seconds": float(data_config["future_horizon_seconds"]),
            "terminal_endpose": False,
        },
        "result_role": result_role,
        "predictions_csv": str(output_csv),
        "predictions_csv_sha256": sha256_file(output_csv),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_selection_value": float(checkpoint["selection_value"]),
        "architecture": {
            "fusion_mode": str(
                getattr(model, "fusion_mode", "temporal_channel_gated")
            ),
            "intention_head_mode": str(
                getattr(model, "intention_head_mode", "hierarchical")
            ),
            "modality_names": list(getattr(model, "modality_names", ())),
            "modality_schema_fingerprint": current_modality_fingerprint,
        },
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "source_content_fingerprint": bundle.provenance[
            "source_content_fingerprint"
        ],
        "artifact_freeze": {
            "manifest": str(freeze_path.resolve()),
            "manifest_sha256": sha256_file(freeze_path),
            "manifest_fingerprint": freeze["manifest_fingerprint"],
            "protocol": freeze["protocol"],
        },
        "final_test_authorization": final_test_binding,
        "visual_artifacts": visual_provenance,
        "future_pose_loss_enabled": future_pose_loss_enabled,
        "split": args.split,
        "full_split_export": full_split_export,
        "frozen_split_endpoint_fingerprint": expected_endpoint_fingerprint,
        "frozen_split_endpoint_count": expected_endpoint_count,
        "exported_endpoint_fingerprint": hashlib.sha256(
            exported_endpoint_payload.encode("utf-8")
        ).hexdigest(),
        "exported_endpoint_count": len(rows),
        "sequence_filter": sorted(selected_sequences),
        "device": str(device),
        "rows": len(rows),
        "sequences": sequence_summary,
        "pose_comparison": {
            "receiving_hand_context": (
                "ground-truth receiving hand for persistence, constant velocity, "
                "and learned candidate selection; the learned residual itself "
                "remains conditioned on predicted-hand probabilities"
            ),
            "primary_sample_set": "fair_common",
            "valid_target_denominator": valid_target_count,
            "fair_common_samples": int(len(fair_frame)),
            "fair_common_coverage": (
                float(len(fair_frame) / valid_target_count)
                if valid_target_count
                else None
            ),
            "fair_common_sample_key_fingerprint": sample_key_fingerprint(
                fair_keys
            ),
            "methods": pose_comparison,
        },
        "learned_end_to_end_diagnostic": pose_metric_summary(
            pose_values(end_to_end, "learned_end_to_end"),
            pose_values(end_to_end, "target"),
            coverage_denominator=valid_target_count,
        ),
        "baseline_policy": {
            "fallbacks": "none",
            "maximum_observation_age_seconds": (
                args.maximum_observation_age_seconds
            ),
            "velocity_lookback_seconds": args.velocity_lookback_seconds,
            "minimum_velocity_fit_span_seconds": (
                args.minimum_velocity_fit_span_seconds
            ),
            "timestamp_basis": "hand_timestamp_ns source captures",
            "constant_velocity_prediction_horizon": (
                "endpoint timestamp_ns + future_horizon_seconds; the future "
                "target capture timestamp is evaluation metadata only"
            ),
            "constant_velocity_orientation": (
                "zero angular velocity; latest valid quaternion is held"
            ),
        },
        "probability_definition": (
            "softmax over continue/fetch/handover logits"
            if getattr(model, "intention_head_mode", "hierarchical") == "flat"
            else (
                "continue=P(no assistance), fetch=P(assistance)*P(fetch|assistance), "
                "handover=P(assistance)*P(handover|assistance)"
            )
        ),
        "modality_weight_interpretation": (
            "per-window learned internal conditioning weights; not causal "
            "modality contributions"
        ),
        "timestamp_alignment": (
            "pose baselines use absolute device timestamps; video_time_s is only "
            "the master START-relative display time and is not CLIP alignment"
        ),
    }
    report["report_fingerprint"] = canonical_json_hash(report)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Device: {device}; rows: {len(rows)}; sequences: {len(sequence_summary)}")
    print(f"Predictions: {output_csv}")
    print(f"Report:      {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
