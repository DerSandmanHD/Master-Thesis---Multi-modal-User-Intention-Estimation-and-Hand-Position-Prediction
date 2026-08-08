#!/usr/bin/env python3
"""Export window-level residual-v2 probabilities, hands, and future poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import INTENTION_NAMES, RECEIVING_HAND_NAMES, prepare_data, sha256_file
from model import HierarchicalResidualPoseTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_COMPONENTS = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
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


def sequence_elapsed(master_dir: Path, sequence_id: str) -> np.ndarray:
    frame = pd.read_csv(
        master_dir / f"{sequence_id}_master.csv",
        usecols=["timestamp_ns", "time_since_start_s"],
    )
    timestamps = frame["timestamp_ns"].to_numpy(np.int64)
    elapsed = frame["time_since_start_s"].to_numpy(np.float64)
    if np.any(np.diff(timestamps) <= 0) or np.any(np.diff(elapsed) < 0):
        raise ValueError(f"Master timeline is invalid for {sequence_id}")
    return elapsed


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir).resolve()
    output_csv = resolve(args.output_csv).resolve()
    report_path = (
        resolve(args.report_out).resolve()
        if args.report_out
        else output_csv.with_suffix(".json")
    )
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / args.checkpoint
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = dict(config["data"])
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
    selected_sequences = set(args.sequence)
    if selected_sequences:
        available = {record.sequence_id for record in dataset.records}
        missing = sorted(selected_sequences - available)
        if missing:
            raise ValueError(f"Sequences are not in {args.split}: {missing}")
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = len(bundle.normalizer.output_feature_names)
    if int(checkpoint["input_dim"]) != input_dim:
        raise ValueError("Checkpoint and data-loader input dimensions differ")
    model = HierarchicalResidualPoseTransformer(
        input_dim=input_dim,
        window_size=int(checkpoint["window_size"]),
        **checkpoint["model_config"],
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

    elapsed_cache = {
        record.sequence_id: sequence_elapsed(master_dir, record.sequence_id)
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
            assistance = F.softmax(outputs["assistance_logits"], dim=-1)
            assistance_type = F.softmax(outputs["assistance_type_logits"], dim=-1)
            hand_probabilities = F.softmax(outputs["receiving_hand_logits"], dim=-1)
            class_probabilities = torch.stack(
                (
                    assistance[:, 0],
                    assistance[:, 1] * assistance_type[:, 0],
                    assistance[:, 1] * assistance_type[:, 1],
                ),
                dim=-1,
            )
            assistance_choice = assistance.argmax(dim=-1)
            type_choice = assistance_type.argmax(dim=-1)
            predicted_intention = torch.where(
                assistance_choice == 0,
                torch.zeros_like(type_choice),
                type_choice + 1,
            )
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
                gt_hand_id = int(batch["receiving_hand"][batch_index])
                pred_hand_id = int(predicted_hand[batch_index])
                pose_valid = bool(batch["residual_pose_valid"][batch_index])
                target_pose = batch["pose_target"][batch_index].numpy()
                predicted_pose_np = predicted_pose[batch_index].cpu().numpy()
                oracle_pose_np = oracle_pose[batch_index].cpu().numpy()
                row = {
                    "split": args.split,
                    "dataset_index": dataset_index,
                    "sequence_id": sequence_id,
                    "participant": record.participant,
                    "endpoint_row": endpoint,
                    "endpoint_timestamp_ns": int(record.timestamps_ns[endpoint]),
                    "video_time_s": float(elapsed_cache[sequence_id][endpoint]),
                    "target_intention_id": target_intention_id,
                    "target_intention": INTENTION_NAMES[target_intention_id],
                    "predicted_intention_id": prediction_id,
                    "predicted_intention": INTENTION_NAMES[prediction_id],
                    "intention_correct": target_intention_id == prediction_id,
                    "continue_probability": float(class_probabilities[batch_index, 0]),
                    "fetch_probability": float(class_probabilities[batch_index, 1]),
                    "handover_probability": float(class_probabilities[batch_index, 2]),
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
                    "pose_valid": pose_valid,
                    "predicted_hand_reference_valid": bool(
                        batch["hand_reference_valid"][batch_index, pred_hand_id]
                    ),
                    "gate_temporal": float(outputs["gate"][batch_index, 0]),
                    "gate_channel": float(outputs["gate"][batch_index, 1]),
                }
                for component, value in zip(POSE_COMPONENTS, predicted_pose_np):
                    row[f"predicted_{component}"] = float(value)
                for component, value in zip(POSE_COMPONENTS, oracle_pose_np):
                    row[f"oracle_{component}"] = float(value)
                if pose_valid:
                    for component, value in zip(POSE_COMPONENTS, target_pose):
                        row[f"target_{component}"] = float(value)
                    row["predicted_position_error_cm"] = float(
                        np.linalg.norm(predicted_pose_np[:3] - target_pose[:3]) * 100.0
                    )
                    row["oracle_position_error_cm"] = float(
                        np.linalg.norm(oracle_pose_np[:3] - target_pose[:3]) * 100.0
                    )
                else:
                    for component in POSE_COMPONENTS:
                        row[f"target_{component}"] = None
                    row["predicted_position_error_cm"] = None
                    row["oracle_position_error_cm"] = None
                rows.append(row)
            dataset_offset += len(features)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    sequence_summary = {}
    frame = pd.DataFrame(rows)
    for sequence_id, group in frame.groupby("sequence_id"):
        valid_pose = group.loc[group["pose_valid"]]
        sequence_summary[sequence_id] = {
            "windows": int(len(group)),
            "intention_accuracy": float(group["intention_correct"].mean()),
            "pose_windows": int(len(valid_pose)),
            "predicted_pose_mae_cm": (
                float(valid_pose["predicted_position_error_cm"].mean())
                if len(valid_pose)
                else None
            ),
            "first_video_time_s": float(group["video_time_s"].min()),
            "last_video_time_s": float(group["video_time_s"].max()),
        }
    report = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "split": args.split,
        "device": str(device),
        "rows": len(rows),
        "sequences": sequence_summary,
        "probability_definition": (
            "continue=P(no assistance), fetch=P(assistance)*P(fetch|assistance), "
            "handover=P(assistance)*P(handover|assistance)"
        ),
        "timestamp_alignment": "master endpoint time_since_start_s to MP4 frame time",
    }
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
