#!/usr/bin/env python3
"""Explain why future receiving-hand targets are invalid per training window."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import INTENTION_TO_ID, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/hierarchical_baseline_v1.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("Training/reports/pose_target_audit.json"),
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=Path("Training/reports/pose_target_audit.csv"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def truthy(value) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def finite_row(row: pd.Series, columns: list[str]) -> bool:
    if any(column not in row.index for column in columns):
        return False
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def target_reason(
    frame: pd.DataFrame,
    endpoint: int,
    horizon_seconds: float,
    timestamp_to_index: dict[int, int],
) -> tuple[str, int | None]:
    row = frame.iloc[endpoint]
    prefix = f"future_{horizon_seconds:g}s_"
    receiving_hand = str(row.get("receiving_hand", "")).strip().lower()
    if receiving_hand not in {"left", "right"}:
        return "unknown_receiving_hand", None

    desired_timestamp = int(row["timestamp_ns"]) + int(horizon_seconds * 1e9)
    future_timestamp = pd.to_numeric(
        pd.Series([row.get("future_target_timestamp_ns")]), errors="coerce"
    ).iloc[0]
    if pd.isna(future_timestamp):
        recording_end = int(frame["timestamp_ns"].iloc[-1])
        reason = (
            "future_after_recording_end"
            if desired_timestamp > recording_end
            else "future_timeline_unmatched"
        )
        return reason, None

    future_timestamp = int(future_timestamp)
    target_index = timestamp_to_index.get(future_timestamp)
    if target_index is None:
        return "future_target_row_missing", future_timestamp
    target_row = frame.iloc[target_index]

    hand_valid = truthy(target_row.get(f"hand_{receiving_hand}_valid"))
    robot_frame_valid = truthy(target_row.get("robot_frame_valid"))
    if not hand_valid and not robot_frame_valid:
        return "future_hand_and_robot_frame_invalid", future_timestamp
    if not hand_valid:
        return "future_hand_tracking_invalid", future_timestamp
    if not robot_frame_valid:
        slam_columns = [
            "slam_tx_world_device",
            "slam_ty_world_device",
            "slam_tz_world_device",
            "slam_qx_world_device",
            "slam_qy_world_device",
            "slam_qz_world_device",
            "slam_qw_world_device",
        ]
        if not finite_row(target_row, slam_columns):
            return "future_slam_pose_invalid", future_timestamp
        return "future_robot_frame_invalid", future_timestamp

    pose_columns = [
        *(f"{prefix}receiving_wrist_robot_{axis}_m" for axis in "xyz"),
        *(f"{prefix}receiving_wrist_robot_q{component}" for component in "xyzw"),
    ]
    if not finite_row(row, pose_columns):
        return "future_transformed_pose_incomplete", future_timestamp
    quaternion = pd.to_numeric(row[pose_columns[3:7]], errors="coerce").to_numpy(dtype=float)
    if float(np.linalg.norm(quaternion)) <= 1e-6:
        return "future_quaternion_invalid", future_timestamp
    if not truthy(row.get(f"{prefix}receiving_wrist_valid")):
        return "future_validity_flag_false", future_timestamp
    return "valid", future_timestamp


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    master_dir = project_path(Path(data_config["master_dir"]))
    data_config["master_dir"] = str(master_dir)
    seed = int(config["training"]["seed"])
    bundle = prepare_data(data_config, seed)

    frame_cache: dict[str, pd.DataFrame] = {}
    timestamp_indices: dict[str, dict[int, int]] = {}
    detail_rows = []
    split_counts: dict[str, Counter] = {}
    participant_counts: dict[str, Counter] = defaultdict(Counter)
    sequence_counts: dict[str, Counter] = defaultdict(Counter)
    handover_id = INTENTION_TO_ID["handover"]
    horizon_seconds = float(data_config["future_horizon_seconds"])

    for split_name, dataset in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        counts = Counter()
        for record_index, endpoint in dataset.indices:
            record = dataset.records[record_index]
            if int(record.intentions[endpoint]) != handover_id:
                continue
            sequence_id = record.sequence_id
            if sequence_id not in frame_cache:
                path = master_dir / f"{sequence_id}_master.csv"
                frame = pd.read_csv(path, low_memory=False)
                frame_cache[sequence_id] = frame
                timestamp_indices[sequence_id] = {
                    int(timestamp): index
                    for index, timestamp in enumerate(frame["timestamp_ns"])
                }
            frame = frame_cache[sequence_id]
            if int(frame.iloc[endpoint]["timestamp_ns"]) != int(record.timestamps_ns[endpoint]):
                raise ValueError(f"Row alignment mismatch for {sequence_id} at {endpoint}")
            reason, future_timestamp = target_reason(
                frame,
                endpoint,
                horizon_seconds,
                timestamp_indices[sequence_id],
            )
            training_valid = bool(record.pose_valid[endpoint])
            if training_valid != (reason == "valid"):
                raise ValueError(
                    f"Pose-validity audit mismatch for {sequence_id} at row {endpoint}: "
                    f"training_valid={training_valid}, reason={reason}"
                )
            counts[reason] += 1
            participant_counts[record.participant][reason] += 1
            sequence_counts[sequence_id][reason] += 1
            detail_rows.append(
                {
                    "split": split_name,
                    "participant": record.participant,
                    "sequence_id": sequence_id,
                    "endpoint_row": endpoint,
                    "endpoint_timestamp_ns": int(record.timestamps_ns[endpoint]),
                    "future_target_timestamp_ns": future_timestamp,
                    "receiving_hand": str(frame.iloc[endpoint].get("receiving_hand", "")),
                    "reason": reason,
                }
            )
        split_counts[split_name] = counts

    def summary(counts: Counter) -> dict:
        total = sum(counts.values())
        valid = counts.get("valid", 0)
        return {
            "handover_windows": total,
            "valid": valid,
            "invalid": total - valid,
            "valid_ratio": round(valid / total, 4) if total else None,
            "reason_counts": dict(sorted(counts.items())),
        }

    report = {
        "config": str(config_path),
        "master_dir": str(master_dir),
        "future_horizon_seconds": horizon_seconds,
        "splits": {name: summary(counts) for name, counts in split_counts.items()},
        "participants": {
            name: summary(counts) for name, counts in sorted(participant_counts.items())
        },
        "sequences": {
            name: summary(counts) for name, counts in sorted(sequence_counts.items())
        },
    }

    report_path = project_path(args.report_out)
    details_path = project_path(args.details_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False)

    for split_name in ("train", "validation", "test"):
        values = report["splits"][split_name]
        print(
            f"{split_name}: handover={values['handover_windows']}, "
            f"valid={values['valid']}, invalid={values['invalid']}, "
            f"valid_ratio={values['valid_ratio']}"
        )
        for reason, count in values["reason_counts"].items():
            print(f"  {reason}: {count}")
    print(f"Report:  {report_path}")
    print(f"Details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
