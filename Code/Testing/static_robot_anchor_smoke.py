#!/usr/bin/env python3
"""Verify static robot-anchor propagation through a temporary Tag 0 occlusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from build_master_dataset import add_coordinate_transforms  # noqa: E402


def main() -> int:
    rows = 3
    data: dict[str, list[float]] = {
        "slam_tx_world_device": [0.0, 1.0, 2.0],
        "slam_ty_world_device": [0.0] * rows,
        "slam_tz_world_device": [0.0] * rows,
        "slam_qx_world_device": [0.0] * rows,
        "slam_qy_world_device": [0.0] * rows,
        "slam_qz_world_device": [0.0] * rows,
        "slam_qw_world_device": [1.0] * rows,
        "apriltag_0_tx_camera_m": [10.0, np.nan, 8.0],
        "apriltag_0_ty_camera_m": [0.0] * rows,
        "apriltag_0_tz_camera_m": [0.0] * rows,
        "apriltag_0_qx_camera_marker": [0.0, np.nan, 0.0],
        "apriltag_0_qy_camera_marker": [0.0, np.nan, 0.0],
        "apriltag_0_qz_camera_marker": [0.0, np.nan, 0.0],
        "apriltag_0_qw_camera_marker": [1.0, np.nan, 1.0],
        "gaze_origin_device_x_m": [0.0] * rows,
        "gaze_origin_device_y_m": [0.0] * rows,
        "gaze_origin_device_z_m": [0.0] * rows,
        "gaze_direction_device_x": [1.0] * rows,
        "gaze_direction_device_y": [0.0] * rows,
        "gaze_direction_device_z": [0.0] * rows,
    }
    for side in ("left", "right"):
        data[f"hand_tx_{side}_device_wrist"] = [1.0] * rows
        data[f"hand_ty_{side}_device_wrist"] = [0.0] * rows
        data[f"hand_tz_{side}_device_wrist"] = [0.0] * rows
        data[f"hand_qx_{side}_device_wrist"] = [0.0] * rows
        data[f"hand_qy_{side}_device_wrist"] = [0.0] * rows
        data[f"hand_qz_{side}_device_wrist"] = [0.0] * rows
        data[f"hand_qw_{side}_device_wrist"] = [1.0] * rows

    transformed = add_coordinate_transforms(
        pd.DataFrame(data),
        ["apriltag_0"],
        np.eye(4, dtype=np.float64),
    )
    np.testing.assert_allclose(
        transformed["right_wrist_robot_x_m"].to_numpy(),
        [-9.0, -8.0, -7.0],
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        transformed["robot_frame_valid"].to_numpy(), [1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(
        transformed["robot_anchor_interpolated"].to_numpy(), [0.0, 1.0, 0.0]
    )
    assert int(transformed["robot_static_anchor_samples"].iloc[0]) == 2
    print("Static robot anchor smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
