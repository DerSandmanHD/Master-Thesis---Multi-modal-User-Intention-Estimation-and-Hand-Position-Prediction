from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_master_dataset import (  # noqa: E402
    OBSERVATION_ALIGNMENT_VERSION,
    causal_observation_merge,
)

TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from data import load_record, select_feature_columns  # noqa: E402
from smoke_test import synthetic_sequence  # noqa: E402


def test_observation_alignment_never_uses_future_capture() -> None:
    timeline = pd.DataFrame({"timestamp_ns": [100, 200, 300]})
    source = pd.DataFrame(
        {
            "hand_timestamp_ns": [90, 105, 205, 400],
            "value": [9.0, 10.5, 20.5, 40.0],
        }
    )
    merged = causal_observation_merge(
        timeline, source, "hand_timestamp_ns", tolerance_ms=0.0001
    )
    assert merged["hand_timestamp_ns"].tolist() == [90, 105, 205]
    offsets = merged["hand_timestamp_ns"] - merged["timestamp_ns"]
    assert bool((offsets <= 0).all())


def test_observation_alignment_does_not_backfill_before_first_capture() -> None:
    timeline = pd.DataFrame({"timestamp_ns": [50, 100]})
    source = pd.DataFrame(
        {"slam_timestamp_ns": [90], "value": [1.0]}
    )
    merged = causal_observation_merge(
        timeline, source, "slam_timestamp_ns", tolerance_ms=0.0001
    )
    assert np.isnan(merged.loc[0, "slam_timestamp_ns"])
    assert merged.loc[1, "slam_timestamp_ns"] == 90


def test_causal_alignment_version_is_explicit() -> None:
    assert OBSERVATION_ALIGNMENT_VERSION == "causal_backward_device_time_v1"


def test_training_rejects_master_without_or_with_stale_alignment_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "P1_0_master.csv"
    synthetic_sequence(path, "P1", 0)
    frame = pd.read_csv(path).drop(columns=["observation_alignment_version"])
    frame.to_csv(path, index=False)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    features = select_feature_columns(header, "multimodal_robot_frame_v1")
    kwargs = {
        "future_horizon_seconds": 1.0,
        "required_observation_alignment_version": OBSERVATION_ALIGNMENT_VERSION,
    }
    with pytest.raises(ValueError, match="observation_alignment_version"):
        load_record(path, features, **kwargs)

    frame = pd.read_csv(path)
    frame["observation_alignment_version"] = OBSERVATION_ALIGNMENT_VERSION
    frame.to_csv(path, index=False)
    loaded = load_record(path, features, **kwargs)
    assert loaded.sequence_id == "P1_0"

    frame["observation_alignment_version"] = "nearest_legacy_v0"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="required .*causal_backward"):
        load_record(path, features, **kwargs)
