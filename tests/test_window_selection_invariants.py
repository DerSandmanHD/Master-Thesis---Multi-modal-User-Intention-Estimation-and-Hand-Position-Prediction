from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import pytest

from data import SequenceRecord, WindowDataset, split_records  # noqa: E402


def record(features: np.ndarray) -> SequenceRecord:
    rows = len(features)
    return SequenceRecord(
        sequence_id="P1_0",
        participant="P1",
        timestamps_ns=np.arange(rows, dtype=np.int64) * 100_000_000,
        features=features.astype(np.float32),
        intentions=np.full(rows, 2, dtype=np.int64),
        pose_targets=np.tile(
            np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            (rows, 1),
        ),
        pose_valid=np.ones(rows, dtype=bool),
        eligibility_observed=np.asarray(
            [
                [True, True],
                [True, False],
                [True, True],
                [False, False],
                [True, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )


def dataset(features: np.ndarray) -> WindowDataset:
    return WindowDataset(
        [record(features)],
        window_size=2,
        stride=1,
        pose_intent_ids=[2],
        minimum_observed_fraction=0.5,
        max_timestamp_gap_seconds=1.0,
    )


def test_ablation_feature_masks_cannot_change_window_sample_set() -> None:
    # The model-level observation masks intentionally disagree completely.
    # Eligibility must nevertheless come from the common full-sensor mask.
    full = dataset(np.asarray([[1, 2, 1, 1]] * 6, dtype=np.float32))
    ablated = dataset(np.asarray([[9, 0]] * 6, dtype=np.float32))
    assert full.indices == ablated.indices
    assert full.endpoint_fingerprint() == ablated.endpoint_fingerprint()
    assert full.discarded_observation_windows == (
        ablated.discarded_observation_windows
    )


def test_complete_explicit_split_rejects_dataset_participant_drift() -> None:
    records = []
    for participant in ("P1", "P2", "P3", "P4"):
        value = record(np.ones((6, 2), dtype=np.float32))
        value.sequence_id = f"{participant}_0"
        value.participant = participant
        records.append(value)
    with pytest.raises(ValueError, match="does not match the selected dataset"):
        split_records(
            records,
            seed=42,
            validation_fraction=0.2,
            test_fraction=0.2,
            train_participants=["P1"],
            validation_participants=["P2"],
            test_participants=["P3"],
        )

    split, metadata = split_records(
        records,
        seed=42,
        validation_fraction=0.2,
        test_fraction=0.2,
        train_participants=["P1", "P4"],
        validation_participants=["P2"],
        test_participants=["P3"],
    )
    assert metadata["strategy"] == "explicit_complete_participants"
    assert {item.participant for item in split["train"]} == {"P1", "P4"}
