from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "Training" / "evaluation"),
)

from summarize_modality_weights import summarize  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_key": ["test|S1|1", "test|S1|2"],
            "participant": ["P1", "P1"],
            "sequence_id": ["S1", "S1"],
            "endpoint_timestamp_ns": [1, 2],
            "target_intention": ["handover", "handover"],
            "predicted_intention": ["handover", "fetch"],
            "modality_gaze_weight": [0.25, 0.0],
            "modality_gaze_available": [True, False],
            "modality_hands_weight": [0.75, 1.0],
            "modality_hands_available": [True, True],
        }
    )


def test_modality_weight_report_preserves_window_context() -> None:
    windows, report = summarize(_frame())
    assert len(windows) == 2
    assert report["modality_names"] == ["gaze", "hands"]
    assert report["invariants"]["unavailable_weight_zero"]


def test_modality_weight_report_rejects_nonzero_missing_weight() -> None:
    frame = _frame()
    frame.loc[1, "modality_gaze_weight"] = 0.1
    with pytest.raises(ValueError, match="Unavailable modalities"):
        summarize(frame)
