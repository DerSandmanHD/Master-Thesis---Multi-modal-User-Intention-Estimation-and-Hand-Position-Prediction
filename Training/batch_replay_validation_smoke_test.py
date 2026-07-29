#!/usr/bin/env python3
"""Short non-model test for the batch replay orchestrator."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from batch_replay_validation import (
    event_onset_summary,
    load_replay_rows,
    load_split_plan,
)
from replay_stream_inference import build_replay_summary


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifacts = root / "artifacts"
        masters = root / "masters"
        artifacts.mkdir()
        masters.mkdir()
        (artifacts / "data_metadata.json").write_text(
            json.dumps(
                {
                    "split": {
                        "sequences": {
                            "train": [],
                            "validation": [],
                            "test": ["A_1", "B_1"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        for sequence in ("A_1", "B_1"):
            (masters / f"{sequence}_master.csv").write_text(
                "timestamp_ns\n1\n", encoding="utf-8"
            )
        plan = load_split_plan(artifacts, masters, "test")
        assert [sequence for sequence, _ in plan] == ["A_1", "B_1"]

        replay_csv = root / "replay.csv"
        fieldnames = [
            "target_intention",
            "sequence_id",
            "endpoint_row",
            "timestamp_ns",
            "raw_intention",
            "stable_intention",
            "actionable_intention",
            "input_quality_ok",
            "input_quality_reasons",
            "intention_inference_ms",
            "pose_inference_ms",
            "pose_position_error_cm",
            "pose_orientation_error_deg",
        ]
        with replay_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "target_intention": "continue",
                    "sequence_id": "A_1",
                    "endpoint_row": "59",
                    "timestamp_ns": "1000000000",
                    "raw_intention": "continue",
                    "stable_intention": "continue",
                    "actionable_intention": "continue",
                    "input_quality_ok": "True",
                    "input_quality_reasons": "[]",
                    "intention_inference_ms": "3.0",
                }
            )
            writer.writerow(
                {
                    "target_intention": "fetch",
                    "sequence_id": "A_1",
                    "endpoint_row": "69",
                    "timestamp_ns": "1333333333",
                    "raw_intention": "fetch",
                    "stable_intention": "fetch",
                    "actionable_intention": "insufficient_input",
                    "input_quality_ok": "False",
                    "input_quality_reasons": '["gaze_coverage_too_low"]',
                    "intention_inference_ms": "4.0",
                }
            )

        rows = load_replay_rows(replay_csv)
        summary = build_replay_summary(rows)
        assert summary["predictions"] == 2
        assert summary["decision_levels"]["raw"]["end_to_end_accuracy"] == 1.0
        assert summary["decision_levels"]["raw"]["macro_f1"] == 1.0
        assert summary["decision_levels"]["actionable"]["coverage"] == 0.5
        assert summary["input_quality"]["reason_counts"] == {
            "gaze_coverage_too_low": 1
        }
        onset = event_onset_summary(rows, "stable_intention")
        assert onset["continue"]["detection_rate"] == 1.0
        assert onset["fetch"]["detection_rate"] == 1.0

        (masters / "B_1_master.csv").unlink()
        try:
            load_split_plan(artifacts, masters, "test")
        except FileNotFoundError as error:
            assert "B_1" in str(error)
        else:
            raise AssertionError("Missing split sequence was not rejected")

    print("Batch replay validation smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
