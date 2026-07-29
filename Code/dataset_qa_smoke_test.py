#!/usr/bin/env python3
"""Short checks for dataset-QA thresholds and report provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

from dataset_qa import build_report, parse_args, unit_interval


def main() -> int:
    assert unit_interval("0") == 0.0
    assert unit_interval("0.70") == 0.7
    assert unit_interval("1") == 1.0
    for invalid in ("-0.01", "1.01", "invalid"):
        try:
            unit_interval(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"Invalid ratio was accepted: {invalid}")

    with patch.object(sys, "argv", ["dataset_qa.py"]):
        args = parse_args()
    assert args.min_handover_hand_valid_ratio == 0.7

    report = build_report(
        [],
        Path("timestamps.json"),
        Path("annotations.csv"),
        Path("manifest.csv"),
        Path("backup"),
        min_phase_seconds=0.5,
        min_handover_hand_valid_ratio=0.7,
    )
    assert report["quality_thresholds"] == {
        "min_phase_seconds": 0.5,
        "min_handover_hand_valid_ratio": 0.7,
    }
    print("Dataset-QA threshold smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
