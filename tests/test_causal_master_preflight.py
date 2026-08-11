from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from verify_causal_masters import (  # noqa: E402
    EXPECTED_ALIGNMENT_VERSION,
    verify_dataset,
)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence_id",
                "include_in_training",
                "status",
                "next_action",
                "master_csv_exists",
            ),
        )
        writer.writeheader()
        for sequence_id in ("P1_0", "P2_0"):
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "include_in_training": "True",
                    "status": "valid",
                    "next_action": "ready_for_master_merge",
                    "master_csv_exists": "True",
                }
            )
            (master_dir / f"{sequence_id}_master.csv").write_text(
                "timestamp_ns,observation_alignment_version\n"
                f"1,{EXPECTED_ALIGNMENT_VERSION}\n",
                encoding="utf-8",
            )
            (master_dir / f"{sequence_id}_master_report.json").write_text(
                json.dumps(
                    {
                        "observation_alignment": {
                            "version": EXPECTED_ALIGNMENT_VERSION,
                            "future_source_captures_allowed": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
    return master_dir, manifest


def test_causal_master_preflight_binds_selected_sources(tmp_path: Path) -> None:
    master_dir, manifest = fixture(tmp_path)
    report = verify_dataset(master_dir, manifest)
    assert report["passed"] is True
    assert report["selected_sequences"] == 2
    assert len(report["source_content_fingerprint"]) == 64


def test_causal_master_preflight_rejects_legacy_master(tmp_path: Path) -> None:
    master_dir, manifest = fixture(tmp_path)
    path = master_dir / "P2_0_master.csv"
    path.write_text("timestamp_ns,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predates causal"):
        verify_dataset(master_dir, manifest)
