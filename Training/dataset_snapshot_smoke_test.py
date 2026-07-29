#!/usr/bin/env python3
"""Short structural test for validate_dataset_snapshot.py."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from validate_dataset_snapshot import validate_snapshot


def write_master(path: Path, sequence_id: str, participant: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_ns", "sequence_id", "participant", "feature"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_ns": 1,
                "sequence_id": sequence_id,
                "participant": participant,
                "feature": 0.5,
            }
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        snapshot = root / "snapshot"
        masters = snapshot / "master_datasets"
        artifacts = root / "artifacts"
        masters.mkdir(parents=True)
        artifacts.mkdir()

        sequences = (
            ("Train_1", "Train", "train"),
            ("Test_1", "Test", "test"),
        )
        for sequence_id, participant, _ in sequences:
            write_master(
                masters / f"{sequence_id}_master.csv",
                sequence_id,
                participant,
            )

        with (snapshot / "dataset_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sequence_id",
                    "include_in_training",
                    "status",
                    "next_action",
                    "master_csv_exists",
                ],
            )
            writer.writeheader()
            for sequence_id, _, _ in sequences:
                writer.writerow(
                    {
                        "sequence_id": sequence_id,
                        "include_in_training": "True",
                        "status": "valid",
                        "next_action": "ready_for_master_merge",
                        "master_csv_exists": "True",
                    }
                )

        (artifacts / "config.json").write_text(
            json.dumps(
                {
                    "data": {
                        "manifest_filter": {
                            "allowed_statuses": ["valid"],
                            "allowed_next_actions": [
                                "ready_for_master_merge"
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (artifacts / "data_metadata.json").write_text(
            json.dumps(
                {
                    "feature_columns": ["feature"],
                    "split": {
                        "sequences": {
                            "train": ["Train_1"],
                            "validation": [],
                            "test": ["Test_1"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        report = validate_snapshot(
            snapshot,
            artifacts,
            calculate_hashes=True,
        )
        assert report["status"] == "valid"
        assert report["manifest"]["eligible_sequences"] == 2
        assert report["artifact_split"]["total_sequences"] == 2
        assert len(report["manifest"]["sha256"]) == 64
        assert len(report["snapshot_files"]) == 3

        (masters / "Test_1_master.csv").write_text(
            "timestamp_ns,sequence_id,participant\n1,Test_1,Test\n",
            encoding="utf-8",
        )
        invalid = validate_snapshot(
            snapshot,
            artifacts,
            calculate_hashes=False,
        )
        assert invalid["status"] == "invalid"
        assert invalid["blocking_issues"]["schema_issues"] == {
            "Test_1": ["feature"]
        }

    print("Dataset snapshot smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
