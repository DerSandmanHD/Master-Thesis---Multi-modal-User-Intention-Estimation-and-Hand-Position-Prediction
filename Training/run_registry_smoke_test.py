#!/usr/bin/env python3
"""Validate the tracked dataset descriptors and run registry."""

from __future__ import annotations

import json
import re
from pathlib import Path

from run_layout import validate_tag


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "Training" / "run_registry.json"
DATASET_COUNT_PATTERN = re.compile(r"_n(\d+)_")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    registry = read_json(REGISTRY_PATH)
    assert registry["schema_version"] == 2
    assert registry["layout"]["tag_pattern"] == "^[a-z0-9][a-z0-9_-]*$"
    active = registry["active_protocol"]
    assert active["planned_dataset_tag"] == (
        "dataset_v3_causal_20260815_n214_5d136a34"
    )
    assert active["planned_dataset_materialized"] is True
    active_descriptor_path = PROJECT_ROOT / active["dataset_descriptor"]
    active_descriptor = read_json(active_descriptor_path)
    assert active_descriptor["dataset_tag"] == active["planned_dataset_tag"]
    assert active_descriptor["selected_sequences"] == 214
    assert active["new_results_available"] is True
    assert active["required_observation_alignment_version"] == (
        "causal_backward_device_time_v1"
    )
    assert active["required_artifact_freeze_protocol"] == (
        "thesis_artifact_freeze_hash_bound_v2"
    )

    datasets: dict[str, dict] = {}
    for record in registry["datasets"]:
        dataset_tag = validate_tag(record["dataset_tag"], "dataset_tag")
        assert dataset_tag not in datasets
        descriptor_path = PROJECT_ROOT / record["descriptor"]
        descriptor = read_json(descriptor_path)
        assert descriptor["dataset_tag"] == dataset_tag
        assert descriptor["selected_sequences"] > 0
        count_match = DATASET_COUNT_PATTERN.search(dataset_tag)
        assert count_match is not None
        assert int(count_match.group(1)) == descriptor["selected_sequences"]
        sequence_fingerprint = descriptor.get("sequence_fingerprint")
        if sequence_fingerprint:
            assert len(sequence_fingerprint) == 64
            assert dataset_tag.endswith(sequence_fingerprint[:8])
        for key in (
            "manifest_sha256",
            "snapshot_validation_sha256",
            "dataset_content_fingerprint",
        ):
            value = descriptor.get(key)
            if value is not None:
                assert re.fullmatch(r"[0-9a-f]{64}", value), (key, value)
        datasets[dataset_tag] = descriptor

    experiment_keys: set[tuple[str, str]] = set()
    for experiment in registry["experiments"]:
        dataset_tag = validate_tag(
            experiment["dataset_tag"],
            "dataset_tag",
        )
        experiment_tag = validate_tag(
            experiment["experiment_tag"],
            "experiment_tag",
        )
        assert dataset_tag in datasets
        key = (dataset_tag, experiment_tag)
        assert key not in experiment_keys
        experiment_keys.add(key)
        assert experiment["seeds"]
        assert len(experiment["models"]) == len(set(experiment["models"]))
        for model_tag in experiment["models"]:
            validate_tag(model_tag, "model_tag")

    print("run registry smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
