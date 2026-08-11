from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from data import prepare_data  # noqa: E402
from smoke_test import synthetic_sequence  # noqa: E402
from visual_embedding_smoke_test import (  # noqa: E402
    data_config,
    write_visual_artifacts,
)


def artifacts(tmp_path: Path) -> tuple[dict, Path]:
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    sequence_ids = []
    for index in range(6):
        participant = f"P{index + 1}"
        sequence_id = f"{participant}_{index}"
        synthetic_sequence(
            master_dir / f"{sequence_id}_master.csv", participant, index
        )
        sequence_ids.append(sequence_id)
    cache_dir = tmp_path / "cache"
    projection = tmp_path / "projection" / "pca.npz"
    write_visual_artifacts(master_dir, cache_dir, projection, sequence_ids)
    return data_config(master_dir, cache_dir, projection, "append"), projection


def test_projection_is_bound_to_actual_train_sequences(tmp_path: Path) -> None:
    config, projection = artifacts(tmp_path)
    bundle = prepare_data(config, seed=42)
    binding = bundle.provenance["schema"]["visual_features"][
        "projection_split_binding"
    ]
    assert binding["verified"] is True
    assert binding["train_sequence_ids"] == ["P1_0", "P2_1", "P3_2", "P4_3"]

    metadata_path = projection.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["train_sequence_ids"] = [
        "P1_0",
        "P2_1",
        "P3_2",
        "P5_4",
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to the active train split"):
        prepare_data(config, seed=42)


def test_projection_rejects_non_train_only_fit(tmp_path: Path) -> None:
    config, projection = artifacts(tmp_path)
    metadata_path = projection.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fit_split"] = "all_splits"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="train_only"):
        prepare_data(config, seed=42)
