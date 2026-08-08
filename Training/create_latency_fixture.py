#!/usr/bin/env python3
"""Export one real, immutable test window for cross-platform latency tests."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import prepare_data, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    artifacts = resolve(args.artifacts_dir).resolve()
    output = resolve(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Fixture exists; pass --overwrite: {output}")
    config_path = artifacts / "config.json"
    metadata_path = artifacts / "data_metadata.json"
    checkpoint_path = artifacts / "best_intention_model.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    master_dir = Path(data_config["master_dir"]).expanduser()
    if not master_dir.is_absolute():
        master_dir = PROJECT_ROOT / master_dir
    data_config["master_dir"] = str(master_dir)
    bundle = prepare_data(data_config, seed=int(config["training"]["seed"]))
    index = int(args.index)
    if not 0 <= index < len(bundle.test):
        raise IndexError(f"Fixture index {index} outside test set of {len(bundle.test)}")
    item = bundle.test[index]
    fixture_metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "test",
        "source_index": index,
        "sequence_id": item["sequence_id"],
        "participant": item["participant"],
        "timestamp_ns": int(item["timestamp_ns"]),
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "input_shape": list(item["features"].shape),
        "hand_reference_shape": list(item["hand_reference_pose"].shape),
        "config_sha256": sha256_file(config_path),
        "data_metadata_sha256": sha256_file(metadata_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_feature_columns": metadata["model_feature_columns"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=item["features"].numpy().astype(np.float32),
            hand_reference_pose=item["hand_reference_pose"].numpy().astype(
                np.float32
            ),
            metadata_json=np.asarray(
                json.dumps(fixture_metadata, sort_keys=True, ensure_ascii=False)
            ),
        )
    os.replace(temporary, output)
    print(f"Fixture: {output}")
    print(json.dumps(fixture_metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
