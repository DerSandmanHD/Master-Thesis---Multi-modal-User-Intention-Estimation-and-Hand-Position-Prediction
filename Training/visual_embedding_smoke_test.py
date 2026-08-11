#!/usr/bin/env python3
"""Smoke-test visual projection, causal alignment, and data-loader modes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import prepare_data
from model import HierarchicalResidualPoseTransformer
from smoke_test import synthetic_sequence
from clip_alignment import (
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    alignment_specification,
    canonical_json_hash,
)
from visual_embeddings import (
    VISUAL_CACHE_SCHEMA_VERSION,
    VISUAL_PROJECTION_SCHEMA_VERSION,
    VisualFeatureLoader,
    causal_align,
    sequence_fingerprint,
    sha256_file,
)


def write_visual_artifacts(
    master_dir: Path,
    cache_dir: Path,
    projection_path: Path,
    sequence_ids: list[str],
) -> None:
    cache_dir.mkdir()
    encoder_fingerprint = "synthetic_encoder_sha256"
    alignment = alignment_specification(sample_hz=5.0)
    alignment_fingerprint = canonical_json_hash(alignment)
    entries = {}
    for sequence_number, sequence_id in enumerate(sequence_ids):
        master = pd.read_csv(
            master_dir / f"{sequence_id}_master.csv",
            usecols=["timestamp_ns"],
        )
        timestamps = master["timestamp_ns"].to_numpy(np.int64)[::5]
        rng = np.random.default_rng(sequence_number)
        embeddings = rng.normal(size=(len(timestamps), 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        source_files = {
            "master": {
                "file": f"{sequence_id}_master.csv",
                "size_bytes": int(
                    (master_dir / f"{sequence_id}_master.csv").stat().st_size
                ),
                "sha256": sha256_file(
                    master_dir / f"{sequence_id}_master.csv"
                ),
            },
            "vrs": {
                "file": f"{sequence_id}.vrs",
                "size_bytes": 123,
                "sha256": f"synthetic-vrs-{sequence_number}",
            },
        }
        metadata = {
            "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "sequence_id": sequence_id,
            "encoder_fingerprint": encoder_fingerprint,
            "alignment_version": VISUAL_ALIGNMENT_VERSION,
            "time_basis": VISUAL_TIME_BASIS,
            "alignment": alignment,
            "alignment_fingerprint": alignment_fingerprint,
            "source_files": source_files,
        }
        path = cache_dir / f"{sequence_id}.npz"
        np.savez_compressed(
            path,
            timestamps_ns=timestamps,
            rgb_frame_indices=np.arange(len(timestamps), dtype=np.int64) * 5,
            embeddings=embeddings.astype(np.float16),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        entries[sequence_id] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "samples": len(timestamps),
            "embedding_dim": 8,
            "alignment_fingerprint": alignment_fingerprint,
            "source_files": source_files,
        }
    manifest = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "encoder": {"model_name": "synthetic"},
        "encoder_fingerprint": encoder_fingerprint,
        "alignment": alignment,
        "alignment_fingerprint": alignment_fingerprint,
        "sequence_fingerprint": sequence_fingerprint(sequence_ids),
        "entries": entries,
    }
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    projection_path.parent.mkdir()
    np.savez_compressed(
        projection_path,
        mean=np.zeros(8, dtype=np.float32),
        components=np.eye(8, dtype=np.float32)[:4],
    )
    projection_metadata = {
        "schema_version": VISUAL_PROJECTION_SCHEMA_VERSION,
        "input_dim": 8,
        "output_dim": 4,
        "fit_split": "train_only",
        "train_sequence_ids": sorted(sequence_ids[:4]),
        "train_sequence_fingerprint": sequence_fingerprint(sequence_ids[:4]),
        "selected_sequence_fingerprint": sequence_fingerprint(sequence_ids),
        "validation_participants_excluded": ["P5"],
        "test_participants_excluded": ["P6"],
        "encoder_fingerprint": encoder_fingerprint,
        "alignment_version": VISUAL_ALIGNMENT_VERSION,
        "time_basis": VISUAL_TIME_BASIS,
        "alignment_fingerprint": alignment_fingerprint,
        "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
        "projection_sha256": sha256_file(projection_path),
    }
    projection_path.with_suffix(".json").write_text(
        json.dumps(projection_metadata, indent=2) + "\n", encoding="utf-8"
    )


def data_config(
    master_dir: Path,
    cache_dir: Path,
    projection_path: Path,
    mode: str,
) -> dict:
    return {
        "master_dir": str(master_dir),
        "feature_profile": "multimodal_robot_frame_v1",
        "visual_embeddings": {
            "enabled": True,
            "mode": mode,
            "cache_dir": str(cache_dir),
            "projection_path": str(projection_path),
            "expected_output_dim": 4,
            "max_age_seconds": 0.25,
            "verify_cache_hashes": True,
            "random_seed": 123,
        },
        "window_size": 20,
        "stride": 10,
        "future_horizon_seconds": 1.0,
        "pose_intent_ids": [2],
        "include_hand_references": True,
        "minimum_observed_fraction": 0.05,
        "max_timestamp_gap_seconds": 2.0,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "validation_participants": ["P5"],
        "test_participants": ["P6"],
    }


def main() -> int:
    aligned, statistics = causal_align(
        np.asarray([100, 200], dtype=np.int64),
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        np.asarray([50, 100, 150, 250, 400], dtype=np.int64),
        max_age_seconds=1e-7,
    )
    assert np.isnan(aligned[0, 0])
    assert aligned[1:4, 0].tolist() == [1.0, 1.0, 2.0]
    assert np.isnan(aligned[4, 0])
    assert statistics["future_matches"] == 0

    with tempfile.TemporaryDirectory(prefix="aria_visual_smoke_") as directory:
        root = Path(directory)
        master_dir = root / "master_datasets"
        master_dir.mkdir()
        sequence_ids = []
        for index in range(6):
            participant = f"P{index + 1}"
            sequence_id = f"{participant}_{index}"
            synthetic_sequence(
                master_dir / f"{sequence_id}_master.csv",
                participant,
                index,
            )
            sequence_ids.append(sequence_id)
        cache_dir = root / "clip_cache"
        projection_path = root / "projection" / "pca4.npz"
        write_visual_artifacts(
            master_dir,
            cache_dir,
            projection_path,
            sequence_ids,
        )

        sensor_bundle = prepare_data(
            {
                key: value
                for key, value in data_config(
                    master_dir, cache_dir, projection_path, "append"
                ).items()
                if key != "visual_embeddings"
            },
            seed=42,
        )
        for mode in ("append", "only", "random_control"):
            config = data_config(master_dir, cache_dir, projection_path, mode)
            bundle = prepare_data(config, seed=42)
            visual = bundle.provenance["schema"]["visual_features"]
            assert visual["alignment"]["future_matches"] == 0
            assert visual["alignment"]["coverage"] > 0.9
            assert visual["projection_fit_split"] == "train_only"
            assert visual["projection_split_binding"]["verified"] is True
            if mode == "only":
                assert len(bundle.feature_columns) == 4
            else:
                assert len(bundle.feature_columns) == len(
                    sensor_bundle.feature_columns
                ) + 4
            batch = next(iter(DataLoader(bundle.train, batch_size=2)))
            model = HierarchicalResidualPoseTransformer(
                input_dim=len(bundle.normalizer.output_feature_names),
                window_size=20,
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            )
            with torch.inference_mode():
                output = model(
                    batch["features"], batch["hand_reference_pose"]
                )
            assert output["assistance_logits"].shape == (2, 2)
            print(
                f"{mode}: raw={len(bundle.feature_columns)}, "
                f"model={len(bundle.normalizer.output_feature_names)}, "
                f"coverage={visual['alignment']['coverage']:.3f}"
            )

        random_config = data_config(
            master_dir, cache_dir, projection_path, "random_control"
        )["visual_embeddings"]
        first_loader = VisualFeatureLoader.from_config(random_config)
        second_loader = VisualFeatureLoader.from_config(random_config)
        target_timestamps = pd.read_csv(
            master_dir / f"{sequence_ids[0]}_master.csv",
            usecols=["timestamp_ns"],
        )["timestamp_ns"].to_numpy(np.int64)
        first = first_loader.features_for(sequence_ids[0], target_timestamps)
        second = second_loader.features_for(sequence_ids[0], target_timestamps)
        np.testing.assert_allclose(first, second, equal_nan=True)

        changed_master = master_dir / f"{sequence_ids[0]}_master.csv"
        changed_master.write_text(
            changed_master.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        try:
            first_loader.features_for(
                sequence_ids[0],
                target_timestamps,
                source_master_path=changed_master,
            )
        except ValueError as exc:
            assert "differs from the visual cache" in str(exc)
        else:
            raise AssertionError("Changed master source was not invalidated")

    print("Visual embedding smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
