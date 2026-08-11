#!/usr/bin/env python3
"""Fit a train-participant-only PCA projection for frozen visual embeddings."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data import (
    canonical_participant,
    manifest_filtered_master_files,
    sequence_id_from_master_path,
)
from clip_alignment import (
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    canonical_json_hash,
)
from visual_embeddings import (
    VISUAL_CACHE_SCHEMA_VERSION,
    VISUAL_PROJECTION_SCHEMA_VERSION,
    load_cache,
    sequence_fingerprint,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_v2.json"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=20260808)
    parser.add_argument("--expected-sequence-fingerprint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_master_files(data_config: dict) -> tuple[list[Path], str]:
    master_dir = project_path(Path(data_config["master_dir"])).resolve()
    files = sorted(master_dir.glob("*_master.csv"))
    selected, _ = manifest_filtered_master_files(
        files,
        master_dir,
        data_config.get("manifest_filter"),
    )
    ids = [sequence_id_from_master_path(path) for path in selected]
    return selected, sequence_fingerprint(ids)


def participant_for(path: Path) -> str:
    frame = pd.read_csv(path, usecols=["participant"], nrows=1)
    if frame.empty:
        raise ValueError(f"Cannot determine participant from {path}")
    return canonical_participant(str(frame.iloc[0]["participant"]))


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config).resolve()
    cache_dir = project_path(args.cache_dir).resolve()
    output_path = project_path(args.output).resolve()
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Projection already exists; pass --overwrite: {output_path}"
        )
    if args.components <= 0:
        raise ValueError("components must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    selected, selected_fingerprint = selected_master_files(data_config)
    if (
        args.expected_sequence_fingerprint
        and selected_fingerprint != args.expected_sequence_fingerprint
    ):
        raise ValueError(
            "Selected sequence fingerprint differs from the frozen dataset: "
            f"{selected_fingerprint} != {args.expected_sequence_fingerprint}"
        )

    validation = {
        canonical_participant(value)
        for value in data_config.get("validation_participants", [])
    }
    test = {
        canonical_participant(value)
        for value in data_config.get("test_participants", [])
    }
    if not validation or not test or validation & test:
        raise ValueError(
            "Visual PCA requires explicit, disjoint validation/test participants"
        )
    train_paths = [
        path
        for path in selected
        if participant_for(path) not in validation | test
    ]
    train_ids = [sequence_id_from_master_path(path) for path in train_paths]
    if not train_ids:
        raise ValueError("No training sequences available for visual PCA")

    manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(cache_manifest.get("schema_version", -1)) != VISUAL_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported or obsolete visual cache manifest schema")
    alignment = cache_manifest.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Visual cache manifest has no alignment specification")
    if alignment.get("version") != VISUAL_ALIGNMENT_VERSION:
        raise ValueError("Visual caches use an obsolete timestamp alignment")
    if alignment.get("time_basis") != VISUAL_TIME_BASIS:
        raise ValueError("Visual caches do not use absolute Aria device time")
    alignment_fingerprint = canonical_json_hash(alignment)
    if cache_manifest.get("alignment_fingerprint") != alignment_fingerprint:
        raise ValueError("Visual cache alignment fingerprint mismatch")
    arrays: list[np.ndarray] = []
    encoder_fingerprint = cache_manifest.get("encoder_fingerprint")
    input_dim = None
    sample_counts = {}
    for sequence_id in train_ids:
        entry = cache_manifest.get("entries", {}).get(sequence_id)
        if entry is None:
            raise FileNotFoundError(
                f"Visual cache manifest has no training sequence {sequence_id}"
            )
        cache_path = cache_dir / entry.get("file", f"{sequence_id}.npz")
        if sha256_file(cache_path) != entry.get("sha256"):
            raise ValueError(f"Visual cache SHA-256 mismatch: {cache_path}")
        _, embeddings, metadata = load_cache(cache_path)
        if metadata.get("encoder_fingerprint") != encoder_fingerprint:
            raise ValueError(f"Visual encoder mismatch: {cache_path}")
        if metadata.get("alignment_fingerprint") != alignment_fingerprint:
            raise ValueError(f"Visual alignment mismatch: {cache_path}")
        if metadata.get("source_files") != entry.get("source_files"):
            raise ValueError(f"Visual source identity mismatch: {cache_path}")
        if input_dim is None:
            input_dim = int(embeddings.shape[1])
        if embeddings.shape[1] != input_dim:
            raise ValueError(f"Visual embedding dimension mismatch: {cache_path}")
        arrays.append(embeddings)
        sample_counts[sequence_id] = int(len(embeddings))
    matrix = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    if args.components > min(matrix.shape):
        raise ValueError(
            f"Cannot fit {args.components} components to matrix {matrix.shape}"
        )

    estimator = PCA(
        n_components=args.components,
        svd_solver="randomized",
        random_state=args.random_state,
    )
    estimator.fit(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            mean=estimator.mean_.astype(np.float32),
            components=estimator.components_.astype(np.float32),
            explained_variance=estimator.explained_variance_.astype(np.float32),
            explained_variance_ratio=estimator.explained_variance_ratio_.astype(
                np.float32
            ),
            singular_values=estimator.singular_values_.astype(np.float32),
        )
    os.replace(temporary, output_path)
    projection_sha256 = sha256_file(output_path)
    metadata = {
        "schema_version": VISUAL_PROJECTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "sklearn.decomposition.PCA",
        "svd_solver": "randomized",
        "random_state": int(args.random_state),
        "fit_split": "train_only",
        "input_dim": int(input_dim),
        "output_dim": int(args.components),
        "fit_samples": int(len(matrix)),
        "train_sequences": len(train_ids),
        "train_sequence_ids": sorted(train_ids),
        "train_sequence_fingerprint": sequence_fingerprint(train_ids),
        "selected_sequence_fingerprint": selected_fingerprint,
        "validation_participants_excluded": sorted(validation),
        "test_participants_excluded": sorted(test),
        "encoder_fingerprint": encoder_fingerprint,
        "alignment_version": VISUAL_ALIGNMENT_VERSION,
        "time_basis": VISUAL_TIME_BASIS,
        "alignment_fingerprint": alignment_fingerprint,
        "cache_manifest_sha256": sha256_file(manifest_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "projection_sha256": projection_sha256,
        "explained_variance_ratio_sum": float(
            estimator.explained_variance_ratio_.sum()
        ),
        "samples_by_sequence": sample_counts,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Projection: {output_path}")
    print(f"Metadata:   {metadata_path}")
    print(
        f"Fit samples: {len(matrix)} from {len(train_ids)} training sequences; "
        f"explained variance={metadata['explained_variance_ratio_sum']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
