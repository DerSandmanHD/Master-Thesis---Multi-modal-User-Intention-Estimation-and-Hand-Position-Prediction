#!/usr/bin/env python3
"""Load, validate, project, and causally align frozen visual embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clip_alignment import (
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    canonical_json_hash,
)


SUPPORTED_VISUAL_MODES = ("append", "only", "random_control")
VISUAL_CACHE_SCHEMA_VERSION = 2
VISUAL_PROJECTION_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_fingerprint(sequence_ids: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sequence_ids)).encode("utf-8")
    ).hexdigest()


def stable_sequence_seed(sequence_id: str, seed: int) -> int:
    payload = f"{int(seed)}:{sequence_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "timestamps_ns",
            "rgb_frame_indices",
            "embeddings",
            "metadata_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"Visual cache {path} is missing arrays: {', '.join(missing)}"
            )
        timestamps_ns = archive["timestamps_ns"].astype(np.int64, copy=False)
        frame_indices = archive["rgb_frame_indices"].astype(np.int64, copy=False)
        embeddings = archive["embeddings"].astype(np.float32, copy=False)
        metadata_value = archive["metadata_json"]
        metadata = json.loads(str(metadata_value.item()))
    if timestamps_ns.ndim != 1:
        raise ValueError(f"Visual cache timestamps must be one-dimensional: {path}")
    if embeddings.ndim != 2 or len(embeddings) != len(timestamps_ns):
        raise ValueError(f"Visual cache embeddings/timestamps do not align: {path}")
    if frame_indices.ndim != 1 or len(frame_indices) != len(timestamps_ns):
        raise ValueError(f"Visual cache frame indices do not align: {path}")
    if not len(timestamps_ns):
        raise ValueError(f"Visual cache is empty: {path}")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"Visual cache timestamps are not strictly increasing: {path}")
    if np.any(np.diff(frame_indices) <= 0) or np.any(frame_indices < 0):
        raise ValueError(f"Visual cache frame indices are invalid: {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Visual cache contains non-finite embeddings: {path}")
    if int(metadata.get("schema_version", -1)) != VISUAL_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported visual cache schema in {path}")
    if metadata.get("alignment_version") != VISUAL_ALIGNMENT_VERSION:
        raise ValueError(f"Unsupported visual alignment version in {path}")
    if metadata.get("time_basis") != VISUAL_TIME_BASIS:
        raise ValueError(f"Unsupported visual timestamp basis in {path}")
    alignment = metadata.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"Visual cache alignment metadata is missing in {path}")
    if metadata.get("alignment_fingerprint") != canonical_json_hash(alignment):
        raise ValueError(f"Visual cache alignment fingerprint mismatch in {path}")
    source_files = metadata.get("source_files")
    if not isinstance(source_files, dict) or not {"master", "vrs"}.issubset(
        source_files
    ):
        raise ValueError(f"Visual cache source identities are incomplete in {path}")
    return timestamps_ns, embeddings, metadata


def causal_align(
    source_timestamps_ns: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_ns: np.ndarray,
    *,
    max_age_seconds: float,
) -> tuple[np.ndarray, dict]:
    """Align the most recent source sample without ever looking into the future."""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be greater than zero")
    indices = np.searchsorted(
        source_timestamps_ns,
        target_timestamps_ns,
        side="right",
    ) - 1
    safe_indices = np.maximum(indices, 0)
    ages_ns = target_timestamps_ns - source_timestamps_ns[safe_indices]
    valid = (
        (indices >= 0)
        & (ages_ns >= 0)
        & (ages_ns <= int(max_age_seconds * 1e9))
    )
    aligned = np.full(
        (len(target_timestamps_ns), source_values.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    aligned[valid] = source_values[safe_indices[valid]]
    valid_ages_ms = ages_ns[valid].astype(np.float64) / 1e6
    return aligned, {
        "target_rows": int(len(target_timestamps_ns)),
        "valid_rows": int(valid.sum()),
        "missing_rows": int((~valid).sum()),
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "mean_age_ms": (
            float(valid_ages_ms.mean()) if len(valid_ages_ms) else None
        ),
        "max_age_ms": (
            float(valid_ages_ms.max()) if len(valid_ages_ms) else None
        ),
        "future_matches": int((ages_ns[indices >= 0] < 0).sum()),
    }


@dataclass
class VisualFeatureLoader:
    cache_dir: Path
    projection_path: Path
    mode: str
    max_age_seconds: float
    random_seed: int
    verify_cache_hashes: bool
    projection_mean: np.ndarray
    projection_components: np.ndarray
    projection_metadata: dict
    cache_manifest: dict
    cache_manifest_sha256: str
    projection_sha256: str
    alignment_by_sequence: dict[str, dict] = field(default_factory=dict)
    verified_master_sources: set[str] = field(default_factory=set)
    split_binding: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "VisualFeatureLoader":
        mode = str(config.get("mode", "append")).strip().casefold()
        if mode not in SUPPORTED_VISUAL_MODES:
            raise ValueError(
                f"Unknown visual embedding mode {mode!r}; expected one of "
                f"{', '.join(SUPPORTED_VISUAL_MODES)}"
            )
        cache_dir = Path(config["cache_dir"]).expanduser().resolve()
        projection_path = Path(config["projection_path"]).expanduser().resolve()
        projection_metadata_path = projection_path.with_suffix(".json")
        manifest_path = cache_dir / "cache_manifest.json"
        for path in (projection_path, projection_metadata_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(projection_path, allow_pickle=False) as archive:
            if not {"mean", "components"}.issubset(archive.files):
                raise ValueError(f"Projection arrays are incomplete: {projection_path}")
            mean = archive["mean"].astype(np.float32, copy=False)
            components = archive["components"].astype(np.float32, copy=False)
        if mean.ndim != 1 or components.ndim != 2:
            raise ValueError("Visual projection mean/components have invalid shapes")
        if components.shape[1] != len(mean):
            raise ValueError("Visual projection input dimensions do not match")

        projection_metadata = json.loads(
            projection_metadata_path.read_text(encoding="utf-8")
        )
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(projection_metadata.get("schema_version", -1)) != (
            VISUAL_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported visual projection schema")
        if projection_metadata.get("fit_split") != "train_only":
            raise ValueError("Visual projection must be fitted on train_only")
        required_split_metadata = (
            "train_sequence_ids",
            "train_sequence_fingerprint",
            "selected_sequence_fingerprint",
            "validation_participants_excluded",
            "test_participants_excluded",
        )
        missing_split_metadata = [
            key for key in required_split_metadata
            if key not in projection_metadata
        ]
        if missing_split_metadata:
            raise ValueError(
                "Visual projection lacks train-split binding metadata: "
                + ", ".join(missing_split_metadata)
            )
        if int(cache_manifest.get("schema_version", -1)) != (
            VISUAL_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported visual cache manifest schema")
        alignment = cache_manifest.get("alignment")
        if not isinstance(alignment, dict):
            raise ValueError("Visual cache manifest has no alignment specification")
        if alignment.get("version") != VISUAL_ALIGNMENT_VERSION:
            raise ValueError("Unsupported visual cache alignment version")
        if alignment.get("time_basis") != VISUAL_TIME_BASIS:
            raise ValueError("Unsupported visual cache timestamp basis")
        alignment_fingerprint = canonical_json_hash(alignment)
        if cache_manifest.get("alignment_fingerprint") != alignment_fingerprint:
            raise ValueError("Visual cache manifest alignment fingerprint mismatch")
        projection_sha256 = sha256_file(projection_path)
        expected_projection_sha256 = projection_metadata.get("projection_sha256")
        if expected_projection_sha256 != projection_sha256:
            raise ValueError("Visual projection SHA-256 does not match its metadata")
        if int(projection_metadata.get("input_dim", -1)) != len(mean):
            raise ValueError("Visual projection input_dim metadata mismatch")
        if int(projection_metadata.get("output_dim", -1)) != len(components):
            raise ValueError("Visual projection output_dim metadata mismatch")
        if projection_metadata.get("encoder_fingerprint") != cache_manifest.get(
            "encoder_fingerprint"
        ):
            raise ValueError("Projection and cache use different visual encoders")
        if projection_metadata.get("alignment_version") != VISUAL_ALIGNMENT_VERSION:
            raise ValueError("Projection uses an obsolete visual alignment")
        if projection_metadata.get("alignment_fingerprint") != (
            cache_manifest.get("alignment_fingerprint")
        ):
            raise ValueError("Projection and cache use different visual alignments")
        if projection_metadata.get("cache_manifest_sha256") != sha256_file(
            manifest_path
        ):
            raise ValueError("Projection was fitted from a different cache manifest")

        max_age_seconds = float(config.get("max_age_seconds", 0.25))
        if max_age_seconds <= 0:
            raise ValueError("visual_embeddings.max_age_seconds must be positive")
        expected_output_dim = config.get("expected_output_dim")
        if expected_output_dim is not None and int(expected_output_dim) != len(
            components
        ):
            raise ValueError(
                "Visual projection output dimension differs from expected_output_dim"
            )
        return cls(
            cache_dir=cache_dir,
            projection_path=projection_path,
            mode=mode,
            max_age_seconds=max_age_seconds,
            random_seed=int(config.get("random_seed", 20260808)),
            verify_cache_hashes=bool(config.get("verify_cache_hashes", True)),
            projection_mean=mean,
            projection_components=components,
            projection_metadata=projection_metadata,
            cache_manifest=cache_manifest,
            cache_manifest_sha256=sha256_file(manifest_path),
            projection_sha256=projection_sha256,
        )

    def validate_split_binding(
        self,
        split_metadata: dict,
        *,
        selected_sequence_ids: list[str],
    ) -> dict:
        """Prove that the loaded projection was fitted on this exact train split."""
        sequences = split_metadata.get("sequences", {})
        participants = split_metadata.get("participants", {})
        actual_train_ids = sorted(str(value) for value in sequences.get("train", []))
        actual_selected_ids = sorted(str(value) for value in selected_sequence_ids)
        actual_validation = sorted(
            str(value) for value in participants.get("validation", [])
        )
        actual_test = sorted(str(value) for value in participants.get("test", []))
        expected_train_ids = sorted(
            str(value)
            for value in self.projection_metadata.get("train_sequence_ids", [])
        )
        checks = {
            "train_sequence_ids": (expected_train_ids, actual_train_ids),
            "train_sequence_fingerprint": (
                self.projection_metadata.get("train_sequence_fingerprint"),
                sequence_fingerprint(actual_train_ids),
            ),
            "selected_sequence_fingerprint": (
                self.projection_metadata.get("selected_sequence_fingerprint"),
                sequence_fingerprint(actual_selected_ids),
            ),
            "validation_participants_excluded": (
                sorted(
                    str(value)
                    for value in self.projection_metadata.get(
                        "validation_participants_excluded", []
                    )
                ),
                actual_validation,
            ),
            "test_participants_excluded": (
                sorted(
                    str(value)
                    for value in self.projection_metadata.get(
                        "test_participants_excluded", []
                    )
                ),
                actual_test,
            ),
            "cache_sequence_fingerprint": (
                self.cache_manifest.get("sequence_fingerprint"),
                sequence_fingerprint(actual_selected_ids),
            ),
        }
        mismatches = [
            key for key, (expected, actual) in checks.items()
            if expected != actual
        ]
        if mismatches:
            details = "; ".join(
                f"{key}: projection/cache={checks[key][0]!r}, "
                f"active_split={checks[key][1]!r}"
                for key in mismatches
            )
            raise ValueError(
                "Visual projection is not bound to the active train split: "
                + details
            )
        self.split_binding = {
            "version": "visual_projection_active_split_binding_v1",
            "verified": True,
            "train_sequence_ids": actual_train_ids,
            "train_sequence_fingerprint": sequence_fingerprint(actual_train_ids),
            "selected_sequence_fingerprint": sequence_fingerprint(
                actual_selected_ids
            ),
            "validation_participants_excluded": actual_validation,
            "test_participants_excluded": actual_test,
        }
        return dict(self.split_binding)

    @property
    def output_dim(self) -> int:
        return int(self.projection_components.shape[0])

    @property
    def feature_names(self) -> list[str]:
        prefix = "random_visual" if self.mode == "random_control" else "clip_pca"
        return [f"{prefix}_{index:03d}" for index in range(self.output_dim)]

    def _entry(self, sequence_id: str) -> tuple[Path, dict]:
        entries = self.cache_manifest.get("entries", {})
        if sequence_id not in entries:
            raise FileNotFoundError(
                f"Visual cache manifest has no entry for {sequence_id}"
            )
        entry = entries[sequence_id]
        path = (self.cache_dir / entry.get("file", f"{sequence_id}.npz")).resolve()
        if path.parent != self.cache_dir:
            raise ValueError(f"Visual cache path escapes cache directory: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.verify_cache_hashes and sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Visual cache SHA-256 mismatch: {path}")
        return path, entry

    def features_for(
        self,
        sequence_id: str,
        target_timestamps_ns: np.ndarray,
        *,
        source_master_path: Path | None = None,
    ) -> np.ndarray:
        cache_path, entry = self._entry(sequence_id)
        if source_master_path is not None:
            expected_master = entry.get("source_files", {}).get("master")
            source_master_path = Path(source_master_path).expanduser().resolve()
            if not isinstance(expected_master, dict):
                raise ValueError(
                    f"Visual cache master identity is missing: {cache_path}"
                )
            actual_master = {
                "file": source_master_path.name,
                "size_bytes": int(source_master_path.stat().st_size),
                "sha256": sha256_file(source_master_path),
            }
            if actual_master != expected_master:
                raise ValueError(
                    "Current master source differs from the visual cache: "
                    f"{source_master_path}"
                )
            self.verified_master_sources.add(sequence_id)
        timestamps_ns, embeddings, metadata = load_cache(cache_path)
        if metadata.get("sequence_id") != sequence_id:
            raise ValueError(f"Visual cache sequence mismatch: {cache_path}")
        encoder_fingerprint = self.cache_manifest.get("encoder_fingerprint")
        if metadata.get("encoder_fingerprint") != encoder_fingerprint:
            raise ValueError(f"Visual cache encoder mismatch: {cache_path}")
        if metadata.get("alignment_fingerprint") != self.cache_manifest.get(
            "alignment_fingerprint"
        ):
            raise ValueError(f"Visual cache alignment mismatch: {cache_path}")
        if metadata.get("source_files") != entry.get("source_files"):
            raise ValueError(f"Visual cache source identity mismatch: {cache_path}")
        if embeddings.shape[1] != len(self.projection_mean):
            raise ValueError(f"Visual cache dimension mismatch: {cache_path}")
        if int(entry.get("samples", -1)) != len(embeddings):
            raise ValueError(f"Visual cache sample count mismatch: {cache_path}")

        if self.mode == "random_control":
            rng = np.random.default_rng(
                stable_sequence_seed(sequence_id, self.random_seed)
            )
            values = rng.standard_normal(
                (len(embeddings), self.output_dim),
                dtype=np.float32,
            )
        else:
            values = (
                (embeddings - self.projection_mean)
                @ self.projection_components.T
            ).astype(np.float32)
        aligned, statistics = causal_align(
            timestamps_ns,
            values,
            np.asarray(target_timestamps_ns, dtype=np.int64),
            max_age_seconds=self.max_age_seconds,
        )
        if statistics["future_matches"]:
            raise ValueError(f"Causal alignment used future frames for {sequence_id}")
        self.alignment_by_sequence[sequence_id] = statistics
        return aligned

    def provenance(self) -> dict:
        total_rows = sum(
            int(value["target_rows"])
            for value in self.alignment_by_sequence.values()
        )
        valid_rows = sum(
            int(value["valid_rows"])
            for value in self.alignment_by_sequence.values()
        )
        return {
            "enabled": True,
            "mode": self.mode,
            "cache_manifest_path": str(self.cache_dir / "cache_manifest.json"),
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_sequence_fingerprint": self.cache_manifest.get(
                "sequence_fingerprint"
            ),
            "encoder": self.cache_manifest.get("encoder"),
            "encoder_fingerprint": self.cache_manifest.get(
                "encoder_fingerprint"
            ),
            "alignment_version": self.cache_manifest.get("alignment", {}).get(
                "version"
            ),
            "alignment_fingerprint": self.cache_manifest.get(
                "alignment_fingerprint"
            ),
            "time_basis": self.cache_manifest.get("alignment", {}).get(
                "time_basis"
            ),
            "projection_sha256": self.projection_sha256,
            "projection_path": str(self.projection_path),
            "projection_metadata_path": str(
                self.projection_path.with_suffix(".json")
            ),
            "projection_metadata_sha256": sha256_file(
                self.projection_path.with_suffix(".json")
            ),
            "projection_fit_split": self.projection_metadata.get("fit_split"),
            "projection_train_sequence_fingerprint": self.projection_metadata.get(
                "train_sequence_fingerprint"
            ),
            "projection_split_binding": dict(self.split_binding),
            "raw_embedding_dim": int(len(self.projection_mean)),
            "projected_embedding_dim": self.output_dim,
            "max_age_seconds": self.max_age_seconds,
            "random_seed": (
                self.random_seed if self.mode == "random_control" else None
            ),
            "verify_cache_hashes": self.verify_cache_hashes,
            "current_master_sources_verified": len(
                self.verified_master_sources
            ),
            "alignment": {
                "sequences": len(self.alignment_by_sequence),
                "target_rows": total_rows,
                "valid_rows": valid_rows,
                "missing_rows": total_rows - valid_rows,
                "coverage": valid_rows / total_rows if total_rows else 0.0,
                "future_matches": 0,
            },
        }
