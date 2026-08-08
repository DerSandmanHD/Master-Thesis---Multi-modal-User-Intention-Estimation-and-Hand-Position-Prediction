#!/usr/bin/env python3
"""Load, validate, project, and causally align frozen visual embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SUPPORTED_VISUAL_MODES = ("append", "only", "random_control")
VISUAL_CACHE_SCHEMA_VERSION = 1
VISUAL_PROJECTION_SCHEMA_VERSION = 1


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
        required = {"timestamps_ns", "embeddings", "metadata_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"Visual cache {path} is missing arrays: {', '.join(missing)}"
            )
        timestamps_ns = archive["timestamps_ns"].astype(np.int64, copy=False)
        embeddings = archive["embeddings"].astype(np.float32, copy=False)
        metadata_value = archive["metadata_json"]
        metadata = json.loads(str(metadata_value.item()))
    if timestamps_ns.ndim != 1:
        raise ValueError(f"Visual cache timestamps must be one-dimensional: {path}")
    if embeddings.ndim != 2 or len(embeddings) != len(timestamps_ns):
        raise ValueError(f"Visual cache embeddings/timestamps do not align: {path}")
    if not len(timestamps_ns):
        raise ValueError(f"Visual cache is empty: {path}")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"Visual cache timestamps are not strictly increasing: {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Visual cache contains non-finite embeddings: {path}")
    if int(metadata.get("schema_version", -1)) != VISUAL_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported visual cache schema in {path}")
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
        if int(cache_manifest.get("schema_version", -1)) != (
            VISUAL_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported visual cache manifest schema")
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
    ) -> np.ndarray:
        cache_path, entry = self._entry(sequence_id)
        timestamps_ns, embeddings, metadata = load_cache(cache_path)
        if metadata.get("sequence_id") != sequence_id:
            raise ValueError(f"Visual cache sequence mismatch: {cache_path}")
        encoder_fingerprint = self.cache_manifest.get("encoder_fingerprint")
        if metadata.get("encoder_fingerprint") != encoder_fingerprint:
            raise ValueError(f"Visual cache encoder mismatch: {cache_path}")
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
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_sequence_fingerprint": self.cache_manifest.get(
                "sequence_fingerprint"
            ),
            "encoder": self.cache_manifest.get("encoder"),
            "encoder_fingerprint": self.cache_manifest.get(
                "encoder_fingerprint"
            ),
            "projection_sha256": self.projection_sha256,
            "projection_fit_split": self.projection_metadata.get("fit_split"),
            "projection_train_sequence_fingerprint": self.projection_metadata.get(
                "train_sequence_fingerprint"
            ),
            "raw_embedding_dim": int(len(self.projection_mean)),
            "projected_embedding_dim": self.output_dim,
            "max_age_seconds": self.max_age_seconds,
            "random_seed": (
                self.random_seed if self.mode == "random_control" else None
            ),
            "verify_cache_hashes": self.verify_cache_hashes,
            "alignment": {
                "sequences": len(self.alignment_by_sequence),
                "target_rows": total_rows,
                "valid_rows": valid_rows,
                "missing_rows": total_rows - valid_rows,
                "coverage": valid_rows / total_rows if total_rows else 0.0,
                "future_matches": 0,
            },
        }
