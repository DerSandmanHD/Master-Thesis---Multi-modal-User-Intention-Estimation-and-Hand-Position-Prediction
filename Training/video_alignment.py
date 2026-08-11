#!/usr/bin/env python3
"""Versioned MP4-frame to Project Aria DEVICE_TIME alignment sidecars."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from clip_alignment import (
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    canonical_json_hash,
    causal_source_indices,
    validate_strict_timestamps,
)


VIDEO_ALIGNMENT_SCHEMA_VERSION = "rgb_mp4_device_time_alignment_v1"
VIDEO_ALIGNMENT_FILE_SUFFIX = ".rgb_device_time_v2.json"
VIDEO_MAPPING_POLICY = (
    "MP4 frame ordinal equals VRS RGB stream record ordinal; frame counts are "
    "validated before and after rendering"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "file": resolved.name,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def validate_visual_manifest(manifest: Mapping[str, object]) -> tuple[dict, str]:
    alignment = manifest.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Visual cache manifest has no alignment specification")
    if alignment.get("version") != VISUAL_ALIGNMENT_VERSION:
        raise ValueError("Visual cache manifest uses an obsolete alignment version")
    if alignment.get("time_basis") != VISUAL_TIME_BASIS:
        raise ValueError("Visual cache manifest uses the wrong timestamp domain")
    fingerprint = canonical_json_hash(alignment)
    if manifest.get("alignment_fingerprint") != fingerprint:
        raise ValueError("Visual cache alignment fingerprint is invalid")
    return alignment, fingerprint


def _sidecar_fingerprint(sidecar: Mapping[str, object]) -> str:
    payload = dict(sidecar)
    payload.pop("sidecar_fingerprint", None)
    return canonical_json_hash(payload)


def build_video_alignment_sidecar(
    *,
    sequence_id: str,
    rgb_capture_timestamps_ns: np.ndarray,
    source_files: dict,
    visual_manifest: Mapping[str, object],
    visual_manifest_sha256: str,
) -> dict:
    timestamps = validate_strict_timestamps(
        rgb_capture_timestamps_ns, name="VRS RGB capture"
    )
    alignment, alignment_fingerprint = validate_visual_manifest(visual_manifest)
    entries = visual_manifest.get("entries")
    if not isinstance(entries, dict) or sequence_id not in entries:
        raise ValueError(
            f"Corrected visual cache manifest has no sequence {sequence_id}"
        )
    cache_entry = entries[sequence_id]
    if not isinstance(cache_entry, dict):
        raise ValueError(f"Invalid visual cache entry for {sequence_id}")
    cache_sources = cache_entry.get("source_files")
    expected_cache_sources = {
        name: source_files[name] for name in ("master", "vrs")
    }
    if cache_sources != expected_cache_sources:
        raise ValueError(
            f"Visual cache sources differ from current master/VRS for {sequence_id}"
        )
    timestamp_digest = hashlib.sha256(
        timestamps.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    sidecar = {
        "schema_version": VIDEO_ALIGNMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_id": sequence_id,
        "time_basis": VISUAL_TIME_BASIS,
        "clip_alignment_version": VISUAL_ALIGNMENT_VERSION,
        "clip_alignment_fingerprint": alignment_fingerprint,
        "visual_cache_manifest_sha256": visual_manifest_sha256,
        "rgb_stream_id": alignment.get("rgb_stream_id"),
        "rgb_timestamp_source": "VRS image_record.capture_timestamp_ns",
        "mapping_policy": VIDEO_MAPPING_POLICY,
        "frame_count": int(len(timestamps)),
        "first_capture_timestamp_ns": int(timestamps[0]),
        "last_capture_timestamp_ns": int(timestamps[-1]),
        "capture_timestamps_sha256": timestamp_digest,
        "rgb_capture_timestamps_ns": timestamps.tolist(),
        "source_files": source_files,
    }
    return {**sidecar, "sidecar_fingerprint": _sidecar_fingerprint(sidecar)}


def validate_video_alignment_sidecar(
    sidecar: Mapping[str, object],
    *,
    sequence_id: str,
    expected_source_files: dict,
    visual_manifest: Mapping[str, object],
    visual_manifest_sha256: str,
) -> np.ndarray:
    _, alignment_fingerprint = validate_visual_manifest(visual_manifest)
    if sidecar.get("schema_version") != VIDEO_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError("Legacy or unsupported video alignment sidecar")
    if sidecar.get("sequence_id") != sequence_id:
        raise ValueError("Video alignment sidecar sequence identity differs")
    if sidecar.get("time_basis") != VISUAL_TIME_BASIS:
        raise ValueError("Video alignment sidecar is not in Aria DEVICE_TIME")
    if sidecar.get("clip_alignment_version") != VISUAL_ALIGNMENT_VERSION:
        raise ValueError("Video sidecar uses an obsolete CLIP alignment")
    if sidecar.get("clip_alignment_fingerprint") != alignment_fingerprint:
        raise ValueError("Video sidecar and corrected CLIP alignment differ")
    if sidecar.get("visual_cache_manifest_sha256") != visual_manifest_sha256:
        raise ValueError("Video sidecar was built from another visual manifest")
    if sidecar.get("source_files") != expected_source_files:
        raise ValueError("Video sidecar sources changed; regenerate the sidecar")
    if sidecar.get("mapping_policy") != VIDEO_MAPPING_POLICY:
        raise ValueError("Video sidecar mapping policy is unsupported")
    if sidecar.get("sidecar_fingerprint") != _sidecar_fingerprint(sidecar):
        raise ValueError("Video sidecar fingerprint mismatch")
    timestamps = validate_strict_timestamps(
        np.asarray(sidecar.get("rgb_capture_timestamps_ns"), dtype=np.int64),
        name="sidecar RGB capture",
    )
    if int(sidecar.get("frame_count", -1)) != len(timestamps):
        raise ValueError("Video sidecar frame count does not match timestamps")
    timestamp_digest = hashlib.sha256(
        timestamps.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    if sidecar.get("capture_timestamps_sha256") != timestamp_digest:
        raise ValueError("Video sidecar timestamp-array hash mismatch")
    if int(sidecar.get("first_capture_timestamp_ns", -1)) != int(timestamps[0]):
        raise ValueError("Video sidecar first timestamp is inconsistent")
    if int(sidecar.get("last_capture_timestamp_ns", -1)) != int(timestamps[-1]):
        raise ValueError("Video sidecar last timestamp is inconsistent")
    return timestamps


def load_video_alignment_sidecar(
    path: Path,
    *,
    sequence_id: str,
    expected_source_files: dict,
    visual_manifest: Mapping[str, object],
    visual_manifest_sha256: str,
) -> tuple[dict, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required absolute-device-time video sidecar is missing: {path}. "
            "No MP4-zero/START fallback is permitted."
        )
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    timestamps = validate_video_alignment_sidecar(
        sidecar,
        sequence_id=sequence_id,
        expected_source_files=expected_source_files,
        visual_manifest=visual_manifest,
        visual_manifest_sha256=visual_manifest_sha256,
    )
    return sidecar, timestamps


def prediction_indices_for_rgb_frames(
    prediction_timestamps_ns: np.ndarray,
    rgb_capture_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Latest prediction at/before every absolute RGB capture timestamp."""

    predictions = validate_strict_timestamps(
        prediction_timestamps_ns, name="prediction endpoint"
    )
    rgb = validate_strict_timestamps(
        rgb_capture_timestamps_ns, name="RGB capture"
    )
    return causal_source_indices(predictions, rgb)


def first_rgb_frame_at_or_after(
    rgb_capture_timestamps_ns: np.ndarray,
    target_timestamp_ns: int,
) -> int:
    rgb = validate_strict_timestamps(
        rgb_capture_timestamps_ns, name="RGB capture"
    )
    index = int(np.searchsorted(rgb, int(target_timestamp_ns), side="left"))
    if index >= len(rgb):
        raise ValueError(
            "Target timestamp lies after the final VRS RGB capture; no frame "
            "may be substituted silently"
        )
    return index
