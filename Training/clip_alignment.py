#!/usr/bin/env python3
"""Pure helpers for absolute-device-time visual feature alignment.

The CLIP extraction path deliberately keeps Project Aria RGB capture timestamps
in their native ``DEVICE_TIME`` domain.  Master rows use the same absolute
nanosecond domain, so no video-relative or START-relative clocks are compared.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


VISUAL_ALIGNMENT_VERSION = "vrs_rgb_device_time_v2"
VISUAL_TIME_BASIS = "project_aria_device_time_capture_timestamp_ns"
DEFAULT_RGB_STREAM_ID = "214-1"


def canonical_json_hash(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def alignment_specification(
    *,
    rgb_stream_id: str = DEFAULT_RGB_STREAM_ID,
    sample_hz: float,
) -> dict:
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    return {
        "version": VISUAL_ALIGNMENT_VERSION,
        "time_basis": VISUAL_TIME_BASIS,
        "rgb_stream_id": str(rgb_stream_id),
        "rgb_timestamp_source": "VRS image_record.capture_timestamp_ns",
        "frame_source": "VRS RGB frames decoded by projectaria_tools",
        "sampling_policy": "device_time_grid_first_frame_plus_final_frame",
        "sampling_hz": float(sample_hz),
        "master_alignment": (
            "latest RGB capture_timestamp_ns at or before absolute "
            "master timestamp_ns"
        ),
    }


def validate_strict_timestamps(
    timestamps_ns: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} timestamps must be a non-empty 1D array")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} timestamps must be strictly increasing")
    return values


def infer_master_start_timestamp_ns(
    master_timestamps_ns: np.ndarray,
    time_since_start_s: np.ndarray,
    *,
    tolerance_ns: int = 100_000,
) -> int:
    """Infer and validate the absolute spoken-START timestamp.

    Every master row must imply the same origin via
    ``timestamp_ns - time_since_start_s * 1e9``.  The small tolerance permits
    CSV floating-point round trips while still detecting mixed clock origins.
    """
    timestamps = validate_strict_timestamps(
        master_timestamps_ns,
        name="master",
    )
    elapsed = np.asarray(time_since_start_s, dtype=np.float64)
    if elapsed.ndim != 1 or len(elapsed) != len(timestamps):
        raise ValueError("Master timestamps and elapsed values do not align")
    if not np.isfinite(elapsed).all() or np.any(elapsed < 0):
        raise ValueError("Master elapsed values must be finite and non-negative")
    if np.any(np.diff(elapsed) < 0):
        raise ValueError("Master elapsed values must be non-decreasing")
    implied = timestamps - np.rint(elapsed * 1e9).astype(np.int64)
    origin = int(np.rint(np.median(implied)))
    maximum_deviation = int(np.max(np.abs(implied - origin)))
    if maximum_deviation > int(tolerance_ns):
        raise ValueError(
            "Master timestamp_ns and time_since_start_s use inconsistent origins: "
            f"maximum deviation is {maximum_deviation} ns"
        )
    return origin


@dataclass
class DeviceTimeSampler:
    """Streaming device-time sampler shared by production and unit tests."""

    sample_hz: float
    _next_regular_ns: int | None = None
    _previous_ns: int | None = None

    def __post_init__(self) -> None:
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self._period_ns = max(1, int(round(1e9 / float(self.sample_hz))))

    def should_sample(self, timestamp_ns: int, *, is_final: bool = False) -> bool:
        timestamp_ns = int(timestamp_ns)
        if self._previous_ns is not None and timestamp_ns <= self._previous_ns:
            raise ValueError("RGB capture timestamps must be strictly increasing")
        self._previous_ns = timestamp_ns
        if self._next_regular_ns is None:
            self._next_regular_ns = timestamp_ns
        regular = timestamp_ns >= self._next_regular_ns
        if regular:
            skipped_periods = (
                (timestamp_ns - self._next_regular_ns) // self._period_ns
            )
            self._next_regular_ns += (skipped_periods + 1) * self._period_ns
        return bool(regular or is_final)


def select_device_time_sample_indices(
    rgb_timestamps_ns: np.ndarray,
    *,
    sample_hz: float,
) -> np.ndarray:
    timestamps = validate_strict_timestamps(rgb_timestamps_ns, name="RGB")
    sampler = DeviceTimeSampler(sample_hz)
    selected = [
        index
        for index, timestamp_ns in enumerate(timestamps)
        if sampler.should_sample(
            int(timestamp_ns),
            is_final=index == len(timestamps) - 1,
        )
    ]
    return np.asarray(selected, dtype=np.int64)


def causal_source_indices(
    source_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Return the latest source index at/before each target, or -1."""
    source = validate_strict_timestamps(source_timestamps_ns, name="source")
    targets = np.asarray(target_timestamps_ns, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("Target timestamps must be one-dimensional")
    return np.searchsorted(source, targets, side="right") - 1


def absolute_alignment_statistics(
    source_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> dict:
    source = validate_strict_timestamps(source_timestamps_ns, name="source")
    targets = np.asarray(target_timestamps_ns, dtype=np.int64)
    indices = causal_source_indices(source, targets)
    valid = indices >= 0
    ages_ns = np.zeros(len(targets), dtype=np.int64)
    ages_ns[valid] = targets[valid] - source[indices[valid]]
    if np.any(ages_ns[valid] < 0):
        raise ValueError("Absolute alignment selected a future RGB frame")
    ages_ms = ages_ns[valid].astype(np.float64) / 1e6
    return {
        "target_rows": int(len(targets)),
        "valid_rows": int(valid.sum()),
        "missing_rows": int((~valid).sum()),
        "coverage_without_age_limit": float(valid.mean()) if len(valid) else 0.0,
        "mean_age_ms": float(ages_ms.mean()) if len(ages_ms) else None,
        "max_age_ms": float(ages_ms.max()) if len(ages_ms) else None,
        "future_matches": 0,
    }


def cache_metadata_matches(
    metadata: dict,
    *,
    sequence_id: str,
    encoder_fingerprint: str,
    alignment_fingerprint: str,
    source_files: dict,
) -> bool:
    """Strict cache identity check; absent legacy fields invalidate a cache."""
    return bool(
        metadata.get("sequence_id") == sequence_id
        and metadata.get("encoder_fingerprint") == encoder_fingerprint
        and metadata.get("alignment_version") == VISUAL_ALIGNMENT_VERSION
        and metadata.get("time_basis") == VISUAL_TIME_BASIS
        and metadata.get("alignment_fingerprint") == alignment_fingerprint
        and metadata.get("source_files") == source_files
    )
