#!/usr/bin/env python3
"""Focused scientific-invariant tests for CLIP/master timestamp alignment."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from clip_alignment import (
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    absolute_alignment_statistics,
    alignment_specification,
    cache_metadata_matches,
    canonical_json_hash,
    causal_source_indices,
    infer_master_start_timestamp_ns,
    select_device_time_sample_indices,
)
from extract_clip_embeddings import valid_existing_cache, write_cache
from visual_embeddings import VISUAL_CACHE_SCHEMA_VERSION


class ClipAlignmentTests(unittest.TestCase):
    def test_start_at_rgb_origin_uses_absolute_device_time(self) -> None:
        rgb_origin = 10_000_000_000
        master = np.asarray([rgb_origin, rgb_origin + 500_000_000], dtype=np.int64)
        elapsed = np.asarray([0.0, 0.5], dtype=np.float64)
        self.assertEqual(
            infer_master_start_timestamp_ns(master, elapsed),
            rgb_origin,
        )

    def test_nonzero_start_offset_preserves_absolute_origin(self) -> None:
        rgb_origin = 10_000_000_000
        start = rgb_origin + 2_000_000_000
        master = np.asarray(
            [start + 100_000_000, start + 700_000_000],
            dtype=np.int64,
        )
        elapsed = np.asarray([0.1, 0.7], dtype=np.float64)
        self.assertEqual(infer_master_start_timestamp_ns(master, elapsed), start)
        self.assertEqual(start - rgb_origin, 2_000_000_000)

    def test_known_rgb_timestamp_maps_to_latest_causal_frame(self) -> None:
        rgb = np.asarray([1_000, 2_000, 3_000], dtype=np.int64)
        targets = np.asarray([999, 1_000, 2_500, 3_000], dtype=np.int64)
        self.assertEqual(
            causal_source_indices(rgb, targets).tolist(),
            [-1, 0, 1, 2],
        )

    def test_start_offset_is_not_subtracted_from_rgb_timestamps(self) -> None:
        second = 1_000_000_000
        rgb = np.asarray([10, 11, 12, 13], dtype=np.int64) * second
        start = 12 * second
        target = np.asarray([start + 100_000_000], dtype=np.int64)
        selected = causal_source_indices(rgb, target)
        self.assertEqual(int(selected[0]), 2)
        self.assertEqual(int(rgb[selected[0]]), 12 * second)
        statistics = absolute_alignment_statistics(rgb, target)
        self.assertEqual(statistics["future_matches"], 0)
        self.assertAlmostEqual(statistics["mean_age_ms"], 100.0)

    def test_alignment_and_source_changes_invalidate_cache(self) -> None:
        alignment = alignment_specification(sample_hz=5.0)
        fingerprint = canonical_json_hash(alignment)
        sources = {
            "master": {"file": "S_master.csv", "size_bytes": 1, "sha256": "m1"},
            "vrs": {"file": "S.vrs", "size_bytes": 2, "sha256": "v1"},
        }
        metadata = {
            "sequence_id": "S",
            "encoder_fingerprint": "encoder",
            "alignment_version": VISUAL_ALIGNMENT_VERSION,
            "time_basis": VISUAL_TIME_BASIS,
            "alignment_fingerprint": fingerprint,
            "source_files": sources,
        }
        expected = {
            "sequence_id": "S",
            "encoder_fingerprint": "encoder",
            "alignment_fingerprint": fingerprint,
            "source_files": sources,
        }
        self.assertTrue(cache_metadata_matches(metadata, **expected))
        legacy = dict(metadata)
        legacy.pop("alignment_version")
        self.assertFalse(cache_metadata_matches(legacy, **expected))
        changed_alignment = dict(expected, alignment_fingerprint="changed")
        self.assertFalse(cache_metadata_matches(metadata, **changed_alignment))
        changed_sources = {
            **sources,
            "master": {**sources["master"], "sha256": "m2"},
        }
        self.assertFalse(
            cache_metadata_matches(
                metadata,
                **dict(expected, source_files=changed_sources),
            )
        )

    def test_sampling_includes_true_end_of_vrs(self) -> None:
        rgb = np.asarray(
            [
                5_000_000_000,
                5_100_000_000,
                5_500_000_000,
                5_900_000_000,
                6_010_000_000,
                6_130_000_000,
            ],
            dtype=np.int64,
        )
        selected = select_device_time_sample_indices(rgb, sample_hz=2.0)
        self.assertEqual(int(selected[0]), 0)
        self.assertEqual(int(selected[-1]), len(rgb) - 1)
        self.assertEqual(int(rgb[selected[-1]]), int(rgb[-1]))

    def test_production_cache_validator_rejects_legacy_and_changed_source(self) -> None:
        alignment = alignment_specification(sample_hz=5.0)
        fingerprint = canonical_json_hash(alignment)
        sources = {
            "master": {"file": "S_master.csv", "size_bytes": 1, "sha256": "m1"},
            "vrs": {"file": "S.vrs", "size_bytes": 2, "sha256": "v1"},
        }
        metadata = {
            "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "sequence_id": "S",
            "encoder_fingerprint": "encoder",
            "alignment_version": VISUAL_ALIGNMENT_VERSION,
            "time_basis": VISUAL_TIME_BASIS,
            "alignment": alignment,
            "alignment_fingerprint": fingerprint,
            "source_files": sources,
        }
        with tempfile.TemporaryDirectory(prefix="clip_cache_test_") as directory:
            cache = Path(directory) / "S.npz"
            write_cache(
                cache,
                timestamps_ns=np.asarray([1, 2], dtype=np.int64),
                frame_indices=np.asarray([0, 1], dtype=np.int64),
                embeddings=np.ones((2, 4), dtype=np.float32),
                metadata=metadata,
            )
            self.assertTrue(
                valid_existing_cache(
                    cache,
                    sequence_id="S",
                    encoder_fingerprint="encoder",
                    alignment_fingerprint=fingerprint,
                    source_files=sources,
                )
            )
            changed = {
                **sources,
                "master": {**sources["master"], "sha256": "m2"},
            }
            self.assertFalse(
                valid_existing_cache(
                    cache,
                    sequence_id="S",
                    encoder_fingerprint="encoder",
                    alignment_fingerprint=fingerprint,
                    source_files=changed,
                )
            )
            legacy = Path(directory) / "legacy.npz"
            np.savez_compressed(
                legacy,
                timestamps_ns=np.asarray([1], dtype=np.int64),
                embeddings=np.ones((1, 4), dtype=np.float32),
                metadata_json=np.asarray("{}"),
            )
            self.assertFalse(
                valid_existing_cache(
                    legacy,
                    sequence_id="S",
                    encoder_fingerprint="encoder",
                    alignment_fingerprint=fingerprint,
                    source_files=sources,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
