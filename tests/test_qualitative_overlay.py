from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from clip_alignment import (  # noqa: E402
    VISUAL_ALIGNMENT_VERSION,
    VISUAL_TIME_BASIS,
    alignment_specification,
    canonical_json_hash,
)
from render_prediction_overlay import (  # noqa: E402
    OVERLAY_SCHEMA_VERSION,
    choose_qualitative_cases,
    pose_bounds,
    qualitative_case_record,
    render_sequence,
    validate_prediction_report,
    validate_qualitative_columns,
)
from video_alignment import (  # noqa: E402
    VIDEO_ALIGNMENT_SCHEMA_VERSION,
    build_video_alignment_sidecar,
    first_rgb_frame_at_or_after,
    load_video_alignment_sidecar,
    prediction_indices_for_rgb_frames,
    validate_video_alignment_sidecar,
)


def source_files() -> dict:
    return {
        "master": {"file": "S_master.csv", "size_bytes": 10, "sha256": "m" * 64},
        "vrs": {"file": "S.vrs", "size_bytes": 20, "sha256": "v" * 64},
        "mp4": {"file": "S.mp4", "size_bytes": 30, "sha256": "p" * 64},
    }


def visual_manifest() -> dict:
    alignment = alignment_specification(sample_hz=5.0)
    return {
        "alignment": alignment,
        "alignment_fingerprint": canonical_json_hash(alignment),
        "entries": {
            "S": {
                "source_files": {
                    name: source_files()[name] for name in ("master", "vrs")
                }
            }
        },
    }


def qualitative_frame() -> pd.DataFrame:
    rows = []
    specifications = (
        ("a", True, 1.0, True),
        ("b", True, 4.0, True),
        ("c", False, 12.0, True),
        ("ignored", True, 0.1, False),
    )
    for index, (key, correct, error, available) in enumerate(specifications):
        row = {
            "sample_key": key,
            "participant": "P1" if index < 2 else "P2",
            "sequence_id": "S",
            "endpoint_timestamp_ns": 12_000_000_000 + index * 100_000_000,
            "target_intention": "handover",
            "predicted_intention": "handover" if correct else "fetch",
            "intention_correct": correct,
            "continue_probability": 0.05,
            "fetch_probability": 0.1 if correct else 0.8,
            "handover_probability": 0.85 if correct else 0.15,
            "sequence_receiving_hand": "left",
            "target_receiving_hand": "left",
            "predicted_receiving_hand": "left" if correct else "right",
            "predicted_receiving_hand_probability": 0.8,
            "pose_valid": True,
            "learned_end_to_end_available": available,
            "predicted_position_error_cm": error,
            "predicted_orientation_error_deg": error * 2,
            "modality_gaze_weight": 0.4,
            "modality_gaze_available": True,
            "modality_clip_weight": 0.6 if available else 0.0,
            "modality_clip_available": available,
        }
        for component_index, component in enumerate(
            ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
        ):
            value = 1.0 if component == "qw" else component_index * 0.01
            row[f"target_{component}"] = value
            row[f"predicted_{component}"] = value + (
                0.001 if component.endswith("_m") else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


class QualitativeOverlayTests(unittest.TestCase):
    def test_prediction_sidecar_binds_csv_content_not_only_row_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualitative_binding_") as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            qualitative_frame().to_csv(predictions, index=False)
            import hashlib

            run_dir = root / "run"
            selection_path = root / "validation_selection.json"
            selection = {
                "schema_version": 2,
                "matrix_id": "synthetic_matrix",
                "complete": True,
                "selection_split": "validation",
                "test_metrics_read": False,
                "final_test_runs": [
                    {
                        "experiment_id": "synthetic_model",
                        "seed": 42,
                        "run_dir": str(run_dir),
                        "checkpoint_name": "best_intention",
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_epoch": 3,
                        "checkpoint_selection_metric": (
                            "validation_intention_macro_f1"
                        ),
                        "checkpoint_selection_value": 0.5,
                        "artifact_manifest_fingerprint": "f" * 64,
                    }
                ],
            }
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            matrix_authorization = {
                "selection_file": str(selection_path),
                "selection_file_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
                "matrix_id": "synthetic_matrix",
                "experiment_id": "synthetic_model",
                "seed": 42,
                "authorized_checkpoint_sha256": "a" * 64,
                "test_metrics_read_during_authorization": False,
            }
            final_report_path = root / "final_test.json"
            final_report = {
                "schema_version": 2,
                "evaluation_protocol": (
                    "validation_frozen_checkpoint_single_test_v2"
                ),
                "report_fingerprint": None,
                "split": "test",
                "source_run": str(run_dir),
                "checkpoint": {
                    "name": "best_intention",
                    "sha256": "a" * 64,
                    "epoch": 3,
                    "selection_metric": "validation_intention_macro_f1",
                    "selection_value": 0.5,
                },
                "source_artifact_manifest_fingerprint": "f" * 64,
                "matrix_authorization": matrix_authorization,
                "test_used_for_model_or_checkpoint_selection": False,
            }
            final_report["report_fingerprint"] = canonical_json_hash(final_report)
            final_report_path.write_text(
                json.dumps(final_report), encoding="utf-8"
            )
            report = {
                "schema_version": 3,
                "report_fingerprint": None,
                "result_role": "primary_validation_selected_checkpoint",
                "checkpoint": "missing_checkpoint.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_epoch": 3,
                "checkpoint_selection_split": "validation",
                "checkpoint_selection_metric": "validation_intention_macro_f1",
                "checkpoint_selection_value": 0.5,
                "predictions_csv_sha256": hashlib.sha256(
                    predictions.read_bytes()
                ).hexdigest(),
                "dataset_content_fingerprint": "dataset",
                "source_content_fingerprint": "source",
                "artifact_freeze": {
                    "manifest": str(root / "artifact_manifest.json"),
                    "manifest_fingerprint": "f" * 64,
                },
                "final_test_authorization": {
                    "path": str(final_report_path),
                    "sha256": hashlib.sha256(
                        final_report_path.read_bytes()
                    ).hexdigest(),
                    "report_fingerprint": final_report["report_fingerprint"],
                    "evaluation_protocol": (
                        "validation_frozen_checkpoint_single_test_v2"
                    ),
                    "matrix_authorization": matrix_authorization,
                },
                "visual_artifacts": {"enabled": False},
                "architecture": {"fusion_mode": "modality_gated"},
                "split": "test",
                "full_split_export": True,
                "sequence_filter": [],
                "rows": 4,
            }
            report["report_fingerprint"] = canonical_json_hash(report)
            report_path = root / "predictions.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            artifact = {
                "manifest_fingerprint": "f" * 64,
                "dataset": {
                    "dataset_content_fingerprint": "dataset",
                    "source_content_fingerprint": "source",
                },
                "output_artifacts": {
                    "checkpoints": {
                        "best_intention": {"sha256": "a" * 64}
                    }
                },
            }
            validate_prediction_report(
                report,
                report_path=report_path,
                predictions_path=predictions,
                prediction_rows=4,
                artifact_validator=lambda _: artifact,
            )
            changed = pd.read_csv(predictions)
            changed.loc[0, "predicted_intention"] = "continue"
            changed.to_csv(predictions, index=False)
            with self.assertRaisesRegex(ValueError, "CSV SHA-256 differ"):
                validate_prediction_report(
                    report,
                    report_path=report_path,
                    predictions_path=predictions,
                    prediction_rows=4,
                    artifact_validator=lambda _: artifact,
                )

    def test_nonzero_start_and_known_rgb_mapping_use_absolute_device_time(self) -> None:
        second = 1_000_000_000
        rgb = np.asarray([10, 11, 12, 13], dtype=np.int64) * second
        # Spoken START is at 12 s, not at MP4 frame zero (10 s).
        predictions = np.asarray([12 * second, 12 * second + 500_000_000])
        indices = prediction_indices_for_rgb_frames(predictions, rgb)
        self.assertEqual(indices.tolist(), [-1, -1, 0, 1])
        self.assertEqual(first_rgb_frame_at_or_after(rgb, 12 * second), 2)

    def test_video_end_is_not_silently_replaced(self) -> None:
        rgb = np.asarray([10, 11, 12], dtype=np.int64)
        with self.assertRaises(ValueError):
            first_rgb_frame_at_or_after(rgb, 13)

    def test_sidecar_requires_corrected_manifest_and_invalidates_sources(self) -> None:
        timestamps = np.asarray([10, 11, 12], dtype=np.int64) * 1_000_000_000
        manifest = visual_manifest()
        sidecar = build_video_alignment_sidecar(
            sequence_id="S",
            rgb_capture_timestamps_ns=timestamps,
            source_files=source_files(),
            visual_manifest=manifest,
            visual_manifest_sha256="c" * 64,
        )
        validated = validate_video_alignment_sidecar(
            sidecar,
            sequence_id="S",
            expected_source_files=source_files(),
            visual_manifest=manifest,
            visual_manifest_sha256="c" * 64,
        )
        self.assertTrue(np.array_equal(validated, timestamps))
        self.assertEqual(sidecar["schema_version"], VIDEO_ALIGNMENT_SCHEMA_VERSION)
        changed = source_files()
        changed["mp4"] = {**changed["mp4"], "sha256": "x" * 64}
        with self.assertRaises(ValueError):
            validate_video_alignment_sidecar(
                sidecar,
                sequence_id="S",
                expected_source_files=changed,
                visual_manifest=manifest,
                visual_manifest_sha256="c" * 64,
            )
        legacy = dict(sidecar)
        legacy["schema_version"] = "legacy_start_relative_v1"
        with self.assertRaises(ValueError):
            validate_video_alignment_sidecar(
                legacy,
                sequence_id="S",
                expected_source_files=source_files(),
                visual_manifest=manifest,
                visual_manifest_sha256="c" * 64,
            )

    def test_missing_sidecar_has_no_mp4_start_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="missing_sidecar_") as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No MP4-zero/START fallback"):
                load_video_alignment_sidecar(
                    Path(directory) / "missing.json",
                    sequence_id="S",
                    expected_source_files=source_files(),
                    visual_manifest=visual_manifest(),
                    visual_manifest_sha256="c" * 64,
                )

    def test_cases_are_reproducible_complete_and_pose_available_only(self) -> None:
        frame = qualitative_frame()
        validate_qualitative_columns(frame)
        first = choose_qualitative_cases(frame, seed=42)
        second = choose_qualitative_cases(frame, seed=42)
        self.assertEqual(
            {name: row["sample_key"] for name, row in first.items()},
            {name: row["sample_key"] for name, row in second.items()},
        )
        self.assertEqual(set(first), {"good", "typical", "failure"})
        self.assertTrue(
            all(bool(row["learned_end_to_end_available"]) for row in first.values())
        )
        self.assertNotIn("ignored", {row["sample_key"] for row in first.values()})
        rgb = np.arange(10_000_000_000, 14_000_000_001, 100_000_000)
        record = qualitative_case_record(
            "good",
            first["good"],
            rgb_capture_timestamps_ns=rgb,
            checkpoint_provenance={"checkpoint_sha256": "a" * 64},
        )
        expected = {
            "participant",
            "sequence_id",
            "ground_truth_intention",
            "predicted_intention",
            "class_probabilities",
            "ground_truth_receiving_hand",
            "predicted_receiving_hand",
            "ground_truth_future_wrist",
            "predicted_future_wrist",
            "position_error_cm",
            "orientation_error_deg",
            "modality_weights",
            "modality_available",
            "available_modalities",
            "missing_modalities",
            "checkpoint_provenance",
        }
        self.assertTrue(expected.issubset(record))
        self.assertTrue(record["learned_end_to_end_available"])
        self.assertEqual(record["modality_weights"], {"clip": 0.6, "gaze": 0.4})
        self.assertEqual(OVERLAY_SCHEMA_VERSION, "qualitative_overlay_device_time_v2")

        bounds = pose_bounds(frame)
        # The unavailable row cannot inject its values into pose visualization.
        self.assertLess(float(np.max(np.abs(np.concatenate(bounds)))), 2.0)

    @unittest.skipIf(cv2 is None, "OpenCV is unavailable in the lightweight test env")
    def test_renderer_matches_predictions_to_vrs_frame_timestamps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="device_time_overlay_") as directory:
            root = Path(directory)
            video = root / "S.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for value in range(4):
                writer.write(np.full((48, 64, 3), value * 30, dtype=np.uint8))
            writer.release()
            sidecar_path = root / "S.sidecar.json"
            sidecar_path.write_text("{}", encoding="utf-8")
            rgb = np.asarray([10, 11, 12, 13], dtype=np.int64) * 1_000_000_000
            group = qualitative_frame().iloc[:2].copy()
            group["endpoint_timestamp_ns"] = [
                11_500_000_000,
                11_750_000_000,
            ]
            seen = []

            def fake_annotate(frame, row, bounds, *, prediction_age_s):
                seen.append(None if row is None else str(row["sample_key"]))
                panel_h = int(185 * max(0.75, frame.shape[1] / 1800.0))
                return np.zeros(
                    (frame.shape[0] + panel_h, frame.shape[1], 3), dtype=np.uint8
                )

            sidecar = {
                "time_basis": VISUAL_TIME_BASIS,
                "clip_alignment_version": VISUAL_ALIGNMENT_VERSION,
                "clip_alignment_fingerprint": "f" * 64,
                "schema_version": VIDEO_ALIGNMENT_SCHEMA_VERSION,
                "sidecar_fingerprint": "s" * 64,
            }
            with patch(
                "render_prediction_overlay.annotate_frame",
                side_effect=fake_annotate,
            ):
                report = render_sequence(
                    "S",
                    group,
                    video,
                    root,
                    rgb_capture_timestamps_ns=rgb,
                    alignment_sidecar=sidecar,
                    alignment_sidecar_path=sidecar_path,
                    requested_stills={
                        "good": {
                            "sample_key": "a",
                            "display_rgb_frame_index": 2,
                            "display_rgb_capture_timestamp_ns": 12_000_000_000,
                        }
                    },
                    max_prediction_age_s=1.0,
                    use_transcode=False,
                )
            # Video frame 2 uses latest causal row b, while the selected still
            # is deliberately re-annotated with exact case row a.
            self.assertEqual(seen, [None, None, "b", "a", None])
            self.assertEqual(report["stills"]["good"]["sample_key"], "a")
            self.assertEqual(report["future_prediction_matches"], 0)
            self.assertIn("endpoint_timestamp_ns", report["alignment_policy"])


if __name__ == "__main__":
    unittest.main()
