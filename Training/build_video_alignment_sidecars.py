#!/usr/bin/env python3
"""Build hashed MP4-frame/VRS-capture DEVICE_TIME alignment sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_alignment import (
    VIDEO_ALIGNMENT_FILE_SUFFIX,
    build_video_alignment_sidecar,
    file_identity,
    sha256_file,
    validate_video_alignment_sidecar,
    validate_visual_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-cache-manifest", type=Path, required=True)
    parser.add_argument("--vrs-dir", type=Path, default=Path("Data_collection/Data_vrs"))
    parser.add_argument("--video-dir", type=Path, default=Path("Data_collection/Data_mp4"))
    parser.add_argument(
        "--master-dir", type=Path, default=Path("Data_collection/master_datasets")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def vrs_rgb_timestamps(vrs_path: Path, rgb_stream_id: str) -> np.ndarray:
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as exc:
        raise RuntimeError(
            "projectaria_tools is required to enumerate absolute VRS RGB timestamps"
        ) from exc
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not create VRS provider: {vrs_path}")
    stream_id = StreamId(rgb_stream_id)
    count = int(provider.get_num_data(stream_id))
    if count <= 0:
        raise ValueError(f"No RGB captures in {vrs_path}")
    timestamps = []
    for index in range(count):
        _, record = provider.get_image_data_by_index(stream_id, index)
        timestamps.append(int(record.capture_timestamp_ns))
    return np.asarray(timestamps, dtype=np.int64)


def reported_video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open MP4: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"MP4 reports no frames: {path}")
    return count


def main() -> int:
    args = parse_args()
    manifest_path = resolve(args.visual_cache_manifest)
    vrs_dir = resolve(args.vrs_dir)
    video_dir = resolve(args.video_dir)
    master_dir = resolve(args.master_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alignment, _ = validate_visual_manifest(manifest)
    manifest_sha256 = sha256_file(manifest_path)
    entries = manifest.get("entries", {})
    selected = list(dict.fromkeys(args.sequence)) if args.sequence else sorted(entries)
    if not selected:
        raise ValueError("No visual-cache sequences selected")

    for sequence_id in selected:
        if sequence_id not in entries:
            raise ValueError(f"Sequence absent from visual cache: {sequence_id}")
        paths = {
            "master": master_dir / f"{sequence_id}_master.csv",
            "vrs": vrs_dir / f"{sequence_id}.vrs",
            "mp4": video_dir / f"{sequence_id}.mp4",
        }
        source_files = {name: file_identity(path) for name, path in paths.items()}
        timestamps = vrs_rgb_timestamps(
            paths["vrs"], str(alignment["rgb_stream_id"])
        )
        video_frames = reported_video_frame_count(paths["mp4"])
        if video_frames != len(timestamps):
            raise ValueError(
                f"MP4/VRS RGB frame count mismatch for {sequence_id}: "
                f"{video_frames} != {len(timestamps)}"
            )
        sidecar = build_video_alignment_sidecar(
            sequence_id=sequence_id,
            rgb_capture_timestamps_ns=timestamps,
            source_files=source_files,
            visual_manifest=manifest,
            visual_manifest_sha256=manifest_sha256,
        )
        output_path = output_dir / f"{sequence_id}{VIDEO_ALIGNMENT_FILE_SUFFIX}"
        if output_path.exists() and not args.overwrite:
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                validate_video_alignment_sidecar(
                    existing,
                    sequence_id=sequence_id,
                    expected_source_files=source_files,
                    visual_manifest=manifest,
                    visual_manifest_sha256=manifest_sha256,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"Invalid sidecar, regenerating {sequence_id}: {exc}")
            else:
                print(f"Valid cached sidecar: {output_path}")
                continue
        output_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Created: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
