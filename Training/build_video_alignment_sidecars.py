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
    map_vrs_rgb_frames_to_video,
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
    parser.add_argument(
        "--exclude-vrs-rgb-frame",
        action="append",
        default=[],
        metavar="SEQUENCE_ID:FRAME_INDEX",
        help=(
            "Reviewed VRS RGB ordinal with no MP4 counterpart. This is explicit "
            "rather than a silent frame-count truncation."
        ),
    )
    parser.add_argument(
        "--frame-exclusion-reason",
        default=None,
        help="Required review note when --exclude-vrs-rgb-frame is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_frame_exclusions(values: list[str]) -> dict[str, tuple[int, ...]]:
    exclusions: dict[str, list[int]] = {}
    for value in values:
        sequence_id, separator, index_text = value.rpartition(":")
        if not separator or not sequence_id or not index_text.isdigit():
            raise ValueError(
                "--exclude-vrs-rgb-frame must be SEQUENCE_ID:FRAME_INDEX"
            )
        exclusions.setdefault(sequence_id, []).append(int(index_text))
    parsed = {key: tuple(sorted(indices)) for key, indices in exclusions.items()}
    if any(len(indices) != len(set(indices)) for indices in parsed.values()):
        raise ValueError("Duplicate VRS RGB frame exclusion")
    return parsed


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
    exclusions = parse_frame_exclusions(args.exclude_vrs_rgb_frame)
    if exclusions and not args.frame_exclusion_reason:
        raise ValueError(
            "--frame-exclusion-reason is required with --exclude-vrs-rgb-frame"
        )
    unknown_exclusions = sorted(set(exclusions) - set(selected))
    if unknown_exclusions:
        raise ValueError(
            "VRS RGB frame exclusions refer to unselected sequences: "
            + ", ".join(unknown_exclusions)
        )

    for sequence_id in selected:
        if sequence_id not in entries:
            raise ValueError(f"Sequence absent from visual cache: {sequence_id}")
        paths = {
            "master": master_dir / f"{sequence_id}_master.csv",
            "vrs": vrs_dir / f"{sequence_id}.vrs",
            "mp4": video_dir / f"{sequence_id}.mp4",
        }
        source_files = {name: file_identity(path) for name, path in paths.items()}
        source_timestamps = vrs_rgb_timestamps(
            paths["vrs"], str(alignment["rgb_stream_id"])
        )
        video_frames = reported_video_frame_count(paths["mp4"])
        excluded = exclusions.get(sequence_id, ())
        timestamps = map_vrs_rgb_frames_to_video(
            source_timestamps,
            video_frame_count=video_frames,
            excluded_vrs_rgb_frame_indices=excluded,
        )
        sidecar = build_video_alignment_sidecar(
            sequence_id=sequence_id,
            rgb_capture_timestamps_ns=timestamps,
            source_files=source_files,
            visual_manifest=manifest,
            visual_manifest_sha256=manifest_sha256,
            source_vrs_rgb_frame_count=len(source_timestamps),
            excluded_vrs_rgb_frame_indices=excluded,
            frame_exclusion_reason=(args.frame_exclusion_reason if excluded else None),
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
