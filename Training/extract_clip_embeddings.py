#!/usr/bin/env python3
"""Extract frozen CLIP image embeddings from timestamp-aligned MP4 frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import manifest_filtered_master_files, sequence_id_from_master_path
from visual_embeddings import (
    VISUAL_CACHE_SCHEMA_VERSION,
    load_cache,
    sequence_fingerprint,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-dir", type=Path, default=Path("Data_collection/master_datasets")
    )
    parser.add_argument(
        "--video-dir", type=Path, default=Path("Data_collection/Data_mp4")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("Data_collection/dataset_manifest.csv")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sequence-fingerprint", default=None)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--weights-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--expected-weights-sha256",
        required=True,
        help="Pinned SHA-256 that must match before the checkpoint is loaded.",
    )
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def canonical_json_hash(data: dict) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocessing_specification(preprocess) -> list[dict]:
    """Return a stable, address-free description of torchvision transforms."""
    specification = []
    for transform in getattr(preprocess, "transforms", [preprocess]):
        if isinstance(transform, types.FunctionType):
            specification.append(
                {
                    "type": "callable",
                    "name": (
                        f"{getattr(transform, '__module__', '')}."
                        f"{getattr(transform, '__qualname__', type(transform).__name__)}"
                    ).strip("."),
                }
            )
            continue
        item = {"type": type(transform).__name__}
        for name in ("size", "interpolation", "max_size", "antialias", "mean", "std"):
            if not hasattr(transform, name):
                continue
            value = getattr(transform, name)
            if isinstance(value, (tuple, list)):
                item[name] = [float(entry) for entry in value]
            elif isinstance(value, (str, int, float, bool)) or value is None:
                item[name] = value
            else:
                item[name] = str(value)
        specification.append(item)
    return specification


def resolve_sequences(args: argparse.Namespace) -> tuple[list[Path], str]:
    master_dir = project_path(args.master_dir).resolve()
    manifest = project_path(args.manifest).resolve()
    files = sorted(master_dir.glob("*_master.csv"))
    selected, metadata = manifest_filtered_master_files(
        files,
        master_dir,
        {
            "path": str(manifest),
            "allowed_statuses": ["valid"],
            "allowed_next_actions": ["ready_for_master_merge"],
            "strict": True,
        },
    )
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    ids = [sequence_id_from_master_path(path) for path in selected]
    fingerprint = sequence_fingerprint(ids)
    if args.limit is None and args.expected_sequence_fingerprint:
        if fingerprint != args.expected_sequence_fingerprint:
            raise ValueError(
                "Selected sequence fingerprint differs from the frozen dataset: "
                f"{fingerprint} != {args.expected_sequence_fingerprint}"
            )
    if not selected:
        raise ValueError("No master datasets were selected")
    return selected, fingerprint


def load_master_clock(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = pd.read_csv(path, nrows=0).columns
    if "time_since_start_s" not in header:
        raise ValueError(f"{path.name} has no time_since_start_s column")
    frame = pd.read_csv(path, usecols=["timestamp_ns", "time_since_start_s"])
    timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(
        np.int64
    )
    elapsed = pd.to_numeric(
        frame["time_since_start_s"], errors="raise"
    ).to_numpy(np.float64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Master clock is empty or unsorted: {path}")
    if np.any(np.diff(elapsed) < 0) or not np.isfinite(elapsed).all():
        raise ValueError(f"Master elapsed time is invalid: {path}")
    return timestamps, elapsed


def flush_batch(
    model,
    tensors: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    if not tensors:
        return np.empty((0, 0), dtype=np.float32)
    batch = torch.stack(tensors).to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        encoded = model.encode_image(batch)
        encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
    return encoded.cpu().numpy().astype(np.float32)


def extract_video(
    *,
    sequence_id: str,
    video_path: Path,
    master_path: Path,
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    sample_hz: float,
    common_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    processing_started = time.perf_counter()
    master_timestamps, master_elapsed = load_master_clock(master_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video FPS is invalid: {video_path}")

    next_sample_time = 0.0
    frame_index = 0
    batch_tensors: list[torch.Tensor] = []
    batch_times: list[float] = []
    embedding_chunks: list[np.ndarray] = []
    sampled_times: list[float] = []
    while True:
        ok = capture.grab()
        if not ok:
            break
        frame_time = frame_index / fps
        if frame_time + 0.5 / fps >= next_sample_time:
            ok, frame = capture.retrieve()
            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_tensors.append(preprocess(Image.fromarray(rgb)))
            batch_times.append(frame_time)
            next_sample_time += 1.0 / sample_hz
            if len(batch_tensors) >= batch_size:
                embedding_chunks.append(flush_batch(model, batch_tensors, device))
                sampled_times.extend(batch_times)
                batch_tensors.clear()
                batch_times.clear()
        frame_index += 1
    capture.release()
    if batch_tensors:
        embedding_chunks.append(flush_batch(model, batch_tensors, device))
        sampled_times.extend(batch_times)
    if not embedding_chunks:
        raise ValueError(f"No frames were sampled from {video_path}")

    embeddings = np.concatenate(embedding_chunks, axis=0)
    frame_times = np.asarray(sampled_times, dtype=np.float64)
    # A frame becomes available only at the first sensor timestamp at or after
    # its presentation time. The later loader uses a backward lookup, so this
    # ceil mapping prevents future-frame leakage.
    master_indices = np.searchsorted(master_elapsed, frame_times, side="left")
    in_range = master_indices < len(master_timestamps)
    embeddings = embeddings[in_range]
    frame_times = frame_times[in_range]
    master_indices = master_indices[in_range]
    timestamps = master_timestamps[master_indices]
    sync_error_ms = (master_elapsed[master_indices] - frame_times) * 1000.0

    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps = timestamps[unique]
    embeddings = embeddings[unique]
    frame_times = frame_times[unique]
    sync_error_ms = sync_error_ms[unique]
    if np.any(sync_error_ms < -1e-6):
        raise ValueError(f"Non-causal video/master alignment in {sequence_id}")
    processing_wall_seconds = time.perf_counter() - processing_started

    metadata = {
        **common_metadata,
        "sequence_id": sequence_id,
        "video_file": video_path.name,
        "video_size_bytes": video_path.stat().st_size,
        "video_fps": fps,
        "video_reported_frame_count": frame_count,
        "video_decoded_frame_count": frame_index,
        "master_file": master_path.name,
        "master_sha256": sha256_file(master_path),
        "samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "frame_time_first_s": float(frame_times[0]),
        "frame_time_last_s": float(frame_times[-1]),
        "master_duration_s": float(master_elapsed[-1]),
        "alignment_policy": "frame_time_ceil_to_master_then_causal_backward_fill",
        "alignment_error_ms_mean": float(sync_error_ms.mean()),
        "alignment_error_ms_max": float(sync_error_ms.max()),
        "future_frame_matches": int((sync_error_ms < -1e-6).sum()),
        "decode_preprocess_encoder_wall_seconds": processing_wall_seconds,
        "embeddings_per_processing_second": float(
            len(embeddings) / processing_wall_seconds
        ),
        "source_duration_to_processing_wall_ratio": float(
            master_elapsed[-1] / processing_wall_seconds
        ),
    }
    return timestamps, frame_times, embeddings, metadata


def valid_existing_cache(
    path: Path,
    *,
    sequence_id: str,
    encoder_fingerprint: str,
) -> bool:
    try:
        _, _, metadata = load_cache(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        metadata.get("sequence_id") == sequence_id
        and metadata.get("encoder_fingerprint") == encoder_fingerprint
    )


def write_cache(
    path: Path,
    *,
    timestamps_ns: np.ndarray,
    frame_times_s: np.ndarray,
    embeddings: np.ndarray,
    metadata: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            timestamps_ns=timestamps_ns.astype(np.int64),
            frame_times_s=frame_times_s.astype(np.float64),
            embeddings=embeddings.astype(np.float16),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            ),
        )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.sample_hz <= 0 or args.batch_size <= 0:
        raise ValueError("sample_hz and batch_size must be positive")
    selected, fingerprint = resolve_sequences(args)
    output_dir = project_path(args.output_dir).resolve()
    video_dir = project_path(args.video_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip is required; install Training/clip_requirements.txt "
            "into the configured PYTHONPATH"
        ) from exc
    weights_cache = (
        project_path(args.weights_cache_dir).resolve()
        if args.weights_cache_dir is not None
        else None
    )
    pretrained_config = open_clip.get_pretrained_cfg(
        args.model_name, args.pretrained
    )
    if pretrained_config is None:
        raise ValueError(
            f"Unknown pretrained CLIP pair: {args.model_name}/{args.pretrained}"
        )
    checkpoint_path = Path(
        open_clip.download_pretrained(
            pretrained_config,
            cache_dir=str(weights_cache) if weights_cache else None,
        )
    )
    weights_sha256 = sha256_file(checkpoint_path)
    if weights_sha256.lower() != args.expected_weights_sha256.lower():
        raise RuntimeError(
            "Pinned CLIP checkpoint hash mismatch: "
            f"expected {args.expected_weights_sha256}, got {weights_sha256}"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=str(checkpoint_path),
        device=device,
        cache_dir=str(weights_cache) if weights_cache else None,
    )
    model.eval()
    encoder = {
        "library": "open_clip_torch",
        "library_version": getattr(open_clip, "__version__", "unknown"),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "weights_file": checkpoint_path.name,
        "weights_sha256": weights_sha256,
        "expected_weights_sha256": args.expected_weights_sha256,
        "hash_verified_before_deserialization": True,
        "preprocess": preprocessing_specification(preprocess),
        "output_normalization": "L2",
        "sampling_hz": float(args.sample_hz),
        "frozen": True,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": 0,
    }
    encoder_fingerprint = canonical_json_hash(encoder)
    common_metadata = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder,
        "encoder_fingerprint": encoder_fingerprint,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    print(f"Device: {device}")
    print(f"Sequences: {len(selected)} ({fingerprint})")
    print(f"Encoder fingerprint: {encoder_fingerprint}")
    errors: dict[str, str] = {}
    for index, master_path in enumerate(selected, start=1):
        sequence_id = sequence_id_from_master_path(master_path)
        output_path = output_dir / f"{sequence_id}.npz"
        if (
            output_path.is_file()
            and not args.overwrite
            and valid_existing_cache(
                output_path,
                sequence_id=sequence_id,
                encoder_fingerprint=encoder_fingerprint,
            )
        ):
            print(f"[{index}/{len(selected)}] {sequence_id}: cached", flush=True)
            continue
        video_path = video_dir / f"{sequence_id}.mp4"
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            timestamps, frame_times, embeddings, metadata = extract_video(
                sequence_id=sequence_id,
                video_path=video_path,
                master_path=master_path,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
                sample_hz=args.sample_hz,
                common_metadata=common_metadata,
            )
            write_cache(
                output_path,
                timestamps_ns=timestamps,
                frame_times_s=frame_times,
                embeddings=embeddings,
                metadata=metadata,
            )
            print(
                f"[{index}/{len(selected)}] {sequence_id}: "
                f"{len(embeddings)} embeddings",
                flush=True,
            )
        except Exception as exc:  # continue to produce a complete error report
            errors[sequence_id] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index}/{len(selected)}] {sequence_id}: ERROR {errors[sequence_id]}",
                flush=True,
            )

    entries = {}
    for master_path in selected:
        sequence_id = sequence_id_from_master_path(master_path)
        cache_path = output_dir / f"{sequence_id}.npz"
        if not cache_path.is_file():
            continue
        try:
            timestamps, embeddings, metadata = load_cache(cache_path)
            if metadata.get("encoder_fingerprint") != encoder_fingerprint:
                raise ValueError("encoder fingerprint mismatch")
            entries[sequence_id] = {
                "file": cache_path.name,
                "sha256": sha256_file(cache_path),
                "samples": int(len(timestamps)),
                "embedding_dim": int(embeddings.shape[1]),
                "first_timestamp_ns": int(timestamps[0]),
                "last_timestamp_ns": int(timestamps[-1]),
                "decode_preprocess_encoder_wall_seconds": metadata.get(
                    "decode_preprocess_encoder_wall_seconds"
                ),
                "embeddings_per_processing_second": metadata.get(
                    "embeddings_per_processing_second"
                ),
            }
        except Exception as exc:
            errors[sequence_id] = f"{type(exc).__name__}: {exc}"
    timed_entries = [
        entry
        for entry in entries.values()
        if entry.get("decode_preprocess_encoder_wall_seconds") is not None
    ]
    total_processing_seconds = sum(
        float(entry["decode_preprocess_encoder_wall_seconds"])
        for entry in timed_entries
    )
    total_timed_embeddings = sum(int(entry["samples"]) for entry in timed_entries)
    manifest = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder,
        "encoder_fingerprint": encoder_fingerprint,
        "selected_sequences": len(selected),
        "completed_sequences": len(entries),
        "sequence_fingerprint": fingerprint,
        "entries": dict(sorted(entries.items())),
        "errors": dict(sorted(errors.items())),
        "extraction_performance": {
            "scope": "video_decode_plus_preprocess_plus_frozen_encoder; excludes NPZ write",
            "timed_sequences": len(timed_entries),
            "timed_embeddings": total_timed_embeddings,
            "wall_seconds_sum": total_processing_seconds,
            "embeddings_per_second": (
                total_timed_embeddings / total_processing_seconds
                if total_processing_seconds > 0
                else None
            ),
        },
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {output_dir / 'cache_manifest.json'}")
    print(f"Completed: {len(entries)}/{len(selected)}; errors: {len(errors)}")
    return 1 if errors or len(entries) != len(selected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
