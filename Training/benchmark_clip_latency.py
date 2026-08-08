#!/usr/bin/env python3
"""Benchmark frozen CLIP preprocessing and encoding on one real RGB frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from benchmark_model_latency import hardware_metadata, summarize, synchronize
from data import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--weights-cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--visual-update-budget-ms",
        type=float,
        default=200.0,
        help="Budget implied by the preregistered 5-Hz visual sampling rate.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def first_rgb_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode first frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def timed_loop(callback, *, warmup: int, repeats: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        callback()
    synchronize(device)
    values = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter_ns()
        callback()
        synchronize(device)
        values.append((time.perf_counter_ns() - started) / 1e6)
    return values


def main() -> int:
    args = parse_args()
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.cpu_threads <= 0
        or args.visual_update_budget_ms <= 0
    ):
        raise ValueError("Invalid benchmark protocol")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open_clip is required for the CLIP benchmark") from exc

    video = resolve(args.video).resolve()
    output = resolve(args.output).resolve()
    weights_cache = resolve(args.weights_cache_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    pretrained_config = open_clip.get_pretrained_cfg(args.model_name, args.pretrained)
    if pretrained_config is None:
        raise ValueError(f"Unknown CLIP pair: {args.model_name}/{args.pretrained}")
    checkpoint_path = Path(
        open_clip.download_pretrained(pretrained_config, cache_dir=str(weights_cache))
    )
    load_started = time.perf_counter_ns()
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=str(checkpoint_path),
        device=device,
        cache_dir=str(weights_cache),
    )
    model.eval()
    synchronize(device)
    load_time_ms = (time.perf_counter_ns() - load_started) / 1e6
    rgb = first_rgb_frame(video)
    frame_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
    preprocessed = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)

    def preprocess_only() -> torch.Tensor:
        return preprocess(Image.fromarray(rgb))

    def encoder_only() -> torch.Tensor:
        with torch.inference_mode():
            encoded = model.encode_image(preprocessed)
            return torch.nn.functional.normalize(encoded.float(), dim=-1)

    def full_rgb_pipeline() -> np.ndarray:
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(tensor)
            normalized = torch.nn.functional.normalize(encoded.float(), dim=-1)
        return normalized.cpu().numpy()

    preprocessing_ms = timed_loop(
        preprocess_only,
        warmup=args.warmup,
        repeats=args.repeats,
        device=torch.device("cpu"),
    )
    encoder_ms = timed_loop(
        encoder_only,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    full_ms = timed_loop(
        full_rgb_pipeline,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": {
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_threads": args.cpu_threads,
            "synchronize_before_and_after_each_measurement": True,
            "visual_sampling_hz": 5.0,
            "visual_update_budget_ms": args.visual_update_budget_ms,
        },
        "hardware": hardware_metadata(device),
        "encoder": {
            "library": "open_clip_torch",
            "library_version": getattr(open_clip, "__version__", "unknown"),
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "weights": str(checkpoint_path),
            "weights_sha256": sha256_file(checkpoint_path),
            "frozen": True,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameters": 0,
            "load_time_ms": load_time_ms,
        },
        "source": {
            "video": str(video),
            "video_sha256": sha256_file(video),
            "frame_index": 0,
            "decoded_rgb_shape": list(rgb.shape),
            "decoded_rgb_sha256": frame_sha256,
        },
        "preprocessing_cpu": summarize(preprocessing_ms, args.visual_update_budget_ms),
        "encoder_forward": summarize(encoder_ms, args.visual_update_budget_ms),
        "rgb_to_embedding": summarize(full_ms, args.visual_update_budget_ms),
        "raw_preprocessing_cpu_ms": preprocessing_ms,
        "raw_encoder_forward_ms": encoder_ms,
        "raw_rgb_to_embedding_ms": full_ms,
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{device}: encoder median={report['encoder_forward']['median_ms']:.3f} ms; "
        f"RGB-to-embedding median={report['rgb_to_embedding']['median_ms']:.3f} ms"
    )
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
