#!/usr/bin/env python3
"""Benchmark identical residual-v2 windows across CPU, CUDA, and MPS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import sha256_file
from model import HierarchicalResidualPoseTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_intention_model.pt")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "std_ms": float(statistics.pstdev(values)),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "throughput_windows_per_second": 1000.0 / statistics.fmean(values),
    }


def device_available(name: str) -> tuple[bool, str | None]:
    if name == "cuda":
        return torch.cuda.is_available(), "torch.cuda.is_available() is false"
    if name == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return available, "torch.backends.mps.is_available() is false"
    return True, None


def load_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        hand = archive["hand_reference_pose"].astype(np.float32, copy=False)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if features.ndim != 2 or hand.shape != (2, 7):
        raise ValueError("Latency fixture has invalid shapes")
    return features, hand, metadata


def model_forward(model, features, hand):
    output = model(features, hand)
    assistance = torch.softmax(output["assistance_logits"], dim=-1)
    assistance_type = torch.softmax(output["assistance_type_logits"], dim=-1)
    receiving_hand = torch.softmax(output["receiving_hand_logits"], dim=-1)
    return assistance, assistance_type, receiving_hand, output["pose_candidates"]


def benchmark_forward(
    model,
    features: torch.Tensor,
    hand: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model_forward(model, features, hand)
        synchronize(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter_ns()
            model_forward(model, features, hand)
            synchronize(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
    return timings


def benchmark_offline_window(
    model,
    features_np: np.ndarray,
    hand_np: np.ndarray,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    def once() -> tuple[float, ...]:
        features = torch.from_numpy(features_np.copy()).unsqueeze(0).to(device)
        hand = torch.from_numpy(hand_np.copy()).unsqueeze(0).to(device)
        outputs = model_forward(model, features, hand)
        # Materialize the small decision output on the host as an offline caller does.
        probabilities = torch.cat([value.flatten() for value in outputs[:3]])
        return tuple(float(value) for value in probabilities.cpu())

    with torch.inference_mode():
        for _ in range(warmup):
            once()
        synchronize(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter_ns()
            once()
            synchronize(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
    return timings


def hardware_metadata(device: torch.device) -> dict:
    data = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        data["accelerator"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    elif device.type == "mps":
        data["accelerator"] = {"name": "Apple Metal Performance Shaders"}
    return data


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.cpu_threads <= 0:
        raise ValueError("warmup, repeats, and cpu-threads are invalid")
    output = resolve(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    available, reason = device_available(args.device)
    if not available:
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "requested_device": args.device,
            "reason": reason,
            "hardware": hardware_metadata(torch.device("cpu")),
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Unavailable: {reason}; report: {output}")
        return 0

    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    artifacts = resolve(args.artifacts_dir).resolve()
    fixture_path = resolve(args.fixture).resolve()
    checkpoint_path = artifacts / args.checkpoint
    config_path = artifacts / "config.json"
    features_np, hand_np, fixture_metadata = load_fixture(fixture_path)
    load_started = time.perf_counter_ns()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HierarchicalResidualPoseTransformer(
        input_dim=int(checkpoint["input_dim"]),
        window_size=int(checkpoint["window_size"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    synchronize(device)
    load_time_ms = (time.perf_counter_ns() - load_started) / 1e6
    if tuple(features_np.shape) != (model.window_size, model.input_dim):
        raise ValueError(
            f"Fixture shape {features_np.shape} != model "
            f"({model.window_size}, {model.input_dim})"
        )
    if fixture_metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Fixture was created for a different checkpoint")
    features = torch.from_numpy(features_np).unsqueeze(0).to(device)
    hand = torch.from_numpy(hand_np).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    forward_values = benchmark_forward(
        model, features, hand, device, args.warmup, args.repeats
    )
    offline_values = benchmark_offline_window(
        model,
        features_np,
        hand_np,
        device,
        args.warmup,
        args.repeats,
    )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = {
        "process_max_rss_before": rss_before,
        "process_max_rss_after": rss_after,
        "process_max_rss_unit": "KiB on Linux; bytes on macOS",
    }
    if device.type == "cuda":
        peak_memory["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        peak_memory["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_memory["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": {
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "synchronize_before_and_after_each_measurement": True,
            "cpu_threads": args.cpu_threads,
        },
        "hardware": hardware_metadata(device),
        "artifacts_dir": str(artifacts),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "fixture": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_metadata": fixture_metadata,
        "model": {
            "input_dim": model.input_dim,
            "window_size": model.window_size,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "load_time_ms": load_time_ms,
        },
        "model_forward": summarize(forward_values),
        "offline_window": summarize(offline_values),
        "peak_memory": peak_memory,
        "raw_model_forward_ms": forward_values,
        "raw_offline_window_ms": offline_values,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{device}: forward median={report['model_forward']['median_ms']:.3f} ms, "
        f"p95={report['model_forward']['p95_ms']:.3f} ms; "
        f"offline median={report['offline_window']['median_ms']:.3f} ms"
    )
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
