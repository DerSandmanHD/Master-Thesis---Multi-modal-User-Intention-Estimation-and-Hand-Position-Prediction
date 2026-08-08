#!/usr/bin/env python3
"""Export one immutable decoded RGB frame for CLIP latency benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from data import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    if args.frame_index < 0:
        raise ValueError("frame-index must be non-negative")
    video = resolve(args.video).resolve()
    output = resolve(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Fixture exists; pass --overwrite: {output}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Could not decode frame {args.frame_index} from {video}"
        )
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": video.name,
        "source_video_size_bytes": video.stat().st_size,
        "source_video_sha256": sha256_file(video),
        "frame_index": int(args.frame_index),
        "color_order": "RGB",
        "decoded_rgb_shape": list(rgb.shape),
        "decoded_rgb_dtype": str(rgb.dtype),
        "decoded_rgb_sha256": rgb_sha256,
        "decoder": {
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            rgb=rgb,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            ),
        )
    os.replace(temporary, output)
    print(f"RGB fixture: {output}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
