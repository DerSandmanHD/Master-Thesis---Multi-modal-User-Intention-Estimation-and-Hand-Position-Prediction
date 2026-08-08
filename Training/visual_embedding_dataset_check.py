#!/usr/bin/env python3
"""Validate visual caches, causal alignment, split isolation, and model shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import prepare_data
from model import HierarchicalResidualPoseTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DATASET_FINGERPRINT = (
    "5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-sequence-fingerprint", default=EXPECTED_DATASET_FINGERPRINT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    data_config["master_dir"] = str(resolve(Path(data_config["master_dir"])))
    bundle = prepare_data(data_config, seed=int(config["training"]["seed"]))
    dataset_filter = bundle.split_metadata["dataset_filter"]
    if dataset_filter["sequence_fingerprint"] != args.expected_sequence_fingerprint:
        raise ValueError("Visual dataset does not match the frozen n214 sequence set")
    visual = bundle.provenance["schema"].get("visual_features")
    if not visual or not visual.get("enabled"):
        raise ValueError("Visual provenance is missing")
    alignment = visual["alignment"]
    if alignment["future_matches"] != 0:
        raise ValueError("Visual alignment used future frames")
    if alignment["coverage"] < 0.95:
        raise ValueError(
            f"Visual alignment coverage is unexpectedly low: {alignment['coverage']:.3f}"
        )
    if visual["projection_fit_split"] != "train_only":
        raise ValueError("Visual PCA was not declared as train-only")
    batch = next(iter(DataLoader(bundle.train, batch_size=2, shuffle=False)))
    model = HierarchicalResidualPoseTransformer(
        input_dim=len(bundle.normalizer.output_feature_names),
        window_size=int(data_config["window_size"]),
        **config["model"],
    )
    with torch.inference_mode():
        outputs = model(batch["features"], batch["hand_reference_pose"])
    if outputs["assistance_logits"].shape != (2, 2):
        raise ValueError("Visual model output shape is invalid")
    print(
        "Visual dataset valid: "
        f"mode={visual['mode']}, raw_features={len(bundle.feature_columns)}, "
        f"model_features={len(bundle.normalizer.output_feature_names)}, "
        f"coverage={alignment['coverage']:.4f}, future_matches=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
