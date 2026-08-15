#!/usr/bin/env python3
"""Validate modality-ablation configs, feature removal, and model shapes."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import (
    SUPPORTED_MODALITY_ABLATIONS,
    feature_modalities,
    prepare_data,
    select_feature_columns,
)
from model import HierarchicalResidualPoseTransformer
from smoke_test import synthetic_sequence


CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "ablations"
CONFIGS = {
    "gaze": CONFIG_DIR / "residual_v2_no_gaze.json",
    "hands": CONFIG_DIR / "residual_v2_no_hands.json",
    "objects": CONFIG_DIR / "residual_v2_no_objects.json",
    "vio": CONFIG_DIR / "residual_v2_no_vio.json",
}


def main() -> int:
    assert set(CONFIGS) == set(SUPPORTED_MODALITY_ABLATIONS)
    try:
        select_feature_columns([], "multimodal_robot_frame_v1", ["unknown"])
    except ValueError as exc:
        assert "Unknown ablation modality" in str(exc)
    else:
        raise AssertionError("Unknown modality was accepted")

    with tempfile.TemporaryDirectory(prefix="aria_ablation_smoke_") as directory:
        master_dir = Path(directory) / "master_datasets"
        master_dir.mkdir()
        for index in range(6):
            participant = f"P{index + 1}"
            synthetic_sequence(
                master_dir / f"{participant}_{index}_master.csv",
                participant,
                index,
            )
        first_master = master_dir / "P1_0_master.csv"
        with first_master.open(newline="", encoding="utf-8") as handle:
            full_feature_count = len(
                select_feature_columns(
                    next(csv.reader(handle)),
                    "multimodal_robot_frame_v1",
                )
            )

        for modality, config_path in CONFIGS.items():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            assert config["model_type"] == (
                "hierarchical_residual_pose_transformer_v2"
            )
            assert config["data"]["ablation_exclude_modalities"] == [modality]

            data_config = dict(config["data"])
            data_config["master_dir"] = str(master_dir)
            data_config.pop("manifest_filter")
            # The fixture intentionally contains six synthetic sequences, not
            # the frozen n214 thesis cohort bound by the production config.
            data_config.pop("dataset_contract")
            data_config.update(
                {
                    "window_size": 20,
                    "stride": 10,
                    "validation_participants": [],
                    "test_participants": [],
                }
            )
            bundle = prepare_data(data_config, seed=42)
            metadata = bundle.split_metadata["feature_ablation"]
            assert metadata["excluded_modalities"] == [modality]
            assert metadata["full_raw_feature_count"] == full_feature_count
            assert metadata["excluded_feature_columns"]
            assert metadata["retained_raw_feature_count"] == len(
                bundle.feature_columns
            )
            assert metadata["retained_model_feature_count_with_masks"] == len(
                bundle.normalizer.output_feature_names
            )
            assert metadata == bundle.provenance["schema"]["feature_ablation"]
            assert all(
                modality not in feature_modalities(column)
                for column in bundle.feature_columns
            )
            if modality == "gaze":
                assert not any(
                    "_gaze_" in column for column in bundle.feature_columns
                )

            batch = next(
                iter(DataLoader(bundle.train, batch_size=2, shuffle=False))
            )
            model = HierarchicalResidualPoseTransformer(
                input_dim=len(bundle.normalizer.output_feature_names),
                window_size=20,
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            )
            outputs = model(
                batch["features"],
                batch["hand_reference_pose"],
            )
            assert outputs["assistance_logits"].shape == (2, 2)
            assert outputs["receiving_hand_logits"].shape == (2, 2)
            assert outputs["pose_candidates"].shape == (2, 2, 7)
            assert torch.isfinite(outputs["pose_candidates"]).all()
            print(
                f"{modality}: "
                f"{metadata['retained_raw_feature_count']} raw / "
                f"{metadata['retained_model_feature_count_with_masks']} model features"
            )

    print("Modality-ablation smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
