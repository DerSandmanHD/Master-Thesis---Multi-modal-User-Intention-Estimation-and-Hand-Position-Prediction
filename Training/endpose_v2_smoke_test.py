#!/usr/bin/env python3
"""Exercise the balanced dual-horizon terminal end-pose training path."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import prepare_data
from endpose_smoke_test import TARGET_CONFIG, make_stable
from endpose_v2 import (
    DUAL_HORIZON_MODEL_TYPE,
    residual_position_scale_m,
    wrap_endpose_v2_bundle,
)
from model import HierarchicalDualHorizonResidualPoseTransformer
from smoke_test import synthetic_sequence
from train_residual import residual_multitask_loss, train


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aria_endpose_v2_smoke_") as directory:
        root = Path(directory)
        for index in range(6):
            participant = f"P{index + 1}"
            path = root / f"{participant}_{index}_master.csv"
            synthetic_sequence(path, participant, index)
            make_stable(path, index)
        data_config = {
            "master_dir": str(root),
            "feature_profile": "multimodal_robot_frame_v1",
            "window_size": 20,
            "stride": 5,
            "future_horizon_seconds": 1.0,
            "pose_target": TARGET_CONFIG,
            "auxiliary_pose_target": {
                "mode": "future_offset",
                "future_horizon_seconds": 0.2,
                "maximum_target_gap_seconds": 0.1,
            },
            "pose_intent_ids": [2],
            "include_hand_references": True,
            "minimum_observed_fraction": 0.05,
            "max_timestamp_gap_seconds": 0.2,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "validation_participants": [],
            "test_participants": [],
        }
        bundle = wrap_endpose_v2_bundle(
            prepare_data(data_config, seed=42), data_config
        )
        assert bundle.train.auxiliary_pose_count() > 0
        sampling_sums = defaultdict(float)
        for weight, (record_index, _) in zip(
            bundle.train.sequence_sampling_weights().tolist(),
            bundle.train.indices,
        ):
            sampling_sums[record_index] += weight
        assert np.ptp(list(sampling_sums.values())) < 1e-6

        loader = DataLoader(bundle.train, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        model = HierarchicalDualHorizonResidualPoseTransformer(
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=20,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        outputs = model(batch["features"], batch["hand_reference_pose"])
        assert outputs["pose_candidates"].shape == (8, 2, 7)
        assert outputs["auxiliary_pose_candidates"].shape == (8, 2, 7)
        assert outputs["position_delta"].shape == (8, 2, 3)
        assert torch.allclose(
            outputs["pose_candidates"], batch["hand_reference_pose"], atol=1e-6
        )
        training_config = {
            "assistance_loss_weight": 1.0,
            "assistance_type_loss_weight": 1.0,
            "receiving_hand_loss_weight": 1.0,
            "pose_loss_weight": 2.0,
            "orientation_loss_weight": 0.25,
            "auxiliary_pose_loss_weight": 0.25,
            "auxiliary_orientation_loss_weight": 0.25,
            "position_loss": {
                "type": "normalized_smooth_l1",
                "beta": 1.0,
                "minimum_scale_m": 0.02,
            },
            "orientation_loss": {"type": "geodesic_radians"},
            "resolved_position_scale_m": residual_position_scale_m(
                bundle.train,
                target_key="pose_target",
                valid_key="residual_pose_valid",
                minimum_scale_m=0.02,
            ),
            "resolved_auxiliary_position_scale_m": residual_position_scale_m(
                bundle.train,
                target_key="auxiliary_pose_target",
                valid_key="auxiliary_residual_pose_valid",
                minimum_scale_m=0.02,
            ),
            "gradient_clip_norm": 1.0,
        }
        loss, components = residual_multitask_loss(
            outputs,
            batch,
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            training_config,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(np.isfinite(value) for value in components.values())

        config = {
            "run_name": "endpose_v2_smoke",
            "model_type": DUAL_HORIZON_MODEL_TYPE,
            "data": data_config,
            "model": {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            "training": {
                **training_config,
                "seed": 42,
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "sampling_mode": "sequence_balanced",
                "early_stopping_patience": 2,
                "num_workers": 0,
            },
        }
        config["training"].pop("resolved_position_scale_m")
        config["training"].pop("resolved_auxiliary_position_scale_m")
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_dir = root / "run"
        train(
            SimpleNamespace(
                config=config_path,
                run_dir=run_dir,
                dataset_tag=None,
                experiment_tag=None,
                epochs=None,
                seed=None,
                device="cpu",
                limit_sequences=None,
                skip_test_evaluation=False,
            )
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["model_type"] == DUAL_HORIZON_MODEL_TYPE
        assert metrics["test"]["best_pose"]["auxiliary_t_plus_1"][
            "pose_oracle"
        ]["samples"] > 0
        checkpoint = torch.load(
            run_dir / "best_pose_model.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["model_type"] == DUAL_HORIZON_MODEL_TYPE
    print("Terminal end-pose v2 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
