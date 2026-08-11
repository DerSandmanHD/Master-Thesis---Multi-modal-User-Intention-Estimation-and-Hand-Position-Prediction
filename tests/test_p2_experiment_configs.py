from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Training/configs/models/residual_transformer_v2.json"
ARCHITECTURE = ROOT / "Training/configs/architecture"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _without(mapping: dict, *keys: str) -> dict:
    return {key: value for key, value in mapping.items() if key not in keys}


def test_p2_architecture_ablations_keep_data_and_training_budget_fixed() -> None:
    base = _load(BASE)
    variants = {
        path.stem: _load(path)
        for path in sorted(ARCHITECTURE.glob("residual_v2_*.json"))
    }
    assert set(variants) == {
        "residual_v2_flat",
        "residual_v2_fusion_simple",
        "residual_v2_modality_gated",
        "residual_v2_temporal_only",
        "residual_v2_without_pose_loss",
    }
    for name, config in variants.items():
        assert config["data"] == base["data"], name
        assert config["model_type"] == base["model_type"], name
        assert _without(
            config["model"], "fusion_mode", "intention_head_mode"
        ) == _without(base["model"], "fusion_mode", "intention_head_mode"), name
        assert config["training"]["seed"] == base["training"]["seed"], name
        assert config["training"]["epochs"] == base["training"]["epochs"], name
        assert config["training"]["batch_size"] == base["training"]["batch_size"], name
        assert config["experiment_definition"]["selection"].startswith(
            "validation"
        ), name


def test_each_p2_config_changes_only_its_declared_factor() -> None:
    base = _load(BASE)
    simple = _load(ARCHITECTURE / "residual_v2_fusion_simple.json")
    modality = _load(ARCHITECTURE / "residual_v2_modality_gated.json")
    temporal = _load(ARCHITECTURE / "residual_v2_temporal_only.json")
    flat = _load(ARCHITECTURE / "residual_v2_flat.json")
    pose_off = _load(ARCHITECTURE / "residual_v2_without_pose_loss.json")

    assert simple["model"]["fusion_mode"] == "temporal_channel_simple"
    assert modality["model"]["fusion_mode"] == "modality_gated"
    assert temporal["model"]["fusion_mode"] == "temporal_only"
    assert flat["model"]["intention_head_mode"] == "flat"
    assert pose_off["training"]["pose_loss_weight"] == 0.0
    assert pose_off["training"]["receiving_hand_loss_weight"] == base["training"][
        "receiving_hand_loss_weight"
    ]
