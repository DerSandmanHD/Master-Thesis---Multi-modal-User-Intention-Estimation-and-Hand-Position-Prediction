from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from experiment_matrix import (  # noqa: E402
    ExperimentMatrixError,
    final_test_commands,
    validate_matrix,
    validation_commands,
)
from select_matrix_checkpoints import (  # noqa: E402
    select_candidate,
    validate_candidate_identity,
    validate_final_test_authorization,
)


MATRIX = ROOT / "Training/configs/experiment_matrix_v2.json"


def test_matrix_is_minimal_predeclared_and_executable() -> None:
    matrix = validate_matrix(MATRIX)
    ids = [entry["id"] for entry in matrix["training_experiments"]]
    assert len(ids) == 16
    assert {
        "baseline_mlp",
        "baseline_gru",
        "baseline_transformer",
        "residual_current_gate",
        "residual_simple_gate",
        "residual_modality_gated",
        "residual_temporal_only",
        "residual_flat",
        "residual_without_pose_aux",
        "visual_corrected_clip_current_gate",
        "visual_corrected_clip_modality_gate",
        "terminal_endpose_learned",
    } <= set(ids)
    for entry in matrix["training_experiments"]:
        config = json.loads((ROOT / entry["config"]).read_text(encoding="utf-8"))
        assert config["data"]["required_observation_alignment_version"] == (
            "causal_backward_device_time_v1"
        )


def test_validation_commands_never_evaluate_test() -> None:
    matrix = validate_matrix(MATRIX)
    rows = validation_commands(matrix)
    assert len(rows) == len(matrix["training_experiments"]) * 3
    assert len({row["run_dir"] for row in rows}) == len(rows)
    for row in rows:
        assert "--skip-test-evaluation" in row["command"]
        assert "evaluate_frozen_run.py" not in row["command"]


def test_final_test_commands_only_consume_frozen_runs() -> None:
    matrix = validate_matrix(MATRIX)
    rows = final_test_commands(matrix)
    assert len({row["output"] for row in rows}) == len(rows)
    for row in rows:
        assert "Training/evaluate_frozen_run.py" in row["command"]
        assert "--checkpoint best_intention" in row["command"]
        assert "--selection-file" in row["command"]
        assert "--experiment-id" in row["command"]
        assert "train_residual.py" not in row["command"]
        assert "Training/train.py" not in row["command"]


def test_test_based_matrix_selection_is_rejected(tmp_path: Path) -> None:
    value = json.loads(MATRIX.read_text(encoding="utf-8"))
    value["policy"]["selection_split"] = "test"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ExperimentMatrixError, match="validation-only"):
        validate_matrix(invalid)


def test_sensor_only_is_an_alias_not_a_duplicate_run() -> None:
    matrix = validate_matrix(MATRIX)
    aliases = {
        entry["id"]: entry["source_experiment"]
        for entry in matrix["aliases_without_training"]
    }
    assert aliases["modality_no_clip_sensor_only"] == "residual_current_gate"
    assert aliases["baseline_residual_transformer"] == "residual_current_gate"


def test_t1_three_way_matrix_entry_uses_checkpoint_bound_export() -> None:
    matrix = validate_matrix(MATRIX)
    entry = next(
        value for value in matrix["evaluation_only"]
        if value["id"] == "t1_pose_baselines"
    )
    assert "Training/export_residual_predictions.py" in entry["command"]
    assert entry["methods"] == [
        "persistence",
        "constant_velocity",
        "learned_model_oracle_hand",
    ]
    assert "fair_common" in entry["primary_comparison"]


def test_seed_checkpoint_selection_uses_validation_tradeoff_only() -> None:
    rows = [
        {
            "seed": 42,
            "validation_intention_macro_f1": 0.900,
            "validation_pose_mae_cm": 8.0,
            "validation_receiving_hand_macro_f1": 0.9,
        },
        {
            "seed": 43,
            "validation_intention_macro_f1": 0.904,
            "validation_pose_mae_cm": 9.0,
            "validation_receiving_hand_macro_f1": 0.95,
        },
        {
            "seed": 44,
            "validation_intention_macro_f1": 0.890,
            "validation_pose_mae_cm": 1.0,
            "validation_receiving_hand_macro_f1": 1.0,
        },
    ]
    assert select_candidate(rows)["seed"] == 42


def test_final_test_authorization_binds_run_seed_and_hash() -> None:
    row = {
        "experiment_id": "residual_current_gate",
        "seed": 42,
        "run_dir": "Training/runs/dataset/validation/residual/seed42",
        "checkpoint_name": "best_intention",
        "checkpoint_sha256": "abc123",
        "artifact_manifest_fingerprint": "manifest123",
    }
    report = {
        "complete": True,
        "selection_split": "validation",
        "test_metrics_read": False,
        "final_test_runs": [row],
    }
    assert validate_final_test_authorization(
        report,
        experiment_id="residual_current_gate",
        seed=42,
        run_dir=row["run_dir"],
        checkpoint_sha256="abc123",
        artifact_manifest_fingerprint="manifest123",
    ) == row
    with pytest.raises(ValueError, match="hash differs"):
        validate_final_test_authorization(
            report,
            experiment_id="residual_current_gate",
            seed=42,
            run_dir=row["run_dir"],
            checkpoint_sha256="changed",
            artifact_manifest_fingerprint="manifest123",
        )
    windows_row = {**row, "run_dir": row["run_dir"].replace("/", "\\")}
    windows_report = {**report, "final_test_runs": [windows_row]}
    assert validate_final_test_authorization(
        windows_report,
        experiment_id="residual_current_gate",
        seed=42,
        run_dir=row["run_dir"],
        checkpoint_sha256="abc123",
        artifact_manifest_fingerprint="manifest123",
    ) == windows_row


def test_matrix_candidate_rejects_wrong_seed_or_source_config(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text("{}", encoding="utf-8")
    from artifact_freeze import sha256_file

    freeze = {
        "seed": 42,
        "run_context": {
            "dataset_tag": "dataset",
            "experiment_tag": "validation",
        },
        "configuration": {
            "source": {
                "path": str(expected),
                "sha256": sha256_file(expected),
            }
        },
    }
    validate_candidate_identity(
        freeze,
        seed=42,
        expected_config=expected,
        expected_dataset_tag="dataset",
        expected_experiment_tag="validation",
    )
    with pytest.raises(ValueError, match="seed differs"):
        validate_candidate_identity(
            freeze,
            seed=43,
            expected_config=expected,
            expected_dataset_tag="dataset",
            expected_experiment_tag="validation",
        )
