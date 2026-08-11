from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "Training/jobs"
MATRIX = json.loads(
    (ROOT / "Training/configs/experiment_matrix_v2.json").read_text(encoding="utf-8")
)


def text(name: str) -> str:
    return (JOBS / name).read_text(encoding="utf-8")


def test_validation_array_matches_predeclared_matrix() -> None:
    value = text("thesis_v2_validation_matrix.sbatch")
    expected_tasks = len(MATRIX["training_experiments"]) * len(MATRIX["seeds"])
    assert f"#SBATCH --array=0-{expected_tasks - 1}%4" in value
    assert "set -euo pipefail" in value
    assert "--skip-test-evaluation" in value
    assert "Training/artifact_freeze.py" in value
    assert "Training/configs/experiment_matrix_v2.json" in value
    assert 'p["training_experiments"]' in value
    assert 'print(e["id"])' in value
    assert 'print(e["entrypoint"])' in value
    assert 'print(e["config"])' in value
    assert "IDS=(" not in value
    assert "CONFIGS=(" not in value


def test_final_test_array_never_retrains() -> None:
    value = text("thesis_v2_final_test_matrix.sbatch")
    assert "Training/evaluate_frozen_run.py" in value
    assert "--checkpoint best_intention" in value
    assert "--selection-file" in value
    assert "--experiment-id" in value
    assert "SELECTION_FILE" in value
    assert "matrix_authorization" in value
    assert "selection_file_sha256" in value
    assert 'p["training_experiments"]' in value
    assert 'print(e["id"])' in value
    assert "IDS=(" not in value
    assert "Training/train_residual.py" not in value
    assert "Training/train.py" not in value
    assert "test_used_for_model_or_checkpoint_selection" in value


def test_jobs_use_unique_outputs_and_propagate_failures() -> None:
    validation = text("thesis_v2_validation_matrix.sbatch")
    final_test = text("thesis_v2_final_test_matrix.sbatch")
    assert "thesis_v2_validation.%A_%a" in validation
    assert "thesis_v2_final_test.%A_%a" in final_test
    assert "set -euo pipefail" in validation
    assert "set -euo pipefail" in final_test
    assert "${ID}_seed${SEED}" in validation
    assert "${ID}_seed${SEED}.json" in final_test


def test_group_cv_job_uses_inner_validation_before_outer_evaluation() -> None:
    value = text("thesis_v2_group_cv.sbatch")
    assert "nested_participant_group_cv_executable_v1" in value
    assert "--skip-test-evaluation" in value
    assert "Training/evaluate_frozen_run.py" in value
    assert "outer_evaluation_used_for_selection" in value
    assert 'len(p["runs"]) == 15' in value
    assert "config_sha256" in value
    assert "EXPECTED_CONFIG_SHA256" in value
    assert "source_artifact_manifest_fingerprint" in value
    assert 'out["checkpoint"]["name"] == "best_intention"' in value
    assert "test_used_for_model_or_checkpoint_selection" in value
    assert "set -euo pipefail" in value


def test_active_prerequisite_paths_are_environment_overridable() -> None:
    clip = text("prepare_clip_embeddings.sbatch")
    terminal = text("audit_endpose_v2.sbatch")
    for variable in ("REPO_DIR", "IMAGE", "PYTHON_DEPS", "WEIGHTS_CACHE"):
        assert f'${{{variable}:-' in clip
    assert '${IMAGE:-' in terminal


def test_matrix_summary_job_consumes_only_frozen_reports() -> None:
    value = text("thesis_v2_summarize_matrix.sbatch")
    assert "set -euo pipefail" in value
    assert "summarize_thesis_v2_matrix.py" in value
    assert "--selection" in value
    assert "--final-test-dir" in value
    assert "--postprocess-root" in value
    assert "train_residual.py" not in value
    assert "Training/train.py" not in value
