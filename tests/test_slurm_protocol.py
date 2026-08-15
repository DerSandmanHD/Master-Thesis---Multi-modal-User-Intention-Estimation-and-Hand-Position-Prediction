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


def root_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


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
    assert "validation_frozen_checkpoint_single_test_v2" in value
    assert "validation_frozen_checkpoint_single_test_v1" not in value
    assert "report_fingerprint" in value


def test_validation_selection_job_is_fail_closed_and_test_free() -> None:
    value = text("thesis_v2_select_checkpoints.sbatch")
    assert "set -euo pipefail" in value
    assert "Training/select_matrix_checkpoints.py" in value
    assert "--require-complete" in value
    assert 's["schema_version"] == 2' in value
    assert 's["complete"] is True' in value
    assert 's["selection_split"] == "validation"' in value
    assert 's["test_metrics_read"] is False' in value
    assert 's["matrix_sha256"] == h' in value
    assert "refusing to overwrite" in value
    assert "evaluate_frozen_run.py" not in value
    assert "train_residual.py" not in value
    assert "--gres=gpu" not in value


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
    assert 'len(runs) == int(p["fold_count"])*len(p["seeds"])' in value
    assert "SLURM_ARRAY_TASK_ID:?Submit with --array" in value
    assert "#SBATCH --array=" not in value
    assert 'len(r["outer_evaluation_participants"]) == 1' in value
    assert "config_sha256" in value
    assert "plan_fingerprint" in value
    assert "EXPECTED_CONFIG_SHA256" in value
    assert "source_artifact_manifest_fingerprint" in value
    assert 'out["checkpoint"]["name"] == "best_intention"' in value
    assert "test_used_for_model_or_checkpoint_selection" in value
    assert "set -euo pipefail" in value
    assert "validation_frozen_checkpoint_single_test_v2" in value
    assert "validation_frozen_checkpoint_single_test_v1" not in value
    assert "report_fingerprint" in value
    protocol = root_text("Training/THESIS_FINAL_PROTOCOL_V2.md")
    assert "--leave-one-participant-out" in protocol
    assert "GROUP_CV_TASKS=" in protocol
    assert "--array=0-${GROUP_CV_LAST_INDEX}%3" in protocol


def test_group_cv_summary_job_is_fail_closed_and_cpu_only() -> None:
    value = text("thesis_v2_summarize_group_cv.sbatch")
    assert "set -euo pipefail" in value
    assert "summarize_group_cv.py" in value
    assert "GROUP_CV_PLAN:?" in value
    assert "GROUP_CV_SUMMARY_DIR:?" in value
    assert "--gres=gpu" not in value


def test_reduced_lopo_plan_and_full_submission_graph_are_predeclared() -> None:
    plan = text("thesis_v2_prepare_group_cv.sbatch")
    submit = text("submit_thesis_v2_pipeline.sh")
    assert "--leave-one-participant-out" in text("thesis_v2_split_audit.sbatch")
    assert "--seeds 42" in plan
    assert 'p["fold_count"] == 25' in plan
    assert 'p["seeds"] == [42]' in plan
    assert "--array=0-24%3" in submit
    assert "GROUP_CV_PLAN_JOB" in submit
    assert "POSTPROCESS_EXPERIMENTS=residual_modality_gated:visual_corrected_clip_modality_gate" in submit
    assert 'afterok:${CLIP_JOB}:${ENDPOSE_AUDIT_JOB}:${SPLIT_AUDIT_JOB}' in submit


def test_postprocess_requires_authorized_final_test_binding() -> None:
    value = text("thesis_v2_postprocess_selected.sbatch")
    expected_tasks = len(MATRIX["postprocessing"]["required_t1_experiments"]) * len(
        MATRIX["seeds"]
    )
    assert f"#SBATCH --array=0-{expected_tasks - 1}%3" in value
    assert 'm["postprocessing"]["required_t1_experiments"]' in value
    assert 's["selection_split"] == "validation"' in value
    assert "SELECTION_FILE" in value
    assert "--final-test-report" in value
    assert "validation_frozen_checkpoint_single_test_v2" in value
    assert "report_fingerprint" in value


def test_active_prerequisite_paths_are_environment_overridable() -> None:
    clip = text("prepare_clip_embeddings.sbatch")
    terminal = text("audit_endpose_v2.sbatch")
    for variable in ("REPO_DIR", "IMAGE", "PYTHON_DEPS", "WEIGHTS_CACHE"):
        assert f'${{{variable}:-' in clip
    assert '${IMAGE:-' in terminal


def test_causal_master_rebuild_precedes_clip_cache_protocol() -> None:
    master = root_text("singularity/aria_build_master_dataset.sbatch")
    batch = root_text("Code/build_master_dataset_batch.py")
    clip = text("prepare_clip_embeddings.sbatch")
    active_tag = MATRIX["dataset_tag"]
    assert '${IMAGE:-' in master
    assert 'OVERWRITE="${OVERWRITE:-1}"' in master
    assert 'ALLOW_UNREVIEWED_TIMESTAMPS="${ALLOW_UNREVIEWED_TIMESTAMPS:-0}"' in master
    assert 'ALLOW_UNREVIEWED_TIMESTAMPS" != "1"' in master
    assert "Reviewed timestamps are required for the active thesis pipeline" in master
    assert "Timestamp source SHA-256" in master
    assert "--allow-unreviewed-timestamps" in batch
    assert 'timestamp_review_status = "unreviewed_legacy_opt_in"' in batch
    assert '"timestamp_summary": source_file_identity(timestamps_path)' in batch
    assert "master_datasets_history" in master
    assert "Training/verify_causal_masters.py" in master
    assert "causal_backward_device_time_v1" in master
    assert "Training/verify_causal_masters.py" in clip
    assert "causal_backward_device_time_v1" in clip
    assert active_tag in clip


def test_legacy_visual_jobs_require_explicit_historical_opt_in() -> None:
    for name in (
        "screen_visual_embeddings_residual_v2.sbatch",
        "final_evaluate_visual_variant.sbatch",
    ):
        value = text(name)
        assert 'ALLOW_LEGACY_EXPERIMENT:-0' in value
        assert "not authorized by thesis protocol v2" in value


def test_matrix_summary_job_consumes_only_frozen_reports() -> None:
    value = text("thesis_v2_summarize_matrix.sbatch")
    assert "set -euo pipefail" in value
    assert "summarize_thesis_v2_matrix.py" in value
    assert "--selection" in value
    assert "--final-test-dir" in value
    assert "--postprocess-root" in value
    assert "Required t+1 postprocess directory missing" in value
    assert 'if [[ -d "$POSTPROCESS_ROOT" ]]' not in value
    assert "train_residual.py" not in value
    assert "Training/train.py" not in value
