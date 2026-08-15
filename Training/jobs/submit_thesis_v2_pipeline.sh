#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-thesis_final_v2_validation}"
REPORT_TAG="${REPORT_TAG:-thesis_final_v2_corrected_alignment}"
GROUP_CV_ROOT="Training/reports/${DATASET_TAG}/thesis_v2_group_cv_seed42"
GROUP_CV_PLAN="${GROUP_CV_ROOT}/group_cv_plan.json"

cd "$REPO_DIR"
[[ -z "$(git status --porcelain)" ]] || {
  echo "Repository must be clean before submitting hash-bound experiments." >&2
  git status --short >&2
  exit 2
}
mkdir -p Training/slurm_logs

MASTER_JOB=$(sbatch --parsable \
  --export=ALL,REPO_DIR="$REPO_DIR",OVERWRITE=1 \
  singularity/aria_build_master_dataset.sbatch)

CLIP_JOB=$(sbatch --parsable --dependency="afterok:${MASTER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG" \
  Training/jobs/prepare_clip_embeddings.sbatch)

ENDPOSE_AUDIT_JOB=$(sbatch --parsable --dependency="afterok:${MASTER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG=terminal_endpose_corrected_v2 \
  Training/jobs/audit_endpose_v2.sbatch)

SPLIT_AUDIT_JOB=$(sbatch --parsable --dependency="afterok:${MASTER_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG" \
  Training/jobs/thesis_v2_split_audit.sbatch)

GROUP_CV_PLAN_JOB=$(sbatch --parsable --dependency="afterok:${SPLIT_AUDIT_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",OUTPUT_DIR="$GROUP_CV_ROOT" \
  Training/jobs/thesis_v2_prepare_group_cv.sbatch)

VALIDATION_JOB=$(sbatch --parsable --dependency="afterok:${CLIP_JOB}:${ENDPOSE_AUDIT_JOB}:${SPLIT_AUDIT_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/thesis_v2_validation_matrix.sbatch)

SELECTION_JOB=$(sbatch --parsable --dependency="afterok:${VALIDATION_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR" \
  Training/jobs/thesis_v2_select_checkpoints.sbatch)

TEST_JOB=$(sbatch --parsable --dependency="afterok:${SELECTION_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG",REPORT_TAG="$REPORT_TAG" \
  Training/jobs/thesis_v2_final_test_matrix.sbatch)

POST_JOB=$(sbatch --parsable --dependency="afterok:${TEST_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",REPORT_TAG="$REPORT_TAG" \
  Training/jobs/thesis_v2_postprocess_selected.sbatch)

GATE_JOB=$(sbatch --parsable --array=0-5%3 --dependency="afterok:${TEST_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",REPORT_TAG="$REPORT_TAG",POSTPROCESS_EXPERIMENTS=residual_modality_gated:visual_corrected_clip_modality_gate \
  Training/jobs/thesis_v2_postprocess_selected.sbatch)

SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${POST_JOB}:${GATE_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",REPORT_TAG="$REPORT_TAG" \
  Training/jobs/thesis_v2_summarize_matrix.sbatch)

QUALITATIVE_JOB=$(sbatch --parsable --dependency="afterok:${POST_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",DATASET_TAG="$DATASET_TAG",REPORT_TAG="$REPORT_TAG" \
  Training/jobs/thesis_v2_qualitative.sbatch)

GROUP_CV_JOB=$(sbatch --parsable --array=0-24%3 --dependency="afterok:${GROUP_CV_PLAN_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",GROUP_CV_PLAN="$GROUP_CV_PLAN" \
  Training/jobs/thesis_v2_group_cv.sbatch)

GROUP_CV_SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${GROUP_CV_JOB}" \
  --export=ALL,REPO_DIR="$REPO_DIR",GROUP_CV_PLAN="$GROUP_CV_PLAN",GROUP_CV_SUMMARY_DIR="${GROUP_CV_ROOT}/summary" \
  Training/jobs/thesis_v2_summarize_group_cv.sbatch)

printf '%s\n' \
  "MASTER_JOB=$MASTER_JOB" \
  "CLIP_JOB=$CLIP_JOB" \
  "ENDPOSE_AUDIT_JOB=$ENDPOSE_AUDIT_JOB" \
  "SPLIT_AUDIT_JOB=$SPLIT_AUDIT_JOB" \
  "GROUP_CV_PLAN_JOB=$GROUP_CV_PLAN_JOB" \
  "VALIDATION_JOB=$VALIDATION_JOB" \
  "SELECTION_JOB=$SELECTION_JOB" \
  "TEST_JOB=$TEST_JOB" \
  "POST_JOB=$POST_JOB" \
  "GATE_JOB=$GATE_JOB" \
  "SUMMARY_JOB=$SUMMARY_JOB" \
  "QUALITATIVE_JOB=$QUALITATIVE_JOB" \
  "GROUP_CV_JOB=$GROUP_CV_JOB" \
  "GROUP_CV_SUMMARY_JOB=$GROUP_CV_SUMMARY_JOB"
