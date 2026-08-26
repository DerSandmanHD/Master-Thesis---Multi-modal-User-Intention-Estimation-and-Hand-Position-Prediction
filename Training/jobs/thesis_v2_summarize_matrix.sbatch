#!/bin/bash
#SBATCH --job-name=thesis_v2_summary
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=Training/slurm_logs/thesis_v2_summary.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_summary.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
REPORT_TAG="${REPORT_TAG:-thesis_final_v2_corrected_alignment}"
REPORT_ROOT="${REPORT_ROOT:-Training/reports/${DATASET_TAG}/${REPORT_TAG}}"
SELECTION_FILE="${SELECTION_FILE:-${REPORT_ROOT}/validation_selection.json}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${REPORT_ROOT}/final_test}"
POSTPROCESS_ROOT="${POSTPROCESS_ROOT:-${REPORT_ROOT}/postprocess}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPORT_ROOT}/final_summary}"
DATASET_DESCRIPTOR="${DATASET_DESCRIPTOR:-Training/datasets/${DATASET_TAG}.json}"
SPLIT_AUDIT="${SPLIT_AUDIT:-Training/reports/${DATASET_TAG}/split_confounding_v2/split_audit.json}"
GROUP_CV_SUMMARY="${GROUP_CV_SUMMARY:-Training/reports/${DATASET_TAG}/thesis_v2_group_cv_seed42/summary_v2/group_cv_summary.json}"
INTENTION_BASELINES="${INTENTION_BASELINES:-Training/reports/${DATASET_TAG}/causal_intention_baselines_v1/intention_baselines.json}"
SAMPLING_AUDIT="${SAMPLING_AUDIT:-Training/reports/${DATASET_TAG}/sampling_window_audit_v1/sampling_window_audit.json}"
POSE_LEARNING_DIAGNOSIS="${POSE_LEARNING_DIAGNOSIS:-Training/reports/${DATASET_TAG}/pose_learning_diagnosis_v1/pose_learning_diagnosis.json}"
QUALITATIVE_MANIFEST="${QUALITATIVE_MANIFEST:-${REPORT_ROOT}/qualitative/qualitative_artifact_manifest.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Experiment matrix missing: $MATRIX" >&2; exit 2; }
[[ -f "$SELECTION_FILE" ]] || { echo "Validation selection missing: $SELECTION_FILE" >&2; exit 2; }
[[ -d "$FINAL_TEST_DIR" ]] || { echo "Final-test directory missing: $FINAL_TEST_DIR" >&2; exit 2; }
[[ -d "$POSTPROCESS_ROOT" ]] || { echo "Required t+1 postprocess directory missing: $POSTPROCESS_ROOT" >&2; exit 2; }
for path in "$DATASET_DESCRIPTOR" "$SPLIT_AUDIT" "$GROUP_CV_SUMMARY" "$INTENTION_BASELINES" "$SAMPLING_AUDIT" "$POSE_LEARNING_DIAGNOSIS" "$QUALITATIVE_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Required authoritative supplement missing: $path" >&2; exit 2; }
done

ARGS=(
  --matrix "$MATRIX"
  --selection "$SELECTION_FILE"
  --final-test-dir "$FINAL_TEST_DIR"
  --postprocess-root "$POSTPROCESS_ROOT"
  --output-dir "$OUTPUT_DIR"
  --dataset-descriptor "$DATASET_DESCRIPTOR"
  --split-audit "$SPLIT_AUDIT"
  --group-cv-summary "$GROUP_CV_SUMMARY"
  --intention-baselines "$INTENTION_BASELINES"
  --sampling-audit "$SAMPLING_AUDIT"
  --pose-learning-diagnosis "$POSE_LEARNING_DIAGNOSIS"
  --qualitative-manifest "$QUALITATIVE_MANIFEST"
)

singularity exec "$IMAGE" \
  python3 Training/evaluation/summarize_thesis_v2_matrix.py "${ARGS[@]}"
echo "Authoritative matrix summary complete: $OUTPUT_DIR"
