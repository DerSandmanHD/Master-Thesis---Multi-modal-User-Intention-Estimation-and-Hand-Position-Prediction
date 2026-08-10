#!/bin/bash
#SBATCH --job-name=aria_endpose_audit
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/endpose_audit_v1.%j.out
#SBATCH --error=Training/slurm_logs/endpose_audit_v1.%j.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v1}"
CONFIG="${CONFIG:-Training/configs/models/residual_transformer_endpose_v1.json}"
REPORT_DIR="Training/reports/${DATASET_TAG}/${EXPERIMENT_TAG}"

cd "$REPO_DIR"
mkdir -p Training/slurm_logs "$REPORT_DIR/audit"

singularity exec "$IMAGE" \
    python3 Training/audit_endpose_targets.py \
    --config "$CONFIG" \
    --output-dir "$REPORT_DIR/audit" \
    --require-sufficient

singularity exec "$IMAGE" \
    python3 Training/dry_run_endpose_experiment.py \
    --config "$CONFIG" \
    --audit-report "$REPORT_DIR/audit/endpose_target_audit.json" \
    --output "$REPORT_DIR/audit/dry_run.json" \
    --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG"

echo "End-pose audit and dry-run passed. No training was started."
