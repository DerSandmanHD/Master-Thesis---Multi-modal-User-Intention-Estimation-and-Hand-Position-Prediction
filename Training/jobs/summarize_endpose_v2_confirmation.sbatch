#!/bin/bash
#SBATCH --job-name=aria_endpose_v2_select
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/endpose_v2_select.%j.out
#SBATCH --error=Training/slurm_logs/endpose_v2_select.%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
STAGE_A_TAG="${STAGE_A_TAG:-residual_v2_endpose_v2_hp_search}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v2_hp_confirm}"
STAGE_A_REPORT_TAG="${STAGE_A_REPORT_TAG:-${STAGE_A_TAG}_checkpoint_coherent_v2}"
REPORT_TAG="${REPORT_TAG:-${EXPERIMENT_TAG}_checkpoint_coherent_v2}"
cd "$REPO_DIR"
singularity exec "$IMAGE" python3 Training/evaluation/summarize_endpose_v2_search.py \
    --stage confirmation --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --report-tag "$REPORT_TAG" \
    --stage-a-summary "Training/reports/${DATASET_TAG}/${STAGE_A_REPORT_TAG}/summary.json" \
    --require-complete
