#!/bin/bash
#SBATCH --job-name=thesis_v2_groupcv_summary
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=Training/slurm_logs/thesis_v2_group_cv_summary.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_group_cv_summary.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
: "${GROUP_CV_PLAN:?Set GROUP_CV_PLAN to generated group_cv_plan.json}"
: "${GROUP_CV_SUMMARY_DIR:?Set GROUP_CV_SUMMARY_DIR to a new output directory}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$GROUP_CV_PLAN" ]] || { echo "Group-CV plan missing: $GROUP_CV_PLAN" >&2; exit 2; }

singularity exec "$IMAGE" python3 Training/evaluation/summarize_group_cv.py \
  --plan "$GROUP_CV_PLAN" \
  --output-dir "$GROUP_CV_SUMMARY_DIR"
