#!/bin/bash
#SBATCH --job-name=aria_export_pred
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=aria_export_pred.%J.out
#SBATCH --error=aria_export_pred.%J.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
RUN_DIR="${RUN_DIR:-Training/runs/hierarchical_baseline_20260712_101448}"
RUNS_ROOT="${RUNS_ROOT:-Training/runs}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/export_checkpoint_predictions.py \
    --run-dir "$RUN_DIR" \
    --runs-root "$RUNS_ROOT" \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --num-workers 2
