#!/bin/bash
#SBATCH --job-name=aria_residual_v2
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=aria_residual_v2.%J.out
#SBATCH --error=aria_residual_v2.%J.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
CONFIG="${CONFIG:-Training/configs/hierarchical_residual_v2.json}"
ARGS=(--config "$CONFIG")

if [[ -n "${EPOCHS:-}" ]]; then
    ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${RUN_DIR:-}" ]]; then
    ARGS+=(--run-dir "$RUN_DIR")
fi

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/train_residual.py "${ARGS[@]}"
