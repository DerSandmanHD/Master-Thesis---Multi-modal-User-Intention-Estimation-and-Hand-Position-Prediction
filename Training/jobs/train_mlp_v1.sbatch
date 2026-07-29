#!/bin/bash
#SBATCH --job-name=aria_mlp
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=Training/slurm_logs/aria_mlp_v1.%J.out
#SBATCH --error=Training/slurm_logs/aria_mlp_v1.%J.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-manual_v1}"
MODEL_TAG="mlp_v1"
CONFIG="${CONFIG:-Training/configs/models/mlp_v1.json}"
SEED_VALUE="${SEED:-42}"
RUN_ID="${RUN_ID:-${EXPERIMENT_TAG}_${MODEL_TAG}_seed${SEED_VALUE}_job${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${MODEL_TAG}/${RUN_ID}}"
ARGS=(
    --config "$CONFIG"
    --dataset-tag "$DATASET_TAG"
    --experiment-tag "$EXPERIMENT_TAG"
    --seed "$SEED_VALUE"
    --run-dir "$RUN_DIR"
)

if [[ -n "${EPOCHS:-}" ]]; then
    ARGS+=(--epochs "$EPOCHS")
fi
cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/train.py "${ARGS[@]}"
