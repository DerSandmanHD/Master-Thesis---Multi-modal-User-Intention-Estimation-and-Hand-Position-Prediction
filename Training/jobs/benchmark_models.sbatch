#!/bin/bash
#SBATCH --job-name=aria_final_compare
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --output=Training/slurm_logs/benchmark_models.%A_%a.out
#SBATCH --error=Training/slurm_logs/benchmark_models.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-benchmark_v1}"

MODELS=(transformer_v1 mlp_v1 gru_v1 residual_v2)
CONFIGS=(
    Training/configs/models/transformer_v1.json
    Training/configs/models/mlp_v1.json
    Training/configs/models/gru_v1.json
    Training/configs/models/residual_transformer_v2.json
)
ENTRYPOINTS=(
    Training/train.py
    Training/train.py
    Training/train.py
    Training/train_residual.py
)
SEEDS=(42 43 44)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
MODEL="${MODELS[$MODEL_INDEX]}"
CONFIG="${CONFIGS[$MODEL_INDEX]}"
ENTRYPOINT="${ENTRYPOINTS[$MODEL_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_ID="${EXPERIMENT_TAG}_${MODEL}_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${MODEL}/${RUN_ID}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 "$ENTRYPOINT" \
    --config "$CONFIG" \
    --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"
