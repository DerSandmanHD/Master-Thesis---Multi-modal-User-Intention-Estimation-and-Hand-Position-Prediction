#!/bin/bash
#SBATCH --job-name=aria_final_compare
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --output=final_compare.%A_%a.out
#SBATCH --error=final_compare.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
FINAL_TAG="${FINAL_TAG:-final_clean_v1}"

MODELS=(transformer_v1 mlp_v1 gru_v1 residual_v2)
CONFIGS=(
    Training/configs/hierarchical_baseline_v1.json
    Training/configs/hierarchical_mlp_v1.json
    Training/configs/hierarchical_gru_v1.json
    Training/configs/hierarchical_residual_v2.json
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
RUN_DIR="Training/runs/${FINAL_TAG}_${MODEL}_seed${SEED}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 "$ENTRYPOINT" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"
