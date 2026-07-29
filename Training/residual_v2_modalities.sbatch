#!/bin/bash
#SBATCH --job-name=aria_v2_ablation
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --output=aria_v2_ablation.%A_%a.out
#SBATCH --error=aria_v2_ablation.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
ABLATION_TAG="${ABLATION_TAG:-modality_ablation_v1}"

ABLATIONS=(no_gaze no_hands no_objects no_vio)
CONFIGS=(
    Training/configs/ablations/residual_v2_no_gaze.json
    Training/configs/ablations/residual_v2_no_hands.json
    Training/configs/ablations/residual_v2_no_objects.json
    Training/configs/ablations/residual_v2_no_vio.json
)
SEEDS=(42 43 44)

ABLATION_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
ABLATION="${ABLATIONS[$ABLATION_INDEX]}"
CONFIG="${CONFIGS[$ABLATION_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_DIR="Training/runs/ablations/${ABLATION_TAG}/${ABLATION}/seed_${SEED}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/train_residual.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"
