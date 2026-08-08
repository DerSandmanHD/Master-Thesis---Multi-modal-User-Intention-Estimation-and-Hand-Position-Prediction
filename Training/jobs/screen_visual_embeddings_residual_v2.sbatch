#!/bin/bash
#SBATCH --job-name=aria_v2_visual
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8%3
#SBATCH --output=Training/slurm_logs/visual_embedding_screen_v2.%A_%a.out
#SBATCH --error=Training/slurm_logs/visual_embedding_screen_v2.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-visual_embedding_screen_v1}"
VARIANTS=(clip_only sensor_plus_clip sensor_plus_random)
CONFIGS=(
    Training/configs/visual/residual_v2_clip_only.json
    Training/configs/visual/residual_v2_sensor_plus_clip.json
    Training/configs/visual/residual_v2_sensor_plus_random.json
)
SEEDS=(42 43 44)

VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
CONFIG="${CONFIGS[$VARIANT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_ID="${EXPERIMENT_TAG}_${VARIANT}_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${VARIANT}/${RUN_ID}"

cd "$REPO_DIR"
[[ -f Training/visual_projections/${DATASET_TAG}/clip_vit_b32_openai_5hz_pca32.npz ]] || {
    echo "CLIP projection is missing; run prepare_clip_embeddings.sbatch first" >&2
    exit 2
}
if [[ -f "$RUN_DIR/metrics.json" ]]; then
    if python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is True
assert "test" not in x
PY
    then
        echo "$VARIANT seed $SEED already completed"
        exit 0
    fi
fi
if [[ -e "$RUN_DIR" ]]; then
    INCOMPLETE_DIR="${RUN_DIR}.incomplete_job${SLURM_JOB_ID}"
    [[ ! -e "$INCOMPLETE_DIR" ]] || { echo "Preservation target exists" >&2; exit 2; }
    mv "$RUN_DIR" "$INCOMPLETE_DIR"
fi

singularity exec --nv "$IMAGE" \
    python3 Training/train_residual.py \
    --config "$CONFIG" \
    --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR" \
    --skip-test-evaluation
