#!/bin/bash
#SBATCH --job-name=aria_v2_tuned_final
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=Training/slurm_logs/final_tuned_residual_v2.%A_%a.out
#SBATCH --error=Training/slurm_logs/final_tuned_residual_v2.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_tuned_v1}"
CONFIG="${CONFIG:-Training/reports/${DATASET_TAG}/residual_v2_hp_confirm_v1/selected_config.json}"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="${EXPERIMENT_TAG}_residual_v2_tuned_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/residual_v2_tuned/${RUN_ID}"

cd "$REPO_DIR"
[[ -f "$CONFIG" ]] || { echo "Selected config missing: $CONFIG" >&2; exit 2; }
if [[ -f "$RUN_DIR/metrics.json" ]]; then
    if python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is False
assert set(x["test"]) == {"best_intention", "best_pose"}
PY
    then
        echo "Final tuned seed $SEED already completed"
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
    --run-dir "$RUN_DIR"
