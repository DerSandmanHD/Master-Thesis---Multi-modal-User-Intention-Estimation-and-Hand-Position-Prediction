#!/bin/bash
#SBATCH --job-name=aria_v2_hp_search
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-23%4
#SBATCH --output=Training/slurm_logs/hyperparameter_search_v2.%A_%a.out
#SBATCH --error=Training/slurm_logs/hyperparameter_search_v2.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_hp_search_v1}"
TRIAL_INDEX="${SLURM_ARRAY_TASK_ID}"
TRIAL_TAG="$(printf 'trial_%03d' "$TRIAL_INDEX")"
CONFIG="Training/configs/hyperparameter_search_v1/${TRIAL_TAG}.json"
RUN_ID="${EXPERIMENT_TAG}_${TRIAL_TAG}_seed42"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${TRIAL_TAG}/${RUN_ID}"

cd "$REPO_DIR"

if [[ -f "$RUN_DIR/metrics.json" ]]; then
    if python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
assert metrics.get("test_evaluation_skipped") is True
assert "test" not in metrics
assert set(metrics["validation_by_checkpoint"]) == {
    "best_intention",
    "best_pose",
}
PY
    then
        echo "$TRIAL_TAG already completed with validation-only metrics"
        exit 0
    fi
fi

if [[ -e "$RUN_DIR" ]]; then
    INCOMPLETE_DIR="${RUN_DIR}.incomplete_job${SLURM_JOB_ID}"
    if [[ -e "$INCOMPLETE_DIR" ]]; then
        echo "Cannot preserve incomplete run; target exists: $INCOMPLETE_DIR" >&2
        exit 2
    fi
    mv "$RUN_DIR" "$INCOMPLETE_DIR"
    echo "Preserved incomplete run as $INCOMPLETE_DIR"
fi

singularity exec --nv "$IMAGE" \
    python3 Training/train_residual.py \
    --config "$CONFIG" \
    --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --seed 42 \
    --run-dir "$RUN_DIR" \
    --skip-test-evaluation
