#!/bin/bash
#SBATCH --job-name=aria_v2_hp_confirm
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --output=Training/slurm_logs/hyperparameter_confirm_v2.%A_%a.out
#SBATCH --error=Training/slurm_logs/hyperparameter_confirm_v2.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
STAGE_A_TAG="${STAGE_A_TAG:-residual_v2_hp_search_v1}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_hp_confirm_v1}"
SUMMARY="Training/reports/${DATASET_TAG}/${STAGE_A_TAG}/summary.json"
SEEDS=(42 43 44)

cd "$REPO_DIR"
TRIAL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED="${SEEDS[$SEED_INDEX]}"
TRIAL_TAG="$(python3 - "$SUMMARY" "$TRIAL_INDEX" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
selected = summary["selected_for_confirmation"]
if len(selected) != 3:
    raise SystemExit(f"Expected 3 selected trials, found {len(selected)}")
print(selected[int(sys.argv[2])])
PY
)"
CONFIG="Training/configs/hyperparameter_search_v1/${TRIAL_TAG}.json"
RUN_ID="${EXPERIMENT_TAG}_${TRIAL_TAG}_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${TRIAL_TAG}/${RUN_ID}"

if [[ -f "$RUN_DIR/metrics.json" ]]; then
    if python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is True
assert "test" not in x
assert set(x["validation_by_checkpoint"]) == {"best_intention", "best_pose"}
PY
    then
        echo "$TRIAL_TAG seed $SEED already completed"
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
