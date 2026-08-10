#!/bin/bash
#SBATCH --job-name=aria_endpose_v2_confirm
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --output=Training/slurm_logs/endpose_v2_confirm.%A_%a.out
#SBATCH --error=Training/slurm_logs/endpose_v2_confirm.%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
STAGE_A_TAG="${STAGE_A_TAG:-residual_v2_endpose_v2_hp_search}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v2_hp_confirm}"
SEEDS=(42 43 44)
TRIAL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
cd "$REPO_DIR"
TRIAL_TAG="$(python3 - "Training/reports/${DATASET_TAG}/${STAGE_A_TAG}/summary.json" "$TRIAL_INDEX" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["complete"] and x["test_metrics_used"] is False
print(x["selected_for_confirmation"][int(sys.argv[2])])
PY
)"
CONFIG="Training/configs/endpose_v2_search/${TRIAL_TAG}.json"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${TRIAL_TAG}/${EXPERIMENT_TAG}_${TRIAL_TAG}_seed${SEED}"
if [[ -f "$RUN_DIR/metrics.json" ]]; then
    python3 - "$RUN_DIR/metrics.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is True and "test" not in x
PY
    echo "$TRIAL_TAG seed $SEED already completed"
    exit 0
fi
if [[ -e "$RUN_DIR" ]]; then
    mv "$RUN_DIR" "${RUN_DIR}.incomplete_job${SLURM_JOB_ID}"
fi
singularity exec --nv "$IMAGE" python3 Training/train_residual.py \
    --config "$CONFIG" --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" --seed "$SEED" --run-dir "$RUN_DIR" \
    --skip-test-evaluation
