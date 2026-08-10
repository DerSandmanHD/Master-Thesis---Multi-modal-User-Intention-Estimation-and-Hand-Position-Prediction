#!/bin/bash
#SBATCH --job-name=aria_endpose_v2_final
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=Training/slurm_logs/endpose_v2_final.%A_%a.out
#SBATCH --error=Training/slurm_logs/endpose_v2_final.%A_%a.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
CONFIRM_TAG="${CONFIRM_TAG:-residual_v2_endpose_v2_hp_confirm}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v2}"
CONFIG="Training/reports/${DATASET_TAG}/${CONFIRM_TAG}/selected_config.json"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/residual_v2_endpose_v2/${EXPERIMENT_TAG}_dual_horizon_endpose_v2_seed${SEED}"
cd "$REPO_DIR"
python3 - "Training/reports/${DATASET_TAG}/${CONFIRM_TAG}/summary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["complete"] and x["test_metrics_used"] is False and x["selected_trial"]
PY
if [[ -f "$RUN_DIR/metrics.json" ]]; then
    python3 - "$RUN_DIR/metrics.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is False and set(x["test"]) == {"best_intention", "best_pose"}
PY
    echo "Final endpose-v2 seed $SEED already completed"
    exit 0
fi
if [[ -e "$RUN_DIR" ]]; then
    mv "$RUN_DIR" "${RUN_DIR}.incomplete_job${SLURM_JOB_ID}"
fi
singularity exec --nv "$IMAGE" python3 Training/train_residual.py \
    --config "$CONFIG" --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" --seed "$SEED" --run-dir "$RUN_DIR"
