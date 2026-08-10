#!/bin/bash
#SBATCH --job-name=aria_endpose_v1
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=Training/slurm_logs/endpose_v1.%A_%a.out
#SBATCH --error=Training/slurm_logs/endpose_v1.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v1}"
CONFIG="${CONFIG:-Training/configs/models/residual_transformer_endpose_v1.json}"
REPORT_DIR="Training/reports/${DATASET_TAG}/${EXPERIMENT_TAG}"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="${EXPERIMENT_TAG}_residual_v2_endpose_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/residual_v2_endpose/${RUN_ID}"
BASELINE_RUN="Training/runs/${DATASET_TAG}/residual_v2_tuned_v1/residual_v2_tuned/residual_v2_tuned_v1_residual_v2_tuned_seed${SEED}"
BASELINE_OUTPUT="${REPORT_DIR}/baseline_terminal_eval/seed${SEED}.json"

cd "$REPO_DIR"
python3 - "$REPORT_DIR/audit/endpose_target_audit.json" "$REPORT_DIR/audit/dry_run.json" <<'PY'
import json
import sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
dry = json.load(open(sys.argv[2], encoding="utf-8"))
assert audit["training_authorized_by_audit"] is True
assert dry["status"] == "passed"
assert dry["training_started"] is False
assert dry["test_metrics_computed"] is False
PY

if [[ -f "$RUN_DIR/metrics.json" ]]; then
    python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x["pose_target_definition"]["mode"] == "terminal_endpose"
assert x.get("test_evaluation_skipped") is False
assert set(x["test"]) == {"best_intention", "best_pose"}
PY
    echo "End-pose seed $SEED already completed"
else
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
fi

mkdir -p "$(dirname "$BASELINE_OUTPUT")"
if [[ ! -f "$BASELINE_OUTPUT" ]]; then
    singularity exec --nv "$IMAGE" \
        python3 Training/evaluate_t1_on_endpose.py \
        --source-run-dir "$BASELINE_RUN" \
        --target-config "$CONFIG" \
        --output "$BASELINE_OUTPUT" \
        --device cuda
fi
