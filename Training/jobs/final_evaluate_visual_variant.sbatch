#!/bin/bash
#SBATCH --job-name=aria_v2_visual_final
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=Training/slurm_logs/final_visual_variant.%A_%a.out
#SBATCH --error=Training/slurm_logs/final_visual_variant.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
SCREEN_TAG="${SCREEN_TAG:-visual_embedding_screen_v1}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-visual_embedding_final_v1}"
SUMMARY="Training/reports/${DATASET_TAG}/${SCREEN_TAG}/summary.json"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd "$REPO_DIR"
VARIANT="$(python3 - "$SUMMARY" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["selected_variant_for_final_test"]
if not value:
    raise SystemExit("Visual screening has no selected variant")
print(value)
PY
)"
if [[ "$VARIANT" == "sensor_baseline" ]]; then
    echo "Validation selected the already-tested tuned sensor baseline; no new test access."
    exit 0
fi
case "$VARIANT" in
    clip_only) CONFIG="Training/configs/visual/residual_v2_clip_only.json" ;;
    sensor_plus_clip) CONFIG="Training/configs/visual/residual_v2_sensor_plus_clip.json" ;;
    *) echo "Unexpected selectable visual variant: $VARIANT" >&2; exit 2 ;;
esac
RUN_ID="${EXPERIMENT_TAG}_${VARIANT}_seed${SEED}"
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${VARIANT}/${RUN_ID}"
if [[ -f "$RUN_DIR/metrics.json" ]]; then
    if python3 - "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("test_evaluation_skipped") is False
assert set(x["test"]) == {"best_intention", "best_pose"}
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
    --run-dir "$RUN_DIR"
