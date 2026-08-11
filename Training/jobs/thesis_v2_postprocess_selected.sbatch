#!/bin/bash
#SBATCH --job-name=thesis_v2_post
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=Training/slurm_logs/thesis_v2_postprocess.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_postprocess.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
: "${RUN_DIR:?Set RUN_DIR to the validation-frozen selected residual run}"
: "${FINAL_TEST_JSON:?Set FINAL_TEST_JSON to its completed frozen test evaluation}"
: "${REPORT_DIR:?Set REPORT_DIR to a unique report directory}"
MASTER_DIR="${MASTER_DIR:-Data_collection/master_datasets}"
PREDICTIONS="$REPORT_DIR/test_predictions.csv"
PREDICTION_REPORT="$REPORT_DIR/test_predictions.json"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$FINAL_TEST_JSON" ]] || { echo "Final test record missing: $FINAL_TEST_JSON" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
singularity exec "$IMAGE" python3 -c 'import json,pathlib,sys; x=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[3])); assert x["checkpoint"]["name"] == "best_intention"; assert x["test_used_for_model_or_checkpoint_selection"] is False; assert x["matrix_authorization"] is not None; assert pathlib.Path(x["source_run"]).resolve() == pathlib.Path(sys.argv[2]).resolve(); assert x["source_artifact_manifest_fingerprint"] == m["manifest_fingerprint"]; assert x["checkpoint"]["sha256"] == x["matrix_authorization"]["authorized_checkpoint_sha256"]' "$FINAL_TEST_JSON" "$RUN_DIR" "$RUN_DIR/artifact_manifest.json"

if [[ -e "$PREDICTIONS" || -e "$PREDICTION_REPORT" ]]; then
  echo "Prediction outputs already exist; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "$REPORT_DIR"
singularity exec --nv "$IMAGE" \
  python3 Training/export_residual_predictions.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best_intention_model.pt \
  --master-dir "$MASTER_DIR" \
  --split test \
  --output-csv "$PREDICTIONS" \
  --report-out "$PREDICTION_REPORT" \
  --device cuda --batch-size 64 --num-workers 2

singularity exec "$IMAGE" \
  python3 Training/evaluation/evaluate_grouped_predictions.py \
  --predictions "$PREDICTIONS" \
  --prediction-report "$PREDICTION_REPORT" \
  --report-out "$REPORT_DIR/grouped_metrics.json" \
  --table-out "$REPORT_DIR/grouped_metrics.csv" \
  --bootstrap-iterations 10000 --bootstrap-seed 42 --sequence-bootstrap

singularity exec "$IMAGE" \
  python3 Training/evaluate_pose_baselines.py \
  --config "$RUN_DIR/config.json" \
  --report-out "$REPORT_DIR/t1_pose_baselines.json" \
  --details-out "$REPORT_DIR/t1_pose_baselines.csv"

if singularity exec "$IMAGE" python3 -c 'import pandas as pd,sys; x=pd.read_csv(sys.argv[1],nrows=1); assert any(c.startswith("modality_") and c.endswith("_weight") for c in x.columns)' "$PREDICTIONS"; then
  singularity exec "$IMAGE" \
    python3 Training/evaluation/summarize_modality_weights.py \
    --predictions "$PREDICTIONS" --output-dir "$REPORT_DIR/modality_weights"
else
  echo "Selected model has no modality-wise weights; skipping weight report."
fi

echo "Postprocessing complete: $REPORT_DIR"
