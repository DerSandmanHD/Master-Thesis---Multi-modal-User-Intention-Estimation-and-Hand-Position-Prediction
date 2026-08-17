#!/bin/bash
#SBATCH --job-name=thesis_v2_post
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
# A grouped 10,000-sample bootstrap can approach three hours.  Every stage is
# resumable and hash-validated, but leave enough headroom for a cold run.
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=Training/slurm_logs/thesis_v2_postprocess.%A_%a.out
#SBATCH --error=Training/slurm_logs/thesis_v2_postprocess.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
REPORT_TAG="${REPORT_TAG:-thesis_final_v2_corrected_alignment}"
REPORT_ROOT="${REPORT_ROOT:-Training/reports/${DATASET_TAG}/${REPORT_TAG}}"
SELECTION_FILE="${SELECTION_FILE:-${REPORT_ROOT}/validation_selection.json}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${REPORT_ROOT}/final_test}"
POSTPROCESS_ROOT="${POSTPROCESS_ROOT:-${REPORT_ROOT}/postprocess}"
POSTPROCESS_EXPERIMENTS="${POSTPROCESS_EXPERIMENTS:-}"
: "${SLURM_ARRAY_TASK_ID:?This job must run as its declared array}"
MASTER_DIR="${MASTER_DIR:-Data_collection/master_datasets}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Experiment matrix missing: $MATRIX" >&2; exit 2; }
[[ -f "$SELECTION_FILE" ]] || { echo "Selection missing: $SELECTION_FILE" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/experiment_matrix.py --matrix "$MATRIX" --json >/dev/null

mapfile -t FIELDS < <(
  singularity exec "$IMAGE" python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); override=[x for x in sys.argv[6].split(":") if x]; ids=override or m["postprocessing"]["required_t1_experiments"]; known={e["id"] for e in m["training_experiments"]}; allowed={"residual_modality_gated","visual_corrected_clip_modality_gate"}; assert set(ids) <= known and (not override or set(ids) <= allowed); pairs=[(x,int(seed)) for x in ids for seed in m["seeds"]]; i=int(sys.argv[3]); assert m["postprocessing"]["seed_policy"] == "all_matrix_seeds" and len(pairs) == len(ids)*len(m["seeds"]) and 0 <= i < len(pairs); experiment,seed=pairs[i]; rows=[r for r in s["final_test_runs"] if r["experiment_id"] == experiment and int(r["seed"]) == seed]; assert s["complete"] is True and s["selection_split"] == "validation" and s["test_metrics_read"] is False and len(rows) == 1; print(experiment); print(seed); print(rows[0]["run_dir"]); print(f"{sys.argv[4]}/{experiment}_seed{seed}.json"); print(f"{sys.argv[5]}/{experiment}_seed{seed}")' "$MATRIX" "$SELECTION_FILE" "$SLURM_ARRAY_TASK_ID" "$FINAL_TEST_DIR" "$POSTPROCESS_ROOT" "$POSTPROCESS_EXPERIMENTS"
)
EXPERIMENT_ID="${FIELDS[0]}"
SEED="${FIELDS[1]}"
RUN_DIR="${FIELDS[2]}"
FINAL_TEST_JSON="${FIELDS[3]}"
REPORT_DIR="${FIELDS[4]}"
PREDICTIONS="$REPORT_DIR/test_predictions.csv"
PREDICTION_REPORT="$REPORT_DIR/test_predictions.json"
GROUPED_REPORT="$REPORT_DIR/grouped_metrics.json"
GROUPED_TABLE="$REPORT_DIR/grouped_metrics.csv"
BASELINE_REPORT="$REPORT_DIR/t1_pose_baselines.json"
BASELINE_TABLE="$REPORT_DIR/t1_pose_baselines.csv"

[[ -f "$FINAL_TEST_JSON" ]] || { echo "Final test record missing: $FINAL_TEST_JSON" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
singularity exec "$IMAGE" python3 -c 'import hashlib,json,pathlib,sys; x=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[3])); payload={**x,"report_fingerprint":None}; fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest(); assert x["schema_version"] == 2 and x["evaluation_protocol"] == "validation_frozen_checkpoint_single_test_v2" and x["report_fingerprint"] == fp; assert x["checkpoint"]["name"] == "best_intention"; assert x["test_used_for_model_or_checkpoint_selection"] is False; assert x["matrix_authorization"] is not None; assert pathlib.Path(x["source_run"]).resolve() == pathlib.Path(sys.argv[2]).resolve(); assert x["source_artifact_manifest_fingerprint"] == m["manifest_fingerprint"]; assert x["checkpoint"]["sha256"] == x["matrix_authorization"]["authorized_checkpoint_sha256"]' "$FINAL_TEST_JSON" "$RUN_DIR" "$RUN_DIR/artifact_manifest.json"

if [[ -f "$PREDICTIONS" && -f "$PREDICTION_REPORT" ]]; then
  singularity exec "$IMAGE" python3 Training/validate_postprocess_resume.py prediction \
    --csv "$PREDICTIONS" --report "$PREDICTION_REPORT" --run-dir "$RUN_DIR"
elif [[ -e "$PREDICTIONS" || -e "$PREDICTION_REPORT" ]]; then
  echo "Incomplete prediction-output pair; refusing unsafe resume." >&2
  exit 2
else
  mkdir -p "$REPORT_DIR"
  singularity exec --nv "$IMAGE" \
    python3 Training/export_residual_predictions.py \
    --run-dir "$RUN_DIR" \
    --checkpoint best_intention_model.pt \
    --master-dir "$MASTER_DIR" \
    --split test \
    --final-test-report "$FINAL_TEST_JSON" \
    --output-csv "$PREDICTIONS" \
    --report-out "$PREDICTION_REPORT" \
    --device cuda --batch-size 64 --num-workers 2
fi

if [[ -f "$GROUPED_REPORT" && -f "$GROUPED_TABLE" ]]; then
  singularity exec "$IMAGE" python3 Training/validate_postprocess_resume.py grouped \
    --csv "$PREDICTIONS" --prediction-report "$PREDICTION_REPORT" \
    --report "$GROUPED_REPORT" --table "$GROUPED_TABLE"
elif [[ -e "$GROUPED_REPORT" || -e "$GROUPED_TABLE" ]]; then
  echo "Incomplete grouped-output pair; refusing unsafe resume." >&2
  exit 2
else
  singularity exec "$IMAGE" \
    python3 Training/evaluation/evaluate_grouped_predictions.py \
    --predictions "$PREDICTIONS" \
    --prediction-report "$PREDICTION_REPORT" \
    --report-out "$GROUPED_REPORT" \
    --table-out "$GROUPED_TABLE" \
    --bootstrap-iterations 10000 --bootstrap-seed 42 --sequence-bootstrap
fi

if [[ -f "$BASELINE_REPORT" && -f "$BASELINE_TABLE" ]]; then
  singularity exec "$IMAGE" python3 Training/validate_postprocess_resume.py baseline \
    --report "$BASELINE_REPORT" --table "$BASELINE_TABLE" \
    --config "$RUN_DIR/config.json"
elif [[ -e "$BASELINE_REPORT" || -e "$BASELINE_TABLE" ]]; then
  echo "Incomplete baseline-output pair; refusing unsafe resume." >&2
  exit 2
else
  singularity exec "$IMAGE" \
    python3 Training/evaluate_pose_baselines.py \
    --config "$RUN_DIR/config.json" \
    --report-out "$BASELINE_REPORT" \
    --details-out "$BASELINE_TABLE"
fi

if singularity exec "$IMAGE" python3 -c 'import pandas as pd,sys; x=pd.read_csv(sys.argv[1],nrows=1); assert any(c.startswith("modality_") and c.endswith("_weight") for c in x.columns)' "$PREDICTIONS"; then
  WEIGHT_REPORT="$REPORT_DIR/modality_weights/summary.json"
  WEIGHT_TABLE="$REPORT_DIR/modality_weights/modality_weights_by_window.csv"
  if [[ -f "$WEIGHT_REPORT" && -f "$WEIGHT_TABLE" ]]; then
    singularity exec "$IMAGE" python3 Training/validate_postprocess_resume.py modality \
      --csv "$PREDICTIONS" --report "$WEIGHT_REPORT" --table "$WEIGHT_TABLE"
  elif [[ -e "$WEIGHT_REPORT" || -e "$WEIGHT_TABLE" ]]; then
    echo "Incomplete modality-weight output pair; refusing unsafe resume." >&2
    exit 2
  else
    singularity exec "$IMAGE" \
      python3 Training/evaluation/summarize_modality_weights.py \
      --predictions "$PREDICTIONS" --output-dir "$REPORT_DIR/modality_weights"
  fi
else
  echo "Selected model has no modality-wise weights; skipping weight report."
fi

echo "Postprocessing complete: experiment=$EXPERIMENT_ID seed=$SEED dir=$REPORT_DIR"
