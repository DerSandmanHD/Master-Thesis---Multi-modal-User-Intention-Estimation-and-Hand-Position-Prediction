#!/bin/bash
#SBATCH --job-name=thesis_v2_test
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-47%4
#SBATCH --output=Training/slurm_logs/thesis_v2_final_test.%A_%a.out
#SBATCH --error=Training/slurm_logs/thesis_v2_final_test.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-thesis_final_v2_validation}"
REPORT_TAG="${REPORT_TAG:-thesis_final_v2_corrected_alignment}"
SELECTION_FILE="${SELECTION_FILE:-Training/reports/${DATASET_TAG}/${REPORT_TAG}/validation_selection.json}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Experiment matrix missing: $MATRIX" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/experiment_matrix.py --matrix "$MATRIX" --stage final-test >/dev/null
mapfile -t FIELDS < <(
  singularity exec "$IMAGE" python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); rows=[(e,s) for e in p["training_experiments"] for s in p["seeds"]]; assert len(rows) == 48; e,s=rows[int(sys.argv[2])]; print(e["id"]); print(s); print(p["dataset_tag"]); print(p["validation_experiment_tag"]); print(p["matrix_id"])' "$MATRIX" "$SLURM_ARRAY_TASK_ID"
)
ID="${FIELDS[0]}"
SEED="${FIELDS[1]}"
[[ "$DATASET_TAG" == "${FIELDS[2]}" ]] || { echo "DATASET_TAG differs from matrix" >&2; exit 2; }
[[ "$EXPERIMENT_TAG" == "${FIELDS[3]}" ]] || { echo "EXPERIMENT_TAG differs from matrix" >&2; exit 2; }
[[ "$REPORT_TAG" == "${FIELDS[4]}" ]] || { echo "REPORT_TAG differs from matrix" >&2; exit 2; }
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${ID}/${ID}_seed${SEED}"
OUTPUT="Training/reports/${DATASET_TAG}/${REPORT_TAG}/final_test/${ID}_seed${SEED}.json"
[[ -f "$RUN_DIR/artifact_manifest.json" ]] || { echo "Frozen validation run missing: $RUN_DIR" >&2; exit 2; }
[[ -f "$SELECTION_FILE" ]] || { echo "Validation selection missing: $SELECTION_FILE" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
singularity exec "$IMAGE" python3 -c 'import hashlib,json,sys; x=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert x["complete"] is True; assert x["selection_split"] == "validation"; assert x["test_metrics_read"] is False; assert x["matrix_id"] == p["matrix_id"] and x["matrix_sha256"] == h' "$SELECTION_FILE" "$MATRIX"

if [[ -f "$OUTPUT" ]]; then
  singularity exec "$IMAGE" python3 -c 'import hashlib,json,pathlib,sys; x=json.load(open(sys.argv[1])); a=x["matrix_authorization"]; m=json.load(open(sys.argv[6])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); payload={**x,"report_fingerprint":None}; fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest(); assert x["schema_version"] == 2 and x["evaluation_protocol"] == "validation_frozen_checkpoint_single_test_v2" and x["report_fingerprint"] == fp; assert x["checkpoint"]["name"] == "best_intention"; assert x["checkpoint"]["selection_split"] == "validation"; assert x["test_used_for_model_or_checkpoint_selection"] is False; assert a["experiment_id"] == sys.argv[3] and a["seed"] == int(sys.argv[4]); assert a["selection_file_sha256"] == h; assert a["authorized_checkpoint_sha256"] == x["checkpoint"]["sha256"]; assert pathlib.Path(x["source_run"]).resolve() == pathlib.Path(sys.argv[5]).resolve(); assert x["source_artifact_manifest_fingerprint"] == m["manifest_fingerprint"]' "$OUTPUT" "$SELECTION_FILE" "$ID" "$SEED" "$RUN_DIR" "$RUN_DIR/artifact_manifest.json"
  echo "Final test already complete: $OUTPUT"
  exit 0
fi

echo "START final-test id=$ID seed=$SEED run=$RUN_DIR"
singularity exec --nv "$IMAGE" \
  python3 Training/evaluate_frozen_run.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best_intention \
  --selection-file "$SELECTION_FILE" \
  --experiment-id "$ID" \
  --output "$OUTPUT"
echo "DONE final-test id=$ID seed=$SEED output=$OUTPUT"
