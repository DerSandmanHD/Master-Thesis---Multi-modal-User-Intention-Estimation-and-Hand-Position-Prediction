#!/bin/bash
#SBATCH --job-name=thesis_v2_val
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-47%4
#SBATCH --output=Training/slurm_logs/thesis_v2_validation.%A_%a.out
#SBATCH --error=Training/slurm_logs/thesis_v2_validation.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-thesis_final_v2_validation}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Experiment matrix missing: $MATRIX" >&2; exit 2; }
singularity exec "$IMAGE" python3 Training/experiment_matrix.py --matrix "$MATRIX" --stage validation >/dev/null
mapfile -t FIELDS < <(
  singularity exec "$IMAGE" python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); rows=[(e,s) for e in p["training_experiments"] for s in p["seeds"]]; assert len(rows) == 48; e,s=rows[int(sys.argv[2])]; print(e["id"]); print(e["entrypoint"]); print(e["config"]); print(s); print(p["dataset_tag"]); print(p["validation_experiment_tag"])' "$MATRIX" "$SLURM_ARRAY_TASK_ID"
)
ID="${FIELDS[0]}"
ENTRYPOINT="${FIELDS[1]}"
CONFIG="${FIELDS[2]}"
SEED="${FIELDS[3]}"
[[ "$DATASET_TAG" == "${FIELDS[4]}" ]] || { echo "DATASET_TAG differs from matrix" >&2; exit 2; }
[[ "$EXPERIMENT_TAG" == "${FIELDS[5]}" ]] || { echo "EXPERIMENT_TAG differs from matrix" >&2; exit 2; }
RUN_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${ID}/${ID}_seed${SEED}"
[[ -f "$CONFIG" ]] || { echo "Config missing: $CONFIG" >&2; exit 2; }

if [[ "$ID" == "terminal_endpose_learned" ]]; then
  AUDIT="Training/reports/${DATASET_TAG}/terminal_endpose_corrected_v2/audit/endpose_target_audit.json"
  [[ -f "$AUDIT" ]] || { echo "Corrected terminal-target audit missing: $AUDIT" >&2; exit 2; }
  singularity exec "$IMAGE" python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["training_authorized_by_audit"] is True; assert x["pose_target_definition"]["target_definition_version"] == "terminal_endpose_unique_hand_capture_v2"' "$AUDIT"
fi

if [[ -e "$RUN_DIR" ]]; then
  if [[ -f "$RUN_DIR/artifact_manifest.json" && -f "$RUN_DIR/metrics.json" ]]; then
    singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
    singularity exec "$IMAGE" python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("test_evaluation_skipped") is True; assert "test" not in x and "test_by_checkpoint" not in x' "$RUN_DIR/metrics.json"
    echo "Validation run already complete: $RUN_DIR"
    exit 0
  fi
  INCOMPLETE_DIR="${RUN_DIR}.incomplete_job${SLURM_JOB_ID}"
  [[ ! -e "$INCOMPLETE_DIR" ]] || { echo "Preservation target exists: $INCOMPLETE_DIR" >&2; exit 2; }
  mv "$RUN_DIR" "$INCOMPLETE_DIR"
fi

echo "START id=$ID seed=$SEED config=$CONFIG run=$RUN_DIR"
singularity exec --nv "$IMAGE" \
  python3 "$ENTRYPOINT" \
  --config "$CONFIG" \
  --dataset-tag "$DATASET_TAG" \
  --experiment-tag "$EXPERIMENT_TAG" \
  --seed "$SEED" \
  --run-dir "$RUN_DIR" \
  --skip-test-evaluation
singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
echo "DONE id=$ID seed=$SEED"
