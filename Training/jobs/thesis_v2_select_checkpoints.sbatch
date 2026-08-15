#!/bin/bash
#SBATCH --job-name=thesis_v2_select
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=Training/slurm_logs/thesis_v2_selection.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_selection.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"
OUTPUT="${OUTPUT:-Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Experiment matrix missing: $MATRIX" >&2; exit 2; }

# Selection is validation-only. The selector reads validation metrics and
# hash-bound run manifests; it never evaluates or reads a test split.
if [[ -e "$OUTPUT" ]]; then
  singularity exec "$IMAGE" python3 -c 'import hashlib,json,sys; s=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert s["schema_version"] == 2; assert s["complete"] is True; assert s["selection_split"] == "validation"; assert s["test_metrics_read"] is False; assert s["matrix_id"] == m["matrix_id"]; assert s["dataset_tag"] == m["dataset_tag"]; assert s["matrix_sha256"] == h; assert len(s["final_test_runs"]) == len(m["training_experiments"]) * len(m["seeds"]); assert len(s["representative_by_experiment"]) == len(m["training_experiments"])' "$OUTPUT" "$MATRIX" || {
    echo "Existing selection is stale or invalid; refusing to overwrite: $OUTPUT" >&2
    exit 2
  }
  echo "Verified existing validation selection: $OUTPUT"
  exit 0
fi

singularity exec "$IMAGE" \
  python3 Training/select_matrix_checkpoints.py \
  --matrix "$MATRIX" --output "$OUTPUT" --require-complete

# Re-run the same strict validation after creation before downstream test jobs
# can depend on this job's successful exit status.
singularity exec "$IMAGE" python3 -c 'import hashlib,json,sys; s=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert s["schema_version"] == 2; assert s["complete"] is True; assert s["selection_split"] == "validation"; assert s["test_metrics_read"] is False; assert s["matrix_id"] == m["matrix_id"]; assert s["dataset_tag"] == m["dataset_tag"]; assert s["matrix_sha256"] == h; assert len(s["final_test_runs"]) == len(m["training_experiments"]) * len(m["seeds"]); assert len(s["representative_by_experiment"]) == len(m["training_experiments"])' "$OUTPUT" "$MATRIX"
echo "Validation-only checkpoint selection complete: $OUTPUT"
