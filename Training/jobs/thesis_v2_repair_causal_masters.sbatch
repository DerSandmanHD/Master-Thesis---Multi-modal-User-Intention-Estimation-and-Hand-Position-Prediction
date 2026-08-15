#!/bin/bash
#SBATCH --job-name=thesis_v2_master_repair
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=Training/slurm_logs/thesis_v2_master_repair.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_master_repair.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/Data_collection}"
MASTER_DIR="${MASTER_DIR:-$DATA_DIR/master_datasets}"
FAILED_BATCH_REPORT="${FAILED_BATCH_REPORT:-$MASTER_DIR/master_batch_report.json}"
TIMESTAMPS="${TIMESTAMPS:-$DATA_DIR/Data_vrs/timestamps_summary.reviewed.json}"
ANNOTATIONS="${ANNOTATIONS:-$DATA_DIR/manual_timestamp_review.csv}"
MANIFEST="${MANIFEST:-$DATA_DIR/dataset_manifest.csv}"
REPAIR_REPORT="$MASTER_DIR/master_repair_job${SLURM_JOB_ID}.json"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$FAILED_BATCH_REPORT" ]] || { echo "Failed batch report missing: $FAILED_BATCH_REPORT" >&2; exit 2; }
[[ -f "$TIMESTAMPS" ]] || { echo "Reviewed timestamps missing: $TIMESTAMPS" >&2; exit 2; }
mkdir -p Training/slurm_logs

mapfile -t FAILED_SEQUENCES < <(
  singularity exec "$IMAGE" python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); rows=sorted(r["sequence_id"] for r in x["records"] if r["status"] == "error"); assert x["errors"] == len(rows) and rows; print("\n".join(rows))' "$FAILED_BATCH_REPORT"
)
echo "Repairing ${#FAILED_SEQUENCES[@]} failed causal masters"
SEQUENCE_ARGS=()
for sequence_id in "${FAILED_SEQUENCES[@]}"; do
  SEQUENCE_ARGS+=(--sequence "$sequence_id")
done

singularity exec "$IMAGE" python3 Code/build_master_dataset_batch.py \
  --data-root "$DATA_DIR" \
  --manifest "$MANIFEST" \
  --timestamps "$TIMESTAMPS" \
  --annotations "$ANNOTATIONS" \
  --output-dir "$MASTER_DIR" \
  --batch-report "$REPAIR_REPORT" \
  --require-semantic-annotations --overwrite \
  "${SEQUENCE_ARGS[@]}"

singularity exec "$IMAGE" python3 Code/dataset_qa.py \
  --data-root "$DATA_DIR" --timestamps "$TIMESTAMPS"

singularity exec "$IMAGE" python3 Training/verify_causal_masters.py \
  --master-dir "$MASTER_DIR" \
  --manifest "$MANIFEST" \
  --expected-alignment-version causal_backward_device_time_v1 \
  --expected-sequence-fingerprint 5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd \
  --output "$MASTER_DIR/causal_master_preflight.repair_job${SLURM_JOB_ID}.json"

echo "Causal master repair and full preflight complete: $REPAIR_REPORT"
