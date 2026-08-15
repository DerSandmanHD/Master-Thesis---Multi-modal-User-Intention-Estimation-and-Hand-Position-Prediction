#!/bin/bash
#SBATCH --job-name=thesis_v2_split
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/thesis_v2_split_audit.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_split_audit.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
CONFIG="${CONFIG:-Training/configs/models/residual_transformer_v2.json}"
OUTPUT="${OUTPUT:-Training/reports/${DATASET_TAG}/split_confounding_v2/split_audit.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Split audit already exists; refusing to overwrite: $OUTPUT" >&2; exit 2; }
mkdir -p Training/slurm_logs "$(dirname "$OUTPUT")"

singularity exec "$IMAGE" python3 Training/verify_causal_masters.py \
  --master-dir Data_collection/master_datasets \
  --manifest Data_collection/dataset_manifest.csv \
  --expected-alignment-version causal_backward_device_time_v1 \
  --expected-sequence-fingerprint 5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd

singularity exec "$IMAGE" python3 Training/audit_participant_splits.py \
  --master-dir Data_collection/master_datasets \
  --manifest Data_collection/dataset_manifest.csv \
  --config "$CONFIG" \
  --balanced-candidate --leave-one-participant-out --seed 42 \
  --output-json "$OUTPUT"

singularity exec "$IMAGE" python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); cv=x["participant_group_cv"]; assert x["sequence_count"] == 214 and x["participant_count"] == 25; assert cv["fold_count"] == 25 and len(cv["folds"]) == 25; assert cv["participant_balanced_aggregation_identifiable"] is True' "$OUTPUT"
echo "Participant split audit complete: $OUTPUT"
