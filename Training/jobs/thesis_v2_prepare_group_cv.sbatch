#!/bin/bash
#SBATCH --job-name=thesis_v2_cvplan
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=Training/slurm_logs/thesis_v2_group_cv_plan.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_group_cv_plan.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
SPLIT_AUDIT="${SPLIT_AUDIT:-Training/reports/${DATASET_TAG}/split_confounding_v2/split_audit.json}"
OUTPUT_DIR="${OUTPUT_DIR:-Training/reports/${DATASET_TAG}/thesis_v2_group_cv_seed42}"
PLAN="${PLAN:-${OUTPUT_DIR}/group_cv_plan.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$SPLIT_AUDIT" ]] || { echo "Split audit missing: $SPLIT_AUDIT" >&2; exit 2; }
[[ ! -e "$PLAN" ]] || { echo "Group-CV plan already exists; refusing to overwrite: $PLAN" >&2; exit 2; }

singularity exec "$IMAGE" python3 Training/prepare_group_cv_runs.py \
  --split-audit "$SPLIT_AUDIT" \
  --base-config Training/configs/models/residual_transformer_v2.json \
  --output-dir "$OUTPUT_DIR" \
  --dataset-tag "$DATASET_TAG" \
  --experiment-tag thesis_v2_group_cv_seed42 \
  --seeds 42

singularity exec "$IMAGE" python3 -c 'import hashlib,json,sys; p=json.load(open(sys.argv[1])); payload={**p,"plan_fingerprint":None}; fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest(); assert p["protocol"] == "nested_participant_group_cv_executable_v1" and p["plan_fingerprint"] == fp; assert p["fold_count"] == 25 and p["seeds"] == [42] and len(p["runs"]) == 25; assert p["outer_evaluation_used_for_selection"] is False' "$PLAN"
echo "Single-seed 25-fold LOPO plan complete: $PLAN"
