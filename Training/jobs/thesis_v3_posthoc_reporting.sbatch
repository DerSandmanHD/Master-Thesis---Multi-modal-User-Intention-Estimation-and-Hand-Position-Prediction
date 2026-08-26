#!/bin/bash
#SBATCH --job-name=thesis_v3_reporting
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/thesis_v3_reporting.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v3_reporting.%j.err

set -euo pipefail

# Reporting only: this job fits deterministic train-only logistic baselines and
# audits timestamps.  It never invokes a neural-model training entry point.
REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
MATRIX="${MATRIX:-Training/configs/experiment_matrix_v2.json}"
CONFIG="${CONFIG:-Training/configs/models/residual_transformer_v2.json}"
DATASET_DESCRIPTOR="${DATASET_DESCRIPTOR:-Training/datasets/${DATASET_TAG}.json}"
REPORT_ROOT="${REPORT_ROOT:-Training/reports/${DATASET_TAG}}"
PREDICTION_REPORT="${PREDICTION_REPORT:-${REPORT_ROOT}/thesis_final_v2_corrected_alignment/postprocess/residual_current_gate_seed42/test_predictions.json}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${REPORT_ROOT}/causal_intention_baselines_v1}"
SAMPLING_OUTPUT_DIR="${SAMPLING_OUTPUT_DIR:-${REPORT_ROOT}/sampling_window_audit_v1}"
BASELINE_MAX_ITERATIONS="${BASELINE_MAX_ITERATIONS:-2500}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
for path in "$MATRIX" "$CONFIG" "$DATASET_DESCRIPTOR" "$PREDICTION_REPORT"; do
  [[ -f "$path" ]] || { echo "Required input missing: $path" >&2; exit 2; }
done

singularity exec "$IMAGE" python3 -c '
import json, pathlib, sys
matrix, descriptor, prediction = (
    json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for path in sys.argv[1:4]
)
tag = sys.argv[4]
assert matrix["dataset_tag"] == descriptor["dataset_tag"] == tag
assert matrix["policy"]["selection_split"] == "validation"
assert matrix["postprocessing"]["required_t1_experiments"] == ["residual_current_gate"]
assert prediction["split"] == "test" and prediction["full_split_export"] is True
' "$MATRIX" "$DATASET_DESCRIPTOR" "$PREDICTION_REPORT" "$DATASET_TAG"

singularity exec "$IMAGE" python3 Training/evaluation/evaluate_causal_intention_baselines.py \
  --config "$CONFIG" \
  --dataset-descriptor "$DATASET_DESCRIPTOR" \
  --prediction-report "$PREDICTION_REPORT" \
  --max-iterations "$BASELINE_MAX_ITERATIONS" \
  --output-dir "$BASELINE_OUTPUT_DIR"

singularity exec "$IMAGE" python3 Training/evaluation/audit_sampling_window_duration.py \
  --config "$CONFIG" \
  --dataset-descriptor "$DATASET_DESCRIPTOR" \
  --output-dir "$SAMPLING_OUTPUT_DIR"

echo "Causal baseline report: $BASELINE_OUTPUT_DIR"
echo "Sampling/window audit: $SAMPLING_OUTPUT_DIR"
