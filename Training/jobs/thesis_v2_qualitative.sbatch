#!/bin/bash
#SBATCH --job-name=thesis_v2_qual
#SBATCH --partition=day
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=Training/slurm_logs/thesis_v2_qualitative.%j.out
#SBATCH --error=Training/slurm_logs/thesis_v2_qualitative.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
REPORT_TAG="${REPORT_TAG:-thesis_final_v2_corrected_alignment}"
REPORT_ROOT="${REPORT_ROOT:-Training/reports/${DATASET_TAG}/${REPORT_TAG}}"
SELECTION_FILE="${SELECTION_FILE:-${REPORT_ROOT}/validation_selection.json}"
POSTPROCESS_ROOT="${POSTPROCESS_ROOT:-${REPORT_ROOT}/postprocess}"
PREDICTIONS="${PREDICTIONS:-}"
PREDICTION_REPORT="${PREDICTION_REPORT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPORT_ROOT}/qualitative}"
VISUAL_MANIFEST="${VISUAL_MANIFEST:-Data_collection/visual_embeddings/${DATASET_TAG}/clip_vit_b32_openai_5hz_device_time_v2/cache_manifest.json}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-Training/reports/${DATASET_TAG}/video_alignment_device_time_v2}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$VISUAL_MANIFEST" ]] || { echo "Corrected visual manifest missing: $VISUAL_MANIFEST" >&2; exit 2; }
if [[ -z "$PREDICTIONS" && -z "$PREDICTION_REPORT" ]]; then
  [[ -f "$SELECTION_FILE" ]] || { echo "Selection missing: $SELECTION_FILE" >&2; exit 2; }
  REPRESENTATIVE_SEED=$(singularity exec "$IMAGE" python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); rows=[r for r in s["representative_by_experiment"] if r["experiment_id"] == "residual_current_gate"]; assert s["complete"] is True and s["selection_split"] == "validation" and s["test_metrics_read"] is False and len(rows) == 1; print(int(rows[0]["seed"]))' "$SELECTION_FILE")
  PREDICTIONS="${POSTPROCESS_ROOT}/residual_current_gate_seed${REPRESENTATIVE_SEED}/test_predictions.csv"
  PREDICTION_REPORT="${POSTPROCESS_ROOT}/residual_current_gate_seed${REPRESENTATIVE_SEED}/test_predictions.json"
elif [[ -z "$PREDICTIONS" || -z "$PREDICTION_REPORT" ]]; then
  echo "Set both PREDICTIONS and PREDICTION_REPORT, or neither." >&2
  exit 2
fi
[[ -f "$PREDICTIONS" ]] || { echo "Predictions missing: $PREDICTIONS" >&2; exit 2; }
[[ -f "$PREDICTION_REPORT" ]] || { echo "Prediction report missing: $PREDICTION_REPORT" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/overlay_report.json" ]] || { echo "Qualitative report already exists" >&2; exit 2; }

mapfile -t SELECTED_SEQUENCES < <(
  singularity exec "$IMAGE" python3 -c 'import pandas as pd,sys; sys.path.insert(0,"Training"); from render_prediction_overlay import choose_qualitative_cases,choose_sequences_from_cases,validate_qualitative_columns; frame=pd.read_csv(sys.argv[1]); validate_qualitative_columns(frame); cases=choose_qualitative_cases(frame,seed=42); selected,_=choose_sequences_from_cases(cases,frame,3); print("\n".join(selected))' "$PREDICTIONS"
)
[[ "${#SELECTED_SEQUENCES[@]}" -gt 0 ]] || { echo "No qualitative sequences selected" >&2; exit 2; }
SIDECAR_SEQUENCE_ARGS=()
FRAME_EXCLUSION_ARGS=()
for sequence_id in "${SELECTED_SEQUENCES[@]}"; do
  SIDECAR_SEQUENCE_ARGS+=(--sequence "$sequence_id")
  if [[ "$sequence_id" == "Jona_7_20260616_182214" ]]; then
    # This reviewed MP4 ends one frame before the source VRS RGB stream.
    # Keep the exception explicit and sidecar-hash-bound; never truncate frames
    # generically or apply it if this sequence was not selected.
    FRAME_EXCLUSION_ARGS+=(
      --exclude-vrs-rgb-frame Jona_7_20260616_182214:1165
      --frame-exclusion-reason "Reviewed MP4 ends one RGB frame before the VRS stream; the unmatched final VRS record is excluded explicitly."
    )
  fi
done

singularity exec "$IMAGE" \
  python3 Training/build_video_alignment_sidecars.py \
  --visual-cache-manifest "$VISUAL_MANIFEST" \
  --vrs-dir Data_collection/Data_vrs \
  --video-dir Data_collection/Data_mp4 \
  --master-dir Data_collection/master_datasets \
  --output-dir "$ALIGNMENT_DIR" \
  "${FRAME_EXCLUSION_ARGS[@]}" \
  "${SIDECAR_SEQUENCE_ARGS[@]}"

singularity exec "$IMAGE" \
  python3 Training/render_prediction_overlay.py \
  --predictions "$PREDICTIONS" \
  --prediction-report "$PREDICTION_REPORT" \
  --visual-cache-manifest "$VISUAL_MANIFEST" \
  --alignment-dir "$ALIGNMENT_DIR" \
  --video-dir Data_collection/Data_mp4 \
  --vrs-dir Data_collection/Data_vrs \
  --master-dir Data_collection/master_datasets \
  --output-dir "$OUTPUT_DIR" \
  --count 3 --selection-seed 42

echo "Qualitative good/typical/failure cases complete: $OUTPUT_DIR"
