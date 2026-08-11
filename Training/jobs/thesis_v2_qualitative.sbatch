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
: "${PREDICTIONS:?Set PREDICTIONS to the selected checkpoint CSV}"
: "${PREDICTION_REPORT:?Set PREDICTION_REPORT to its checkpoint-bound JSON}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new qualitative report directory}"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
VISUAL_MANIFEST="${VISUAL_MANIFEST:-Data_collection/visual_embeddings/${DATASET_TAG}/clip_vit_b32_openai_5hz_device_time_v2/cache_manifest.json}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-Training/reports/${DATASET_TAG}/video_alignment_device_time_v2}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$VISUAL_MANIFEST" ]] || { echo "Corrected visual manifest missing: $VISUAL_MANIFEST" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/overlay_report.json" ]] || { echo "Qualitative report already exists" >&2; exit 2; }

singularity exec "$IMAGE" \
  python3 Training/build_video_alignment_sidecars.py \
  --visual-cache-manifest "$VISUAL_MANIFEST" \
  --vrs-dir Data_collection/Data_vrs \
  --video-dir Data_collection/Data_mp4 \
  --master-dir Data_collection/master_datasets \
  --output-dir "$ALIGNMENT_DIR"

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
