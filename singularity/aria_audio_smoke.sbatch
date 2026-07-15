#!/bin/bash
#SBATCH --job-name=aria_audio_all
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=aria_audio_all.%J.out
#SBATCH --error=aria_audio_all.%J.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"

DATA_DIR="$REPO_DIR/Data_collection"
VRS_DIR="$DATA_DIR/Data_vrs"
RESULT_BACKUP_DIR="$DATA_DIR/audio_batch_backups/job_$SLURM_JOB_ID"

MODEL_SIZE="${MODEL_SIZE:-medium.en}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_TYPE="${COMPUTE_TYPE:-int8}"
KEEP_DEBUG_AUDIO="${KEEP_DEBUG_AUDIO:-0}"

SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
export HF_HOME="$SCRATCH_DIR/huggingface"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$HF_HOME" "$RESULT_BACKUP_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Image: $IMAGE"
echo "VRS dir: $VRS_DIR"
echo "Backup dir: $RESULT_BACKUP_DIR"
echo "Model: $MODEL_SIZE"
echo "Device: $DEVICE"
echo "Compute type: $COMPUTE_TYPE"
echo "Keep debug audio: $KEEP_DEBUG_AUDIO"
echo "Started: $(date)"
echo

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Singularity image not found: $IMAGE" >&2
    exit 2
fi

if [ ! -d "$REPO_DIR/Code" ]; then
    echo "ERROR: Repo dir does not look correct: $REPO_DIR" >&2
    exit 2
fi

if [ ! -d "$VRS_DIR" ]; then
    echo "ERROR: VRS directory not found: $VRS_DIR" >&2
    exit 2
fi

cd "$REPO_DIR"

echo "============================================================"
echo "1) Input overview"
echo "============================================================"
echo "VRS count:"
find "$VRS_DIR" -maxdepth 1 -type f -name "*.vrs" | wc -l
echo

echo "Backing up existing timestamp outputs if present..."
cp "$VRS_DIR/timestamps_summary.json" "$RESULT_BACKUP_DIR/timestamps_summary.before_job_$SLURM_JOB_ID.json" 2>/dev/null || true
cp "$VRS_DIR/timestamps_debug.json" "$RESULT_BACKUP_DIR/timestamps_debug.before_job_$SLURM_JOB_ID.json" 2>/dev/null || true

echo
echo "============================================================"
echo "2) Run speech recognition for all VRS files"
echo "============================================================"

EXTRA_ARGS=()
if [ "$KEEP_DEBUG_AUDIO" = "1" ]; then
    EXTRA_ARGS+=(--keep-debug-audio)
fi

singularity exec --nv "$IMAGE" python3 Code/speech_recognition_demo.py \
    --dir "$VRS_DIR" \
    --model-size "$MODEL_SIZE" \
    --device "$DEVICE" \
    --compute-type "$COMPUTE_TYPE" \
    "${EXTRA_ARGS[@]}"

echo
echo "Copying new timestamp outputs to backup directory..."
cp "$VRS_DIR/timestamps_summary.json" "$RESULT_BACKUP_DIR/" || true
cp "$VRS_DIR/timestamps_debug.json" "$RESULT_BACKUP_DIR/" || true

echo
echo "============================================================"
echo "3) Run dataset QA"
echo "============================================================"

if [ -d "$REPO_DIR/BackUp_Videos" ]; then
    BACKUP_DIR="$REPO_DIR/BackUp_Videos"
elif [ -d "$REPO_DIR/Data_collection_backup" ]; then
    BACKUP_DIR="$REPO_DIR/Data_collection_backup"
else
    BACKUP_DIR="$REPO_DIR/BackUp_Videos"
fi

echo "Using backup dir for QA: $BACKUP_DIR"

singularity exec --nv "$IMAGE" python3 Code/dataset_qa.py \
    --data-root "$DATA_DIR" \
    --backup-dir "$BACKUP_DIR"

echo
echo "============================================================"
echo "4) QA summary"
echo "============================================================"

singularity exec --nv "$IMAGE" python3 - <<PY
from pathlib import Path
import json

report_path = Path("$DATA_DIR") / "dataset_qa_report.json"
manifest_path = Path("$DATA_DIR") / "dataset_manifest.csv"

print(f"Manifest: {manifest_path}")
print(f"Report:   {report_path}")

if not report_path.exists():
    print("No QA report found.")
    raise SystemExit(0)

report = json.loads(report_path.read_text(encoding="utf-8"))

print()
print("Dataset QA:")
print(f"  total_sequences: {report.get('total_sequences')}")
print(f"  training_include_candidates: {report.get('training_include_candidates')}")
print(f"  training_excluded_candidates: {report.get('training_excluded_candidates')}")
print(f"  valid_or_warning_sequences: {report.get('valid_or_warning_sequences')}")

print()
print("Status counts:")
for key, value in sorted(report.get("status_counts", {}).items()):
    print(f"  {key}: {value}")

print()
print("Next actions:")
for key, value in sorted(report.get("next_action_counts", {}).items()):
    print(f"  {key}: {value}")
PY

echo
echo "Finished: $(date)"
echo "Full audio batch and QA done."
