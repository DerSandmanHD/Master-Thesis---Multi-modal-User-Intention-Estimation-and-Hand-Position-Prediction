#!/bin/bash
#SBATCH --job-name=aria_vrs_to_mp4
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00
#SBATCH --output=aria_vrs_to_mp4.%J.out
#SBATCH --error=aria_vrs_to_mp4.%J.err

set -euo pipefail

REPO_DIR="/home/eker/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="/home/eker/singularity/aria_master.simg"

VRS_SRC="$REPO_DIR/Data_collection/Data_vrs"
MP4_OUT="$REPO_DIR/Data_collection/Data_mp4"

SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
VRS_SCRATCH="$SCRATCH_DIR/Data_vrs"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Image: $IMAGE"
echo "VRS source: $VRS_SRC"
echo "MP4 output: $MP4_OUT"
echo "Scratch dir: $SCRATCH_DIR"
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

if [ ! -d "$VRS_SRC" ]; then
    echo "ERROR: VRS source directory not found: $VRS_SRC" >&2
    exit 2
fi

mkdir -p "$MP4_OUT"
mkdir -p "$VRS_SCRATCH"

echo "============================================================"
echo "1) Copy VRS files to local scratch"
echo "============================================================"

rsync -avh --include="*.vrs" --exclude="*" "$VRS_SRC/" "$VRS_SCRATCH/"

echo
echo "VRS files in scratch:"
find "$VRS_SCRATCH" -maxdepth 1 -type f -name "*.vrs" | wc -l

echo
echo "Existing MP4 files before conversion:"
find "$MP4_OUT" -maxdepth 1 -type f -name "*.mp4" | wc -l

echo
echo "============================================================"
echo "2) Dry-run conversion plan"
echo "============================================================"

cd "$REPO_DIR"

singularity exec "$IMAGE" python3 Code/vrs_to_mp4_all.py \
    --input-dir "$VRS_SCRATCH" \
    --output-dir "$MP4_OUT" \
    --dry-run

echo
echo "============================================================"
echo "3) Convert missing VRS recordings to MP4"
echo "============================================================"

singularity exec "$IMAGE" python3 Code/vrs_to_mp4_all.py \
    --input-dir "$VRS_SCRATCH" \
    --output-dir "$MP4_OUT"

echo
echo "============================================================"
echo "4) Result"
echo "============================================================"

echo "MP4 files after conversion:"
find "$MP4_OUT" -maxdepth 1 -type f -name "*.mp4" | wc -l

echo
echo "Finished: $(date)"
echo "VRS to MP4 job done."
