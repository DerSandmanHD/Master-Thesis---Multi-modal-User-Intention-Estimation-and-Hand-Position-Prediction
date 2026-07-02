#!/bin/bash
#SBATCH --job-name=aria_tags_all
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=aria_tags_all.%J.out
#SBATCH --error=aria_tags_all.%J.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"

DATA_DIR="$REPO_DIR/Data_collection"
VRS_DIR="$DATA_DIR/Data_vrs"
ARUCO_OUT_DIR="$DATA_DIR/Aruco_CSV"
RESULT_BACKUP_DIR="$DATA_DIR/aruco_batch_backups/job_$SLURM_JOB_ID"

# Default resume mode: keep valid outputs and process only missing/invalid CSVs.
# Set OVERWRITE=1 only when every sequence must intentionally be recomputed.
OVERWRITE="${OVERWRITE:-0}"
MAX_FRAMES="${MAX_FRAMES:-}"
WRITE_VIDEO="${WRITE_VIDEO:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"

SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
export YOLO_CONFIG_DIR="$SCRATCH_DIR/ultralytics"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$YOLO_CONFIG_DIR" "$ARUCO_OUT_DIR" "$RESULT_BACKUP_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Image: $IMAGE"
echo "VRS dir: $VRS_DIR"
echo "ArUco output dir: $ARUCO_OUT_DIR"
echo "Backup dir: $RESULT_BACKUP_DIR"
echo "Overwrite all: $OVERWRITE"
echo "Max frames: ${MAX_FRAMES:-all}"
echo "Write annotated videos: $WRITE_VIDEO"
echo "Progress every: $PROGRESS_EVERY"
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

is_valid_marker_csv() {
    local csv_path="$1"
    if [ ! -s "$csv_path" ]; then
        return 1
    fi

    local header
    IFS= read -r header < "$csv_path" || return 1
    case "$header" in
        sequence_id,frame_index,timestamp_ns,marker_family,marker_id,*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

echo "============================================================"
echo "1) Input and resume overview"
echo "============================================================"

vrs_count=$(find "$VRS_DIR" -maxdepth 1 -type f -name "*.vrs" | wc -l)
csv_count=$(find "$ARUCO_OUT_DIR" -maxdepth 1 -type f -name "aruco_poses_*.csv" | wc -l)

echo "VRS count: $vrs_count"
echo "Existing CSV count in Aruco_CSV: $csv_count"
echo

missing_before=0
invalid_before=0
valid_before=0

for vrs in "$VRS_DIR"/*.vrs; do
    [ -f "$vrs" ] || continue
    sequence_id="$(basename "$vrs" .vrs)"
    output_csv="$ARUCO_OUT_DIR/aruco_poses_${sequence_id}.csv"

    if is_valid_marker_csv "$output_csv"; then
        valid_before=$((valid_before + 1))
    elif [ -e "$output_csv" ]; then
        echo "INVALID existing CSV will be recomputed: $output_csv"
        invalid_before=$((invalid_before + 1))
    else
        echo "MISSING: $sequence_id"
        missing_before=$((missing_before + 1))
    fi
done

echo
echo "Valid before run:   $valid_before"
echo "Missing before run: $missing_before"
echo "Invalid before run: $invalid_before"

echo
echo "Backing up existing completed CSV files..."
find "$ARUCO_OUT_DIR" -maxdepth 1 -type f -name "aruco_poses_*.csv" \
    -exec cp {} "$RESULT_BACKUP_DIR/" \; 2>/dev/null || true

echo
echo "============================================================"
echo "2) Run calibrated marker extraction"
echo "============================================================"

count=0
processed=0
skipped=0
failed=0
invalid_reprocessed=0

for vrs in "$VRS_DIR"/*.vrs; do
    [ -f "$vrs" ] || continue

    count=$((count + 1))
    sequence_id="$(basename "$vrs" .vrs)"

    output_csv="$ARUCO_OUT_DIR/aruco_poses_${sequence_id}.csv"
    qa_link_csv="$DATA_DIR/aruco_poses_${sequence_id}.csv"
    output_video="$ARUCO_OUT_DIR/aruco_${sequence_id}.mp4"

    echo
    echo "------------------------------------------------------------"
    echo "[$count/$vrs_count] Sequence: $sequence_id"
    echo "VRS:        $vrs"
    echo "Output CSV: $output_csv"
    echo "QA symlink: $qa_link_csv"

    if [ "$WRITE_VIDEO" = "1" ]; then
        echo "Output MP4: $output_video"
    fi

    if [ "$OVERWRITE" != "1" ] && is_valid_marker_csv "$output_csv"; then
        echo "SKIP: valid output already exists."
        ln -sfn "Aruco_CSV/aruco_poses_${sequence_id}.csv" "$qa_link_csv"
        skipped=$((skipped + 1))
        continue
    fi

    EXTRA_ARGS=(
        --input-vrs "$vrs"
        --output-csv "$output_csv"
        --progress-every "$PROGRESS_EVERY"
    )

    if [ "$OVERWRITE" = "1" ]; then
        EXTRA_ARGS+=(--overwrite)
    elif [ -e "$output_csv" ]; then
        # Only this malformed/incomplete final output is replaced. Valid CSVs
        # were already skipped above.
        echo "Recomputing invalid existing output: $output_csv"
        cp "$output_csv" "$RESULT_BACKUP_DIR/$(basename "$output_csv").invalid_before_resume" \
            2>/dev/null || true
        EXTRA_ARGS+=(--overwrite)
        invalid_reprocessed=$((invalid_reprocessed + 1))
    fi

    if [ -n "$MAX_FRAMES" ]; then
        EXTRA_ARGS+=(--max-frames "$MAX_FRAMES")
    fi

    if [ "$WRITE_VIDEO" = "1" ]; then
        EXTRA_ARGS+=(--output-video "$output_video")
    fi

    set +e
    singularity exec --nv "$IMAGE" python3 Code/detect_tags.py "${EXTRA_ARGS[@]}"
    exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ] && is_valid_marker_csv "$output_csv"; then
        echo "Finished and validated: $sequence_id"
        ln -sfn "Aruco_CSV/aruco_poses_${sequence_id}.csv" "$qa_link_csv"
        processed=$((processed + 1))
    else
        echo "ERROR: marker extraction failed or produced an invalid CSV for $sequence_id" >&2
        echo "Exit code: $exit_code" >&2
        failed=$((failed + 1))
    fi
done

echo
echo "Copying final CSV outputs to the job backup directory..."
find "$ARUCO_OUT_DIR" -maxdepth 1 -type f -name "aruco_poses_*.csv" \
    -exec cp {} "$RESULT_BACKUP_DIR/" \; 2>/dev/null || true

echo
echo "============================================================"
echo "3) Validate completeness before dataset QA"
echo "============================================================"

valid_after=0
missing_after=0
invalid_after=0

for vrs in "$VRS_DIR"/*.vrs; do
    [ -f "$vrs" ] || continue
    sequence_id="$(basename "$vrs" .vrs)"
    output_csv="$ARUCO_OUT_DIR/aruco_poses_${sequence_id}.csv"

    if is_valid_marker_csv "$output_csv"; then
        valid_after=$((valid_after + 1))
        ln -sfn "Aruco_CSV/aruco_poses_${sequence_id}.csv" \
            "$DATA_DIR/aruco_poses_${sequence_id}.csv"
    elif [ -e "$output_csv" ]; then
        echo "INVALID AFTER RUN: $sequence_id" >&2
        invalid_after=$((invalid_after + 1))
    else
        echo "MISSING AFTER RUN: $sequence_id" >&2
        missing_after=$((missing_after + 1))
    fi
done

echo "Valid after run:   $valid_after/$vrs_count"
echo "Missing after run: $missing_after"
echo "Invalid after run: $invalid_after"

echo
echo "============================================================"
echo "4) Run dataset QA"
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
echo "5) QA summary"
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
echo "============================================================"
echo "6) Marker extraction summary"
echo "============================================================"
echo "VRS files seen:          $count"
echo "Processed now:           $processed"
echo "Skipped valid existing:  $skipped"
echo "Invalid reprocessed:     $invalid_reprocessed"
echo "Extraction failures:     $failed"
echo "Valid final CSVs:        $valid_after/$vrs_count"
echo "Missing final CSVs:      $missing_after"
echo "Invalid final CSVs:      $invalid_after"
echo "QA symlinks:             $(find "$DATA_DIR" -maxdepth 1 -type l -name 'aruco_poses_*.csv' | wc -l)"
echo "Finished: $(date)"

if [ "$failed" -gt 0 ] || [ "$missing_after" -gt 0 ] || [ "$invalid_after" -gt 0 ]; then
    echo "ERROR: Marker extraction is still incomplete. Check the SLURM logs." >&2
    exit 1
fi

echo "Full marker extraction and QA completed successfully."
