#!/bin/bash
#SBATCH --job-name=aria_master_ds
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=aria_master_ds.%J.out
#SBATCH --error=aria_master_ds.%J.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"

DATA_DIR="$REPO_DIR/Data_collection"
MASTER_DIR="$DATA_DIR/master_datasets"
TIMESTAMPS="$DATA_DIR/Data_vrs/timestamps_summary.reviewed.json"
FALLBACK_TIMESTAMPS="$DATA_DIR/Data_vrs/timestamps_summary.json"
ANNOTATIONS="$DATA_DIR/manual_timestamp_review.csv"
MANIFEST="$DATA_DIR/dataset_manifest.csv"
BATCH_REPORT="$MASTER_DIR/master_batch_report.json"

# Default: build only sequences with target_object_id and receiving_hand,
# because those are the useful training candidates.
REQUIRE_SEMANTIC_ANNOTATIONS="${REQUIRE_SEMANTIC_ANNOTATIONS:-1}"

# Resume mode by default: existing *_master.csv + *_master_report.json are skipped.
# Set OVERWRITE=1 only when intentionally rebuilding existing master datasets.
OVERWRITE="${OVERWRITE:-0}"

# Optional debugging controls:
#   LIMIT=5 sbatch ...
#   DRY_RUN=1 sbatch ...
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$MASTER_DIR" "$SCRATCH_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Image: $IMAGE"
echo "Data dir: $DATA_DIR"
echo "Master output dir: $MASTER_DIR"
echo "Manifest: $MANIFEST"
echo "Annotations: $ANNOTATIONS"
echo "Reviewed timestamps: $TIMESTAMPS"
echo "Require semantic annotations: $REQUIRE_SEMANTIC_ANNOTATIONS"
echo "Overwrite: $OVERWRITE"
echo "Limit: ${LIMIT:-all}"
echo "Dry run only: $DRY_RUN"
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

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR" >&2
    exit 2
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Dataset manifest not found: $MANIFEST" >&2
    echo "Run dataset_qa.py first." >&2
    exit 2
fi

if [ ! -f "$ANNOTATIONS" ]; then
    echo "ERROR: Manual annotation/review CSV not found: $ANNOTATIONS" >&2
    exit 2
fi

if [ -f "$TIMESTAMPS" ]; then
    TIMESTAMPS_TO_USE="$TIMESTAMPS"
elif [ -f "$FALLBACK_TIMESTAMPS" ]; then
    TIMESTAMPS_TO_USE="$FALLBACK_TIMESTAMPS"
    echo "WARNING: Reviewed timestamps not found. Falling back to: $TIMESTAMPS_TO_USE"
else
    echo "ERROR: No timestamp summary found." >&2
    exit 2
fi

cd "$REPO_DIR"

echo "============================================================"
echo "1) Input overview"
echo "============================================================"

echo "Existing master CSV files:"
find "$MASTER_DIR" -maxdepth 1 -type f -name "*_master.csv" 2>/dev/null | wc -l

echo
echo "Current QA next actions:"
python3 - <<PY
import json
from pathlib import Path

report_path = Path("$DATA_DIR") / "dataset_qa_report.json"
if not report_path.exists():
    print("No dataset_qa_report.json found.")
    raise SystemExit(0)

report = json.loads(report_path.read_text(encoding="utf-8"))
for key, value in sorted(report.get("next_action_counts", {}).items()):
    print(f"  {key}: {value}")
PY

echo
echo "============================================================"
echo "2) Dry-run build plan"
echo "============================================================"

BUILD_ARGS=(
    --data-root "$DATA_DIR"
    --manifest "$MANIFEST"
    --timestamps "$TIMESTAMPS_TO_USE"
    --annotations "$ANNOTATIONS"
    --output-dir "$MASTER_DIR"
    --batch-report "$BATCH_REPORT"
)

if [ "$REQUIRE_SEMANTIC_ANNOTATIONS" = "1" ]; then
    BUILD_ARGS+=(--require-semantic-annotations)
fi

if [ "$OVERWRITE" = "1" ]; then
    BUILD_ARGS+=(--overwrite)
fi

if [ -n "$LIMIT" ]; then
    BUILD_ARGS+=(--limit "$LIMIT")
fi

singularity exec "$IMAGE" python3 Code/build_master_dataset_batch.py \
    "${BUILD_ARGS[@]}" \
    --dry-run

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "DRY_RUN=1, stopping after dry-run plan."
    exit 0
fi

echo
echo "============================================================"
echo "3) Build missing master datasets"
echo "============================================================"

set +e
singularity exec "$IMAGE" python3 Code/build_master_dataset_batch.py "${BUILD_ARGS[@]}"
build_exit=$?
set -e

echo
echo "============================================================"
echo "4) Run dataset QA after master build"
echo "============================================================"

singularity exec "$IMAGE" python3 Code/dataset_qa.py \
    --data-root "$DATA_DIR" \
    --timestamps "$TIMESTAMPS_TO_USE"

echo
echo "============================================================"
echo "5) Build and QA summary"
echo "============================================================"

singularity exec "$IMAGE" python3 - <<PY
from pathlib import Path
import json

batch_report = Path("$BATCH_REPORT")
qa_report = Path("$DATA_DIR") / "dataset_qa_report.json"

print(f"Batch report: {batch_report}")
if batch_report.exists():
    report = json.loads(batch_report.read_text(encoding="utf-8"))
    print(f"  selected: {report.get('selected')}")
    print(f"  built: {report.get('built')}")
    print(f"  already_exists: {report.get('already_exists')}")
    print(f"  errors: {report.get('errors')}")
else:
    print("  missing")

print()
print(f"QA report: {qa_report}")
if qa_report.exists():
    qa = json.loads(qa_report.read_text(encoding="utf-8"))
    print(f"  master_datasets.csv_available: {qa.get('master_datasets', {}).get('csv_available')}")
    print("  next_actions:")
    for key, value in sorted(qa.get("next_action_counts", {}).items()):
        print(f"    {key}: {value}")
else:
    print("  missing")
PY

echo
echo "Finished: $(date)"

if [ "$build_exit" -ne 0 ]; then
    echo "ERROR: Master dataset build finished with errors. See $BATCH_REPORT" >&2
    exit "$build_exit"
fi

echo "Master dataset batch job done."
