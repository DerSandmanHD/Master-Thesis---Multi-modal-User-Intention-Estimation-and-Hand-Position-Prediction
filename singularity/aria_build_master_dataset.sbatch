#!/bin/bash
#SBATCH --job-name=aria_master_ds
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=Training/slurm_logs/aria_master_ds.%j.out
#SBATCH --error=Training/slurm_logs/aria_master_ds.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"

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

# Causal alignment is a schema migration. The active thesis-v2 job rebuilds
# all selected derived masters by default, but first snapshots the previous
# derived directory. Raw VRS/MPS inputs are never modified.
OVERWRITE="${OVERWRITE:-1}"
ALLOW_UNREVIEWED_TIMESTAMPS="${ALLOW_UNREVIEWED_TIMESTAMPS:-0}"

# Optional debugging controls:
#   LIMIT=5 sbatch ...
#   DRY_RUN=1 sbatch ...
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$MASTER_DIR" "$SCRATCH_DIR" "$REPO_DIR/Training/slurm_logs"

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
echo "Allow unreviewed timestamps: $ALLOW_UNREVIEWED_TIMESTAMPS"
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
    if [ "$ALLOW_UNREVIEWED_TIMESTAMPS" != "1" ]; then
        echo "ERROR: Reviewed timestamps are required for the active thesis pipeline: $TIMESTAMPS" >&2
        echo "Legacy diagnostics may opt in with ALLOW_UNREVIEWED_TIMESTAMPS=1." >&2
        exit 2
    fi
    TIMESTAMPS_TO_USE="$FALLBACK_TIMESTAMPS"
    echo "WARNING: Explicit legacy diagnostic uses unreviewed timestamps: $TIMESTAMPS_TO_USE"
else
    echo "ERROR: No timestamp summary found." >&2
    exit 2
fi

cd "$REPO_DIR"

if [ "$DRY_RUN" != "1" ] && [ "$OVERWRITE" = "1" ] && find "$MASTER_DIR" -maxdepth 1 -type f \( -name "*_master.csv" -o -name "*_master_report.json" \) -print -quit | grep -q .; then
    HISTORY_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    HISTORY_DIR="$DATA_DIR/master_datasets_history/${HISTORY_STAMP}_pre_causal_rebuild_job${SLURM_JOB_ID}"
    mkdir -p "$HISTORY_DIR"
    find "$MASTER_DIR" -maxdepth 1 -type f \( -name "*_master.csv" -o -name "*_master_report.json" -o -name "master_batch_report.json" \) -exec cp -p {} "$HISTORY_DIR" \;
    echo "Preserved previous derived masters in: $HISTORY_DIR"
fi

TIMESTAMPS_SHA256=$(singularity exec "$IMAGE" python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$TIMESTAMPS_TO_USE")
echo "Timestamp source used: $TIMESTAMPS_TO_USE"
echo "Timestamp source SHA-256: $TIMESTAMPS_SHA256"

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

if [ "$ALLOW_UNREVIEWED_TIMESTAMPS" = "1" ]; then
    BUILD_ARGS+=(--allow-unreviewed-timestamps)
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
echo "============================================================"
echo "6) Verify causal master-source joins"
echo "============================================================"

CAUSAL_REPORT="$MASTER_DIR/causal_master_preflight.job${SLURM_JOB_ID}.json"
singularity exec "$IMAGE" python3 Training/verify_causal_masters.py \
    --master-dir "$MASTER_DIR" \
    --manifest "$MANIFEST" \
    --expected-alignment-version causal_backward_device_time_v1 \
    --output "$CAUSAL_REPORT"

echo
echo "Finished: $(date)"

if [ "$build_exit" -ne 0 ]; then
    echo "ERROR: Master dataset build finished with errors. See $BATCH_REPORT" >&2
    exit "$build_exit"
fi

echo "Master dataset batch job done."
