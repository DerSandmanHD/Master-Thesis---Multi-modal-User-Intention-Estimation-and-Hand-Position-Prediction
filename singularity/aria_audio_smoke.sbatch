#!/bin/bash
#SBATCH --job-name=aria_audio_smoke
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=aria_audio_smoke.%J.out
#SBATCH --error=aria_audio_smoke.%J.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE="$HOME/singularity/aria_master.simg"

DATA_DIR="$REPO_DIR/Data_collection"
VRS_SRC="$DATA_DIR/Data_vrs"
MANIFEST="$DATA_DIR/dataset_manifest.csv"
SCRATCH_DIR="/scratch/$SLURM_JOB_ID"
TEST_VRS_DIR="$SCRATCH_DIR/audio_problem_vrs"
RESULT_DIR="$DATA_DIR/audio_smoke_results/job_$SLURM_JOB_ID"

LIMIT="${LIMIT:-10}"
MODEL_SIZE="${MODEL_SIZE:-medium.en}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"

export HF_HOME="$SCRATCH_DIR/huggingface"
export YOLO_CONFIG_DIR="$SCRATCH_DIR/ultralytics"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$TEST_VRS_DIR" "$HF_HOME" "$YOLO_CONFIG_DIR" "$RESULT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Submit dir: $SUBMIT_DIR"
echo "Repo dir: $REPO_DIR"
echo "Image: $IMAGE"
echo "VRS source: $VRS_SRC"
echo "Manifest: $MANIFEST"
echo "Test VRS dir: $TEST_VRS_DIR"
echo "Result dir: $RESULT_DIR"
echo "Limit: $LIMIT"
echo "Model: $MODEL_SIZE"
echo "Device: $DEVICE"
echo "Compute type: $COMPUTE_TYPE"
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

cd "$REPO_DIR"

echo "============================================================"
echo "1) Select problematic VRS files from dataset_manifest.csv"
echo "============================================================"

singularity exec --nv "$IMAGE" python3 - <<PY
from pathlib import Path
import pandas as pd

manifest_path = Path("$MANIFEST")
vrs_dir = Path("$VRS_SRC")
out_list = Path("$SCRATCH_DIR") / "problem_vrs_list.txt"
limit = int("$LIMIT")

if not manifest_path.exists():
    print(f"Manifest not found: {manifest_path}")
    print("Fallback: selecting first VRS files.")
    files = sorted(vrs_dir.glob("*.vrs"))[:limit]
else:
    df = pd.read_csv(manifest_path)
    print(f"Manifest rows: {len(df)}")

    required = {"sequence_id", "status", "next_action", "issues", "missing_commands"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        print(f"WARNING: manifest is missing expected columns: {missing_cols}")

    text = df.astype(str)
    empty_col = pd.Series("", index=df.index)
    next_action = text["next_action"] if "next_action" in text.columns else empty_col
    status = text["status"] if "status" in text.columns else empty_col
    issues = text["issues"] if "issues" in text.columns else empty_col
    missing_commands = text["missing_commands"] if "missing_commands" in text.columns else empty_col

    mask = (
        next_action.str.contains("fix_timestamps", case=False, na=False)
        | status.str.contains("partial_timestamps|missing_timestamps|bad_timestamp_order", case=False, regex=True, na=False)
        | issues.str.contains("partial_timestamps|missing_timestamps|bad_timestamp_order", case=False, regex=True, na=False)
        | missing_commands.str.len().gt(0)
    )

    candidates = df[mask].copy()
    if "sequence_id" not in candidates.columns or candidates.empty:
        print("No timestamp-problem rows found. Fallback: selecting first VRS files.")
        files = sorted(vrs_dir.glob("*.vrs"))[:limit]
    else:
        files = []
        for sequence_id in candidates["sequence_id"].astype(str):
            path = vrs_dir / f"{sequence_id}.vrs"
            if path.exists():
                files.append(path)
            else:
                print(f"WARNING: selected in manifest but VRS not found: {path}")
            if len(files) >= limit:
                break
        if not files:
            print("No selected files exist on disk. Fallback: selecting first VRS files.")
            files = sorted(vrs_dir.glob("*.vrs"))[:limit]

out_list.write_text("\n".join(str(path) for path in files) + "\n", encoding="utf-8")

print()
print("Selected files:")
for path in files:
    print(f"  {path.name}")
print()
print(f"Wrote list to: {out_list}")
print(f"Number selected: {len(files)}")
PY

echo
echo "============================================================"
echo "2) Copy selected VRS files to scratch"
echo "============================================================"

while IFS= read -r vrs_file; do
    if [ -n "$vrs_file" ]; then
        echo "Copying: $vrs_file"
        cp "$vrs_file" "$TEST_VRS_DIR/"
    fi
done < "$SCRATCH_DIR/problem_vrs_list.txt"

echo
echo "VRS files copied to scratch:"
find "$TEST_VRS_DIR" -maxdepth 1 -type f -name "*.vrs" -print | sort
echo
echo "Count:"
find "$TEST_VRS_DIR" -maxdepth 1 -type f -name "*.vrs" | wc -l

echo
echo "============================================================"
echo "3) Run speech recognition on selected difficult files"
echo "============================================================"

singularity exec --nv "$IMAGE" python3 Code/speech_recognition_demo.py \
    --dir "$TEST_VRS_DIR" \
    --model-size "$MODEL_SIZE" \
    --device "$DEVICE" \
    --compute-type "$COMPUTE_TYPE" \
    --keep-debug-audio

echo
echo "============================================================"
echo "4) Copy recognition results back to repo"
echo "============================================================"

cp "$TEST_VRS_DIR/timestamps_summary.json" "$RESULT_DIR/" || true
cp "$TEST_VRS_DIR/timestamps_debug.json" "$RESULT_DIR/" || true
cp "$SCRATCH_DIR/problem_vrs_list.txt" "$RESULT_DIR/"

if [ -d "$TEST_VRS_DIR/debug_audio" ]; then
    cp -R "$TEST_VRS_DIR/debug_audio" "$RESULT_DIR/"
fi

echo "Results copied to:"
echo "$RESULT_DIR"

echo
echo "============================================================"
echo "5) Quick result overview"
echo "============================================================"

singularity exec --nv "$IMAGE" python3 - <<PY
from pathlib import Path
import json

result_dir = Path("$RESULT_DIR")
summary_path = result_dir / "timestamps_summary.json"
debug_path = result_dir / "timestamps_debug.json"
expected = ["START", "SECOND", "DONE", "THIRD"]

if not summary_path.exists():
    print("No timestamps_summary.json found.")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
debug = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else {}

complete = 0
partial = 0
empty = 0

print("Per-file overview:")
for filename, commands in sorted(summary.items()):
    found = [cmd for cmd in expected if cmd in commands]
    missing = [cmd for cmd in expected if cmd not in commands]
    warnings = debug.get(filename, {}).get("warnings", [])

    if len(found) == 4:
        complete += 1
        status = "complete"
    elif len(found) == 0:
        empty += 1
        status = "empty"
    else:
        partial += 1
        status = "partial"

    print(f"  {filename}: {status}, found={found}, missing={missing}, warnings={warnings}")

print()
print("Summary:")
print(f"  complete: {complete}")
print(f"  partial: {partial}")
print(f"  empty: {empty}")
print(f"  total: {len(summary)}")
PY

echo
echo "Finished: $(date)"
echo "Audio smoke test done."
