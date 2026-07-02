#!/bin/bash
#SBATCH --job-name=aria_code_check
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=aria_code_check.%J.out
#SBATCH --error=aria_code_check.%J.err

set -euo pipefail

REPO_DIR="/home/eker/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="/home/eker/singularity/aria_master.simg"

cd "$REPO_DIR"

export HF_HOME="/scratch/$SLURM_JOB_ID/huggingface"
export YOLO_CONFIG_DIR="/scratch/$SLURM_JOB_ID/ultralytics"
mkdir -p "$HF_HOME" "$YOLO_CONFIG_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repo dir: $(pwd)"
echo "Image: $IMAGE"
echo "Started: $(date)"
echo

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Singularity image not found: $IMAGE" >&2
    exit 2
fi

if [ ! -d "$REPO_DIR/Code" ]; then
    echo "ERROR: Code directory not found: $REPO_DIR/Code" >&2
    exit 2
fi

echo "============================================================"
echo "1) Container import check"
echo "============================================================"

singularity exec --nv "$IMAGE" python3 - <<'PY'
import cv2
import numpy as np
import pandas as pd
import scipy
import torch

print("Basic imports: OK")
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("opencv aruco available:", hasattr(cv2, "aruco"))

try:
    import faster_whisper
    print("faster_whisper: OK")
except Exception as e:
    print("faster_whisper: FAILED", repr(e))
    raise

try:
    import ultralytics
    print("ultralytics: OK")
except Exception as e:
    print("ultralytics: FAILED", repr(e))
    raise

try:
    from projectaria_tools.core import data_provider
    print("projectaria_tools: OK")
except Exception as e:
    print("projectaria_tools: FAILED", repr(e))
    raise

print("Container import check passed")
PY

echo
echo "============================================================"
echo "2) Python syntax compile check for Code/*.py"
echo "   Excluding Code/Testing and __pycache__"
echo "============================================================"

singularity exec --nv "$IMAGE" bash -lc '
set -euo pipefail
cd "/home/eker/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"

for file in Code/*.py; do
    [ -f "$file" ] || continue
    echo "Checking syntax: $file"
    python3 -m py_compile "$file"
done

echo "All top-level Code/*.py files compiled successfully."
'

echo
echo "============================================================"
echo "3) List checked files"
echo "============================================================"

find Code -maxdepth 1 -type f -name "*.py" -print | sort

echo
echo "CODE CHECK PASSED"
echo "Finished: $(date)"
