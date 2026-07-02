#!/bin/bash
#SBATCH --job-name=aria_smoke
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00
#SBATCH --output=aria_smoke.%J.out
#SBATCH --error=aria_smoke.%J.err

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

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Singularity image not found: $IMAGE" >&2
    exit 2
fi

if [ ! -d "$REPO_DIR/Code" ]; then
    echo "ERROR: Repo dir does not look correct: $REPO_DIR" >&2
    exit 2
fi

singularity exec --nv "$IMAGE" python3 - <<'PY'
import shutil
from importlib.metadata import version

import cv2
import faster_whisper
import numpy as np
import pandas as pd
import scipy
import torch
import ultralytics
from projectaria_tools.core import data_provider

print("Python imports: OK")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"pandas: {pd.__version__}")
print(f"torch: {torch.__version__}")
print(f"torch cuda runtime: {torch.version.cuda}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"opencv: {cv2.__version__}")
print(f"opencv aruco available: {hasattr(cv2, 'aruco')}")
print(f"faster-whisper: {version('faster-whisper')}")
print(f"projectaria-tools: {version('projectaria-tools')}")
print(f"ultralytics: {version('ultralytics')}")

ffmpeg = shutil.which("ffmpeg")
vrs_to_mp4 = shutil.which("vrs_to_mp4")

print(f"ffmpeg: {ffmpeg}")
print(f"vrs_to_mp4: {vrs_to_mp4}")

assert hasattr(cv2, "aruco"), "OpenCV aruco module is missing"
assert ffmpeg is not None, "ffmpeg is missing"
assert vrs_to_mp4 is not None, "vrs_to_mp4 is missing"

print("SMOKE TEST PASSED")
PY

echo "Finished: $(date)"
