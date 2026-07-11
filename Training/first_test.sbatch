#!/bin/bash
#SBATCH --job-name=aria_gtn_test
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=aria_gtn_test.%J.out
#SBATCH --error=aria_gtn_test.%J.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
CONFIG="${CONFIG:-Training/configs/participant_split_v1.json}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/train.py --config "$CONFIG"
