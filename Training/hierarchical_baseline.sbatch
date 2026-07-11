#!/bin/bash
#SBATCH --job-name=aria_hier_base
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=aria_hier_base.%J.out
#SBATCH --error=aria_hier_base.%J.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
CONFIG="${CONFIG:-Training/configs/hierarchical_baseline_v1.json}"

cd "$REPO_DIR"
singularity exec --nv "$IMAGE" \
    python3 Training/train.py --config "$CONFIG"
