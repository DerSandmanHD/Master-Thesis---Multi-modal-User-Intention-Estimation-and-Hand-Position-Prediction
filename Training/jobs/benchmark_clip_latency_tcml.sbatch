#!/bin/bash
#SBATCH --job-name=aria_clip_latency
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/benchmark_clip_latency.%j.out
#SBATCH --error=Training/slurm_logs/benchmark_clip_latency.%j.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
PYTHON_DEPS="$HOME/.cache/aria_python/open_clip_2_26_1"
WEIGHTS_CACHE="$HOME/.cache/clip"
VIDEO="${VIDEO:-Data_collection/Data_mp4/Angelina_10_20260715_153801.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-Training/reports/latency/clip_vit_b32_openai_5hz}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" Training/slurm_logs
for DEVICE in cpu cuda; do
    NV_ARGS=()
    if [[ "$DEVICE" == "cuda" ]]; then
        NV_ARGS=(--nv)
    fi
    singularity exec "${NV_ARGS[@]}" "$IMAGE" \
        env PYTHONPATH="$PYTHON_DEPS" \
        python3 Training/benchmark_clip_latency.py \
        --video "$VIDEO" \
        --output "$OUTPUT_DIR/tcml_${DEVICE}.json" \
        --model-name ViT-B-32 \
        --pretrained openai \
        --weights-cache-dir "$WEIGHTS_CACHE" \
        --device "$DEVICE" \
        --cpu-threads 4 \
        --warmup 100 \
        --repeats 1000
done

echo "CLIP encoder latency benchmark complete."
