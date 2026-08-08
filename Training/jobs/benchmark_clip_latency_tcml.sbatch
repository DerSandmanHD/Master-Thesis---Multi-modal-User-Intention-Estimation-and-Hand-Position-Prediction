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
WEIGHTS_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
OUTPUT_DIR="${OUTPUT_DIR:-Training/reports/latency/clip_vit_b32_openai_5hz}"
RGB_FIXTURE="${RGB_FIXTURE:-${OUTPUT_DIR}/rgb_fixture.npz}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" Training/slurm_logs
[[ -f "$RGB_FIXTURE" ]] || {
    echo "Pinned RGB fixture is missing: $RGB_FIXTURE" >&2
    exit 2
}
for DEVICE in cpu cuda; do
    NV_ARGS=()
    if [[ "$DEVICE" == "cuda" ]]; then
        NV_ARGS=(--nv)
    fi
    singularity exec "${NV_ARGS[@]}" "$IMAGE" \
        env PYTHONPATH="$PYTHON_DEPS" \
        python3 Training/benchmark_clip_latency.py \
        --rgb-fixture "$RGB_FIXTURE" \
        --output "$OUTPUT_DIR/tcml_${DEVICE}.json" \
        --model-name ViT-B-32 \
        --pretrained openai \
        --weights-cache-dir "$WEIGHTS_CACHE" \
        --expected-weights-sha256 "$WEIGHTS_SHA256" \
        --device "$DEVICE" \
        --cpu-threads 4 \
        --warmup 100 \
        --repeats 1000
done

echo "CLIP encoder latency benchmark complete."
