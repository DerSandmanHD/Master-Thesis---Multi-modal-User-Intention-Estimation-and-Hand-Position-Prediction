#!/bin/bash
#SBATCH --job-name=aria_v2_latency
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=Training/slurm_logs/benchmark_latency_tcml.%j.out
#SBATCH --error=Training/slurm_logs/benchmark_latency_tcml.%j.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${ARTIFACTS_DIR:?Set ARTIFACTS_DIR to the final residual-v2 run}"
: "${FIXTURE:?Set FIXTURE to the exported latency fixture}"
OUTPUT_DIR="${OUTPUT_DIR:-Training/reports/latency/final_residual_v2}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" Training/slurm_logs
singularity exec "$IMAGE" \
    python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --fixture "$FIXTURE" \
    --output "$OUTPUT_DIR/tcml_compute_cpu.json" \
    --device cpu \
    --cpu-threads 4 \
    --warmup 100 \
    --repeats 1000

singularity exec --nv "$IMAGE" \
    python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --fixture "$FIXTURE" \
    --output "$OUTPUT_DIR/tcml_compute_cuda.json" \
    --device cuda \
    --cpu-threads 4 \
    --warmup 100 \
    --repeats 1000
