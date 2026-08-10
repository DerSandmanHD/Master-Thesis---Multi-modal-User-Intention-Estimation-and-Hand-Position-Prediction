#!/bin/bash
#SBATCH --job-name=aria_endpose_v2_report
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=Training/slurm_logs/endpose_v2_report.%j.out
#SBATCH --error=Training/slurm_logs/endpose_v2_report.%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v2}"
REPORT_DIR="Training/reports/${DATASET_TAG}/${EXPERIMENT_TAG}"
RUN="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/residual_v2_endpose_v2/${EXPERIMENT_TAG}_dual_horizon_endpose_v2_seed42"
cd "$REPO_DIR"
mkdir -p "$REPORT_DIR/latency/endpose_v2"
FIXTURE="$REPORT_DIR/latency/endpose_v2/fixture.npz"
singularity exec "$IMAGE" python3 Training/create_latency_fixture.py \
    --artifacts-dir "$RUN" --output "$FIXTURE" --overwrite
singularity exec "$IMAGE" python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$RUN" --fixture "$FIXTURE" \
    --output "$REPORT_DIR/latency/endpose_v2/tcml_cpu.json" \
    --device cpu --warmup 100 --repeats 1000 --cpu-threads 1
singularity exec --nv "$IMAGE" python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$RUN" --fixture "$FIXTURE" \
    --output "$REPORT_DIR/latency/endpose_v2/tcml_cuda.json" \
    --device cuda --warmup 100 --repeats 1000 --cpu-threads 1
singularity exec "$IMAGE" python3 Training/evaluation/summarize_endpose_v2_experiment.py \
    --dataset-tag "$DATASET_TAG" --experiment-tag "$EXPERIMENT_TAG" \
    --require-complete
