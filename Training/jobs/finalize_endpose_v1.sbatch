#!/bin/bash
#SBATCH --job-name=aria_endpose_final
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=Training/slurm_logs/endpose_finalize_v1.%j.out
#SBATCH --error=Training/slurm_logs/endpose_finalize_v1.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="$HOME/singularity/aria_master.simg"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-residual_v2_endpose_v1}"
REPORT_DIR="Training/reports/${DATASET_TAG}/${EXPERIMENT_TAG}"
ENDPOSE_RUN="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/residual_v2_endpose/${EXPERIMENT_TAG}_residual_v2_endpose_seed42"
T1_RUN="Training/runs/${DATASET_TAG}/residual_v2_tuned_v1/residual_v2_tuned/residual_v2_tuned_v1_residual_v2_tuned_seed42"

cd "$REPO_DIR"
mkdir -p "$REPORT_DIR/latency/endpose" "$REPORT_DIR/latency/t_plus_1"

for MODEL in endpose t_plus_1; do
    if [[ "$MODEL" == "endpose" ]]; then
        ARTIFACTS="$ENDPOSE_RUN"
    else
        ARTIFACTS="$T1_RUN"
    fi
    FIXTURE="$REPORT_DIR/latency/$MODEL/fixture.npz"
    singularity exec "$IMAGE" \
        python3 Training/create_latency_fixture.py \
        --artifacts-dir "$ARTIFACTS" \
        --output "$FIXTURE" \
        --overwrite
    singularity exec "$IMAGE" \
        python3 Training/benchmark_model_latency.py \
        --artifacts-dir "$ARTIFACTS" \
        --fixture "$FIXTURE" \
        --output "$REPORT_DIR/latency/$MODEL/tcml_cpu.json" \
        --device cpu --warmup 100 --repeats 1000 --cpu-threads 1
    singularity exec --nv "$IMAGE" \
        python3 Training/benchmark_model_latency.py \
        --artifacts-dir "$ARTIFACTS" \
        --fixture "$FIXTURE" \
        --output "$REPORT_DIR/latency/$MODEL/tcml_cuda.json" \
        --device cuda --warmup 100 --repeats 1000 --cpu-threads 1
done

singularity exec "$IMAGE" \
    python3 Training/evaluation/summarize_endpose_experiment.py \
    --dataset-tag "$DATASET_TAG" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --require-complete
