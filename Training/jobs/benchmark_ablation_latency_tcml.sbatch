#!/bin/bash
#SBATCH --job-name=aria_v2_abl_latency
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-4%3
#SBATCH --output=Training/slurm_logs/benchmark_ablation_latency.%A_%a.out
#SBATCH --error=Training/slurm_logs/benchmark_ablation_latency.%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
: "${DATASET_TAG:?Set DATASET_TAG to an immutable dataset version}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-modality_ablation_v1}"
SEED="${SEED:-42}"
VARIANTS=(full no_gaze no_hands no_objects no_vio)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

if [[ "$VARIANT" == "full" ]]; then
    ARTIFACTS_DIR="Training/runs/${DATASET_TAG}/benchmark_v2/residual_v2/benchmark_v2_residual_v2_seed${SEED}"
else
    ARTIFACTS_DIR="Training/runs/${DATASET_TAG}/${EXPERIMENT_TAG}/${VARIANT}/${EXPERIMENT_TAG}_${VARIANT}_seed${SEED}"
fi
OUTPUT_DIR="Training/reports/${DATASET_TAG}/${EXPERIMENT_TAG}/latency/${VARIANT}"
FIXTURE="$OUTPUT_DIR/fixture.npz"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" Training/slurm_logs
if [[ ! -e "$FIXTURE" ]]; then
    singularity exec "$IMAGE" \
        python3 Training/create_latency_fixture.py \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --output "$FIXTURE" \
        --index 0
fi

singularity exec "$IMAGE" \
    python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --fixture "$FIXTURE" \
    --output "$OUTPUT_DIR/tcml_cpu.json" \
    --device cpu \
    --cpu-threads 4 \
    --warmup 100 \
    --repeats 1000

singularity exec --nv "$IMAGE" \
    python3 Training/benchmark_model_latency.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --fixture "$FIXTURE" \
    --output "$OUTPUT_DIR/tcml_cuda.json" \
    --device cuda \
    --cpu-threads 4 \
    --warmup 100 \
    --repeats 1000

echo "Ablation latency complete: $VARIANT"
