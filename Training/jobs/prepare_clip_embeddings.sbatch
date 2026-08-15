#!/bin/bash
#SBATCH --job-name=aria_clip_cache
#SBATCH --partition=day
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=Training/slurm_logs/prepare_clip_embeddings.%j.out
#SBATCH --error=Training/slurm_logs/prepare_clip_embeddings.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
PYTHON_DEPS="${PYTHON_DEPS:-$HOME/.cache/aria_python/open_clip_2_26_1}"
WEIGHTS_CACHE="${WEIGHTS_CACHE:-$HOME/.cache/clip}"
WEIGHTS_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
DATASET_TAG="${DATASET_TAG:-dataset_v3_causal_20260815_n214_5d136a34}"
SEQUENCE_FINGERPRINT="${SEQUENCE_FINGERPRINT:-5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd}"
CACHE_DIR="Data_collection/visual_embeddings/${DATASET_TAG}/clip_vit_b32_openai_5hz_device_time_v2"
PROJECTION="Training/visual_projections/${DATASET_TAG}/clip_vit_b32_openai_5hz_device_time_v2_pca32.npz"
DATA_CONFIG="${DATA_CONFIG:-Training/configs/models/residual_transformer_v2.json}"

cd "$REPO_DIR"
mkdir -p Training/slurm_logs "$(dirname "$PROJECTION")"

if [[ ! -d "$PYTHON_DEPS/open_clip" ]]; then
    echo "Missing isolated CLIP dependencies: $PYTHON_DEPS" >&2
    echo "Install the pinned packages from Training/clip_requirements.txt with --no-deps." >&2
    exit 2
fi

# Cheap CPU preflight before requesting hours of GPU work. This rejects old
# nearest/backfilled masters before any CLIP cache can be created from them.
singularity exec "$IMAGE" \
    python3 Training/verify_causal_masters.py \
    --master-dir Data_collection/master_datasets \
    --manifest Data_collection/dataset_manifest.csv \
    --expected-alignment-version causal_backward_device_time_v1 \
    --expected-sequence-fingerprint "$SEQUENCE_FINGERPRINT"

singularity exec --nv "$IMAGE" \
    env PYTHONPATH="$PYTHON_DEPS" TORCH_HOME="$HOME/.cache/aria_torch" \
    python3 Training/extract_clip_embeddings.py \
    --master-dir Data_collection/master_datasets \
    --vrs-dir Data_collection/Data_vrs \
    --manifest Data_collection/dataset_manifest.csv \
    --output-dir "$CACHE_DIR" \
    --expected-sequence-fingerprint "$SEQUENCE_FINGERPRINT" \
    --model-name ViT-B-32 \
    --pretrained openai \
    --weights-cache-dir "$WEIGHTS_CACHE" \
    --expected-weights-sha256 "$WEIGHTS_SHA256" \
    --sample-hz 5 \
    --batch-size 64 \
    --device cuda

# The projection is content-bound to cache_manifest.json.  Refit it after every
# cache preparation run; --overwrite never touches the historical v1 path.
singularity exec "$IMAGE" \
    python3 Training/fit_visual_projection.py \
    --config "$DATA_CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --output "$PROJECTION" \
    --components 32 \
    --random-state 20260808 \
    --expected-sequence-fingerprint "$SEQUENCE_FINGERPRINT" \
    --overwrite

singularity exec "$IMAGE" \
    python3 Training/visual_embedding_dataset_check.py \
    --config Training/configs/visual/residual_v2_sensor_plus_clip.json \
    --visual-cache-dir "$CACHE_DIR" \
    --visual-projection-path "$PROJECTION" \
    --expected-sequence-fingerprint "$SEQUENCE_FINGERPRINT"

echo "CLIP cache and train-only PCA projection are ready."
