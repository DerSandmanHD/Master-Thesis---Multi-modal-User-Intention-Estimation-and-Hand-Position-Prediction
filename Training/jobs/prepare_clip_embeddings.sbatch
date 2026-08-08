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

REPO_DIR="$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction"
IMAGE="$HOME/singularity/aria_master.simg"
PYTHON_DEPS="$HOME/.cache/aria_python/open_clip_2_26_1"
WEIGHTS_CACHE="$HOME/.cache/clip"
WEIGHTS_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
DATASET_TAG="${DATASET_TAG:-dataset_v2_20260802_n214_5d136a34}"
SEQUENCE_FINGERPRINT="${SEQUENCE_FINGERPRINT:-5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd}"
CACHE_DIR="Data_collection/visual_embeddings/clip_vit_b32_openai_5hz"
PROJECTION="Training/visual_projections/${DATASET_TAG}/clip_vit_b32_openai_5hz_pca32.npz"

cd "$REPO_DIR"
mkdir -p Training/slurm_logs "$(dirname "$PROJECTION")"

if [[ ! -d "$PYTHON_DEPS/open_clip" ]]; then
    echo "Missing isolated CLIP dependencies: $PYTHON_DEPS" >&2
    echo "Install the pinned packages from Training/clip_requirements.txt with --no-deps." >&2
    exit 2
fi

singularity exec --nv "$IMAGE" \
    env PYTHONPATH="$PYTHON_DEPS" TORCH_HOME="$HOME/.cache/aria_torch" \
    python3 Training/extract_clip_embeddings.py \
    --master-dir Data_collection/master_datasets \
    --video-dir Data_collection/Data_mp4 \
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

if [[ -e "$PROJECTION" ]]; then
    echo "Projection already exists and will be validated by the data smoke test: $PROJECTION"
else
    singularity exec "$IMAGE" \
        python3 Training/fit_visual_projection.py \
        --config Training/configs/models/residual_transformer_v2.json \
        --cache-dir "$CACHE_DIR" \
        --output "$PROJECTION" \
        --components 32 \
        --random-state 20260808 \
        --expected-sequence-fingerprint "$SEQUENCE_FINGERPRINT"
fi

singularity exec "$IMAGE" \
    python3 Training/visual_embedding_dataset_check.py \
    --config Training/configs/visual/residual_v2_sensor_plus_clip.json

echo "CLIP cache and train-only PCA projection are ready."
