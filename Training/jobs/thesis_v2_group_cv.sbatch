#!/bin/bash
#SBATCH --job-name=thesis_v2_groupcv
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-14%3
#SBATCH --output=Training/slurm_logs/thesis_v2_group_cv.%A_%a.out
#SBATCH --error=Training/slurm_logs/thesis_v2_group_cv.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
IMAGE="${IMAGE:-$HOME/singularity/aria_master.simg}"
: "${GROUP_CV_PLAN:?Set GROUP_CV_PLAN to generated group_cv_plan.json}"

cd "$REPO_DIR"
[[ -f "$IMAGE" ]] || { echo "Container missing: $IMAGE" >&2; exit 2; }
[[ -f "$GROUP_CV_PLAN" ]] || { echo "Group-CV plan missing: $GROUP_CV_PLAN" >&2; exit 2; }

mapfile -t FIELDS < <(
  singularity exec "$IMAGE" python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["protocol"] == "nested_participant_group_cv_executable_v1" and p["outer_evaluation_used_for_selection"] is False and len(p["runs"]) == 15; r=p["runs"][int(sys.argv[2])]; print(r["config"]); print(r["config_sha256"]); print(r["run_dir"]); print(r["outer_evaluation_output"]); print(r["validation_command"].split()[1])' "$GROUP_CV_PLAN" "$SLURM_ARRAY_TASK_ID"
)
CONFIG="${FIELDS[0]}"
EXPECTED_CONFIG_SHA256="${FIELDS[1]}"
RUN_DIR="${FIELDS[2]}"
OUTPUT="${FIELDS[3]}"
ENTRYPOINT="${FIELDS[4]}"

[[ -f "$CONFIG" ]] || { echo "Fold config missing: $CONFIG" >&2; exit 2; }
ACTUAL_CONFIG_SHA256=$(singularity exec "$IMAGE" python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONFIG")
[[ "$ACTUAL_CONFIG_SHA256" == "$EXPECTED_CONFIG_SHA256" ]] || {
  echo "Fold config changed after plan generation: $CONFIG" >&2
  exit 2
}
if [[ ! -e "$RUN_DIR" ]]; then
  DATASET_TAG=$(singularity exec "$IMAGE" python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_tag"])' "$GROUP_CV_PLAN")
  EXPERIMENT_TAG=$(singularity exec "$IMAGE" python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_tag"])' "$GROUP_CV_PLAN")
  SEED=$(singularity exec "$IMAGE" python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["seed"])' "$GROUP_CV_PLAN" "$SLURM_ARRAY_TASK_ID")
  singularity exec --nv "$IMAGE" \
    python3 "$ENTRYPOINT" --config "$CONFIG" \
    --dataset-tag "$DATASET_TAG" --experiment-tag "$EXPERIMENT_TAG" \
    --seed "$SEED" --run-dir "$RUN_DIR" --skip-test-evaluation
fi
singularity exec "$IMAGE" python3 Training/artifact_freeze.py "$RUN_DIR/artifact_manifest.json"
if [[ -e "$OUTPUT" ]]; then
  singularity exec "$IMAGE" python3 -c 'import json,pathlib,sys; run=pathlib.Path(sys.argv[1]).resolve(); out=json.load(open(sys.argv[2])); freeze=json.load(open(run/"artifact_manifest.json")); metrics=json.load(open(run/"metrics.json")); cp=metrics["checkpoints"]["best_intention"]; frozen=freeze["output_artifacts"]["checkpoints"]["best_intention"]; source=pathlib.Path(out["source_run"]).resolve(); assert source == run; assert out["split"] == "test" and out["evaluation_protocol"] == "validation_frozen_checkpoint_single_test_v1"; assert out["source_artifact_manifest_fingerprint"] == freeze["manifest_fingerprint"]; assert out["checkpoint"]["name"] == "best_intention" and out["checkpoint"]["sha256"] == frozen["sha256"] == cp["sha256"]; assert out["test_used_for_model_or_checkpoint_selection"] is False; assert out.get("matrix_authorization") is None; assert metrics["test_evaluation_skipped"] is True and "test" not in metrics and "test_by_checkpoint" not in metrics' "$RUN_DIR" "$OUTPUT" || {
    echo "Existing outer evaluation is stale or mismatched: $OUTPUT" >&2
    exit 2
  }
  echo "Verified existing outer evaluation: $OUTPUT"
  exit 0
fi
singularity exec --nv "$IMAGE" \
  python3 Training/evaluate_frozen_run.py \
  --run-dir "$RUN_DIR" --checkpoint best_intention --output "$OUTPUT"
echo "Group-CV task complete: $OUTPUT"
