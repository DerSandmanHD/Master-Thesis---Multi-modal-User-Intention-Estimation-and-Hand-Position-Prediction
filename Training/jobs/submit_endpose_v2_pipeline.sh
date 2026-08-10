#!/bin/bash
# Submit the complete leakage-safe endpose-v2 Slurm dependency chain.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Master-Thesis---Multi-modal-User-Intention-Estimation-and-Hand-Position-Prediction}"
cd "$REPO_DIR"

submit() {
    sbatch --parsable --export="ALL,REPO_DIR=${REPO_DIR}" "$@"
}

AUDIT_JOB="$(submit Training/jobs/audit_endpose_v2.sbatch)"
SEARCH_JOB="$(submit --dependency="afterok:${AUDIT_JOB}" Training/jobs/search_endpose_v2.sbatch)"
RANK_JOB="$(submit --dependency="afterok:${SEARCH_JOB}" Training/jobs/summarize_endpose_v2_search.sbatch)"
CONFIRM_JOB="$(submit --dependency="afterok:${RANK_JOB}" Training/jobs/confirm_endpose_v2.sbatch)"
SELECT_JOB="$(submit --dependency="afterok:${CONFIRM_JOB}" Training/jobs/summarize_endpose_v2_confirmation.sbatch)"
FINAL_JOB="$(submit --dependency="afterok:${SELECT_JOB}" Training/jobs/train_endpose_v2.sbatch)"
REPORT_JOB="$(submit --dependency="afterok:${FINAL_JOB}" Training/jobs/finalize_endpose_v2.sbatch)"

printf 'audit=%s\nsearch=%s\nrank=%s\nconfirm=%s\nselect=%s\nfinal=%s\nreport=%s\n' \
    "$AUDIT_JOB" "$SEARCH_JOB" "$RANK_JOB" "$CONFIRM_JOB" "$SELECT_JOB" \
    "$FINAL_JOB" "$REPORT_JOB"
