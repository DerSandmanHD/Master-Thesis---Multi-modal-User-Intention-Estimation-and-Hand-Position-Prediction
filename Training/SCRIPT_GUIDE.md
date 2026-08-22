# Active Python script guide

For the active causal v3 thesis protocol, start from
`THESIS_FINAL_PROTOCOL_V2.md` or `jobs/submit_thesis_v2_pipeline.sh` rather
than invoking individual historical scripts.

## Active entry points

| Purpose | Python entry points |
|---|---|
| Build and verify data | `Code/build_master_dataset_batch.py`, `Code/dataset_qa.py`, `verify_causal_masters.py` |
| Create visual features | `extract_clip_embeddings.py`, `fit_visual_projection.py`, `visual_embedding_dataset_check.py` |
| Train and select models | `train.py`, `train_residual.py`, `experiment_matrix.py`, `select_matrix_checkpoints.py` |
| Evaluate final checkpoints | `evaluate_frozen_run.py`, `export_residual_predictions.py`, `evaluate_pose_baselines.py` |
| Participant statistics | `audit_participant_splits.py`, `prepare_group_cv_runs.py`, `evaluation/evaluate_grouped_predictions.py`, `evaluation/summarize_group_cv.py` |
| Final reports | `evaluation/summarize_thesis_v2_matrix.py`, `evaluation/summarize_modality_weights.py` |
| Qualitative videos | `build_video_alignment_sidecars.py`, `render_prediction_overlay.py` |

## Required supporting modules

The active entry points import shared modules such as `data.py`, `model.py`,
`metrics.py`, `artifact_freeze.py`, `clip_alignment.py`, `visual_embeddings.py`,
`endpose_v2.py`, `participant_splits.py`, `pose_baselines.py`, and
`video_alignment.py`.  These are library modules, even when they also provide
a command-line interface, and should remain in place.

## Historical and development tools

Older v1/v2 experiment scripts, latency tools, live-demo utilities, and
`*_smoke_test.py` files are not part of the normal v3 execution path.  They
remain available for historical reproduction or development checks.  Clearly
unreferenced historical standalone tools are stored in `legacy_tools/`.
