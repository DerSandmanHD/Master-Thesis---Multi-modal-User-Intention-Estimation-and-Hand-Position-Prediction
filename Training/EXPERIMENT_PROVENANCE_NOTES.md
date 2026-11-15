# Clarifications for the thesis and reproduction package

These notes describe existing experiments; no new training or model selection
was performed. They do not replace the frozen reports.

## Adaptation source: a separate v3 search

The adaptation source is the `trial_000` family of `residual_v3_hp_search_v1`,
not the main matrix's `residual_current_gate` checkpoints. The v3 search tested
the reference and 23 additional candidates, using three participant-grouped
validation folds, with subsequent multi-seed confirmation. The saved selection
identifies `trial_000` as both the reference and selected candidate, records
`test_metrics_read=false`, and binds the stage-B summary by SHA-256. The sources
used for adaptation are the matching-seed original-split runs.

Evidence (external result package / research archive):
- `Training/reports/dataset_v3_causal_20260815_n214_5d136a34/residual_v3_hp_search_v1/selection.json`
- same directory: `stage_b_summary.json`, `final_summary.json`, `SEARCH_RESULTS.md`
- `Data_collection/object_marker_free_v1/adaptation_protocol.json`
- source checkpoint hashes in the processed-artifact lock and per-seed protocols.

The complete search archive is not claimed to be included by the processed
reproduction lock. The adaptation source checkpoints and their run metadata are
indexed. Repeating the entire hyperparameter search requires its additional
configs, fold manifests, schedules and search code from the research archive.

## Terminal learning rate: historical v2 inheritance

The value `3.81605632501e-4` occurs in `trial_022` of the older fixed-horizon
`residual_v2_hp_search_v1` and in the selected configuration of
`residual_v2_hp_confirm_v1`. Screening comprised 24 candidates; confirmation
used seeds 42, 43 and 44. This is the origin of the retained learning rate and
backbone settings, not evidence of a search on the corrected terminal target.

Evidence in `Training/reports/dataset_v2_20260802_n214_5d136a34/`:
- `residual_v2_hp_search_v1/summary.json`
- `residual_v2_hp_confirm_v1/summary.json` and `selected_config.json`
- compare `Training/configs/models/residual_transformer_endpose_v2.json`.

A different historical terminal search,
`residual_v2_endpose_v2_hp_confirm`, selected `trial_003` with LR
`1.82289290736e-4`. It must not be cited as the selection of the corrected
terminal configuration reported in the thesis. The corrected terminal task
retained settings without a new search; architecture, losses and sampling also
differ from the main fixed-horizon model.

## Development pilot

The pilot is the first new-domain recording used during development/checking of
the on-device conversion pipeline. It remains in the training recording split,
not validation or test. Evidence: `Training/jobs/OBJECT_MARKER_FREE_ONDEVICE.md`
in the research archive and `adaptation_protocol.json` in the external package.
It is not an additional participant or an independent test cohort.

## Legacy feature name: robot_anchor_interpolated

In `Code/build_master_dataset.py`, `add_coordinate_transforms` sets this feature
to 1 when a static world-to-marker anchor and a valid SLAM pose supply the robot
transform but no valid instantaneous marker transform is available at the aligned
sample. Otherwise it remains 0. This does **not** interpolate a trajectory or
measurement. Availability is determined after timestamp association and validity
checks, so it is more precise than saying only that a marker is invisible in the
current RGB image. The field name stays unchanged for checkpoint/schema
compatibility; describe it as a static-anchor fallback indicator in prose.

## Existing latency measurements do not cover robot execution

- `Training/reports/latency/final_sensor_plus_clip_v1/mac_cpu.json` binds a
  historical v2 Sensor+CLIP checkpoint and a cached-feature input fixture.
  The report's word "final" is historical, not the active v3 primary model.
- `Training/reports/latency/clip_vit_b32_openai_5hz/` separately measures the
  visual frontend on identical decoded pixels. It is not the complete pipeline.
- `Training/reports/latency/existing_mac_live/summary.json` summarizes 1,116
  exploratory records, explicitly sets `final_tuned_model_claimed=false` and
  `capture_to_host_latency_available=false`. Device/host clock alignment was
  not validated.

These measurements do not include calibrated capture-to-host delay, planning and
physical robot motion. End-to-end latency of the current system therefore still
needs measurement. Existing logs are not a checkpoint-bound replay validation of
the active v3 system; do not equate replay tooling with a completed replay study.
