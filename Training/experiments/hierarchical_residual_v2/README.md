# Hierarchical Residual v2

Status: first full cluster run completed; run artifacts still need to be copied
into this immutable experiment directory.

## Purpose

This experiment tests whether residual future-pose prediction can preserve the
early-handover advantage of the absolute transformer while falling back to the
last observed hand pose once motion becomes small.

## Fixed design

- participant split identical to hierarchical tracking baseline v1
- observation window: 60 samples
- future horizon: 1 second
- learned receiving-hand classification: left versus right
- shared position delta relative to the selected hand reference
- relative quaternion composed with the selected hand reference
- oracle-hand and predicted-hand pose metrics
- separate hand-side and handover-progress metrics
- best-intention and best-validation-pose checkpoints

## Files

- active config: `Training/configs/models/residual_transformer_v2.json`
- training entry point: `Training/train_residual.py`
- cluster job: `Training/jobs/train_residual_v2.sbatch`

## First full cluster run

- run directory: `Training/runs/hierarchical_residual_v2_20260712_145907`
- best-intention checkpoint: intention macro F1 `0.8420`, receiving-hand F1
  `0.8206`, oracle pose MAE `17.18 cm`, end-to-end pose MAE `17.86 cm`
- best-pose checkpoint: intention macro F1 `0.7853`, receiving-hand F1
  `0.8304`, oracle pose MAE `16.08 cm`, end-to-end pose MAE `16.55 cm`

The exact source commit, cluster `config.json`, `data_metadata.json` and
`metrics.json` must still be added before this result is considered an
immutable accepted experiment. The existing v1 experiment is not overwritten.
