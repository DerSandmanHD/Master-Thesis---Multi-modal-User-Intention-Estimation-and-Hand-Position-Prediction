# Hierarchical Residual v2

Status: implementation complete, cluster result pending.

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

- active config: `Training/configs/hierarchical_residual_v2.json`
- training entry point: `Training/train_residual.py`
- cluster job: `Training/hierarchical_residual_v2.sbatch`

The exact source commit, config snapshot, run directory and metrics must be
added here after the first accepted cluster run. The existing v1 experiment is
not overwritten.
