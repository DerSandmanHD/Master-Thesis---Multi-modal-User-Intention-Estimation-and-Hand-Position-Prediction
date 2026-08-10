# Improved terminal end-pose experiment v2

This experiment is separate from both the frozen t+1 model and
`residual_v2_endpose_v1`. It never overwrites either run family.

## Targets

- Primary target: the same robust terminal receiving-hand pose used by
  endpose-v1 (latest stable 0.5-second segment after `THIRD`).
- Auxiliary target: receiving-hand pose at t+1 second, derived from the
  current hand-tracking stream with a maximum timestamp mismatch of 0.1 s.
- Rejected terminal targets remain in intent/hand training but contribute no
  primary or auxiliary pose loss.

The unchanged primary target makes v1/v2 terminal-pose results directly
comparable.

## Improvements under test

1. Position residuals are standardized per axis using **training data only**.
2. Quaternion supervision uses sign-invariant geodesic angular loss.
3. The training sampler gives every sequence equal expected sampling mass.
4. Terminal pose loss balances the available remaining-time bins per sequence.
5. Left and right hands have separate residual candidates.
6. A separate t+1 pose head supplies an auxiliary local-motion signal.

No true remaining time is supplied as a model feature; that value is available
only for stratified loss weighting and evaluation and therefore cannot leak
future timing into live inference.

## Leakage-safe protocol

- Dataset: frozen `dataset_v2_20260802_n214_5d136a34` (214 sequences).
- Split: the same participant-wise train/validation/test split as t+1 and
  endpose-v1.
- Stage A: 12 deterministic validation-only trials with seed 42.
- Confirmation: the three selected trials on seeds 42, 43 and 44,
  validation-only.
- Final: one validation-selected configuration on seeds 42, 43 and 44; test is
  evaluated only in this final stage.
- Selection: minimize mean validation terminal position error; retain trials
  within 0.5 cm, then minimize orientation error, then maximize intent and hand
  macro-F1.

## Audit and dry-run

```bash
sbatch --export=ALL,REPO_DIR="$PWD" Training/jobs/audit_endpose_v2.sbatch
```

The job must produce `status=passed`, `training_started=false` and
`test_metrics_computed=false` before search begins.

## Complete cluster pipeline

```bash
REPO_DIR="$PWD" Training/jobs/submit_endpose_v2_pipeline.sh
```

The submission script connects audit, Stage A, ranking, confirmation,
validation selection, final three-seed training and reporting using `afterok`
dependencies.

Final outputs are written to:

```text
Training/runs/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v2/
Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v2/
```

## Local checks

```bash
conda run -n aria_conda python Training/endpose_v2_smoke_test.py
conda run -n aria_conda python Training/residual_smoke_test.py
conda run -n aria_conda python Training/endpose_smoke_test.py
```
