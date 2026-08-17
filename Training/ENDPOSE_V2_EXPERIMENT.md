# Improved terminal end-pose experiment v2

## Historischer Status

Die unten dokumentierten Completed Results stammen aus dem vor der Korrektur
ausgeführten Endpose-v2-Strang. Sie bleiben als historische Evidenz erhalten,
sind aber wegen der alten Capture-Duplizierung und früheren
Checkpoint-/Oracle-Semantik keine Ergebnisse des korrigierten Protokolls.
Neue Terminal-Läufe verwenden `terminal_endpose_unique_hand_capture_v2`, einen
einzigen validation-ausgewählten Hauptcheckpoint und werden getrennt vom
primären t+1-Strang berichtet; siehe `THESIS_FINAL_PROTOCOL_V2.md`.

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

## Completed result (2026-08-10)

The full pipeline completed on the TCML cluster. The audit accepted 208/214
handover sequences (97.2%). The validation-only search selected `trial_003`
after confirmation on seeds 42, 43 and 44. Its main settings are `d_model=32`,
one transformer layer, eight heads, dropout 0.3, batch size 32, learning rate
0.000182289, terminal pose loss weight 4.0, orientation weight 0.1 and auxiliary
t+1 weight 0.25.

On the held-out test participants, endpose-v2 achieved 16.30 ± 1.34 cm terminal
position error and 42.79 ± 1.61° orientation error. This improves endpose-v1 by
2.81 cm and 11.17°, respectively. Compared with t+1 evaluated as a terminal
predictor, position is similar (+0.29 cm) and orientation is better (-6.33°),
while intent macro-F1 is 0.020 lower.

The complete tables, per-seed values, remaining-time curves, latency results and
PNG/PDF figures are in:

```text
Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v2/
```
