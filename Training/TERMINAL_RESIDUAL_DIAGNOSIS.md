# Supplementary terminal residual diagnosis

Authorized 2026-11-11 after reviewing the earlier timing analysis. This is a
post-hoc diagnostic of the corrected v3 terminal receiving-wrist task, separate
from primary t+1 prediction and the new marker-free fine-tuning runs.

Use seeds 42/43/44 of `terminal_endpose_learned` in
`thesis_final_v2_validation`. Compare the already saved validation-selected
`best_intention_model.pt` and `best_pose_model.pt` without training or selecting
another checkpoint. The latter was selected on the original broad validation
end-to-end pose metric, which includes valid partial-evidence targets; it was
not specifically selected on the pure-future GT-hand evaluation below.

Reconstruct the original 23-recording held-out participant split, preserve the
frozen normalizer and all window identities, and reproduce the official main
checkpoint metrics and previous timing exports. Apply the same model-independent
paired cohort to both checkpoints: **201 pure-future 60-sample/stride-10 windows
from 20 recordings**, with both wrist references valid. Partially observed
target intervals remain in raw exports but are excluded from primary residual
diagnosis. Official thesis main checkpoints and reports are not overwritten.

For every eligible window, with the GT receiving hand held fixed, compute:

```
d = terminal_target - latest_observed_wrist
r = predicted_terminal_position - latest_observed_wrist
persistence_error = ||d||
learned_error = ||r-d||
```

Report lengths, absolute length mismatch, model-minus-persistence error,
fractions of improved/worsened windows, and direction angle. The angle is only
reported when **both** vector lengths are at least **2 cm**, a fixed descriptive
threshold rather than a validated sensor-noise bound. Small and zero actual
motions do not get artificial direction angles. Angle denominators can vary
across checkpoints and seeds and are retained in summary tables.

The squared-error identity is checked numerically on every window:

```
||r-d||² = (||r||-||d||)² + 2 (||r|| ||d|| - dot(r,d))
```

This is a geometric decomposition, not a causal explanation of training
failure. Perfect target direction with the predicted length is an explicitly
non-deployable oracle thought experiment. It is not a third trained model.

Fixed displacement bins: [0,2), [2,5), [5,10), [10,20), [20,40), [40,∞) cm.
Fixed time-remaining bins: [0,.5), [.5,1), [1,2), [2,3), [3,∞) seconds.
Show per-seed window means and equal-recording means. Seed SD is not uncertainty
over participants, and overlapping windows are not independent trials.
Predicted-hand model/persistence diagnostics and the difference from GT-hand
predictions are reported separately so hand-selection errors do not masquerade
as displacement-direction errors.

The protocol is frozen in `configs/evaluation/terminal_residual_diagnosis_v1.json`.
Pure functions and tests: `terminal_residual_diagnostics.py` and
`tests/test_terminal_residual_diagnostics.py`. Inference:
`evaluate_terminal_checkpoints.py`. Summary/figures:
`summarize_terminal_residuals.py`. Thirteen local displacement/timing invariant
tests passed before submission. The remote source snapshot has SHA-256 checksums.

GPU array: **2396083**. Dependent summary: **2396086**.
Output root:
`Training/reports/dataset_v3_causal_20260815_n214_5d136a34/terminal_residual_diagnosis_v1/`.
It contains per-checkpoint window exports, checkpoint hashes/epochs, unchanged
saved validation metrics, the vector table, grouped/recording summaries, a
German report and PNG/PDF figures. `jobs.json` records actual submitted IDs.

## Completed findings

All GPU tasks completed (1:42, 1:42, 0:38), summary job 2396086 completed
(14 seconds), with empty stderr. All 31 original result artifacts downloaded
locally passed checksum validation. Both checkpoints reproduce identical input
and target identities; main-checkpoint metrics reproduce the official report.
Reporting-only revision **2396102** adds explicit angle denominators and plot
counts and removes an unexplained scatter color scale. The evaluated model
code remains in the original snapshot; the revised reporter is retained
separately under `code/reporting_v2/`. Numerical results are unchanged.

Primary held-out pure-future comparison (GT hand, 201 identical windows):

| Checkpoint/method | Mean position error cm | Seed sample SD cm |
|---|---:|---:|
| Original best_intention | 19.6777 | 0.9823 |
| Diagnostic best_pose | 20.8651 | 1.7426 |
| GT-hand persistence | 18.2506 | 0.0000 |

Predicted-hand model errors are 19.8075 and 20.6693 cm, respectively; matching
predicted-hand persistence is 18.7146 and 18.8284 cm. The GT-hand comparison
isolates the residual correction from reference-hand selection. Lower saved
validation error does not translate to lower average test error in this
supplementary comparison. One seed's GT-hand result improves, so this is an
aggregate result rather than deterioration in every individual run.

The original best_intention checkpoint predicts displacement length 9.44 cm
on average versus actual remaining displacement 18.25 cm. Direction error is
59.97 degrees where both vector lengths are at least 2 cm (111–118 eligible
angles per seed); the geometric direction term contributes 38.1% of total
squared error. This does not establish a causal training explanation.

Behavior changes with actual motion size: at [0,2) cm actual displacement,
predicted length averages 8.57 cm and model error exceeds persistence by
7.69 cm (51 windows from 12 recordings). At [20,40) cm it improves the mean
error by 6.26 cm (52 windows from 15 recordings); 82.7% of window/seed cases
improve. At >=40 cm it underestimates length (12.35 versus 47.18 cm on average),
and rare bad cases prevent a mean benefit despite 65.4% improved cases.
Thus both unnecessary corrections near the target and insufficient/poorly
directed corrections farther away occur; neither a universal overestimation
nor direction-only explanation fits the results.

Overall 48.6% of original checkpoint window/seed cases improve over GT-hand
persistence. With equal video weighting, the mean model-minus-persistence
error is +3.31 cm; 9/20 recordings improve. These are descriptive post-hoc
findings, not independent-trial success estimates or significance claims.
