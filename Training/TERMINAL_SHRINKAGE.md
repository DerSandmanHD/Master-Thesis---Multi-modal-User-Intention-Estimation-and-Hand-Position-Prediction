# Validation-selected attenuation of terminal residuals

Authorized after the terminal displacement diagnosis on 2026-11-11. This is
an exploratory post-hoc follow-up, not an untouched final holdout experiment.
It does not retrain a model or replace the original thesis checkpoints.

For each fixed terminal checkpoint family (`best_intention` primary and
`best_pose` diagnostic), predict position as:

```
p_alpha = p_last + alpha * (p_model - p_last)
```

Use one common alpha for seeds 42/43/44 per family. The fixed grid is
0, 0.05, ..., 1. Alpha=0 exactly reproduces persistence, alpha=1 reproduces
the previously exported model prediction. Intermediate factors attenuate only
the position correction. Hand choice, intention, orientation and weights
remain unchanged. The formula uses no future target, true remaining time or
true distance to the end position.

New GPU inference exports **validation only** for both fixed checkpoint
families, using the original source normalizer, feature schema and split.
Artifact/weight hashes are verified; original stored validation metrics are
reproduced. The selection objective is mean seed validation mean Euclidean
error with the **predicted hand**, on the common pure-future both-reference
cohort of 256 windows. Ties within 1e-9 cm choose the smaller alpha.

The selection function accepts only validation exports and has no path to test
predictions. Before any test export is opened, it persists `selection.json`
with chosen alphas, source checkpoint identities and validation export hashes.
Only then are the existing test exports from `terminal_residual_diagnosis_v1`
loaded and validated against the same checkpoint identities. Validate disjoint
validation/test sequence and sample identities. Evaluate only three labels:
alpha=0, the frozen selected alpha, and alpha=1, with no subsequent test-driven
parameter or checkpoint-family selection. If alpha=0/1 is selected, the
selected label intentionally coincides with the corresponding reference.

Test cohort: the same 201 pure-future 60-sample stride-10 windows from 20 old
held-out recordings. Primary comparison uses model-predicted hand for all three
variants. GT-hand results are diagnostic and use their matching references;
GT-hand persistence is not substituted for predicted-hand persistence.
Report window means and equal-recording means, with sample SD across seeds.
Overlapping windows are dependent and seed SD is not person uncertainty.

Protocol: `configs/evaluation/terminal_shrinkage_v1.json`.
Pure functions: `terminal_shrinkage.py`. GPU validation export:
`export_terminal_validation.py`. Selection/test/reporting:
`evaluate_terminal_shrinkage.py`. Job files:
`jobs/terminal_shrinkage_validation.sbatch`, `jobs/terminal_shrinkage_summary.sbatch`.
Fifteen local attenuation/vector invariant tests passed before submission.

Output root:
`Training/reports/dataset_v3_causal_20260815_n214_5d136a34/terminal_residual_shrinkage_v1/`.
The outcome is reported regardless of whether attenuation improves on
persistence; no further alpha adjustment is made from these test results.

## Completed result

Initial L40S jobs 2396136/2396137 were cancelled while still pending because
resources were occupied. The same scripts were resubmitted with SLURM resource
overrides `--partition=day --gres=gpu:A4000:1`: validation array **2396140**,
selection/test job **2396143**. All completed successfully (GPU 1:36–1:38;
selection/test/report 10 seconds), with empty stderr. Checkpoint and original
validation errors reproduced within the frozen numerical tolerances.

Validation selected alpha=**0.05** for best_intention and **0.15** for best_pose.
Validation predicted-hand errors: 12.10 -> 12.01 cm (persistence -> selected)
and 12.06 -> 11.92 cm, respectively. Those factors were saved in selection.json
before test prediction files were opened; timestamp ordering and SHA-256 were
verified locally. No test alpha grid was computed.

Test predicted-hand means over seeds 42/43/44, same 201 windows:

| Family | alpha=0 persistence | Selected alpha | alpha=1 original |
|---|---:|---:|---:|
| best_intention | 18.7146 cm | 18.6278 cm (0.05) | 19.8075 cm |
| best_pose | 18.8284 cm | 18.8354 cm (0.15) | 20.6693 cm |

GT-hand diagnostic means: best_intention 18.2506 / 18.1707 / 19.6777 cm;
best_pose 18.2506 / 18.3411 / 20.8651 cm (same column order). Predicted-hand
persistence depends on each family's hand decisions and must not be replaced
by the GT-hand baseline when assessing end-to-end improvement.

The primary selected factor retains only 5% of the original correction. It
reduces test error by about 1.18 cm versus the original model, but by only
0.0868 cm (0.868 mm) versus matching persistence. The diagnostic family is
about 0.0069 cm worse than matching persistence. This is predominantly a
return toward persistence, not evidence of a substantial learned forecasting
advantage. Do not claim significance or a deployment-quality improvement from
the tiny difference; this post-hoc experiment uses previously observed tests.

All **26 downloaded artifacts** passed recorded SHA-256 verification. Local
audits recomputed the validation-selected factors from validation_grid.csv,
verified their frozen-selection hash, checked that selection preceded the
recorded first test read, and confirmed all 18 family/seed/variant rows use
201 windows. PNG layout was visually inspected; standalone PDF is available.
