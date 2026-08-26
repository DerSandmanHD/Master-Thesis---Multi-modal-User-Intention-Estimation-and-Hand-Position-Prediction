# Primary t+1 pose-learning diagnosis

Dataset: `dataset_v3_causal_20260815_n214_5d136a34`. This is a retrospective diagnosis of the primary receiving-wrist pose at t+1 s; terminal endpose is excluded.

| Seed | Epochs | Best intention | Best pose | Train error first→last (cm) | Validation error at best intention→best pose (cm) |
|---:|---:|---:|---:|---:|---:|
| 42 | 14 | 7 | 13 | 9.514→8.954 | 8.899→7.837 |
| 43 | 10 | 3 | 5 | 9.431→9.278 | 9.386→7.866 |
| 44 | 17 | 10 | 10 | 9.426→9.203 | 7.575→7.575 |

## Decision

The checklist's 14–15 cm train-underfitting trigger is not met. Existing curves do not justify a new normalized-loss run; checkpoint-selection timing explains part of the pose gap where the validation pose optimum occurs after best_intention.

A new `normalized_smooth_l1` run is therefore not recommended.

The raw position-loss magnitude alone is not used as evidence: the pose objective also contains the weighted orientation term, and metre-scale Smooth-L1 is quadratic for these sub-metre residuals.

Guardrail: Any normalized-loss run initiated after final-test inspection must be labelled post-hoc sensitivity analysis, never the original primary model.
