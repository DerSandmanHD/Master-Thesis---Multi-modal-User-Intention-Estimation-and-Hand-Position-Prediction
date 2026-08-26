# Active-v3 authoritative thesis report

Every seed row is bound to exactly one executable validation-selected `best_intention` checkpoint. Aggregate rows are mean ± sample SD across seeds and are not executable checkpoints.
Matrix SHA-256: `20ebefa13be76635b2f78139a6a8506dcf1eae6fedd8fdc1c7bdcf4d53e7e6d2`; validation-selection SHA-256: `a87834b20f7275293e3cba47e96565ef55f27895dff49dc82499d5a20efe4d21`.
Checkpoint-coherent seed-results SHA-256: `7b975744db612cf1f2cf87f12f56baa94e1a0263125b8ad43401f2c5a3d0d027`.

| Task | Experiment | Seeds | Intent macro-F1 | Assistance macro-F1 | Receiving-hand macro-F1 | Pose mean cm | Orientation mean deg | Pose coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary_t_plus_1_future_wrist | baseline_mlp | 3 | 0.8225 ± 0.0184 | 0.8821 ± 0.0107 | 0.8965 ± 0.0095 | 19.5021 ± 0.9514 | 56.0420 ± 2.1715 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | baseline_gru | 3 | 0.8027 ± 0.0182 | 0.8671 ± 0.0074 | 0.9106 ± 0.0170 | 18.7473 ± 0.9769 | 45.1326 ± 1.4650 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | baseline_transformer | 3 | 0.8069 ± 0.0141 | 0.8694 ± 0.0062 | 0.9577 ± 0.0130 | 19.4269 ± 0.9939 | 47.8042 ± 1.9851 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | residual_current_gate | 3 | 0.8280 ± 0.0152 | 0.8799 ± 0.0209 | 0.9477 ± 0.0132 | 16.5447 ± 0.3084 | 48.0455 ± 1.6828 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | residual_simple_gate | 3 | 0.8215 ± 0.0163 | 0.8789 ± 0.0120 | 0.9284 ± 0.0328 | 16.2681 ± 0.3692 | 47.4862 ± 1.6232 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | residual_modality_gated | 3 | 0.8315 ± 0.0075 | 0.8913 ± 0.0149 | 0.9128 ± 0.0292 | 16.0080 ± 0.1834 | 46.0234 ± 1.0447 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | visual_corrected_random_current_gate | 3 | 0.8135 ± 0.0113 | 0.8860 ± 0.0188 | 0.9592 ± 0.0072 | 16.5872 ± 0.2966 | 47.6300 ± 0.2532 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | residual_flat | 3 | 0.8193 ± 0.0148 | 0.8759 ± 0.0154 | 0.9551 ± 0.0045 | 17.2548 ± 1.3219 | 45.8598 ± 2.6145 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | residual_without_pose_aux | 3 | 0.8267 ± 0.0104 | 0.8763 ± 0.0095 | 0.9535 ± 0.0025 | — | — | — |
| primary_t_plus_1_future_wrist | modality_no_gaze | 3 | 0.7995 ± 0.0145 | 0.8715 ± 0.0100 | 0.9540 ± 0.0121 | 15.9642 ± 0.4273 | 43.2840 ± 1.5778 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | modality_no_hands | 3 | 0.7344 ± 0.0301 | 0.8447 ± 0.0249 | 0.6021 ± 0.1054 | 19.3206 ± 0.4133 | 61.4648 ± 3.4187 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | modality_no_objects | 3 | 0.8465 ± 0.0152 | 0.8950 ± 0.0108 | 0.9598 ± 0.0070 | 15.8495 ± 0.8513 | 44.5420 ± 1.7235 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | modality_no_vio | 3 | 0.7935 ± 0.0173 | 0.8595 ± 0.0098 | 0.9242 ± 0.0119 | 16.3246 ± 0.4428 | 46.0650 ± 1.9258 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | visual_corrected_clip_current_gate | 3 | 0.8465 ± 0.0111 | 0.9135 ± 0.0090 | 0.9336 ± 0.0135 | 16.0349 ± 0.5971 | 45.8275 ± 1.3940 | 0.8341 ± 0.0000 |
| primary_t_plus_1_future_wrist | visual_corrected_clip_modality_gate | 3 | 0.8630 ± 0.0107 | 0.9203 ± 0.0065 | 0.8683 ± 0.0658 | 16.8916 ± 0.8792 | 48.5877 ± 5.3698 | 0.8341 ± 0.0000 |
| secondary_terminal_endpose | terminal_endpose_learned | 3 | 0.8271 ± 0.0309 | 0.8822 ± 0.0168 | 0.9544 ± 0.0080 | 19.8075 ± 0.5203 | 56.1830 ± 2.2233 | 0.8517 ± 0.0000 |

## Secondary terminal/endpose paired diagnostic

The overview uses the strictly-before-aggregation (pure-future) regime. The pooled row below is diagnostic because it mixes pure forecasting and partial-target-evidence estimation.

| Experiment | Learned GT-hand mean cm | Learned end-to-end mean cm | Persistence mean cm | Shared samples | Coverage |
|---|---:|---:|---:|---:|---:|
| terminal_endpose_learned | 18.3365 ± 1.0141 | 18.6102 ± 0.5416 | 16.6532 ± 0.0000 | 221.0000 ± 0.0000 | 0.8500 ± 0.0000 |

| Experiment | Target regime | Learned GT-hand mean cm | Learned end-to-end mean cm | Persistence (GT-hand) mean cm | Shared samples | Coverage |
|---|---|---:|---:|---:|---:|---:|
| terminal_endpose_learned | pure future | 19.6777 ± 0.9823 | 19.8075 ± 0.5203 | 18.2506 ± 0.0000 | 201.0000 ± 0.0000 | 0.8517 ± 0.0000 |
| terminal_endpose_learned | partial target evidence | 4.8575 ± 1.4248 | 6.5772 ± 1.4829 | 0.5988 ± 0.0000 | 20.0000 ± 0.0000 | 0.8333 ± 0.0000 |

## t+1 paired baseline availability

Persistence/constant-velocity/learned values are included only when a checkpoint-bound grouped prediction report exists. Missing sidecars remain unavailable and are not estimated from other reports.

| Experiment | Bound seed reports | Learned (GT hand) mean cm | Persistence mean cm | Constant velocity mean cm |
|---|---:|---:|---:|---:|
| baseline_mlp | 0/3 | — | — | — |
| baseline_gru | 0/3 | — | — | — |
| baseline_transformer | 0/3 | — | — | — |
| residual_current_gate | 3/3 | 14.9033 ± 0.3078 | 14.8042 ± 0.0000 | 33.4904 ± 0.0000 |
| residual_simple_gate | 0/3 | — | — | — |
| residual_modality_gated | 3/3 | 15.2408 ± 0.1874 | 14.8042 ± 0.0000 | 33.4904 ± 0.0000 |
| visual_corrected_random_current_gate | 0/3 | — | — | — |
| residual_flat | 0/3 | — | — | — |
| residual_without_pose_aux | 0/3 | — | — | — |
| modality_no_gaze | 0/3 | — | — | — |
| modality_no_hands | 0/3 | — | — | — |
| modality_no_objects | 0/3 | — | — | — |
| modality_no_vio | 0/3 | — | — | — |
| visual_corrected_clip_current_gate | 0/3 | — | — | — |
| visual_corrected_clip_modality_gate | 3/3 | 15.2716 ± 0.4025 | 14.8042 ± 0.0000 | 33.4904 ± 0.0000 |

## Full intention -> hand -> t+1 pose cascade

Every rate uses the same ground-truth handover windows with a valid t+1 receiving-wrist target. Pose thresholds are strict Euclidean errors (`error < threshold`) from the learned predicted-hand output.

| Experiment | Bound seeds | Handover correct | + correct hand | Success@20 cm | Success@15 cm | Success@10 cm | Success@5 cm | Evaluable windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_current_gate | 3/3 | 0.8986 ± 0.0092 | 0.8848 ± 0.0092 | 0.6406 ± 0.0092 | 0.5899 ± 0.0046 | 0.5207 ± 0.0122 | 0.3318 ± 0.0997 | 217.0000 ± 0.0000 |
| residual_modality_gated | 3/3 | 0.9478 ± 0.0266 | 0.9247 ± 0.0349 | 0.6482 ± 0.0270 | 0.5975 ± 0.0148 | 0.4854 ± 0.0186 | 0.2381 ± 0.0692 | 217.0000 ± 0.0000 |
| visual_corrected_clip_modality_gate | 3/3 | 0.9309 ± 0.0440 | 0.8618 ± 0.0697 | 0.6190 ± 0.0662 | 0.5760 ± 0.0679 | 0.4624 ± 0.0820 | 0.2949 ± 0.1004 | 217.0000 ± 0.0000 |

## Retrospective causal intention baselines

These descriptive baselines fit train windows only. No test metric selected features, hyperparameters, or a model.

| Method | Test accuracy | Test macro-F1 | Continue F1 | Fetch F1 | Handover F1 |
|---|---:|---:|---:|---:|---:|
| majority_class | 0.7058 | 0.2758 | 0.8275 | 0.0000 | 0.0000 |
| elapsed_time_since_start_logistic | 0.7294 | 0.4660 | 0.8742 | 0.0000 | 0.5239 |
| last_sensor_frame_logistic | 0.8558 | 0.7884 | 0.9156 | 0.7355 | 0.7141 |

## Participant-balanced LOPO generalisation

Fixed-test and LOPO estimates are separate evidence. Hand metrics retain all three predeclared interpretations.

| LOPO metric | Participant-balanced estimate |
|---|---:|
| Intention accuracy | 0.8629 |
| Intention macro-F1 | 0.8152 |
| Receiving hand, fixed two-class/all participants | 0.6011 |
| Receiving hand, supported classes/all participants | 0.9579 |
| Receiving hand, fixed two-class/mixed-hand participants | 0.8723 |

## Dataset split and participant–hand confounding

Frozen participant-disjoint split: 170 train / 21 validation / 23 test sequences (19 / 3 / 3 participants; 25 total).
Participant × receiving-hand Cramér's V: 0.7620; participant-majority-hand accuracy: 0.8505; mixed-hand participants: 7.

Identity-provenance warning: Confirm manually that 'Test' is an intentional participant pseudonym rather than technical test data.

## Empirical sampling and observation duration

Median Δt is 0.033333 s (30.0003 Hz). A 60-sample window spans 59 intervals and has a measured median duration of 1.966667 s (IQR 0.000001 s).

## Pose-loss decision and qualitative evidence

The checklist's 14–15 cm train-underfitting trigger is not met. Existing curves do not justify a new normalized-loss run; checkpoint-selection timing explains part of the pose gap where the validation pose optimum occurs after best_intention.
Normalized Smooth-L1 sensitivity run recommended: false.
Three device-time-v2 overlays are hash-bound and synchronized: Jona_7_20260616_182214, Edu_3_20260604_170622, Mona_6_20260624_123930.

## Offline versus deployment reporting

All values above are offline model/window or retrospective grouped metrics. No checkpoint-bound replay report is included; these values must not be presented as raw, stable, or actionable deployment performance.
