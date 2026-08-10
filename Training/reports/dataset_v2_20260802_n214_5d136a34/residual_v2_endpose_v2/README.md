# Improved terminal end-pose experiment v2 (n214)

Status: **complete**.

Endpose-v2 keeps the exact robust terminal target and participant split from v1, but adds train-fitted position scaling, geodesic orientation loss, sequence/time-bin balancing, hand-specific residuals, and an auxiliary t+1 head. Hyperparameters and checkpoints were selected using validation only.

Target audit: **208/214 (97.2%)** stable handover sequences.

| Model | Intent F1 | Hand F1 | Position (cm) | Orientation (deg) | Coverage | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| t+1 as terminal baseline | 0.863 ± 0.004 | 0.935 ± 0.037 | 16.00 ± 0.74 | 49.12 ± 3.35 | 92.8% | 63,023 |
| Terminal endpose v1 | 0.847 ± 0.007 | 0.905 ± 0.050 | 19.11 ± 2.30 | 53.96 ± 4.82 | 92.8% | 63,023 |
| Improved dual-horizon endpose v2 | 0.843 ± 0.012 | 0.933 ± 0.009 | 16.30 ± 1.34 | 42.79 ± 1.61 | 92.8% | 65,860 |

## Result

Endpose-v2 is **better** than endpose-v1 on aggregate terminal position: change -2.81 cm; orientation change -11.17°. Negative changes are improvements.

Compared with the t+1 model evaluated as a terminal predictor, endpose-v2 changes position by +0.29 cm, orientation by -6.33°, intent macro-F1 by -0.020, and receiving-hand macro-F1 by -0.002. Thus its aggregate position is effectively similar, while terminal orientation is substantially better; intent remains slightly weaker.

The full remaining-time curves are in `data/error_vs_time_remaining.csv` and `figures/02_error_vs_time_remaining.{png,pdf}`.

## Validation-selected configuration

The validation-only search selected **trial_003** with 11.17 ± 0.22 cm over seeds 42, 43 and 44. Test metrics were not used for selection.

| Hyperparameter | Value |
|---|---:|
| Model dimension | 32 |
| Transformer layers / heads | 1 / 8 |
| Feed-forward dimension | 256 |
| Dropout | 0.3 |
| Batch size | 32 |
| Learning rate | 0.000182289 |
| Terminal pose loss weight | 4.0 |
| Orientation loss weight | 0.1 |
| Auxiliary t+1 loss weight | 0.25 |

## Latency

| Model | TCML CPU mean (ms) | TCML CUDA mean (ms) |
|---|---:|---:|
| t+1 as terminal baseline | 1.857 | 1.582 |
| Terminal endpose v1 | 1.820 | 1.609 |
| Improved dual-horizon endpose v2 | 3.181 | 3.026 |

All test values are mean ± population standard deviation over seeds 42, 43 and 44.
