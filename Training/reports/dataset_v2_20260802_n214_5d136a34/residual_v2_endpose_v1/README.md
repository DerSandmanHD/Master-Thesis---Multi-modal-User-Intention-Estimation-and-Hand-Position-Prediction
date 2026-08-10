# Terminal end-pose experiment (n214)

Status: **complete**.

The existing Residual-v2 model predicts the receiving-hand pose at **t+1 second**. The new, separate experiment predicts one **robust terminal handover pose**, formed from the latest stable 0.5-second receiving-hand segment after `THIRD`. Existing t+1 runs and checkpoints were not modified.

## Target audit

Accepted terminal targets: **208/214 (97.2%)**. Rejected sequences remain in the intent/hand tasks but do not contribute pose loss.

## Test comparison on the same terminal target

| Model | Intent macro-F1 | Hand macro-F1 | Position error (cm) | Orientation error (deg) | Target coverage | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| t+1 model (as terminal baseline) | 0.863 ± 0.004 | 0.935 ± 0.037 | 16.00 ± 0.74 | 49.12 ± 3.35 | 92.8% | 63,023 |
| Terminal endpose model | 0.847 ± 0.007 | 0.905 ± 0.050 | 19.11 ± 2.30 | 53.96 ± 4.82 | 92.8% | 63,023 |

Values are mean ± population standard deviation across seeds 42, 43 and 44. Intent and hand use the best-intention checkpoint; pose uses the best-pose checkpoint. Both checkpoints were selected exclusively on validation.

## Result

The dedicated terminal model did **not** improve the aggregate terminal-pose result. Relative to the existing t+1 checkpoint evaluated against the same terminal target, its position error is 3.10 cm higher and its orientation error is 4.85° higher. Intent macro-F1 changes by -0.016 and receiving-hand macro-F1 by -0.030.

The remaining-time analysis reveals a narrower benefit: at **>=3 seconds** before sequence end, the terminal model reaches 25.84 cm / 73.07° versus 27.74 cm / 83.72° for the t+1 baseline. The t+1 baseline is better in every bin from 0 to 3 seconds. Thus the terminal objective shows some long-horizon anticipation, but the overall hypothesis is not supported by this experiment.

For context only, the original model's native t+1 position error was 15.21 ± 0.44 cm. This native metric has a different target and must not be compared directly with terminal pose error; the table above re-evaluates that checkpoint on the shared terminal target.

## Latency

| Model | TCML CPU mean (ms) | TCML CUDA mean (ms) |
|---|---:|---:|
| t+1 model (as terminal baseline) | 1.857 | 1.582 |
| Terminal endpose model | 1.820 | 1.609 |

Both models use the same Residual-v2 architecture; only the learned target and weights differ.

## Files

- `data/model_runs.csv`: all seed-level metrics
- `data/model_summary.csv`: mean and standard deviation
- `data/error_vs_time_remaining.csv`: terminal error by remaining-time bin
- `comparison.json`: machine-readable complete report
- `figures/`: matching PNG and PDF figures
