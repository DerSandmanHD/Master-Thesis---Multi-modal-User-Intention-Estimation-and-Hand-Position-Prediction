# Finale Auswertung der getunten Residual-v2-Sensor-Baseline

Dataset: `dataset_v2_20260802_n214_5d136a34`

Seeds: 42, 43, 44

Hyperparameterauswahl: ausschließlich Validation

| Modell | Test Intent Macro-F1 | Test Accuracy | Test Hand Macro-F1 | Test Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| ursprüngliche Residual-v2-Baseline | 0,8579 ± 0,0012 | 0,9031 ± 0,0032 | **0,9567 ± 0,0035** | **14,62 ± 0,11 cm** | 184.015 |
| getunte Residual-v2-Variante | **0,8631 ± 0,0039** | **0,9065 ± 0,0044** | 0,9349 ± 0,0374 | 15,21 ± 0,44 cm | **63.023** |

Das Tuning erhöht den primären Intentions-Macro-F1 um `+0,0052` und reduziert
die Parameterzahl um 120.992 (`-65,8 %`). Der Gewinn überträgt sich jedoch
nicht auf alle Köpfe: Hand-F1 fällt um `0,0218`, und der Pose-MAE des
Validation-selektierten Best-Pose-Checkpoints verschlechtert sich um
`0,58 cm`. Dieser Zielkonflikt wird deshalb gemeinsam mit dem F1-Gewinn
berichtet.

`summary.json` und `data/` enthalten die maschinenlesbaren Werte. Die
Abbildungen unter `figures/` zeigen Mittelwert ± Populationsstandardabweichung
sowie den gepaarten Vergleich der identischen Seeds.
