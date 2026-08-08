# Residual-v2 Hyperparameterbestätigung (`n214`)

Status: vollständig, 9/9 Läufe erfolgreich

SLURM-Job: `2204122`

Auswahl-Split: ausschließlich Validation; alle neun Läufe wurden mit
`--skip-test-evaluation` ausgeführt und enthalten keine Testmetriken.

| Trial | Val. Intention Macro-F1 | Val. Hand Macro-F1 | Val. Pose-MAE | Parameter |
|---|---:|---:|---:|---:|
| `trial_022` | **0,9311 ± 0,0064** | 0,9964 ± 0,0029 | 8,794 ± 0,695 cm | **63.023** |
| `trial_020` | 0,9272 ± 0,0012 | 0,9985 ± 0,0012 | 9,604 ± 1,357 cm | 531.343 |
| `trial_001` | 0,9254 ± 0,0013 | 0,9985 ± 0,0012 | **8,588 ± 0,278 cm** | 662.927 |

Angegeben sind Mittelwert ± Populationsstandardabweichung über Seeds 42, 43
und 44 am jeweils anhand Validation gewählten Best-Intention-Checkpoint.

## Eingefrorene Auswahl

`trial_022` wurde gewählt. Die Regel maximiert zunächst den mittleren
Validation-Intention-Macro-F1, behält Varianten innerhalb von 0,005 des besten
Werts und verwendet dann Pose-MAE, Hand-F1 und Parameterzahl als vorab
festgelegte Tie-Breaker. `trial_022` besitzt zugleich die höchste primäre
Metrik und mit großem Abstand die wenigsten Parameter.

Die eingefrorene Konfiguration liegt in `selected_config.json`. Ihre zentralen
Werte sind:

- `d_model=32`, acht Attention-Heads, eine Transformer-Schicht
- Feedforward-Dimension 256, Dropout 0,15
- Batchgröße 64, Learning Rate 0,000381605632501, Weight Decay 0,0001
- Receiving-Hand-Lossgewicht 2,0, Orientierungs-Lossgewicht 0,5

Maschinenlesbare Quellen:

- `summary.json`
- `selected_config.json`
- `data/confirmation_runs.csv`
- `data/confirmation_summary.csv`
