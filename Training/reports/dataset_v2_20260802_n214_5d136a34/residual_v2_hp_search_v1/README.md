# Residual-v2-Hyperparametersuche – Stage A

- Dataset: `dataset_v2_20260802_n214_5d136a34`
- SLURM-Array: `2204077`
- Such-Seed / Trainings-Seed: `20260808` / `42`
Trainingscode: Commit `7d168e7aecd3796b80de67475517bf721396bf87`

Alle 24 vorab erzeugten Random-Search-Konfigurationen wurden erfolgreich
trainiert. Die Auswertung enthält ausschließlich Validation-Metriken;
`test_evaluation_skipped=true` wurde für jeden Lauf geprüft und kein Trial
enthält einen `test`-Block.

## Für Stage B ausgewählt

| Rang | Trial | Val. Intention Macro-F1 | Val. Pose-MAE am Best-Intention-Checkpoint | Val. Hand Macro-F1 | Parameter |
|---:|---|---:|---:|---:|---:|
| 1 | `trial_022` | 0,93383 | 8,820 cm | 0,99848 | 63.023 |
| 2 | `trial_001` | 0,93057 | 9,287 cm | 0,99848 | 662.927 |
| 3 | `trial_020` | 0,92898 | 10,059 cm | 0,99848 | 531.343 |

Die Auswahl folgt der vorab festgelegten Regel: höchste Validation-Intention-
Macro-F1; Kandidaten innerhalb von 0,005 bleiben im F1-Toleranzband, danach
entscheiden Pose-MAE am Best-Intention-Checkpoint, Hand-F1 und Parameterzahl.
Die Einzel-Seed-Werte sind noch keine finale Modellentscheidung; genau diese
drei Trials werden in Stage B mit Seeds 42, 43 und 44 bestätigt.

Maschinenlesbare Quellen:

- `summary.json`: Vollständigkeit, Auswahlregel und ausgewählte Trials
- `data/stage_a_trials.csv`: alle 24 Läufe in Manifestreihenfolge
- `data/stage_a_ranking.csv`: vollständiges Ranking und Hyperparameter
- `figures/01_validation_pareto.*`: F1/Pose/Hand/Parameter-Paretoansicht
- `figures/02_hyperparameter_effects.*`: einzelne Parametereffekte
- `figures/03_parallel_coordinates.*`: gemeinsamer Suchraum

Der Repository-Status der Clusterläufe war wegen bereits vorhandener,
experimentfremder QA-/Datenartefakte als `dirty` protokolliert. Der exakte
Trainingscommit, jede Trial-Konfiguration, deren SHA-256, der Dataset-
Fingerprint sowie sämtliche Laufmetriken sind dennoch je Run gespeichert.
