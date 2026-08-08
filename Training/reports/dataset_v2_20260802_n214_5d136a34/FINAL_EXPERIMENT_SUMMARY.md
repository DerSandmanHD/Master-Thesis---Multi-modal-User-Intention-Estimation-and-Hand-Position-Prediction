# Abschlusszusammenfassung der n214-Experimente

Dataset: `dataset_v2_20260802_n214_5d136a34`

Seeds: 42, 43, 44

Auswahl: ausschließlich Validation; keine Testmetrik wurde zur Auswahl verwendet

## Zentrale Testergebnisse

| Modell | Intent Macro-F1 | Accuracy | Hand Macro-F1 | Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| Residual v2 (ursprünglich) | 0.8579 ± 0.0012 | 0.9031 ± 0.0032 | 0.9567 ± 0.0035 | 14.62 ± 0.11 cm | 184,015 |
| Residual v2 (getunt) | 0.8631 ± 0.0039 | 0.9065 ± 0.0044 | 0.9349 ± 0.0374 | 15.21 ± 0.44 cm | 63,023 |
| Sensor + CLIP | 0.8405 ± 0.0082 | 0.8892 ± 0.0030 | 0.9216 ± 0.0349 | 14.88 ± 0.18 cm | 67,119 |

Das Tuning verbessert den Intentions-Macro-F1 um
`0.0052`
bei 63,023 statt 184,015
Parametern, verschlechtert jedoch Hand-F1 und Pose-MAE. Sensor+CLIP gewinnt
das Validation-Screening, überträgt den Gewinn aber nicht auf Test:
`-0.0226`
Intentions-F1 gegenüber der getunten Sensor-Baseline. Der Best-Pose-MAE
verbessert sich dabei um
`0.33 cm`.

## Hyperparametersuche

- Stufe A: 24 vollständige, testfreie Random-Search-Trials.
- Stufe B: drei Konfigurationen × drei Seeds, ebenfalls testfrei.
- Gewinner: `trial_022` mit Validation-F1
  `0.9311 ± 0.0064`.
- Architektur: `d_model=32`, 8 Heads, 1 Layer, Feedforward 256,
  Dropout 0,15, Batchgröße 64, Learning Rate 0,0003816056,
  Hand-Lossgewicht 2,0 und Orientierungs-Lossgewicht 0,5.

## CLIP und visuelle Features

- Frozen OpenAI CLIP ViT-B/32 bei 5 Hz: 36,874
  Embeddings aus 214 Sequenzen, keine Cachefehler.
- PCA 512→32 ausschließlich auf 28,996 Samples aus
  170 Trainingssequenzen; erklärte Varianz
  `0.787`.
- Validation-F1: Sensor `0.9311`,
  CLIP-only `0.6155`,
  Sensor+CLIP `0.9391`,
  Random-Control `0.9178`.

## Ablationen

| Modell | Intent Macro-F1 | Accuracy | Hand Macro-F1 | Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| Residual v2 (ursprünglich) | 0.8579 ± 0.0012 | 0.9031 ± 0.0032 | 0.9567 ± 0.0035 | 14.62 ± 0.11 cm | 184,015 |
| Ablation ohne Gaze | 0.7946 ± 0.0341 | 0.8455 ± 0.0284 | 0.9638 ± 0.0089 | 14.29 ± 0.36 cm | 176,847 |
| Ablation ohne Handfeatures | 0.7519 ± 0.0282 | 0.8304 ± 0.0208 | 0.4193 ± 0.0048 | 16.53 ± 0.55 cm | 179,407 |
| Ablation ohne Objektfeatures | 0.8845 ± 0.0019 | 0.9200 ± 0.0032 | 0.9814 ± 0.0084 | 14.26 ± 0.23 cm | 170,191 |
| Ablation ohne VIO | 0.8222 ± 0.0159 | 0.8777 ± 0.0072 | 0.9034 ± 0.0522 | 15.10 ± 0.95 cm | 181,711 |

Handfeatures sind für Intentions- und Handklassifikation am wichtigsten
(`no_hands`: ΔF1 `-0.1060`).
Das positive `no_objects`-Testdelta ist deskriptiv und wird nicht nachträglich
zur Architekturauswahl verwendet.

## Latenz und Visualisierung

- Fünf identische Modellbenchmarks: alle 5.000 Offlinefenster unter 33,3 ms.
- Modell-Forward-Median: 1,112 ms (Mac CPU) bis 2,602 ms (Mac MPS).
- Separates RGB→CLIP-Median: 25,943 ms (TCML CUDA) bis 74,393 ms
  (TCML CPU), jeweils vollständig innerhalb des 200-ms-/5-Hz-Budgets.
- Drei H.264-Overlays plus Thesis-Stills; streng kausal synchronisiert und
  insgesamt 0 Future-Matches.

## Final eingefrorener Studiencheckpoint

`Training/runs/dataset_v2_20260802_n214_5d136a34/visual_embedding_final_v1/sensor_plus_clip/visual_embedding_final_v1_sensor_plus_clip_seed42/best_intention_model.pt`

SHA-256: `c9de5f091b1230bd0117a99a3fbbd69ae2c28ac67353fc5e644bec56bf73967b`

Die Wahl von Sensor+CLIP/Seed 42 bleibt trotz des später beobachteten
Testverlusts unverändert, weil Architektur und Seed ausschließlich auf
Validation festgelegt wurden.

## Grenzen

- Kein belastbarer 3D-in-RGB-Poseplot; stattdessen validiertes Robot-Frame-Inset.
- Keine neue Live-End-to-End-Aufnahme des finalen Checkpoints ohne physisch
  angeschlossene Aria-Brille; vorhandene Mac-Live-Sitzungen bleiben separat
  als explorative Messung dokumentiert.
- CLIP-Frontend und zeitliches Modell haben unterschiedliche Taktraten und
  werden daher getrennt berichtet.

Maschinenlesbare Quelle: `FINAL_EXPERIMENT_SUMMARY.json` und
`final_test_metrics.csv`. Alle Werte werden beim Erzeugen gegen die
zugrunde liegenden Reports und Provenienzhashes geprüft.
