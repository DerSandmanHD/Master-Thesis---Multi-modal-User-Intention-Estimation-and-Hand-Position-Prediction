# CLIP-Embedding-Screening (Validation only)

Dataset: `dataset_v2_20260802_n214_5d136a34`

Seeds: 42, 43, 44

Auswahl-Split: ausschließlich Validation

Der Vergleich verwendet die auf Validation ausgewählte Residual-v2-
Konfiguration (`trial_022`). CLIP ViT-B/32 ist eingefroren, wird mit 5 Hz
abgetastet und über eine ausschließlich auf dem Trainingssplit angepasste
PCA-Projektion von 512 auf 32 Dimensionen reduziert.

| Variante | Val. Intent Macro-F1 | Val. Hand Macro-F1 | Val. Pose-MAE |
|---|---:|---:|---:|
| Sensor-Baseline | 0,9311 ± 0,0064 | 0,9964 ± 0,0029 | 8,79 ± 0,69 cm |
| CLIP-only | 0,6155 ± 0,0119 | 0,9893 ± 0,0022 | 11,31 ± 0,39 cm |
| Sensor + CLIP | **0,9391 ± 0,0054** | **0,9970 ± 0,0022** | 8,85 ± 0,55 cm |
| Sensor + Random-Control | 0,9178 ± 0,0099 | 0,9959 ± 0,0047 | **8,46 ± 0,37 cm** |

Nach der vorab festgelegten Auswahlregel wurde `sensor_plus_clip` für die
einmalige finale Testauswertung ausgewählt. Die zufälligen Features sind nur
eine diagnostische Kontrolle und waren nicht auswählbar.

Alle neun Screening-Runs enthalten `test_evaluation_skipped: true` und kein
`test`-Feld. Die maschinenlesbaren Einzel- und Aggregatwerte liegen unter
`data/`, die Entscheidung unter `summary.json` und die Abbildung unter
`figures/`. Provenienz und Hashes des visuellen Caches stehen im benachbarten
Bericht `../visual_embedding_cache_v1/`.
