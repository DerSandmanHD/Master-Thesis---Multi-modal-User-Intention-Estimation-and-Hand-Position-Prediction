# Sensor-Modalitätsablation Residual v2 (`n214`)

Status: vollständig, 12/12 Trainingsläufe und 10/10 Latenzmessungen erfolgreich

SLURM-Jobs: Training `2204078`, Latenz `2204124`

Seeds: 42, 43, 44

Split: unveränderter participant-wise Split von
`dataset_v2_20260802_n214_5d136a34`

## Checkpoint- und Metrikregel

Intentions- und Receiving-Hand-Metriken stammen aus dem anhand der Validation
gewählten `best_intention`-Checkpoint. Der primäre Pose-MAE stammt aus dem
separat anhand Validation gewählten `best_pose`-Checkpoint. Zusätzlich enthält
die CSV den Posefehler des `best_intention`-Checkpoints, damit die Leistung des
einzelnen späteren Deployment-Checkpoints nicht mit der dedizierten
Poseauswertung verwechselt wird.

Alle Fehlerbalken sind Populationsstandardabweichungen über die drei Seeds.
Mit drei Seeds und einem einzigen fixen Participant-Testsplit werden keine
Signifikanzbehauptungen abgeleitet.

## Ergebnis

| Variante | Intent Macro-F1 | Δ F1 vs. Full | Hand Macro-F1 | Pose-MAE, best pose | Δ Pose |
|---|---:|---:|---:|---:|---:|
| Full | 0,8579 | 0,0000 | 0,9567 | 14,624 cm | 0,000 cm |
| ohne Gaze | 0,7946 | −0,0633 | 0,9638 | 14,288 cm | −0,337 cm |
| ohne Hände | 0,7519 | −0,1060 | 0,4193 | 16,527 cm | +1,903 cm |
| ohne Objekte | 0,8845 | +0,0266 | 0,9814 | 14,255 cm | −0,369 cm |
| ohne VIO | 0,8222 | −0,0357 | 0,9034 | 15,099 cm | +0,474 cm |

Die Handfeatures tragen am stärksten zur Gesamtaufgabe bei: Ohne sie fallen
sowohl Intentions- als auch Handklassifikation deutlich ab, und der
Posefehler steigt. Gaze und VIO liefern ebenfalls relevante Information für
die Intention. Dass das Entfernen der Objektfeatures auf dem bestehenden
Testsplit alle drei Hauptmetriken verbessert, ist ein Hinweis auf potenziell
rauschende oder überangepasste Objektfeatures.

Dieses `no_objects`-Resultat wird **nicht** nachträglich zur Auswahl der finalen
Architektur verwendet: Die Testwerte wurden im Rahmen der vorab festgelegten
Ablation bereits geöffnet. Eine belastbare Modellentscheidung dazu benötigt
einen neuen Holdout oder ein verschachteltes Cross-Validation-Protokoll.

Die Trainingslaufzeit der älteren Full-Baseline wurde im damaligen
`metrics.json` noch nicht gespeichert und ist deshalb im Effizienzdiagramm als
`n/a` markiert. Parameterzahlen sind vollständig.

## Parameter und Modelllatenz

Alle Varianten wurden anschließend auf demselben realen Testfenster
`Edu_10_20260616_150412`, Timestamp `3106145555000`, mit Batchgröße 1, 100
Warm-ups und 1.000 synchronisierten Messungen geprüft. Die verschiedenen
Dataset-Fingerprints sind beabsichtigt: Sie kodieren die jeweils entfernten
Feature-Spalten; Sequenz, Timestamp und Split-Index stimmen überein.

| Variante | Parameter | TCML CPU Median | TCML CUDA Median |
|---|---:|---:|---:|
| Full | 184.015 | 2,873 ms | 3,230 ms |
| ohne Gaze | 176.847 | 2,826 ms | 3,750 ms |
| ohne Hände | 179.407 | 2,858 ms | 3,723 ms |
| ohne Objekte | 170.191 | 2,563 ms | 3,317 ms |
| ohne VIO | 181.711 | 2,849 ms | 3,575 ms |

Alle 10.000 Forward-Messungen blieben unter der vorab festgelegten
33,33-ms-Grenze; das höchste P95 beträgt 4,292 ms. Dass CUDA bei diesen sehr
kleinen Modellen und Batchgröße 1 nicht schneller ist, ist plausibel durch
Kernel-Start- und Synchronisationskosten und keine allgemeine Aussage gegen
GPU-Inferenz.

Maschinenlesbare Quellen:

- `summary.json`
- `data/ablation_runs.csv`
- `data/ablation_summary.csv`
- `figures/01_sensor_ablation_metrics.{png,pdf}`
- `figures/02_sensor_ablation_efficiency.{png,pdf}`
- `latency/ablation_latency_summary.csv`
- `latency/latency_summary.json`
- `latency/figures/01_ablation_latency.{png,pdf}`
