# Methoden- und Ergebnisentwurf: finale n214-Experimente

Stand: 8. August 2026

Dieses Dokument ist ein direkt in die Thesis übertragbarer Entwurf. Alle
Zahlen stammen aus versionierten JSON-/CSV-Berichten; die zentrale
Auswertung prüft ihre Vollständigkeit und Provenienz automatisch.

## 1. Datensatz und Auswertungsprotokoll

Verwendet wurde der eingefrorene Datensatz
`dataset_v2_20260802_n214_5d136a34` mit 214 qualitätsgeprüften Sequenzen. Der
Split erfolgt teilnehmerweise und bleibt für alle Experimente unverändert:

| Split | Sequenzen | Fenster | Teilnehmer |
|---|---:|---:|---|
| Training | 170 | 15.189 | 19 |
| Validation | 21 | 1.978 | Atilla, Ermal, Vanessa |
| Test | 23 | 2.199 | Edu, Jona, Mona |

Jedes Modell verwendet ein Beobachtungsfenster von 60 Frames, entsprechend
ungefähr zwei Sekunden, und prognostiziert mit einem Horizont von einer
Sekunde. Die Hauptmetrik ist der Macro-F1 der drei Intentionen `continue`,
`fetch` und `handover`. Zusätzlich werden Accuracy, Receiving-Hand-Macro-F1
und der mittlere euklidische Positionsfehler der Handpose in Zentimetern
berichtet. Finale Vergleiche verwenden die Seeds 42, 43 und 44 und geben
Mittelwert plus Populationsstandardabweichung an.

Alle Hyperparameter- und Architekturentscheidungen wurden ausschließlich auf
Validation getroffen. Der Testsplit wurde erst nach dem Einfrieren der
jeweiligen Entscheidung ausgewertet. Damit verhindert das Protokoll eine
nachträgliche Optimierung auf den Testdaten.

## 2. Sensorbasierte Baseline und Hyperparametersuche

Die ursprüngliche Residual-v2-Baseline besitzt 184.015 trainierbare
Parameter. Eine reproduzierbare Random Search umfasste 24 testfreie Trials.
Die besten drei Konfigurationen wurden anschließend mit den Seeds 42, 43 und
44 ebenfalls ohne Testauswertung bestätigt. Die vorab festgelegte Auswahlregel
maximierte zunächst Validation-Intent-Macro-F1, verwendete innerhalb einer
Toleranz von 0,005 den Posefehler als Tie-Breaker und danach Hand-F1 und
Parameterzahl.

Die gewählte Konfiguration `trial_022` verwendet `d_model=32`, acht
Attention-Heads, einen Transformer-Layer, eine Feedforward-Dimension von 256,
Dropout 0,15, Batchgröße 64, Learning Rate 0,0003816056, Weight Decay 0,0001,
Receiving-Hand-Lossgewicht 2,0 und Orientierungs-Lossgewicht 0,5. Sie erreichte
auf Validation `0,9311 ± 0,0064` Intent-Macro-F1.

| Modell | Test Intent-F1 | Accuracy | Hand-F1 | Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| Residual v2, ursprünglich | 0,8579 ± 0,0012 | 0,9031 ± 0,0032 | 0,9567 ± 0,0035 | 14,62 ± 0,11 cm | 184.015 |
| Residual v2, getunt | **0,8631 ± 0,0039** | **0,9065 ± 0,0044** | 0,9349 ± 0,0374 | 15,21 ± 0,44 cm | **63.023** |

Das Tuning verbessert den Intent-Macro-F1 um 0,0052 und reduziert die
Parameterzahl um 120.992 beziehungsweise 65,8 %. Gleichzeitig sinkt der
Hand-F1 um 0,0218 und der Posefehler steigt um 0,58 cm. Das Ergebnis ist daher
kein einheitlicher Gewinn über alle Aufgaben, sondern ein Trade-off zugunsten
von Intentionsklassifikation und Modellgröße.

Der frühere n156-Benchmark erreichte `0,8620 ± 0,0141` Intent-Macro-F1 und
`14,57 ± 1,22 cm` Pose-MAE. n214 erhöht den Mittelwert nicht eindeutig, senkt
aber vor allem die Streuung zwischen Seeds. Mehr Daten verbesserten in diesem
Vergleich daher primär die Stabilität statt die durchschnittliche Güte.

## 3. Sensorablationen

Jede Ablation entfernt genau eine Eingangsmodalität aus der ursprünglichen
Residual-v2-Architektur. Datensatz, Split, Seeds und Trainingsparameter bleiben
gleich. Bei `no_hands` werden Handfeatures nur aus dem Eingang entfernt; die
Ground-Truth-Handreferenz bleibt notwendiges Trainingsziel der Poseaufgabe.

| Variante | Intent-F1 | Delta zu Full | Hand-F1 | Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| Full | 0,8579 ± 0,0012 | 0,0000 | 0,9567 ± 0,0035 | 14,62 ± 0,11 cm | 184.015 |
| ohne Gaze | 0,7946 ± 0,0341 | -0,0633 | 0,9638 ± 0,0089 | 14,29 ± 0,36 cm | 176.847 |
| ohne Handfeatures | 0,7519 ± 0,0282 | **-0,1060** | 0,4193 ± 0,0048 | 16,53 ± 0,55 cm | 179.407 |
| ohne Objektfeatures | 0,8845 ± 0,0019 | +0,0266 | 0,9814 ± 0,0084 | 14,26 ± 0,23 cm | 170.191 |
| ohne VIO | 0,8222 ± 0,0159 | -0,0357 | 0,9034 ± 0,0522 | 15,10 ± 0,95 cm | 181.711 |

Handfeatures tragen am stärksten zur Intentions- und Handklassifikation bei.
Gaze und VIO liefern ebenfalls positive Beiträge. Das bessere Testergebnis von
`no_objects` wird nur deskriptiv berichtet: Da es erst auf dem Testsplit
sichtbar wurde, darf es nicht nachträglich zur Auswahl einer neuen Architektur
verwendet werden. Für eine belastbare Entscheidung wäre ein neues,
vorregistriertes Experiment mit zusätzlichem Holdout erforderlich.

## 4. Visuelle CLIP-Features

Die visuelle Erweiterung folgt den Versuchsprinzipien aus
[CLIP](https://proceedings.mlr.press/v139/radford21a.html),
[EgoVLP](https://proceedings.neurips.cc/paper_files/paper/2022/file/31fb284a0aaaad837d2930a610cd5e50-Paper-Conference.pdf),
[LaViLa](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Learning_Video_Representations_From_Large_Language_Models_CVPR_2023_paper.pdf),
[AVT](https://openaccess.thecvf.com/content/ICCV2021/papers/Girdhar_Anticipative_Video_Transformer_ICCV_2021_paper.pdf),
[RU-LSTM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Furnari_What_Would_You_Expect_Anticipating_Egocentric_Actions_With_Rolling-Unrolling_LSTMs_ICCV_2019_paper.pdf)
und dem
[Object-Centric Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.pdf):
sparse und strikt kausale RGB-Frames, ein zunächst eingefrorener Encoder,
explizite Einzelablationen und getrennte Validation-/Testentscheidungen.

OpenAI CLIP ViT-B/32 wurde eingefroren und lokal mit 5 Hz ausgeführt. Der
Cache umfasst 36.874 normalisierte 512-dimensionale Embeddings aus allen 214
Sequenzen ohne Fehler. Eine PCA reduzierte 512 auf 32 Dimensionen. Sie wurde
ausschließlich mit 28.996 Samples aus 170 Trainingssequenzen angepasst und
erklärt 78,7 % der Varianz; Validation und Test waren vom Fit ausgeschlossen.

Das testfreie Screening ergab:

| Variante | Validation Intent-F1 | Hand-F1 | Pose-MAE |
|---|---:|---:|---:|
| Sensorbaseline | 0,9311 ± 0,0064 | 0,9964 | 8,79 cm |
| CLIP-only | 0,6155 ± 0,0119 | 0,9893 | 11,31 cm |
| Sensor + CLIP | **0,9391 ± 0,0054** | 0,9970 | 8,85 cm |
| Sensor + frozen Random-Control | 0,9178 ± 0,0099 | 0,9959 | 8,46 cm |

Nach dieser Validation-Auswahl wurde Sensor+CLIP einmalig auf Test evaluiert:

| Modell | Test Intent-F1 | Accuracy | Hand-F1 | Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| getunte Sensorbaseline | **0,8631 ± 0,0039** | **0,9065 ± 0,0044** | **0,9349 ± 0,0374** | 15,21 ± 0,44 cm | 63.023 |
| Sensor + CLIP | 0,8405 ± 0,0082 | 0,8892 ± 0,0030 | 0,9216 ± 0,0349 | **14,88 ± 0,18 cm** | 67.119 |

Der Validation-Gewinn von CLIP generalisiert nicht auf die
Intentionsklassifikation: Test-F1 fällt um 0,0226. Der am besten auf
Validation ausgewählte Posecheckpoint verbessert den Pose-MAE dagegen um
0,33 cm. Bei dem für die Studienvisualisierung verwendeten
Best-Intention-Checkpoint beträgt der Posegewinn 0,91 cm. Insgesamt liefern
die RGB-Features in der aktuellen Datenmenge keinen belastbaren Vorteil für
die primäre Intentionsmetrik.

## 5. Eingefrorener Studiencheckpoint

Architektur und Seed wurden ausschließlich anhand Validation festgelegt. Der
reproduzierbare Checkpoint für die qualitative Studienauswertung ist:

```text
Training/runs/dataset_v2_20260802_n214_5d136a34/visual_embedding_final_v1/sensor_plus_clip/visual_embedding_final_v1_sensor_plus_clip_seed42/best_intention_model.pt
```

SHA-256:
`c9de5f091b1230bd0117a99a3fbbd69ae2c28ac67353fc5e644bec56bf73967b`.

Diese Festlegung wird trotz des anschließend beobachteten Testverlusts nicht
geändert. Andernfalls würde der Testsplit nachträglich zum Validation-Split.
Für eine spätere praktische Bereitstellungsentscheidung ist die getunte
Sensorbaseline aufgrund des aktuell höheren beobachteten Intent-F1 ein
wichtiger Kandidat, muss aber auf neuen Teilnehmern bestätigt werden.

## 6. Qualitative Visualisierung

Für alle 2.199 Testfenster wurden Vorhersagen exportiert. Drei automatisch
gewählte Sequenzen zeigen ein Erfolgs-, Median- und Fehlerbeispiel:

| Sequenz | Rolle | Fenster-Accuracy | Pose-MAE |
|---|---|---:|---:|
| `Edu_5_20260604_170944` | Erfolg | 0,989 | 15,84 cm |
| `Mona_3_20260624_123548` | Median | 0,924 | 16,09 cm |
| `Jona_7_20260616_182214` | Fehler | 0,663 | 8,47 cm |

Die H.264-Videos zeigen RGB, Ground Truth, Intention mit
Wahrscheinlichkeitsbalken, Empfangshand sowie Ground-Truth- und vorhergesagte
Handposition in einem Robot-Frame-XY-Inset. Eine 3D-Projektion in das RGB-Bild
wird nicht behauptet, da keine für alle Aufnahmen validierte zeitvariable
Kameraprojektion vorliegt. Die Synchronisationsprüfung bestätigt streng
steigende Timestamps und null Zuordnungen aus der Zukunft.

## 7. Latenzanalyse

Modelllatenz und RGB-Frontend werden getrennt gemessen, weil CLIP mit 5 Hz
arbeitet und das zeitliche Sensormodell mit 30 Hz das letzte kausal verfügbare
Embedding hält. Alle Modellbenchmarks nutzen denselben Checkpoint, dasselbe
reale Testfenster, Batchgröße 1, 100 Warm-ups und 1.000 synchronisierte
Messungen.

| Plattform | Modell Forward Median / p95 | Offlinefenster Median / p95 |
|---|---:|---:|
| Mac CPU | 1,112 / 1,223 ms | 1,118 / 1,208 ms |
| Mac MPS | 2,602 / 2,851 ms | 2,755 / 2,937 ms |
| Uni `login3` CPU | 2,395 / 2,792 ms | 2,464 / 2,562 ms |
| TCML Compute CPU | 1,730 / 1,893 ms | 1,780 / 1,974 ms |
| TCML Compute CUDA, RTX 2080 Ti | 1,585 / 1,611 ms | 1,732 / 1,759 ms |

Alle 5.000 Offlinefenster liegen unter dem 33,3-ms-Budget. Auf `login3` ist
keine PyTorch-CUDA-GPU verfügbar; dies ist als `unavailable` und nicht als
fehlender Messwert protokolliert.

Die vollständige RGB-zu-CLIP-Pipeline benötigt im Median 50,52 ms auf Mac CPU,
40,78 ms auf Mac MPS, 74,39 ms auf TCML CPU und 25,94 ms auf TCML CUDA. Alle
4.000 Messungen liegen unter dem 200-ms-Budget des 5-Hz-Frontends. Drei
vorhandene Mac-Live-Sitzungen mit 1.116 Vorhersagen wurden zusätzlich
explorativ aggregiert. Eine neue physische End-to-End-Aufnahme des finalen
Checkpoints erfordert eine angeschlossene Aria-Brille und kann nicht durch
synthetische Offlinewerte ersetzt werden. Device- und Host-Uhren werden nicht
unzulässig voneinander subtrahiert.

## 8. Grenzen und Schlussfolgerung

- Drei Seeds quantifizieren Optimierungsstreuung, ersetzen aber keine
  Konfidenzintervalle über neue Teilnehmer.
- Der Testsplit enthält nur drei Teilnehmer; kleine Unterschiede dürfen nicht
  als statistisch gesicherte Überlegenheit interpretiert werden.
- Die positive Validation-Wirkung von CLIP und die negative Testwirkung deuten
  auf begrenzte Generalisierung beziehungsweise Overfitting der
  Architekturentscheidung hin.
- `no_objects` ist ein interessanter Befund, aber aufgrund der Sichtung auf
  Test nur eine Hypothese für ein neues Experiment.
- CLIP-Embeddings sind abgeleitete personenbezogene Forschungsdaten und werden
  mit denselben Zugriffsbeschränkungen wie die RGB-Aufnahmen behandelt.

Die robusteste Aussage ist damit: Das getunte Sensormodell ist deutlich
kleiner und verbessert die primäre Intentionsmetrik leicht; Hand- und
Posequalität zeigen jedoch einen Trade-off. Frozen CLIP-Features verbessern
die Pose teilweise, bringen in diesem Experiment aber keinen generalisierbaren
Gewinn für die Intentionsklassifikation.

## 9. Reproduzierbare Artefakte

- Zentrale Zusammenfassung:
  [`FINAL_EXPERIMENT_SUMMARY.md`](../Training/reports/dataset_v2_20260802_n214_5d136a34/FINAL_EXPERIMENT_SUMMARY.md)
- Maschinenlesbare Evidenz:
  [`FINAL_EXPERIMENT_SUMMARY.json`](../Training/reports/dataset_v2_20260802_n214_5d136a34/FINAL_EXPERIMENT_SUMMARY.json)
- Finale Testtabelle:
  [`final_test_metrics.csv`](../Training/reports/dataset_v2_20260802_n214_5d136a34/final_test_metrics.csv)
- Hyperparametersuche:
  [`residual_v2_hp_search_v1`](../Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_hp_search_v1/)
- Getuntes Sensormodell:
  [`residual_v2_tuned_v1`](../Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_tuned_v1/)
- Ablationen:
  [`modality_ablation_v1`](../Training/reports/dataset_v2_20260802_n214_5d136a34/modality_ablation_v1/)
- CLIP-Screening und finaler Test:
  [`visual_embedding_screen_v1`](../Training/reports/dataset_v2_20260802_n214_5d136a34/visual_embedding_screen_v1/) und
  [`visual_embedding_final_v1`](../Training/reports/dataset_v2_20260802_n214_5d136a34/visual_embedding_final_v1/)
- Modellvisualisierung:
  [`qualitative_overlay_final`](../Training/reports/dataset_v2_20260802_n214_5d136a34/qualitative_overlay_final/)
- Modell- und CLIP-Latenz:
  [`final_sensor_plus_clip_v1`](../Training/reports/latency/final_sensor_plus_clip_v1/) und
  [`clip_vit_b32_openai_5hz`](../Training/reports/latency/clip_vit_b32_openai_5hz/)
- Literatur- und Versuchsmatrix:
  [`experiment_design_matrix.md`](../Training/literature/experiment_design_matrix.md)
