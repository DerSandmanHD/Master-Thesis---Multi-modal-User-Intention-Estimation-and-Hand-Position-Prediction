# Finale Sensor+CLIP-Auswertung

Die Architektur `sensor_plus_clip` wurde vor dem Testzugriff ausschließlich
auf Validation ausgewählt. Die Tabelle vergleicht sie mit der ebenfalls
Validation-selektierten, getunten Sensor-Baseline (jeweils Seeds 42/43/44).

| Variante | Test Intent Macro-F1 | Test Accuracy | Test Hand Macro-F1 | Test Pose-MAE | Parameter |
|---|---:|---:|---:|---:|---:|
| getunte Sensor-Baseline | **0,8631 ± 0,0039** | **0,9065 ± 0,0044** | **0,9349 ± 0,0374** | 15,21 ± 0,44 cm | **63.023** |
| Sensor + CLIP | 0,8405 ± 0,0082 | 0,8892 ± 0,0030 | 0,9216 ± 0,0349 | **14,88 ± 0,18 cm** | 67.119 |

Der Validierungsgewinn von CLIP überträgt sich nicht auf den Testsplit:
Intentions-Macro-F1 fällt um `0,0226`, während der Best-Pose-MAE um `0,33 cm`
sinkt. Am für die gemeinsame Deployment-Ausgabe relevanten
Best-Intention-Checkpoint verbessert sich der Pose-MAE stärker von `15,55`
auf `14,64 cm`, aber auch dort bleibt der Intentionsverlust bestehen.

Die Architektur wird nach Sichtung des Tests nicht rückwirkend geändert.
`final_model_selection.json` im übergeordneten Berichtsordner friert daher
regelkonform Sensor+CLIP, Seed 42 und den Best-Intention-Checkpoint mit
SHA-256 `c9de5f091b1230bd0117a99a3fbbd69ae2c28ac67353fc5e644bec56bf73967b`
ein. Auswahlkriterium und Seedwahl lesen keine Testmetriken.

Maschinenlesbare Einzel- und Aggregatwerte liegen in `data/` und
`summary.json`; die PNG-/PDF-Abbildung liegt unter `figures/`.
