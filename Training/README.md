# Multimodales Training

Dieser Ordner enthaelt die reproduzierbare Trainingspipeline fuer die
hierarchische Assistenzintention und die handover-spezifische Vorhersage der
zukuenftigen Empfangshand.

## Aufgaben

Das Modell loest zwei hierarchische Klassifikationsstufen und eine bedingte
Regressionsaufgabe:

1. `continue` gegen `assistance needed`
2. bei Assistenz: `fetch` gegen `handover`
3. nur bei `handover`: Position und Quaternion der Empfangshand eine Sekunde
   in der Zukunft, relativ zu AprilTag 0 am Roboter

ArUco-IDs 6 bis 14 sind keine Transformer-Zielklassen. Ihre Positionen,
Sichtbarkeit sowie Gaze-Winkel und -Distanzen bleiben Eingabefeatures fuer den
Objekt- und Szenenkontext. Eine Zielobjektauswahl kann separat mit einem
geometrischen Gaze-Marker-Modul evaluiert werden.

AprilTag 0 definiert einen statischen Roboteranker. Der Master-Builder schaetzt
dessen robuste Weltpose aus allen Frames mit gleichzeitig gueltigem Tag und
SLAM. Bei kurzzeitiger Verdeckung wird der Roboterframe ueber die aktuelle
SLAM-Pose fortgefuehrt; `robot_frame_valid` und
`robot_anchor_interpolated` dokumentieren dies pro Zeile.

## Architektur

`model.py` implementiert einen an GTN angelehnten Zwei-Turm-Transformer. Ein
Turm modelliert zeitliche Abhaengigkeiten, der zweite Beziehungen zwischen
Sensorkanaelen. Ein lernbares Gate fusioniert beide Repraesentationen. Die
Klassifikationskoepfe bilden die Assistenzhierarchie explizit ab; der Pose-Kopf
wird nur mit gueltigen Handover-Targets trainiert.

`data.py` stellt Missing-Data-Masken, ausschliesslich auf dem Trainingssplit
angepasste Normalisierung und participant-wise Splits bereit. Die Sensorzeilen
aus `DONE -> THIRD` bleiben als kontinuierlicher Kontext mit dem internen Label
`transition` erhalten, duerfen aber nie Endpunkt eines Trainingsfensters sein.
Nur Sliding Windows mit einem echten Timestamp-Sprung ueber dem konfigurierten
Grenzwert werden verworfen.

### Residual v2

`HierarchicalResidualPoseTransformer` erweitert die Baseline um eine
Handover-spezifische Links/Rechts-Klassifikation. Der Pose-Head sagt keine
absolute Handpose voraus, sondern eine Positionskorrektur und eine relative
Quaternion zur letzten gueltigen Handpose im Beobachtungsfenster:

```text
future_position   = last_position + position_delta
future_quaternion = last_quaternion * quaternion_delta
```

Der Pose-Head ist mit Positionsdelta null und Quaterniondelta Identitaet
initialisiert. Vor dem Lernen entspricht er daher exakt Last Observation. Die
Korrektur wird von der vorhergesagten Handwahrscheinlichkeit konditioniert.

Die Auswertung unterscheidet:

- `pose_oracle`: wahre Empfangshand, isoliert die Bewegungsprognose
- `pose_end_to_end`: vorhergesagte Empfangshand und vorhergesagte Bewegung
- `last_observation_oracle`: direkte Persistenzreferenz
- Pose nach linker/rechter Hand und vier Abschnitten der Handover-Phase

Da die aktuelle Validation nur rechte Handover-Haende enthaelt, wird die linke
Generalisation separat berichtet und nicht zur Modellauswahl verwendet.

## Datenvoraussetzung

Die Eingabe sind aktuelle Dateien unter:

```text
Data_collection/master_datasets/*_master.csv
```

Erforderlich sind Intentionslabels sowie das ausgewaehlte zukuenftige
Empfangshand-Target. Labels und Zukunftswerte werden nicht als Eingabefeatures
verwendet.

## Training

Smoke-Test:

```bash
python3 Training/smoke_test.py
python3 Code/Testing/static_robot_anchor_smoke.py
```

Interaktiver Ein-Epochen-Test:

```bash
python3 Training/train.py \
  --config Training/configs/hierarchical_baseline_v1.json \
  --epochs 1
```

Residual-v2-Smoke-Test und interaktiver Ein-Epochen-Lauf:

```bash
python3 Training/residual_smoke_test.py

python3 Training/train_residual.py \
  --config Training/configs/hierarchical_residual_v2.json \
  --epochs 1
```

Residual-v2-GPU-Job:

```bash
sbatch --export=ALL,EPOCHS=1,RUN_DIR=Training/runs/residual_v2_cluster_smoke \
  Training/hierarchical_residual_v2.sbatch

sbatch Training/hierarchical_residual_v2.sbatch
```

Der erste Befehl ist der einmalige echte Cluster-Smoke-Test. Erst wenn dieser
ohne Schema- oder CUDA-Fehler abgeschlossen ist, wird der vollstaendige Lauf
mit dem zweiten Befehl gestartet.

Jeder v2-Lauf speichert `best_intention_model.pt` nach Validation-Intent-Macro-
F1 und `best_pose_model.pt` nach Validation-Oracle-Position-MAE. Early Stopping
wird nur ausgeloest, wenn sich keines der beiden Validation-Ziele innerhalb der
konfigurierten Patience verbessert.

Pose-Target-Verfuegbarkeit mit denselben Splits und Fensterregeln auditieren:

```bash
python3 Training/audit_pose_targets.py \
  --config Training/configs/hierarchical_baseline_v1.json
```

Der Audit schreibt eine Zusammenfassung nach
`Training/reports/pose_target_audit.json` und alle Handover-Fenster mit ihrer
konkreten Ursache nach `Training/reports/pose_target_audit.csv`.

Naive Pose-Baselines auf exakt denselben gueltigen Zukunftstargets auswerten:

```bash
python3 Training/evaluate_pose_baselines.py \
  --config Training/configs/hierarchical_baseline_v1.json \
  --model-metrics Training/runs/hierarchical_baseline_20260712_101448/metrics.json
```

Der Evaluator vergleicht den Trainingsmittelwert, die letzte beobachtete Pose
und eine konstante lineare Geschwindigkeit. Die beiden bewegungsbasierten
Verfahren verwenden die annotierte Empfangshand und werden deshalb explizit
als Oracle-Receiving-Hand-Baselines dokumentiert. Fehlende Beobachtungen
werden per festgelegter Fallback-Kette aufgefangen, damit alle Metriken dieselbe
Target-Menge verwenden. Direkte Abdeckung und Metriken ohne Fallback werden
separat berichtet.

Fenstergenaue Vorhersagen eines vorhandenen Checkpoints exportieren:

```bash
python3 Training/export_checkpoint_predictions.py \
  --run-dir Training/runs/hierarchical_baseline_20260712_101448
```

Der Export schreibt `test_predictions.csv` und
`test_prediction_analysis.json` in das Run-Verzeichnis. Vor dem Schreiben
werden die erneut berechneten Intention- und Pose-Metriken gegen die vorhandene
`metrics.json` geprueft. Die Analyse gruppiert Transformer und Oracle Last
Observation nach Teilnehmer, Handseite, Sequenz und Fortschritt innerhalb der
Handover-Phase.

GPU-Export des aktuellen Baseline-Checkpoints auf dem Cluster:

```bash
sbatch --export=ALL,RUN_DIR=Training/runs/hierarchical_baseline_20260712_101448 \
  Training/export_checkpoint_predictions.sbatch
```

GPU-Job auf dem Cluster:

```bash
sbatch Training/hierarchical_baseline.sbatch
```

Die historische Flat/Object-Baseline verwendete `first_test.json` und
`participant_split_v1.json`. Diese Konfigurationen bleiben zur
Nachvollziehbarkeit des ersten Laufs erhalten, sind aber nicht die aktuelle
Thesis-Baseline.

## Ergebnisse

Jeder neue Lauf landet unter `Training/runs/hierarchical_baseline_*` und
enthaelt:

- `best_model.pt`: bester Checkpoint nach Validation-Intention-Macro-F1
- `config.json`: tatsaechlich verwendete Konfiguration
- `data_metadata.json`: Features, Normalisierung, Split sowie wegen echter
  Zeitluecken, geringer Beobachtung oder unlabeled Endpunkt verworfene Fenster
- `metrics.json`: Verlauf sowie einmalige Testauswertung

Die Metriken werden getrennt fuer Drei-Klassen-Intention, Assistenzbedarf,
Fetch/Handover und Handover-Pose berichtet. Der Testsplit wird erst nach der
Auswahl des besten Validation-Checkpoints ausgewertet.
