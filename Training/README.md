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

Pose-Target-Verfuegbarkeit mit denselben Splits und Fensterregeln auditieren:

```bash
python3 Training/audit_pose_targets.py \
  --config Training/configs/hierarchical_baseline_v1.json
```

Der Audit schreibt eine Zusammenfassung nach
`Training/reports/pose_target_audit.json` und alle Handover-Fenster mit ihrer
konkreten Ursache nach `Training/reports/pose_target_audit.csv`.

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
