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

### Vergleichsbackbones

`train.py` unterstuetzt neben dem Transformer zwei neuronale
Vergleichsbackbones. Alle drei verwenden dieselben normalisierten Fenster,
Missing-Data-Masken, Teilnehmer-Splits, hierarchischen Klassifikationskoepfe,
Pose-Targets, Losses und Metriken:

- `HierarchicalWindowMLP` flacht das vollstaendige Beobachtungsfenster ab und
  verarbeitet es mit einem Feed-forward-Netz. Es dient als Baseline ohne
  zeitliche Gewichtsteilung.
- `HierarchicalGRU` verarbeitet das Fenster mit einer unidirektionalen GRU.
  Die Unidirektionalitaet verhindert, dass ein Online-Modell Informationen
  ausserhalb des Beobachtungsfensters oder aus zukuenftigen Frames verwendet.
- `HierarchicalGatedMultimodalTransformer` bleibt die GTN-inspirierte
  Transformer-Baseline.

Die Modellwahl steht als `model_type` in der jeweiligen JSON-Konfiguration.
Die v1-Groessen von MLP und GRU sind so gewaehlt, dass ihre Parameterzahl bei
dem aktuellen Featureprofil in derselben Groessenordnung wie die des
Transformers liegt. Dadurch wird der Architekturvergleich nicht allein durch
eine stark unterschiedliche Modellkapazitaet bestimmt.
Jeder Lauf schreibt Modelltyp und Anzahl trainierbarer Parameter in
`metrics.json` und den Checkpoint.

`data.py` stellt Missing-Data-Masken, ausschliesslich auf dem Trainingssplit
angepasste Normalisierung und participant-wise Splits bereit. Die Sensorzeilen
aus `DONE -> THIRD` bleiben als kontinuierlicher Kontext mit dem internen Label
`transition` erhalten, duerfen aber nie Endpunkt eines Trainingsfensters sein.
Nur Sliding Windows mit einem echten Timestamp-Sprung ueber dem konfigurierten
Grenzwert werden verworfen.

Alle aktiven finalen Konfigurationen filtern die Master-CSVs strikt ueber
`Data_collection/dataset_manifest.csv`. Verwendet werden nur Zeilen mit
`include_in_training=True`, `status=valid`,
`next_action=ready_for_master_merge` und vorhandenem Master-Dataset. Dateien,
die nicht im Manifest stehen, lassen den Lauf im strikten Modus abbrechen.
Der Teilnehmername wird fuer den Split case-insensitiv kanonisiert; dadurch
werden beispielsweise `David` und `david` garantiert derselben Person und
demselben Split zugeordnet. `Test` ist ein regulaerer Teilnehmername und keine
automatische Ausschlussregel.

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

## Ablagestruktur und Versionierung

Aktive Vorlagen und Jobs sind nach Zweck getrennt:

```text
Training/
├── configs/models/       # Transformer, MLP, GRU und Residual v2
├── configs/ablations/    # Modalitaetsablationen
├── jobs/                 # SLURM-Einstiegspunkte
├── datasets/             # kleine, versionierte Dataset-Deskriptoren
├── runs/                 # Checkpoints und Run-Artefakte; nicht in Git
├── reports/              # aggregierte Auswertungen
├── live_runs/            # neue Replay-/Live-Sitzungen; nicht in Git
└── slurm_logs/           # neue .out/.err-Dateien; nicht in Git
```

Neue Trainingsläufe verwenden:

```text
Training/runs/<dataset_tag>/<experiment_tag>/<model_tag>/<run_id>/
```

`config.json` enthält zusätzlich `run_context` mit genau diesen drei Tags.
`data_metadata.json`, `dataset_provenance.json` und der Manifest-Snapshot
belegen den tatsächlich geladenen Dateninhalt. Vergleichsreports landen unter
`Training/reports/<dataset_tag>/<experiment_tag>/`.

`Training/run_registry.json` registriert Datasetstände und abgeschlossene
Experimente. Der bisherige `final_clean_v1`-Stand, seine flachen Reports, der
Deployment-Spiegel und die verschachtelten Cluster-Runs bleiben unverändert an
ihren bisherigen Orten. Sie sind Legacy-Artefakte und werden nicht
nachträglich umbenannt.

## Training

Smoke-Test:

```bash
python3 Training/smoke_test.py
python3 tests/integration/static_robot_anchor_smoke.py
```

Interaktiver Ein-Epochen-Test:

```bash
python3 Training/train.py \
  --config Training/configs/models/transformer_v1.json \
  --dataset-tag development \
  --experiment-tag local_smoke \
  --epochs 1
```

MLP- und GRU-Vergleich mit demselben Laufkontext:

```bash
python3 Training/train.py \
  --config Training/configs/models/mlp_v1.json \
  --dataset-tag development \
  --experiment-tag local_smoke

python3 Training/train.py \
  --config Training/configs/models/gru_v1.json \
  --dataset-tag development \
  --experiment-tag local_smoke
```

Separate GPU-Jobs:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_mlp_v1.sbatch
sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_gru_v1.sbatch
```

## Finaler Modellvergleich

Vor dem finalen Lauf muss die QA nach dem letzten Master-Build erneut erzeugt
und der Datasetstand unter `Training/datasets/` eingefroren und in
`Training/run_registry.json` registriert worden sein. Der Beispieltag unten
muss durch diesen tatsächlichen Tag ersetzt werden. Der Array-Job startet
Transformer v1, MLP, GRU und Residual Transformer v2 jeweils mit den Seeds 42,
43 und 44. Maximal vier GPU-Jobs laufen gleichzeitig:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34
EXPERIMENT_TAG=benchmark_v2

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/benchmark_models.sbatch
```

Nach Abschluss aller zwoelf Tasks werden Datensatz-Fingerprint, Sequenzsplits
und Fensterzahlen validiert und Mittelwert sowie Standardabweichung
aggregiert:

```bash
python3 Training/compare_final_runs.py \
  --dataset-tag "$DATASET_TAG" \
  --tag "$EXPERIMENT_TAG" \
  --runs-root "Training/runs/$DATASET_TAG/$EXPERIMENT_TAG"
```

Erzeugt werden:

```text
Training/reports/<dataset_tag>/<experiment_tag>/<experiment_tag>_comparison.json
Training/reports/<dataset_tag>/<experiment_tag>/<experiment_tag>_comparison.csv
```

Der historische Vergleich bleibt weiterhin mit
`python3 Training/compare_final_runs.py --tag final_clean_v1` auswertbar; seine
bereits erzeugten Reports bleiben flach unter `Training/reports/`.

Die Standardmodelle speichern `best_model.pt` als besten Intent-Checkpoint und
zusaetzlich `best_pose_model.pt` nach Validation-Positions-MAE. Residual v2
behaelt analog getrennte Intent- und Pose-Checkpoints. Dadurch werden Intent-
und Poseergebnisse nicht nach dem Testset ausgewaehlt.

Ein-Epochen-Cluster-Smoke-Tests koennen ohne Aenderung der Konfiguration
gestartet werden:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG=cluster_smoke_v1,EPOCHS=1 \
  Training/jobs/train_mlp_v1.sbatch

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG=cluster_smoke_v1,EPOCHS=1 \
  Training/jobs/train_gru_v1.sbatch
```

Residual-v2-Smoke-Test und interaktiver Ein-Epochen-Lauf:

```bash
python3 Training/residual_smoke_test.py

python3 Training/train_residual.py \
  --config Training/configs/models/residual_transformer_v2.json \
  --dataset-tag development \
  --experiment-tag local_smoke \
  --epochs 1
```

Residual-v2-GPU-Job:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG=cluster_smoke_v1,EPOCHS=1 \
  Training/jobs/train_residual_v2.sbatch

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG=residual_full_v1 \
  Training/jobs/train_residual_v2.sbatch
```

Der erste Befehl ist der einmalige echte Cluster-Smoke-Test. Erst wenn dieser
ohne Schema- oder CUDA-Fehler abgeschlossen ist, wird der vollstaendige Lauf
mit dem zweiten Befehl gestartet.

Jeder v2-Lauf speichert `best_intention_model.pt` nach Validation-Intent-Macro-
F1 und `best_pose_model.pt` nach Validation-Oracle-Position-MAE. Early Stopping
wird nur ausgeloest, wenn sich keines der beiden Validation-Ziele innerhalb der
konfigurierten Patience verbessert.

## Modalitaetsablationen

Vier vorbereitete Residual-v2-Konfigurationen entfernen jeweils eine
Featuregruppe vor Normalisierung und Missing-Data-Maskierung:

| Konfiguration | Entfernte Encoderfeatures | Roh-/Modellfeatures |
|---|---|---:|
| `configs/ablations/residual_v2_no_gaze.json` | direkte Gazefeatures sowie gaze-abgeleitete Objektwinkel und -distanzen | 64 / 128 |
| `configs/ablations/residual_v2_no_hands.json` | Handgueltigkeit, Trackingkonfidenz und beide Wrist-Posen | 74 / 148 |
| `configs/ablations/residual_v2_no_objects.json` | alle ArUco-6-bis-14-Positionen, Gaze-Beziehungen und Gueltigkeitsflags | 38 / 76 |
| `configs/ablations/residual_v2_no_vio.json` | direkte SLAM-Bewegungs-/Qualitaetsfeatures sowie Robot-Frame-Gueltigkeitsflags | 83 / 166 |

Alle Varianten behalten Architektur, Split, Fenster, Losses und Hyperparameter
der Residual-v2-Vollvariante. `data_metadata.json` und
`dataset_provenance.json` speichern entfernte Modalitaeten, konkrete Spalten
und die veraenderte Eingabedimension.

Zwei Interpretationsgrenzen sind wichtig:

- `no_hands` ist eine Ablation der Encoder- und damit der
  Intentions-/Handklassifikationsfeatures. Die letzte Handpose bleibt als
  geometrische Referenz fuer den Residual-Pose-Output erforderlich; die
  Posemetrik beschreibt daher kein vollstaendig handloses System.
- `no_vio` entfernt die direkten VIO-Kanaele. Hand- und Objektpositionen
  bleiben im vorab berechneten AprilTag-0-Frame und sind daher nicht
  unabhaengig von der Offline-Koordinatenaufbereitung. Eine vollstaendig
  VIO-freie Variante benoetigt ein neues Featureprofil in einem anderen
  Koordinatensystem.

Kurzer lokaler Konfigurations- und Formtest:

```bash
python3 Training/ablation_smoke_test.py
```

Der Array-Job startet vier Varianten mit jeweils drei Seeds:

```bash
DATASET_TAG=dataset_v2_20260802_n214_5d136a34
EXPERIMENT_TAG=modality_ablation_v1

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/ablate_modalities_residual_v2.sbatch
```

Die Ergebnisse landen getrennt unter:

```text
Training/runs/<dataset_tag>/<experiment_tag>/<ablation>/
  <experiment_tag>_<ablation>_seed<seed>/
```

Der abgeschlossene n214-Bericht mit Tabellen sowie PNG-/PDF-Abbildungen liegt
unter `Training/reports/dataset_v2_20260802_n214_5d136a34/modality_ablation_v1/`.

## Offline-Streaming-Replay

Vor der Anbindung eines echten Aria-Streams kann eine Master-CSV kausal, also
Frame fuer Frame, durch die finalen Residual-v2-Checkpoints abgespielt werden.
Der Replay verwendet nur die letzten 60 Eingabeframes fuer die Vorhersage.
Labels und zukuenftige Pose-Targets werden ausschliesslich fuer die
Offline-Kontrolle ausgegeben und niemals als Modelleingabe verwendet.

Vom Repository-Hauptordner:

```bash
python3 Training/replay_stream_inference.py \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44 \
  --master-csv Data_collection/replay_inputs/Jona_6_final_master.csv \
  --output-csv \
    Training/live_runs/dataset_v1_20260729_n156_seq457a80f1/replay_jona_6/predictions.csv
```

Mit `--realtime` folgt der Replay den Zeitstempeln der Aufnahme. Ohne diese
Option wird die Sequenz so schnell wie moeglich ausgewertet. Das Skript gibt
geglättete Intentionen, Konfidenzen, Inferenzzeiten und bei stabilem
`handover` die vorhergesagte Empfangshand sowie Pose im AprilTag-0-Frame aus.
Es besitzt bewusst keine Schnittstelle zur Robotersteuerung.

Die Replay-Eingabe muss mit dem finalen Master-Builder erzeugt worden sein und
alle im Deployment-Metadatenfile gespeicherten Modellfeatures enthalten.
`--allow-missing-features` ist nur fuer technische Diagnosen vorgesehen; damit
erzeugte Vorhersagen sind keine gueltigen Modellergebnisse.

### Vollstaendiger Testsplit-Replay

Ein finaler Dataset-Snapshot kann gegen Manifest, Artefaktsplit und
Feature-Schema geprueft und mit SHA-256 eingefroren werden:

```bash
python Training/validate_dataset_snapshot.py \
  --snapshot-dir Data_collection/final_dataset_snapshot_20260729 \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44
```

Der folgende Befehl liest die Testsequenzen direkt aus
`data_metadata.json`, verlangt jede der 21 erwarteten Master-CSVs und wertet
Raw, Stable und Actionable getrennt aus:

```bash
python Training/batch_replay_validation.py \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44 \
  --master-dir \
    Data_collection/final_dataset_snapshot_20260729/master_datasets \
  --split test \
  --device cpu
```

Die erzeugten Einzel- und Gesamtberichte liegen standardmaessig unter
`Training/evaluation/deployment_validation_runs/` und werden nicht von Git
versioniert. Raw-/Stable-Ergebnisse sind mit den vorhandenen Masters
auswertbar. Ein striktes Live-Frische-Gate kann offline dagegen blockieren,
wenn ein historischer `nearest`-Merge einen minimal zukuenftigen Sensorwert
ausgewaehlt hat. Ein solcher Wert wird bewusst nicht in eine kausale
Altersangabe umgedeutet.

## Aria-Gen2-Live-Inferenz

`aria_live_inference.py` verbindet den finalen Residual-v2-Checkpoint mit einem
Aria-Gen2-Livestream. Eye Gaze gibt den kausalen 30-Hz-Modelltakt vor. RGB wird
mit `profile9` ungefaehr bei 5 Hz verarbeitet; erkannte Markerposen werden bis
zu 500 ms im AprilTag-0-Koordinatensystem gehalten. Handtracking und
hochfrequentes VIO werden mit denselben Toleranzen wie beim Master-Builder
synchronisiert. Das Modell erhaelt dadurch weiterhin ein 60-Frame-Fenster von
ungefaehr zwei Sekunden.

Voraussetzungen pruefen, ohne einen Stream zu starten:

```bash
conda activate aria_conda
python Training/aria_live_inference.py \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44 \
  --check-only
```

Live-Inferenz vom Repository-Hauptordner starten:

```bash
python Training/aria_live_inference.py \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44 \
  --profile-name profile9 \
  --interface usb \
  --print-mode changes \
  --output-jsonl \
    Training/live_runs/dataset_v1_20260729_n156_seq457a80f1/live_validation_01/predictions.jsonl
```

AprilTag 0 muss zu Beginn so lange im RGB-Bild sichtbar sein, bis der
regelmaessige Status `anchor_ready: true` und mindestens acht
`anchor_samples` meldet. Danach werden 60 gueltige 30-Hz-Frames gesammelt.
`buffer_frames: 60` zeigt an, dass das Modellfenster bereit ist;
`predictions` zaehlt die ausgegebenen Vorhersagen. Mit `Ctrl-C` werden Stream,
Receiver und USB-Verbindung sauber beendet.

Die Live-Ausgabe trennt bewusst die reine Modellantwort von der validierten
Entscheidung:

- `stable_intention` bzw. `model=...` bleibt die zeitlich geglaettete
  Modellvorhersage und wird auch fuer die Fehleranalyse protokolliert.
- `actionable_intention` bzw. der rueckwaertskompatible Alias
  `decision_intention` ist die freigegebene Wahrnehmungsausgabe. Sie wird zu
  `insufficient_input`, wenn Gazeabdeckung, Robot-Frame-Abdeckung oder die
  Frische von VIO und Anker nicht genuegen. Bei `handover` werden Abdeckung,
  aktuelle Gueltigkeit und Alter ausschliesslich fuer die vom Modell
  vorhergesagte Empfangshand geprueft. Bei `fetch` werden sichtbare, aber
  veraltete Objektmarker nicht freigegeben.
- Kurze Blinks werden dadurch toleriert. Laenger geschlossene Augen oder ein
  verlorenes Gaze-Signal werden nicht faelschlich als `continue` behandelt.
  Die Grenzwerte sind Kommandozeilenoptionen und sollen nach weiteren
  Live-Versuchen anhand der JSONL-Logs validiert werden.

Parallel verfolgt eine roboterfreie Wahrnehmungs-Zustandsmaschine die Folge
`continue -> fetch -> handover`. Sie meldet unter
`perception_workflow.state` Kandidaten und bestaetigte Zustaende, fordert aber
niemals eine externe Aktion an. Ein `handover` ohne vorher bestaetigten
`fetch`-Kontext wird explizit als `handover_without_fetch_context`
gekennzeichnet.

Die separate Zielauswahl betrachtet die sichtbaren ArUco-IDs 6 bis 14. Ein
Objekt wird erst nach standardmaessig einer Sekunde eindeutiger Fixation
ausgegeben. Der kleinste Gaze-Winkel muss unter 0,35 rad liegen und mindestens
0,05 rad Abstand zum zweitbesten Objekt haben. Das Ergebnis steht unter
`target_selection`; bei bestaetigtem `fetch` wird die ID im
`perception_workflow` festgehalten. `selection_score` ist ein geometrischer
Heuristikwert und keine trainierte oder kalibrierte Wahrscheinlichkeit.

Die AprilTag-Diagnose unterscheidet ausserdem:

- `apriltag_0_recent`: Tag 0 liegt innerhalb der Marker-Toleranz im Cache,
- `apriltag_0_frame_aligned`: Tagbeobachtung liegt innerhalb von 20 ms zum
  aktuellen Modelltakt,
- `apriltag_0_age_ms`: Alter der verwendbaren Beobachtung.

Die Modellfeatures `apriltag_0_valid` und `robot_anchor_interpolated` behalten
damit unveraendert ihre Trainingssemantik; die neue Diagnose erklaert lediglich
den bei einem 5-Hz-RGB-Stream erwartbaren Unterschied zwischen einem
verwendbaren statischen Anker und einem exakt frame-synchronen Tag.

Die roboterfreie Entscheidungslogik kann ohne Aria-Hardware geprueft werden:

```bash
python3 Training/live_decision_smoke_test.py
```

Ein kontrollierter, gelabelter Liveversuch inklusive monotonichem
Ereignismarker und Latenzauswertung ist in
`Training/live_validation_protocol.md` beschrieben. Die Auswertung trennt
Raw, Stable, Input Quality und Actionable und berechnet die Zeit vom manuell
markierten Ereignis-Onset bis zur jeweiligen Entscheidung.

Die JSONL-Datei wird absichtlich angehaengt. Fuer einen getrennten Versuch
sollte daher ein neuer Dateiname verwendet werden. `profile9` ist der
validierte Standard. `mp_streaming_demo` ist in der lokal getesteten
SDK-Version wegen RGB-Decoderfehlern und einer niedrigeren Handtracking-Rate
nicht fuer diesen Adapter geeignet.

Der Adapter ist ausschliesslich Inferenz und besitzt keine
Robotersteuerung. Insbesondere werden keine Befehle an einen Franka-Arm
gesendet. Vor einer spaeteren Handover-Ausfuehrung sind mindestens
Robot-Base-Kalibrierung, Workspace-/Kollisionspruefung, Confidence-Gating,
Watchdog und ein separater sicherer Controller erforderlich.

Pose-Target-Verfuegbarkeit mit denselben Splits und Fensterregeln auditieren:

```bash
python3 Training/audit_pose_targets.py \
  --config Training/configs/models/transformer_v1.json
```

Der Audit schreibt eine Zusammenfassung nach
`Training/reports/pose_target_audit.json` und alle Handover-Fenster mit ihrer
konkreten Ursache nach `Training/reports/pose_target_audit.csv`.

Naive Pose-Baselines auf exakt denselben gueltigen Zukunftstargets auswerten:

```bash
python3 Training/evaluate_pose_baselines.py \
  --config Training/configs/models/transformer_v1.json \
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
  Training/jobs/export_predictions.sbatch
```

Neuer Transformer-GPU-Job auf dem Cluster:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_transformer_v1.sbatch
```

Die historische Flat/Object-Baseline liegt unter
`archive/legacy_flat_object_baseline/`. Sie bleibt zur Nachvollziehbarkeit des
ersten Laufs erhalten, ist aber nicht die aktuelle Thesis-Baseline.

## Ergebnisse

Jeder neue Lauf landet unter
`Training/runs/<dataset_tag>/<experiment_tag>/<model_tag>/<run_id>/` und
enthaelt mindestens:

- `best_model.pt`: bester Checkpoint nach Validation-Intention-Macro-F1
- `config.json`: tatsaechlich verwendete Konfiguration
- `data_metadata.json`: Features, Normalisierung, Split sowie wegen echter
  Zeitluecken, geringer Beobachtung oder unlabeled Endpunkt verworfene Fenster
- `dataset_provenance.json`: Dateihashes, Manifest, Builder-, Git- und
  Laufzeitinformationen
- `dataset_manifest_snapshot.csv`: das beim Lauf verwendete Manifest
- `metrics.json`: Verlauf sowie einmalige Testauswertung

Die Metriken werden getrennt fuer Drei-Klassen-Intention, Assistenzbedarf,
Fetch/Handover und Handover-Pose berichtet. Der Testsplit wird erst nach der
Auswahl des besten Validation-Checkpoints ausgewertet.

Die schon vorhandenen Dateien unter `Training/Outputs/`,
`Training/evaluation/generated/`, die flachen Reports und alten Run-Pfade sind
historische Ergebnisse. Sie werden nicht verschoben; alle neuen Läufe folgen
der oben beschriebenen Struktur.
