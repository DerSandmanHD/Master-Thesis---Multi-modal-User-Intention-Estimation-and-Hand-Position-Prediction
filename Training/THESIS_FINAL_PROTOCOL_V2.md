# Finales Thesis-Protokoll v2

Dieses Protokoll ist der aktive, noch **nicht ausgeführte** Experimentstand für
„Multi-modal User Intention Estimation and Hand Position Prediction“. Es ersetzt
keine historischen Artefakte. Deren wissenschaftliche Verwendbarkeit steht in
`Training/reports/dataset_v2_20260802_n214_5d136a34/ARTIFACT_VALIDITY_V2.json`.
Es wurden in diesem Arbeitsstand keine Resultate erfunden oder aus fehlenden
Trainingsläufen extrapoliert.

## Forschungsaufgaben

Der primäre Lauf verarbeitet ein vergangenes multimodales Fenster und schätzt
den aktuellen Zustand hierarchisch:

```text
continue vs assistance -> fetch vs handover -> receiving hand -> Wrist-Pose t+1 s
```

Die Intention bleibt ein aktuelles Zustandslabel. Die t+1-Pose ist ein
handover-spezifischer Multi-Task-Output. Die robuste terminale Empfangspose ist
ein getrenntes sekundäres Experiment und niemals ein Alias für das t+1-Ziel.

## Unveränderliche wissenschaftliche Regeln

1. Architektur, Hyperparameter, Variantenauswahl und Checkpoint werden nur mit
   Train/Validation festgelegt.
2. Validation-Läufe verwenden immer `--skip-test-evaluation`.
3. Der finale Test lädt exakt den bereits eingefrorenen
   `best_intention`-Checkpoint über `evaluate_frozen_run.py`; er trainiert nicht
   erneut.
4. Eine Hauptzeile enthält ausschließlich Outputs dieses Checkpoints. Der
   pose-selektierte Checkpoint und Ground-Truth-Hand-Posewerte sind nur
   diagnostisch.
5. Seed-Mittelwert und Seed-Standardabweichung sind Replikationsdiagnostik, keine
   Populationsunsicherheit. Teilnehmer- und Sequenz-Cluster-Bootstrap werden
   separat ausgewiesen.
6. Der historische Testsplit wurde bereits betrachtet. Participant-wise
   Auswertung und ausführbare Group-CV sind deshalb vorbereitet; neue Aussagen
   gelten erst nach deren tatsächlicher Ausführung als zusätzlich abgesichert.

## Zeit- und Zieldefinitionen

- Master, MPS-Hand, VIO und VRS-RGB verwenden absolute Project-Aria
  `DEVICE_TIME`-Nanosekunden.
- CLIP speichert den VRS-RGB-`capture_timestamp_ns`. Für einen Master-Zeitpunkt
  wird nur der letzte RGB-Capture `<= timestamp_ns` verwendet. START wird nicht
  subtrahiert.
- Aktive CLIP-Version: `vrs_rgb_device_time_v2`.
- Primäres Poseziel: physischer Empfangshand-Capture nahe `t + 1 s`.
- Constant Velocity schätzt Geschwindigkeit aus kausalen physischen
  `hand_timestamp_ns`, extrapoliert aber ausschließlich bis zum zur Inferenz
  bekannten nominalen Zeitpunkt `t + 1 s`. Der tatsächliche zukünftige
  Capture-Zeitpunkt ist nur Auswertungsmetadatum und kein Baseline-Input.
- Terminalziel: letzter stabiler 0,5-s-Abschnitt aus **eindeutigen physischen**
  `hand_timestamp_ns`-Captures; Version
  `terminal_endpose_unique_hand_capture_v2`.
- Terminal-Endpunkte innerhalb des Aggregationsintervalls werden als teilweise
  beobachtete Terminalzustandsschätzung gekennzeichnet. Endpunkte am oder nach
  dem letzten Ziel-Capture werden maskiert.

## Artifact Freeze

Jeder neue Lauf erzeugt `artifact_manifest.json`. Das Manifest bindet per
SHA-256:

- Git-Commit, Dirty-Status und Diff-Fingerprint;
- Quell- und aufgelöste Config;
- Dataset-Identifier, Master-Hashes, Sequenzlisten und Participant-Split;
- Feature- und Modalitätsschema, Train-Normalizer;
- CLIP-Cache-Manifest, Alignment-Fingerprint und PCA-Projektion;
- Python-, PyTorch-, CUDA-, Project-Aria- und MPS-Versionen, soweit installiert;
- Seed, vollständiges Kommando, Zeitstempel, Checkpoints, Selection-Regel und
  Metrikdatei.

Validierung:

```bash
python3 Training/artifact_freeze.py \
  Training/runs/<dataset>/<experiment>/<model>/<run>/artifact_manifest.json
```

Eine Datei mit richtigem Namen, aber falschem Inhalt, wird abgewiesen.

## Minimale Experimentmatrix

Die maschinenlesbare Matrix ist
`Training/configs/experiment_matrix_v2.json`. Sie enthält 16 tatsächlich zu
trainierende Konfigurationen × drei Seeds = 48 Validation-Läufe:

- Baselines: MLP, GRU, Transformer, Residual Transformer;
- Fusion: aktuelles Temporal/Channel-Gate, simple Fusion, modality-wise Gate,
  temporal-only;
- Intention: hierarchisch gegen flat;
- Multi-Task: mit und ohne t+1-Pose-Loss;
- Modalitäten: no gaze, no hands, no objects, no direct VIO;
- visuell: Sensor + korrigiertes CLIP mit aktuellem beziehungsweise
  modality-wise Gate;
- sekundär: ein korrigierter gelernter Terminal-Endpose-Lauf.

Residual-Baseline, Sensor-only/no-CLIP und Terminal-Persistence werden als
Aliase beziehungsweise auswertungsbasierte Methoden wiederverwendet. Dadurch
entstehen keine identischen Trainingsduplikate. Die Matrix ist absichtlich kein
kartesisches Produkt aller Faktoren.

Matrix prüfen und exakte Kommandos anzeigen:

```bash
python3 Training/experiment_matrix.py --stage validation
python3 Training/experiment_matrix.py --stage final-test
```

## Lokale ausführbare Reihenfolge

Voraussetzung sind die nicht im aktuellen lokalen Checkout vorhandenen Master-,
VRS-, MPS-, MP4- und CLIP-Artefakte.

```bash
# 1. Invarianten
python3 -m pytest -q tests
python3 Training/run_scientific_tests.py

# 2. Split/Confounding und Group-CV-Plan, ohne Modellmetriken
python3 Training/audit_participant_splits.py \
  --master-dir Data_collection/master_datasets \
  --manifest Data_collection/dataset_manifest.csv \
  --config Training/configs/models/residual_transformer_v2.json \
  --balanced-candidate --group-cv-folds 5 --seed 42 \
  --output-json Training/reports/dataset_v2_20260802_n214_5d136a34/split_confounding_v2/split_audit.json

# 3. Korrigiertes Terminalziel vor jedem Terminaltraining
python3 Training/audit_endpose_targets.py \
  --config Training/configs/models/residual_transformer_endpose_v2.json \
  --output-dir Training/reports/dataset_v2_20260802_n214_5d136a34/terminal_endpose_corrected_v2/audit \
  --require-sufficient

# 4. Beispiel eines Validation-Laufs
python3 Training/train_residual.py \
  --config Training/configs/architecture/residual_v2_modality_gated.json \
  --dataset-tag dataset_v2_20260802_n214_5d136a34 \
  --experiment-tag thesis_final_v2_validation --seed 42 \
  --run-dir Training/runs/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_validation/residual_modality_gated/residual_modality_gated_seed42 \
  --skip-test-evaluation

# 5. Erst wenn alle Validation-Läufe vollständig sind
python3 Training/select_matrix_checkpoints.py --require-complete

# 6. Ausführbare verschachtelte Group-CV für die vor Test festgelegte Architektur
python3 Training/prepare_group_cv_runs.py \
  --split-audit Training/reports/dataset_v2_20260802_n214_5d136a34/split_confounding_v2/split_audit.json \
  --base-config Training/configs/models/residual_transformer_v2.json \
  --output-dir Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_v2_group_cv \
  --dataset-tag dataset_v2_20260802_n214_5d136a34 \
  --experiment-tag thesis_v2_group_cv --seeds 42 43 44

# 7. Finaler Test eines bereits eingefrorenen Laufs
python3 Training/evaluate_frozen_run.py \
  --run-dir Training/runs/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_validation/residual_modality_gated/residual_modality_gated_seed42 \
  --checkpoint best_intention \
  --selection-file Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json \
  --experiment-id residual_modality_gated \
  --output Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/final_test/residual_modality_gated_seed42.json

# 8. Autoritativer Gesamtbericht, nachdem alle 48 Final-Test-Dateien vorliegen
python3 Training/evaluation/summarize_thesis_v2_matrix.py \
  --matrix Training/configs/experiment_matrix_v2.json \
  --selection Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json \
  --final-test-dir Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/final_test \
  --postprocess-root Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/postprocess \
  --output-dir Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary
```

## SLURM-Reihenfolge

CLIP-Neuaufbau und Terminal-Audit sind voneinander unabhängig:

```bash
CLIP_JOB=$(sbatch --parsable \
  --export=ALL,DATASET_TAG=dataset_v2_20260802_n214_5d136a34 \
  Training/jobs/prepare_clip_embeddings.sbatch)

ENDPOSE_AUDIT_JOB=$(sbatch --parsable \
  --export=ALL,DATASET_TAG=dataset_v2_20260802_n214_5d136a34,EXPERIMENT_TAG=terminal_endpose_corrected_v2 \
  Training/jobs/audit_endpose_v2.sbatch)

VALIDATION_JOB=$(sbatch --parsable \
  --dependency=afterok:${CLIP_JOB}:${ENDPOSE_AUDIT_JOB} \
  --export=ALL,DATASET_TAG=dataset_v2_20260802_n214_5d136a34,EXPERIMENT_TAG=thesis_final_v2_validation \
  Training/jobs/thesis_v2_validation_matrix.sbatch)
```

Nach erfolgreichem Validation-Job wird **zuerst** die Auswahl erzeugt und
inhaltlich geprüft:

```bash
python3 Training/select_matrix_checkpoints.py --require-complete
```

Erst danach darf der vorab definierte finale Test-Array eingereicht werden:

```bash
TEST_JOB=$(sbatch --parsable \
  --export=ALL,DATASET_TAG=dataset_v2_20260802_n214_5d136a34,EXPERIMENT_TAG=thesis_final_v2_validation,REPORT_TAG=thesis_final_v2_corrected_alignment \
  Training/jobs/thesis_v2_final_test_matrix.sbatch)
```

Das Array verweigert jeden Lauf, der nicht mit exakt gleicher Experiment-ID,
Seed, Run-Pfad und Checkpoint-Hash in `validation_selection.json` autorisiert
ist. Alle drei Seeds der vorab deklarierten finalen Vergleiche bleiben erhalten,
damit Seed-Standardabweichungen getrennt von gruppierten Konfidenzintervallen
berichtet werden können. Die pro Experiment gewählte repräsentative Seed-Zeile
dient nur der deterministischen qualitativen Fallauswahl.

Postprocessing eines anhand Validation gewählten Residual-Laufs:

```bash
POST_JOB=$(sbatch --parsable --export=ALL,\
RUN_DIR=Training/runs/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_validation/residual_modality_gated/residual_modality_gated_seed42,\
FINAL_TEST_JSON=Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/final_test/residual_modality_gated_seed42.json,\
REPORT_DIR=Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/postprocess/residual_modality_gated_seed42 \
Training/jobs/thesis_v2_postprocess_selected.sbatch)
```

Das Beispiel `residual_modality_gated_seed42` ist kein vorweggenommenes
Ergebnis; RUN_DIR und Seed müssen aus `validation_selection.json` übernommen
werden.

Der checkpoint-kohärente Matrixbericht darf erst nach Final-Test und dem
gewünschten t+1-Postprocessing laufen:

```bash
SUMMARY_JOB=$(sbatch --parsable \
  --dependency=afterok:${TEST_JOB}:${POST_JOB} \
  --export=ALL,DATASET_TAG=dataset_v2_20260802_n214_5d136a34,REPORT_TAG=thesis_final_v2_corrected_alignment \
  Training/jobs/thesis_v2_summarize_matrix.sbatch)
```

Die verschachtelte Group-CV verwendet innere Validation für Checkpoints und
den äußeren Participant-Fold ausschließlich zur Evaluation:

```bash
GROUP_CV_JOB=$(sbatch --parsable \
  --export=ALL,GROUP_CV_PLAN=Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_v2_group_cv/group_cv_plan.json \
  Training/jobs/thesis_v2_group_cv.sbatch)
```

Qualitative Videos werden erst aus diesem checkpoint-gebundenen Export erzeugt:

```bash
sbatch --export=ALL,\
DATASET_TAG=dataset_v2_20260802_n214_5d136a34,\
PREDICTIONS=<selected_report_dir>/test_predictions.csv,\
PREDICTION_REPORT=<selected_report_dir>/test_predictions.json,\
OUTPUT_DIR=Training/reports/dataset_v2_20260802_n214_5d136a34/thesis_final_v2_corrected_alignment/qualitative \
Training/jobs/thesis_v2_qualitative.sbatch
```

## Berichtspflicht

Für t+1 werden Persistence, Constant Velocity und das gelernte Modell auf
demselben Fair-Common-Sample-Set mit Mean, Median, cm, Orientierung, Coverage
und Nennern ausgegeben. Für Terminal-Endpose sind Persistence und das gelernte
Modell verpflichtend. Hauptreports enthalten Intent/Assistance/Fetch/Handover,
Receiving Hand, Pose, Orientierung, Coverage, Samplezahlen, Confusion Matrices,
per-class Metriken sowie Window-, Sequence-, Participant- und per-hand-Ebenen.

Modality-Gewichte werden pro Window nur für tatsächlich verfügbare Modalitäten
ausgegeben und summieren sich dort zu eins. Sie sind interne Modellkonditionierung
und dürfen nicht als kausale Wichtigkeit interpretiert werden.

Der maschinenlesbare Implementierungs-/Blockerstatus steht in
`IMPLEMENTATION_STATUS_P0_P5.json`. `PASS` bezeichnet dort verifizierten Code,
nicht einen bereits ausgeführten vollständigen Trainingslauf.

## Aktuell lokal blockiert

Im lokalen `Data_collection/` liegt nur `manual_timestamp_review.csv`. Deshalb
sind Full-Dataset-Audit, CLIP-Neuaufbau, PCA, Training, echte Testauswertung,
MP4-Sidecars und qualitative Videos hier nicht ausführbar. Benötigt werden die
unveränderten Raw-VRS/MPS-Dateien, Master-CSVs samt Manifest, MP4-Dateien, der
Singularity-Container und für Training/CLIP eine GPU. Alle Codepfade und
Kommandos sind vorbereitet; der Status enthält bewusst keine neuen Metriken.
