# Finales Thesis-Protokoll v2

Dieses Protokoll ist der aktive Experimentstand für
„Multi-modal User Intention Estimation and Hand Position Prediction“. Es ersetzt
keine historischen Artefakte. Deren wissenschaftliche Verwendbarkeit steht im
historischen Invaliditätsinventar
`Training/reports/dataset_v2_20260802_n214_5d136a34/ARTIFACT_VALIDITY_V2.json`.
Der aktive kausale Datasetstand
`dataset_v3_causal_20260815_n214_5d136a34` wurde auf dem Cluster materialisiert
und für 214 Sequenzen hashgebunden verifiziert. Der korrigierte CLIP-Cache,
48 Validation-Läufe, die Validation-only-Auswahl, 48 autorisierte Final-Tests
und 25 LOPO-Läufe samt participant-balanced Summary sind abgeschlossen.
Alle neun Postprocessing-Tasks sind ebenfalls abgeschlossen. Der autoritative
Gesamtbericht und die qualitative Ausgabe benötigen zum Statuszeitpunkt
2026-08-17 noch gezielte Wiederholungen: Der Summary-Job lehnte den neueren
Reporting-Checkout ab; die qualitative Pipeline fand für
`Jona_7_20260616_182214` 1165 MP4- gegenüber 1166 VRS-RGB-Frames.
Alle 16 aktiven Matrix-Configs erzwingen vor dem Training zusätzlich exakt 214
ausgewählte Sequenzen und den vollständigen Sequenz-Fingerprint
`5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd`.
Der Tag allein wird daher nicht als Datasetnachweis akzeptiert.
Ergebniszahlen dürfen erst aus den maschinenlesbaren v3-Berichten übernommen
werden; laufende oder fehlende Aggregate werden nicht extrapoliert.

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
- Master-Quellen werden mit `causal_backward_device_time_v1` verbunden: pro
  Masterzeitpunkt darf nur der letzte verfügbare Quell-Capture mit
  `source_timestamp_ns <= timestamp_ns` eingehen. Der Batch-Preflight lehnt
  alte Nearest-/Forward-/Backfill-Master ab.
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

Die statische Transformation `world -> robot` ist eine **offline** aus der
gesamten Sequenz geschätzte Kalibrierung. Die zeitvariablen Sensor-Joins sind
kausal, diese Whole-Sequence-Kalibrierung jedoch nicht. Ergebnisse belegen
daher keine vollständig online-kausale Deployment-Pipeline; für diese Aussage
wäre eine vorab extern kalibrierte oder ausschließlich vergangenheitsbasierte
Transformation erforderlich.

## Artifact Freeze

Jeder neue Lauf erzeugt ein `artifact_manifest.json` nach
`thesis_artifact_freeze_hash_bound_v2`. Das Manifest bindet per
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
- Fusion: aktuelles Temporal/Channel-Gate, simple Fusion und modality-wise Gate;
- Intention: hierarchisch gegen flat;
- Multi-Task: mit und ohne t+1-Pose-Loss;
- Modalitäten: no gaze, no hands, no objects, no direct VIO;
- visuell: Sensor + korrigiertes CLIP mit aktuellem beziehungsweise
  modality-wise Gate sowie ein deterministischer Random-Feature-Control mit
  identischen Zeitstempeln und identischer Dimensionalität;
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
  --balanced-candidate --leave-one-participant-out --seed 42 \
  --output-json Training/reports/dataset_v3_causal_20260815_n214_5d136a34/split_confounding_v2/split_audit.json

# 3. Korrigiertes Terminalziel vor jedem Terminaltraining
python3 Training/audit_endpose_targets.py \
  --config Training/configs/models/residual_transformer_endpose_v2.json \
  --output-dir Training/reports/dataset_v3_causal_20260815_n214_5d136a34/terminal_endpose_corrected_v2/audit \
  --require-sufficient

# 4. Beispiel eines Validation-Laufs
python3 Training/train_residual.py \
  --config Training/configs/architecture/residual_v2_modality_gated.json \
  --dataset-tag dataset_v3_causal_20260815_n214_5d136a34 \
  --experiment-tag thesis_final_v2_validation --seed 42 \
  --run-dir Training/runs/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_validation/residual_modality_gated/residual_modality_gated_seed42 \
  --skip-test-evaluation

# 5. Erst wenn alle Validation-Läufe vollständig sind
python3 Training/select_matrix_checkpoints.py --require-complete

# 6. Ausführbare verschachtelte Group-CV für das vorab deklarierte Primärmodell
python3 Training/prepare_group_cv_runs.py \
  --split-audit Training/reports/dataset_v3_causal_20260815_n214_5d136a34/split_confounding_v2/split_audit.json \
  --base-config Training/configs/models/residual_transformer_v2.json \
  --output-dir Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42 \
  --dataset-tag dataset_v3_causal_20260815_n214_5d136a34 \
  --experiment-tag thesis_v2_group_cv_seed42 --seeds 42

# Nach Abschluss aller 25 äußeren Leave-One-Participant-Out-Auswertungen
python3 Training/evaluation/summarize_group_cv.py \
  --plan Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/group_cv_plan.json \
  --output-dir Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/summary

# 7. Finaler Test eines bereits eingefrorenen Laufs
python3 Training/evaluate_frozen_run.py \
  --run-dir Training/runs/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_validation/residual_modality_gated/residual_modality_gated_seed42 \
  --checkpoint best_intention \
  --selection-file Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json \
  --experiment-id residual_modality_gated \
  --output Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_test/residual_modality_gated_seed42.json

# 8. Autoritativer Gesamtbericht, nachdem alle 48 Final-Test-Dateien vorliegen
python3 Training/evaluation/summarize_thesis_v2_matrix.py \
  --matrix Training/configs/experiment_matrix_v2.json \
  --selection Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json \
  --final-test-dir Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_test \
  --postprocess-root Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/postprocess \
  --output-dir Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary
```

## SLURM-Reihenfolge

Der verbindliche Dependency-Graph ist:

```text
MASTER_JOB
├── CLIP_JOB
└── ENDPOSE_AUDIT_JOB
    └── VALIDATION_JOB (wartet auf CLIP_JOB und ENDPOSE_AUDIT_JOB)
        └── SELECTION_JOB (validation-only, kein Testzugriff)
            ├── TEST_JOB (eingefrorene Checkpoints)
            │   ├── POST_JOB
            │   │   └── QUALITATIVE_JOB
            │   └── SUMMARY_JOB (wartet zusätzlich auf benötigte POST_JOBs)
            └── GROUP_CV_JOB -> GROUP_CV_SUMMARY_JOB
```

Der Master-Rebuild migriert die abgeleiteten Master-CSVs auf kausale Joins,
sichert bestehende abgeleitete Master unter `master_datasets_history/` und
verändert weder Raw-VRS noch originale MPS-Daten. CLIP-Neuaufbau und
Terminal-Audit sind danach voneinander unabhängig:

Für den aktiven Dataset-Tag ist
`Data_vrs/timestamps_summary.reviewed.json` zwingend. Der Job bricht ohne diese
Datei ab und protokolliert Pfad sowie SHA-256. Der alte Fallback ist nur mit
`ALLOW_UNREVIEWED_TIMESTAMPS=1` für ausdrücklich historische Diagnostik
verfügbar und ist nicht als finales Thesis-Dataset zulässig.

```bash
MASTER_JOB=$(sbatch --parsable \
  --export=ALL,OVERWRITE=1 \
  singularity/aria_build_master_dataset.sbatch)

CLIP_JOB=$(sbatch --parsable \
  --dependency=afterok:${MASTER_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34 \
  Training/jobs/prepare_clip_embeddings.sbatch)

ENDPOSE_AUDIT_JOB=$(sbatch --parsable \
  --dependency=afterok:${MASTER_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,EXPERIMENT_TAG=terminal_endpose_corrected_v2 \
  Training/jobs/audit_endpose_v2.sbatch)

VALIDATION_JOB=$(sbatch --parsable \
  --dependency=afterok:${CLIP_JOB}:${ENDPOSE_AUDIT_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,EXPERIMENT_TAG=thesis_final_v2_validation \
  Training/jobs/thesis_v2_validation_matrix.sbatch)
```

Nach erfolgreichem Validation-Job erzeugt ein eigener CPU-Job die
Validation-Auswahl und prüft sie fail-closed. Ein vorhandenes gültiges Manifest
wird nur nach Schema-, Matrix-Hash-, Vollständigkeits- und Splitprüfung
wiederverwendet; ein abweichendes Manifest wird nicht überschrieben:

```bash
SELECTION_JOB=$(sbatch --parsable \
  --dependency=afterok:${VALIDATION_JOB} \
  --export=ALL,MATRIX=Training/configs/experiment_matrix_v2.json,OUTPUT=Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/validation_selection.json \
  Training/jobs/thesis_v2_select_checkpoints.sbatch)
```

Erst danach darf der vorab definierte finale Test-Array eingereicht werden:

```bash
TEST_JOB=$(sbatch --parsable \
  --dependency=afterok:${SELECTION_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,EXPERIMENT_TAG=thesis_final_v2_validation,REPORT_TAG=thesis_final_v2_corrected_alignment \
  Training/jobs/thesis_v2_final_test_matrix.sbatch)
```

Das Array verweigert jeden Lauf, der nicht mit exakt gleicher Experiment-ID,
Seed, Run-Pfad und Checkpoint-Hash in `validation_selection.json` autorisiert
ist. Alle drei Seeds der vorab deklarierten finalen Vergleiche bleiben erhalten,
damit Seed-Standardabweichungen getrennt von gruppierten Konfidenzintervallen
berichtet werden können. Die pro Experiment gewählte repräsentative Seed-Zeile
dient nur der deterministischen qualitativen Fallauswahl.

Das verpflichtende t+1-Postprocessing ist in der Matrix vorab auf
`residual_current_gate` festgelegt und läuft automatisch für alle drei Seeds.
Damit wird keine Architektur anhand des Tests ausgewählt:

```bash
POST_JOB=$(sbatch --parsable \
  --dependency=afterok:${TEST_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,REPORT_TAG=thesis_final_v2_corrected_alignment \
  Training/jobs/thesis_v2_postprocess_selected.sbatch)
```

Der Job liest Experiment, Seeds, Run-Pfade und autorisierte Final-Test-Dateien
aus Matrix und `validation_selection.json`. Der Summary-Job ist fail-closed,
bis alle drei checkpoint-gebundenen Exporte samt Grouped-Report vorliegen.

Der checkpoint-kohärente Matrixbericht darf erst nach Final-Test und dem
verpflichtenden t+1-Postprocessing laufen:

```bash
SUMMARY_JOB=$(sbatch --parsable \
  --dependency=afterok:${TEST_JOB}:${POST_JOB} \
  --export=ALL,DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,REPORT_TAG=thesis_final_v2_corrected_alignment \
  Training/jobs/thesis_v2_summarize_matrix.sbatch)
```

Die verschachtelte Group-CV verwendet innere Validation für Checkpoints und
den äußeren Participant-Fold ausschließlich zur Evaluation:

```bash
GROUP_CV_PLAN=Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/group_cv_plan.json
GROUP_CV_TASKS=$(singularity exec "${IMAGE:-$HOME/singularity/aria_master.simg}" \
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); n=len(p["runs"]); assert n == int(p["fold_count"])*len(p["seeds"]); print(n)' \
  "$GROUP_CV_PLAN")
GROUP_CV_LAST_INDEX=$((GROUP_CV_TASKS - 1))

GROUP_CV_JOB=$(sbatch --parsable \
  --array=0-${GROUP_CV_LAST_INDEX}%3 \
  --dependency=afterok:${GROUP_CV_PLAN_JOB} \
  --export=ALL,GROUP_CV_PLAN=${GROUP_CV_PLAN} \
  Training/jobs/thesis_v2_group_cv.sbatch)

GROUP_CV_SUMMARY_JOB=$(sbatch --parsable \
  --dependency=afterok:${GROUP_CV_JOB} \
  --export=ALL,GROUP_CV_PLAN=Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/group_cv_plan.json,GROUP_CV_SUMMARY_DIR=Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/summary \
  Training/jobs/thesis_v2_summarize_group_cv.sbatch)
```

`GROUP_CV_PLAN` verwendet echte Leave-One-Participant-Out-Folds und die vorab
deklarierte Primärkonfiguration
`residual_transformer_v2.json`; es findet keine nachträgliche Wahl anhand des
äußeren Folds oder historischer Testwerte statt. Der äußere Participant-Fold
darf weder Architektur noch Checkpoint auswählen. Ergebnisse existieren erst,
wenn alle 25 autorisierten LOPO-Auswertungen mit dem vorab festgelegten Seed 42
vollständig vorliegen und der separate Group-CV-Summary
deren Plan- und Report-Bindings geprüft hat.

Die drei Matrix-Seeds bleiben für die kompakte Hauptmatrix erhalten. Für LOPO
wird bewusst nur Seed 42 verwendet: Die 25 voneinander disjunkten äußeren
Teilnehmer-Folds liefern bereits die relevante Between-Participant-Streuung;
eine Verdreifachung auf 75 Trainingsläufe wäre für diese Robustheitsaussage
weitgehend redundant. Die drei bisherigen `temporal_only`-Läufe wurden durch
den wissenschaftlich aussagekräftigeren Random-Visual-Control ersetzt.

Die vollständige, fail-closed verkettete Einreichung ist ausführbar mit:

```bash
bash Training/jobs/submit_thesis_v2_pipeline.sh
```

Sie reicht Master-Rebuild, CLIP, Terminal- und Split-Audit, 48 Validation-
Läufe, eingefrorene Testauswertung, primäres Postprocessing, Gate-Reports,
qualitative Overlays und 25-fold LOPO mitsamt Abhängigkeiten ein.

Qualitative Videos werden erst aus diesem checkpoint-gebundenen Export erzeugt:

```bash
QUALITATIVE_JOB=$(sbatch --parsable \
  --dependency=afterok:${POST_JOB} \
  --export=ALL,\
DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34,\
PREDICTIONS=<selected_report_dir>/test_predictions.csv,\
PREDICTION_REPORT=<selected_report_dir>/test_predictions.json,\
OUTPUT_DIR=Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/qualitative \
Training/jobs/thesis_v2_qualitative.sbatch)
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

## Lokaler Checkout und Clusterstatus

Ein normaler lokaler Git-Checkout enthält nicht die großen Raw-VRS/MPS-Dateien,
Master-CSVs, CLIP-Caches, Checkpoints und vollständigen Reports. Diese liegen
auf dem TCML-Cluster. Der Clusterlauf hat die Kernstufen bereits abgeschlossen;
der genaue maschinenlesbare Zwischenstand steht in
`IMPLEMENTATION_STATUS_P0_P5.json` und `run_registry.json`. Lokal fehlende
Großartefakte bedeuten daher nicht mehr, dass das Experiment unausgeführt ist.
