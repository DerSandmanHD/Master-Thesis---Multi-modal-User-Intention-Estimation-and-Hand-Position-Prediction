# Multi-modal User Intention Estimation and Hand Position Prediction

Dieses Repository enthält die Datenverarbeitungs-, Qualitätsprüfungs- und
Trainingspipeline der Masterarbeit zur multimodalen Assistenzintention mit
Project Aria Gen 2.

Das aktuelle System verarbeitet Blickrichtung, MPS-Handtracking,
SLAM-Trajektorien und markerbasierte Objektposen. Das Modell sagt hierarchisch
voraus, ob Unterstützung benötigt wird, unterscheidet anschließend zwischen
`fetch` und `handover` und schätzt für Handover die zukünftige Pose der
Empfangshand.

## Aktueller Datenfluss

```text
VRS recording
  |-- RGB and eye gaze ------------------- Code/extract_multimodal_data.py
  |-- audio commands --------------------- Code/speech_recognition_demo.py
  |-- MPS hand tracking and SLAM -------- Project Aria MPS
  `-- AprilTag/ArUco poses from RGB ------ Code/detect_tags.py
                         |
                         v
              Code/build_master_dataset.py
                         |
                         v
       Data_collection/master_datasets/*_master.csv
                         |
                         v
                Training/data.py
                         |
              +----------+----------+
              |                     |
      Training/train.py    Training/train_residual.py
```

Alle Modalitäten werden über `DEVICE_TIME` auf die Gaze-Zeitachse
synchronisiert. Fehlende Werte werden über eigene Gültigkeitsmasken abgebildet
und nicht als echte numerische Null interpretiert.

AprilTag 0 definiert den verwendeten Robotermarker-Koordinatenrahmen. Dieser
Frame ist ohne zusätzliche vermessene Transformation nicht identisch mit der
physischen Roboterbasis.

## Intentionssegmente

| Zeitbereich | Aktivität | Target |
|---|---|---|
| `START -> SECOND` | aktuelle Tätigkeit fortsetzen | `continue` |
| `SECOND -> DONE` | Zielobjekt fixieren beziehungsweise anfordern | `fetch` |
| `DONE -> THIRD` | Warte- und Übergangsbereich | kein Fenster-Target |
| `THIRD -> Aufnahmeende` | Hand zum Roboter ausstrecken | `handover` |

Das Handover-Pose-Target ist eine tatsächlich gemessene Handgelenkpose bei
`t + 1 s`, keine aus der aktuellen Pose erzeugte Extrapolation.

## Repository-Struktur

| Pfad | Zweck |
|---|---|
| `Code/` | Produktive Extraktion, Review, QA und Master-Dataset-Erstellung |
| `Training/` | Aktive Modelle, Datenloader, Evaluation und Cluster-Jobs |
| `Training/experiments/` | Kleine, unveränderliche Ergebnis-Snapshots |
| `tests/integration/` | Pipeline-nahe Smoke- und Integrationstests |
| `Thesis/` | Methodische Dokumentation und Ergebnisstände |
| `singularity/` | Reproduzierbare Containerdefinition und SLURM-Jobs |
| `references/` | Externe Implementierungen und Literaturbezug |
| `archive/` | Historischer, nicht produktiver Projektcode |

Rohdaten, Master-CSVs, Checkpoints und normale Cluster-Logs werden nicht in
Git gespeichert. Die entsprechenden lokalen Verzeichnisse sind über
`.gitignore` ausgeschlossen. Nur
`Data_collection/manual_timestamp_review.csv` wird als kompakte,
reproduktionsrelevante Annotationstabelle versioniert.

## Zentrale Befehle

Dataset-QA:

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/dataset_qa.py \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.reviewed.json \
  --min-handover-hand-valid-ratio 0.70
```

Der Hand-Tracking-Grenzwert von 70 % wird sowohl für die allgemeine
Handover-Handabdeckung als auch für die annotierte Empfangshand verwendet.
Der tatsächlich verwendete Grenzwert wird im QA-JSON unter
`quality_thresholds` gespeichert. Der bisherige finale 80-%-Snapshot bleibt
damit als unveränderte Vergleichsbasis erhalten.

Master-Datasets auf dem Cluster bauen:

```bash
sbatch --export=ALL,OVERWRITE=1 \
  singularity/aria_build_master_dataset.sbatch
```

Aktive Smoke-Tests:

```bash
python3 Code/dataset_qa_smoke_test.py
python3 tests/integration/static_robot_anchor_smoke.py
python3 Training/smoke_test.py
python3 Training/residual_smoke_test.py
python3 Training/ablation_smoke_test.py
python3 Training/pose_baselines_smoke_test.py
python3 Training/export_predictions_smoke_test.py
python3 Training/run_layout_smoke_test.py
python3 Training/run_registry_smoke_test.py
python3 Training/run_discovery_smoke_test.py
```

Hierarchische Backbone-Vergleiche:

```bash
python3 Training/train.py \
  --config Training/configs/models/transformer_v1.json

python3 Training/train.py \
  --config Training/configs/models/mlp_v1.json

python3 Training/train.py \
  --config Training/configs/models/gru_v1.json
```

Residual-v2-Modell:

```bash
python3 Training/train_residual.py \
  --config Training/configs/models/residual_transformer_v2.json
```

Cluster-Training für einen zuvor unter `Training/datasets/` eingefrorenen
Datasetstand:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_transformer_v1.sbatch
sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_mlp_v1.sbatch
sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_gru_v1.sbatch
sbatch --export=ALL,DATASET_TAG="$DATASET_TAG" \
  Training/jobs/train_residual_v2.sbatch
```

Finaler, manifestgefilterter Vergleich mit vier Modellen und drei Seeds:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34
EXPERIMENT_TAG=benchmark_v2

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/benchmark_models.sbatch

python3 Training/compare_final_runs.py \
  --dataset-tag "$DATASET_TAG" \
  --tag "$EXPERIMENT_TAG" \
  --runs-root "Training/runs/$DATASET_TAG/$EXPERIMENT_TAG"
```

## Dokumentation

- Neue VRS- und MPS-Aufnahmen verarbeiten:
  [`NEW_RECORDINGS_PIPELINE.md`](NEW_RECORDINGS_PIPELINE.md)
- Rohdaten, Einheiten und Transformationen:
  [`Thesis/raw_data_and_processing.md`](Thesis/raw_data_and_processing.md)
- Aktueller technischer Projektstand: [`Thesis/status.md`](Thesis/status.md)
- Akzeptierte hierarchische Baseline:
  [`Thesis/status_testing_hierarchical_tracking_baseline_v1.md`](Thesis/status_testing_hierarchical_tracking_baseline_v1.md)
- Trainingspipeline: [`Training/README.md`](Training/README.md)
- Literatur und externe Architekturvorbilder: [`Papers.md`](Papers.md)

## Reproduzierbarkeit

Aktiver Quellcode bleibt jeweils einmal unter `Code/` beziehungsweise
`Training/`. Experimente enthalten keine kopierten Source-Trees, sondern
Konfiguration, Metriken, Split-Metadaten und den verwendeten Git-Commit.
Checkpoints und vollständige Run-Verzeichnisse bleiben außerhalb der normalen
Git-Historie. `Training/run_registry.json` ordnet eingefrorene Datasetstände,
Experimente, Reports und ausgewählte Deployment-Artefakte einander zu.
