# Cluster jobs

Alle Jobs werden vom Repository-Hauptverzeichnis eingereicht. Trainingsjobs
verlangen einen expliziten `DATASET_TAG`; dadurch können neue Läufe nicht
versehentlich einem alten Datasetstand zugeordnet werden.

Einzeljobs akzeptieren zusätzlich `EXPERIMENT_TAG`, `SEED`, `EPOCHS`,
`RUN_ID` und `RUN_DIR`. Benchmark- und Ablationsjobs erzeugen die kanonische
verschachtelte Run-Struktur automatisch. SLURM-Ausgaben landen gesammelt unter
`Training/slurm_logs/`.

| Job | Zweck |
|---|---|
| `../../singularity/aria_build_master_dataset.sbatch` | verlangt reviewed Timestamp-Kommandos samt Hash-Provenienz, migriert Derived Master-CSVs auf kausale Backward-Joins, sichert alte Derived Masters und führt den kausalen Preflight aus |
| `train_transformer_v1.sbatch` | einzelner Transformer-v1-Lauf |
| `train_mlp_v1.sbatch` | einzelner MLP-Lauf |
| `train_gru_v1.sbatch` | einzelner GRU-Lauf |
| `train_residual_v2.sbatch` | einzelner Residual-v2-Lauf |
| `benchmark_models.sbatch` | vier Modelle × drei Seeds |
| `ablate_modalities_residual_v2.sbatch` | vier Modalitätsablationen × drei Seeds |
| `hyperparameter_search_residual_v2.sbatch` | 24 validation-only Random-Search-Trials |
| `confirm_hyperparameters_residual_v2.sbatch` | Top-3-Konfigurationen × drei Seeds, weiterhin ohne Test |
| `final_evaluate_tuned_residual_v2.sbatch` | einmalige Testauswertung der final gewählten Konfiguration |
| `prepare_clip_embeddings.sbatch` | frozen CLIP ViT-B/32 mit 5 Hz cachen und Train-only-PCA fitten |
| `screen_visual_embeddings_residual_v2.sbatch` | CLIP-only, Sensor+CLIP und Random-Control auf Validation vergleichen |
| `final_evaluate_visual_variant.sbatch` | nur die anhand Validation gewählte visuelle Variante testen |
| `benchmark_latency_tcml.sbatch` | identisches Fenster auf TCML-CPU und -GPU messen |
| `benchmark_ablation_latency_tcml.sbatch` | Full-Modell und vier Sensorablationen auf einem passenden realen Fenster auf TCML-CPU/GPU messen |
| `benchmark_clip_latency_tcml.sbatch` | RGB-Preprocessing und frozen CLIP ViT-B/32 auf TCML-CPU/GPU messen |
| `export_predictions.sbatch` | Residual-v2-Vorhersagen eines exakten Runs und Checkpoints exportieren |
| `audit_endpose_v1.sbatch` | robuste terminale Handpose-Targets prüfen und einen trainingsfreien Dry-Run ausführen |
| `train_endpose_v1.sbatch` | separates Endpose-Residual-v2 für Seeds 42/43/44 trainieren und das bestehende t+1-Modell auf denselben terminalen Targets auswerten |
| `finalize_endpose_v1.sbatch` | Endpose/t+1-Latenz messen sowie CSV/JSON/PNG/PDF/Markdown-Vergleich erzeugen |
| `thesis_v2_validation_matrix.sbatch` | aktive minimale Matrix: 16 Konfigurationen mal drei Seeds, strikt validation-only |
| `thesis_v2_select_checkpoints.sbatch` | fail-closed Validation-Auswahl; prüft Schema, Matrix-Hash, Vollständigkeit und verweigert stale Outputs |
| `thesis_v2_final_test_matrix.sbatch` | lädt ausschließlich eingefrorene Best-Intent-Checkpoints für den finalen Test; kein Retraining |
| `thesis_v2_postprocess_selected.sbatch` | Matrix-gesteuertes 3-Seed-Array für den vorab deklarierten primären t+1-Lauf: Prediction-Export, Baselines, gruppierte Bootstrap-Metriken und ggf. Modalitätsgewichte |
| `thesis_v2_qualitative.sbatch` | hashgebundene VRS/MP4-Sidecars und gute/typische/fehlerhafte qualitative Fälle |
| `thesis_v2_split_audit.sbatch` | kausaler Preflight und 25-fold Leave-One-Participant-Out-Splitaudit |
| `thesis_v2_repair_causal_masters.sbatch` | baut nach einem fehlgeschlagenen Vollbatch ausschließlich dessen Fehlersequenzen neu und führt QA plus vollständigen kausalen Preflight aus |
| `thesis_v2_prepare_group_cv.sbatch` | materialisiert den vorab festgelegten 25-fold-LOPO-Plan ausschließlich für Seed 42 |
| `thesis_v2_group_cv.sbatch` | ausführbare verschachtelte Leave-One-Participant-Out-CV; Array-Grenzen werden aus dem hashgebundenen Plan gelesen und beim `sbatch`-Aufruf übergeben |
| `thesis_v2_summarize_group_cv.sbatch` | prüft die Vollständigkeit und Plan-/Checkpoint-Bindings aller äußeren Group-CV-Auswertungen und aggregiert sie |
| `thesis_v2_summarize_matrix.sbatch` | prüft alle 48 autorisierten Testartefakte sowie die verpflichtenden t+1-Postprocess-Artefakte, bindet die sieben v3-Supplemente und erzeugt checkpoint-kohärente Seed- und Aggregate-Tabellen |
| `thesis_v3_posthoc_reporting.sbatch` | reporting-only: train-fitted kausale Intention-Baselines und empirischer Sampling-/Window-Audit; startet kein neuronales Training |
| `submit_thesis_v2_pipeline.sh` | reicht den vollständigen Dependency-Graph inklusive Gate-Reports und 25 LOPO-Runs ein |

## Aktive v2-Reihenfolge

Die `thesis_v2_*`-Jobs gehören zum aktiven Protokoll in
`../THESIS_FINAL_PROTOCOL_V2.md`. Verbindlich ist:

```text
Master-Rebuild -> CLIP-Neuaufbau / Terminal-Audit -> Validation
-> Validation-Selection -> eingefrorener Final-Test
-> Postprocess / Summary -> qualitative Fälle
```

Der Master-Job erzwingt `causal_backward_device_time_v1`; CLIP und Terminal-
Audit starten nur nach seinem erfolgreichen Abschluss. Der Selection-Job liest
ausschließlich Validation-Artefakte. Erst danach darf das Final-Test-Array
laufen. Postprocess, Summary und qualitative Fälle sind checkpoint- und
hashgebunden. Die Protokolldatei enthält die exakten `afterok`-Abhängigkeiten.

Beispiel für das aktive Validation-Array, nachdem die beiden Prerequisite-Jobs
erfolgreich abgeschlossen sind:

```bash
DATASET_TAG=dataset_v3_causal_20260815_n214_5d136a34
EXPERIMENT_TAG=thesis_final_v2_validation

sbatch --dependency=afterok:${CLIP_JOB}:${ENDPOSE_AUDIT_JOB} \
  --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/thesis_v2_validation_matrix.sbatch
```

Der Tag ist der aktive kausale Stand. Master-Rebuild, Preflight und finale
Matrixauswertung sind abgeschlossen; für neue Ausführungen bleiben die
fail-closed Voraussetzungen unverändert. Der Name allein belegt weder
vorhandene Derived Artifacts noch Ergebnisse. Ein bereits vorhandenes
Run-Verzeichnis wird von den Trainern nicht überschrieben.

## Testsplit-Schutz bei der Modellauswahl

Die Jobs für Hyperparameter- und visuelle Variantensuche übergeben
`--skip-test-evaluation`. Ihre Resume-Prüfung verwirft Runs, die Testmetriken
enthalten. Erst die jeweiligen `final_evaluate_*.sbatch`-Jobs führen nach der
Validation-Auswahl die Testauswertung aus.

Der aktive Export verwendet den validation-ausgewählten
`best_intention_model.pt`, verlangt auf dem Testsplit zusätzlich das
autorisierende Final-Test-JSON und schreibt in ein neues explizites
Report-Verzeichnis. Ein gefilterter Testexport oder das Überschreiben eines
vorhandenen Exports wird abgewiesen.
`MASTER_DIR` (standardmäßig `Data_collection/master_datasets`) überschreibt
zusätzlich einen nicht portablen absoluten Pfad aus der gespeicherten
Run-Konfiguration.

Für die vollständige Einreichung nach einem sauberen Checkout genügt:

```bash
bash Training/jobs/submit_thesis_v2_pipeline.sh
```

Nach einem erfolgreichen gezielten Master-Repair kann dessen Job-ID als bereits
geprüfte Upstream-Abhängigkeit wiederverwendet werden:

```bash
UPSTREAM_MASTER_JOB=<repair-job-id> bash Training/jobs/submit_thesis_v2_pipeline.sh
```

Der separate Gate-Array wertet `residual_modality_gated` und
`visual_corrected_clip_modality_gate` für alle drei Matrix-Seeds aus. LOPO wird
hingegen bewusst nur mit Seed 42 ausgeführt (25 statt 75 Läufe), weil dort die
Streuung über die äußeren Teilnehmer-Folds die primäre Robustheitsanalyse ist.

## CLIP-Abhängigkeiten

Der Singularity-Container liefert Torch und Torchvision. Die vier zusätzlichen,
fest versionierten Pakete aus `Training/clip_requirements.txt` werden mit
`--no-deps` in ein isoliertes Verzeichnis installiert und dem CLIP-Job über
`PYTHONPATH` übergeben. Gewichte und Embeddings liegen in lokalen Caches; es
werden keine RGB-Frames an einen externen Dienst übertragen.

## Historische visuelle Jobs

`screen_visual_embeddings_residual_v2.sbatch` und
`final_evaluate_visual_variant.sbatch` gehören **nicht** zum finalen v2-
Protokoll. Sie bilden einen historischen Visual-Screening-Workflow ab und sind
für neue Thesis-Ergebnisse fail-closed, solange nicht ausdrücklich
`ALLOW_LEGACY_EXPERIMENT=1` gesetzt wird. Dieses Opt-in macht Ergebnisse nicht
automatisch v2-gültig. Für den finalen Vergleich sind ausschließlich die
beiden korrigierten CLIP-Konfigurationen der Matrix und der autorisierte
Final-Test-/Postprocess-Pfad zu verwenden.

Auch bei kausalen zeitvariablen Source-Joins bleibt die statische
`world -> robot`-Transformation eine offline aus der ganzen Sequenz geschätzte
Kalibrierung. Das aktive Protokoll behauptet daher keine vollständig
online-kausale Deployment-Ausführung.
