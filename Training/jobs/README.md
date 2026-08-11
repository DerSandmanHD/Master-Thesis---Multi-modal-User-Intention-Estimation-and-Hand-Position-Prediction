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
| `thesis_v2_final_test_matrix.sbatch` | lädt ausschließlich eingefrorene Best-Intent-Checkpoints für den finalen Test; kein Retraining |
| `thesis_v2_postprocess_selected.sbatch` | Prediction-Export, t+1-Baselines, gruppierte Bootstrap-Metriken und ggf. Modalitätsgewichte |
| `thesis_v2_qualitative.sbatch` | hashgebundene VRS/MP4-Sidecars und gute/typische/fehlerhafte qualitative Fälle |
| `thesis_v2_group_cv.sbatch` | ausführbare verschachtelte participant-disjoint Group-CV für eine zuvor auf Validation eingefrorene Architektur |
| `thesis_v2_summarize_matrix.sbatch` | prüft alle 48 autorisierten Testartefakte und erzeugt checkpoint-kohärente Seed- und Aggregate-Tabellen |

## Aktive v2-Reihenfolge

Die `thesis_v2_*`-Jobs gehören zum aktiven Protokoll in
`../THESIS_FINAL_PROTOCOL_V2.md`. Zuerst laufen der korrigierte CLIP-Neuaufbau
und der Terminal-Target-Audit, danach das Validation-Array. Anschließend wird
`select_matrix_checkpoints.py --require-complete` lokal oder im Login-Job
ausgeführt und geprüft. Erst danach darf das Final-Test-Array laufen; Postprocess
und qualitative Fälle verwenden ausschließlich dessen eingefrorenen
Best-Intent-Checkpoint. Das Beispiel in der Protokolldatei enthält die exakten
`sbatch`-Abhängigkeiten.

Beispiel für den vollständigen Vergleich:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34
EXPERIMENT_TAG=benchmark_v2

sbatch --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/benchmark_models.sbatch
```

Der Beispieltag ist zu ersetzen. Gültig ist nur ein bereits unter
`Training/datasets/` dokumentierter Datasetstand. Ein bereits vorhandenes
Run-Verzeichnis wird von den Trainern nicht überschrieben.

## Testsplit-Schutz bei der Modellauswahl

Die Jobs für Hyperparameter- und visuelle Variantensuche übergeben
`--skip-test-evaluation`. Ihre Resume-Prüfung verwirft Runs, die Testmetriken
enthalten. Erst die jeweiligen `final_evaluate_*.sbatch`-Jobs führen nach der
Validation-Auswahl die Testauswertung aus.

Der qualitative Exportjob verwendet standardmäßig den validation-ausgewählten
`best_intention_model.pt` und schreibt `test_predictions.csv` sowie
`test_predictions.json` direkt in das angegebene Run-Verzeichnis. Ein anderer
Checkpoint, Split oder Zielpfad muss explizit über `CHECKPOINT`, `SPLIT`,
`OUTPUT_CSV` beziehungsweise `REPORT_OUT` gesetzt werden.
`MASTER_DIR` (standardmäßig `Data_collection/master_datasets`) überschreibt
zusätzlich einen nicht portablen absoluten Pfad aus der gespeicherten
Run-Konfiguration.

## CLIP-Abhängigkeiten

Der Singularity-Container liefert Torch und Torchvision. Die vier zusätzlichen,
fest versionierten Pakete aus `Training/clip_requirements.txt` werden mit
`--no-deps` in ein isoliertes Verzeichnis installiert und dem CLIP-Job über
`PYTHONPATH` übergeben. Gewichte und Embeddings liegen in lokalen Caches; es
werden keine RGB-Frames an einen externen Dienst übertragen.
