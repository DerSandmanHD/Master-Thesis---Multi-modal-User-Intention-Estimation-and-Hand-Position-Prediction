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
| `export_predictions.sbatch` | Vorhersagen eines exakten Runs exportieren |

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

## CLIP-Abhängigkeiten

Der Singularity-Container liefert Torch und Torchvision. Die vier zusätzlichen,
fest versionierten Pakete aus `Training/clip_requirements.txt` werden mit
`--no-deps` in ein isoliertes Verzeichnis installiert und dem CLIP-Job über
`PYTHONPATH` übergeben. Gewichte und Embeddings liegen in lokalen Caches; es
werden keine RGB-Frames an einen externen Dienst übertragen.
