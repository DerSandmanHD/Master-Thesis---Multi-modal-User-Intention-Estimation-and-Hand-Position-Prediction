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
