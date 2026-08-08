# Residual-v2 Hyperparameter Search v1

Die 24 Stage-A-Konfigurationen werden deterministisch erzeugt:

```bash
python3 Training/generate_hyperparameter_trials.py --overwrite
```

Der Such-Seed ist `20260808`, der Trainings-Seed für Stage A ist `42`. Die
Auswahl verwendet ausschließlich Validation-Metriken; der SLURM-Job startet
`train_residual.py` deshalb mit `--skip-test-evaluation`.

Clusterstart:

```bash
DATASET_TAG=dataset_v2_20260802_n214_5d136a34 \
EXPERIMENT_TAG=residual_v2_hp_search_v1 \
sbatch --export=ALL,DATASET_TAG,EXPERIMENT_TAG \
  Training/jobs/hyperparameter_search_residual_v2.sbatch
```

Das verbindliche Auswahlprotokoll steht in
`Training/literature/experiment_design_matrix.md`.
