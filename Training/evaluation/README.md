# Training evaluation

Dieser Ordner erzeugt reproduzierbare Lernkurven aus einem vollständigen
Benchmark mit:

- Transformer v1, MLP, GRU und Residual Transformer v2
- jeweils Seeds 42, 43 und 44
- Train-/Validation-Loss, Validation Intent Macro-F1 und Validation Pose-MAE

Das Skript sucht die Run-Verzeichnisse rekursiv. Dadurch funktioniert auch die
historische Zwischenebene `Training/runs/run_cluster/Ohne Titel/` sowie die
neue Struktur
`Training/runs/<dataset_tag>/<experiment_tag>/<model_tag>/<run_id>/`.

Vom Repository-Hauptordner:

```bash
python3 Training/evaluation/generate_training_diagrams.py
```

Dieser Befehl reproduziert den Legacy-Stand `final_clean_v1` und schreibt wie
bisher nach `Training/evaluation/generated/`. Für einen neuen Benchmark werden
Dataset und Experiment explizit angegeben:

```bash
DATASET_TAG=dataset_v2_20260815_n180_ab12cd34
EXPERIMENT_TAG=benchmark_v2

python3 Training/evaluation/generate_training_diagrams.py \
  --dataset-tag "$DATASET_TAG" \
  --tag "$EXPERIMENT_TAG" \
  --runs-dir "Training/runs/$DATASET_TAG/$EXPERIMENT_TAG"
```

Die neue Standardausgabe liegt dann unter
`Training/reports/<dataset_tag>/<experiment_tag>/training_diagrams/`.
`config.json::run_context` wird gegen die Kommandozeilentags geprüft, damit
keine Runs verschiedener Datasetstände vermischt werden. `--output-dir` kann
weiterhin verwendet werden, wenn bewusst ein anderer Zielpfad benötigt wird.

## Erzeugte Daten

Unter `<output-dir>/data/`:

- `training_history_by_seed.csv`: eine Zeile pro Modell, Seed und Epoche
- `training_history_mean_std.csv`: Mittelwert und Standardabweichung pro Modell
  und gemeinsamer Epoche
- `run_summary.csv`: Checkpoint-Epochen und finale Run-Metriken

Unter `<output-dir>/figures/`, jeweils als PNG mit 300 dpi und als PDF:

1. `01_total_loss_train_validation_by_model`
2. `02_validation_intention_macro_f1_by_model`
3. `03_validation_pose_position_mae_by_model`
4. `04_validation_intention_macro_f1_mean_std`

Sterne markieren die anhand der Validation-Metrik ausgewaehlten Checkpoints.
Da Early Stopping zu unterschiedlich langen Laeufen fuehrt, werden
Mittelwert-und-Standardabweichung-Kurven nur fuer Epochen berechnet, die bei
allen drei Seeds des jeweiligen Modells vorhanden sind. Die einzelnen
Seed-Kurven bleiben separat sichtbar.

Die F1-Diagramme verwenden fuer bessere Lesbarkeit bewusst den dargestellten
Wertebereich 0,85 bis 0,95; die exakten Werte bleiben vollstaendig in den
CSV-Dateien erhalten.
