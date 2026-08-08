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
  --runs-dir "Training/runs/$DATASET_TAG/$EXPERIMENT_TAG" \
  --baseline-comparison-json Training/reports/final_clean_v1_comparison.json
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
- `benchmark_test_summary_mean_std.csv`: Testmetriken als Mittelwert und
  Populationsstandardabweichung über die drei Seeds
- `test_intention_confusion_matrices.csv`: aggregierte Intention-
  Konfusionsmatrizen
- `test_intention_per_class_metrics.csv`: Precision, Recall, F1 und Support je
  Klasse, Modell und Seed
- `test_receiving_hand_confusion_matrix.csv`: Receiving-hand-Auswertung des
  Residual-v2-Modells
- `dataset_comparison.csv`: direkter Vergleich zum optional angegebenen
  älteren Benchmark

Unter `<output-dir>/figures/`, jeweils als PNG mit 300 dpi und als PDF:

1. `01_total_loss_train_validation_by_model`
2. `02_validation_intention_macro_f1_by_model`
3. `03_validation_pose_position_mae_by_model`
4. `04_validation_intention_macro_f1_mean_std`
5. `05_test_intention_macro_f1_by_model`
6. `06_test_pose_position_mae_by_model`
7. `07_test_intention_accuracy_vs_macro_f1`
8. `08_test_intention_confusion_matrices`
9. `09_test_intention_per_class_f1` (Precision, Recall und F1; Support im Titel)
10. `10_test_receiving_hand_confusion_matrix`
11. `11_residual_v2_generalization_gap`
12. `12_dataset_n156_vs_n214` (nur mit `--baseline-comparison-json`)

Sterne markieren die anhand der Validation-Metrik ausgewaehlten Checkpoints.
Da Early Stopping zu unterschiedlich langen Laeufen fuehrt, werden
Mittelwert-und-Standardabweichung-Kurven nur fuer Epochen berechnet, die bei
allen drei Seeds des jeweiligen Modells vorhanden sind. Die einzelnen
Seed-Kurven bleiben separat sichtbar.

Alle Abbildungen tragen Dataset-Tag, Experiment-Tag, Splitdefinition, Seeds und
die Definition der Fehlerbalken. Die exakten Werte bleiben vollständig in den
CSV-Dateien erhalten.

## Weitere Experimentberichte

Die zusätzlichen Auswertungen schreiben ebenfalls maschinenlesbare CSV-/JSON-
Dateien und Abbildungen als PNG und PDF unter
`Training/reports/<dataset_tag>/<experiment_tag>/`:

- `summarize_hyperparameter_search.py`: Stage-A-Ranking, Pareto-Ansicht,
  Parametereffekte und Parallel Coordinates, ausschließlich Validation.
- `summarize_hyperparameter_confirmation.py`: Mittelwert/Standardabweichung der
  Top-3-Konfigurationen über Seeds 42/43/44 und eingefrorene finale Config.
- `freeze_visual_configs.py`: überträgt genau diese validation-ausgewählten
  Sensorhyperparameter auf CLIP-only, Sensor+CLIP und Random-Control und hält
  dabei die visuellen Datenblöcke sowie ihre Provenienz fest.
- `summarize_modality_ablation.py`: Full-vs.-`no_gaze`/`no_hands`/
  `no_objects`/`no_vio` inklusive Deltas und Effizienzindikatoren.
- `summarize_visual_embedding_experiment.py`: Sensorbaseline, CLIP-only,
  Sensor+CLIP und dimensionsgleiche Random-Control, Auswahl nur auf Validation.
- `summarize_visual_final.py`: einmaliger Testvergleich der auf Validation
  eingefrorenen visuellen Variante mit dem getunten Sensormodell.
- `summarize_tuned_residual.py`: gepaarter Drei-Seed-Testvergleich der
  ursprünglichen und validation-ausgewählten Residual-v2-Konfiguration.
- `summarize_latency.py`: identischer Checkpoint und identisches reales Fenster
  über Plattformen, mit Median, Mittelwert, SD, p95, p99, Durchsatz und CDF.
- `summarize_ablation_latency.py`: TCML-CPU/GPU-Latenzwirkung der vier
  Sensorablationen relativ zum Full-Modell.
- `summarize_clip_latency.py`: frozen CLIP-Encoder und vollständige
  RGB-zu-Embedding-Pipeline, getrennt vom temporalen Residual-Transformer.
- `summarize_live_latency_logs.py`: explorative Host-Latenzverteilungen aus
  vorhandenen Mac-Live-Sitzungen; ohne unzulässige Device-/Host-Uhrsubtraktion.

Qualitative Vorhersagen werden mit `export_residual_predictions.py` exportiert.
`render_prediction_overlay.py` erzeugt daraus synchronisierte MP4s und
Thesis-PNGs. Da eine validierte zeitvariable 3D-zu-RGB-Projektion nicht für alle
Aufnahmen garantiert ist, werden Ground Truth und Prediction ausdrücklich in
einem separaten Robot-Frame-Inset gezeigt.

## Zentrale Abschlussauswertung

Nach Abschluss aller Experimente erzeugt folgender Befehl die gemeinsame
Thesis-Tabelle und prüft gleichzeitig Vollständigkeit, testfreie Auswahl,
CLIP-PCA-Split, Checkpoint-/Fixture-Hashes, Overlay-Synchronisation und
Plattformabdeckung:

```bash
python3 Training/evaluation/build_final_experiment_summary.py \
  --dataset-tag dataset_v2_20260802_n214_5d136a34
```

Die Ausgabe liegt unter `Training/reports/<dataset_tag>/`:

- `FINAL_EXPERIMENT_SUMMARY.md`: kompakte, lesbare Ergebnisübersicht;
- `FINAL_EXPERIMENT_SUMMARY.json`: maschinenlesbare Werte und Evidenzchecks;
- `final_test_metrics.csv`: gemeinsame Tabelle aller finalen Modell- und
  Ablationsergebnisse.

Das Skript bricht mit einer Assertion ab, sobald ein erforderlicher Report
unvollständig ist oder eine Provenienzbedingung verletzt wird. Es ersetzt
damit keine Einzelreports, sondern validiert und verknüpft sie.
