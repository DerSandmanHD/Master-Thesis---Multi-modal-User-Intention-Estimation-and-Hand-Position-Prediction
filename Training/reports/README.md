# Evaluation reports

Neue aggregierte Auswertungen werden nach Dataset und Experiment getrennt:

```text
Training/reports/<dataset_tag>/<experiment_tag>/
```

`compare_final_runs.py` und `evaluation/generate_training_diagrams.py`
verwenden diese Struktur automatisch, sobald `--dataset-tag` gesetzt ist.
Nur kleine, endgültig akzeptierte Reports und Abbildungen sollten in Git
aufgenommen werden; vorläufige Auswertungen können lokal bleiben.

Die vorhandenen Dateien `final_clean_v1_comparison.*` sind unveränderte
Legacy-Reports. Auch `Training/evaluation/generated/` bleibt als historischer
Diagrammstand bestehen.
