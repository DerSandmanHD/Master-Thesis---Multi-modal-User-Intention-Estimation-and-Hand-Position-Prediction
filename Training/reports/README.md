# Evaluation reports

Neue aggregierte Auswertungen werden nach Dataset und Experiment getrennt:

```text
Training/reports/<dataset_tag>/<experiment_tag>/
```

Der autoritative v2-Matrixbericht ist erst vollständig, wenn unter
`postprocess/residual_current_gate_seed{42,43,44}/` jeweils der hashgebundene
Prediction-Sidecar und `grouped_metrics.json` vorliegen. Fehlende t+1-Baselines
werden nicht als optionales `n/a` in einen finalen Hauptbericht übernommen.

`compare_final_runs.py` und `evaluation/generate_training_diagrams.py`
verwenden diese Struktur automatisch, sobald `--dataset-tag` gesetzt ist.
Nur kleine, endgültig akzeptierte Reports und Abbildungen sollten in Git
aufgenommen werden; vorläufige Auswertungen können lokal bleiben.

Die vorhandenen Dateien `final_clean_v1_comparison.*` sind unveränderte
Legacy-Reports. Auch `Training/evaluation/generated/` bleibt als historischer
Diagrammstand bestehen.

Der aktive geplante Protokollstand ist
`dataset_v3_causal_20260815_n214_5d136a34`. Solange Master-Rebuild, CLIP-v2,
Training und autorisierte Evaluation nicht tatsächlich gelaufen sind, gibt es
unter diesem Tag keine behaupteten Resultate. Der Report
`dataset_v2_20260802_n214_5d136a34/ARTIFACT_VALIDITY_V2.json` ist ein
historisches Invaliditätsinventar: Er beschreibt, warum alte CLIP-, zentrale
Checkpoint-, Terminal- und qualitative Artefakte nicht als finale v2-
Ergebnisse verwendet werden dürfen.
