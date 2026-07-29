# Projektorganisation für zukünftige Trainingsläufe

**Stand:** 29. Juli 2026

Ziel ist eine eindeutige Zuordnung von Datasetstand, Experiment, Modell und
Seed, ohne historische Runs oder Checkpoints zu verändern.

## Neue Struktur

| Bereich | Pfad |
|---|---|
| Modellkonfigurationen | `Training/configs/models/` |
| Ablationskonfigurationen | `Training/configs/ablations/` |
| SLURM-Jobs | `Training/jobs/` |
| Dataset-Deskriptoren | `Training/datasets/` |
| Trainingsruns | `Training/runs/<dataset>/<experiment>/<model>/<run_id>/` |
| Vergleichsreports | `Training/reports/<dataset>/<experiment>/` |
| Replay-/Live-Sitzungen | `Training/live_runs/<dataset>/<session>/` |
| SLURM-Ausgaben | `Training/slurm_logs/` |

`Training/run_registry.json` registriert eingefrorene Datasetstände und
akzeptierte Experimente. `Training/run_layout.py` erzeugt und validiert die
kanonischen Pfade und Tags.

## Umbenannte Einstiegspunkte

| Vorher | Jetzt |
|---|---|
| `configs/hierarchical_baseline_v1.json` | `configs/models/transformer_v1.json` |
| `configs/hierarchical_mlp_v1.json` | `configs/models/mlp_v1.json` |
| `configs/hierarchical_gru_v1.json` | `configs/models/gru_v1.json` |
| `configs/hierarchical_residual_v2.json` | `configs/models/residual_transformer_v2.json` |
| `final_comparison.sbatch` | `jobs/benchmark_models.sbatch` |
| `residual_v2_modalities.sbatch` | `jobs/ablate_modalities_residual_v2.sbatch` |
| `hierarchical_*.sbatch` | `jobs/train_<model>.sbatch` |
| `export_checkpoint_predictions.sbatch` | `jobs/export_predictions.sbatch` |

Die Modellhyperparameter der vier verschobenen Konfigurationen wurden nicht
verändert. Nur die kurzen `run_name`-Werte wurden an die neuen Modell-Tags
angepasst.

## Legacy-Bestand

Die bisherigen `final_clean_v1`-Runs, der Deployment-Spiegel
`Training/final_clean_v1_residual_v2_seed44`, vorhandene Dateien unter
`Training/Outputs/`, die flachen Vergleichsreports und
`Training/evaluation/generated/` bleiben an ihren bisherigen Orten. Sie sind
im Registry als Legacy-Bestand beschrieben und werden weiterhin rekursiv
gefunden.

## Ablauf für den nächsten Trainingsstand

1. Neue Aufnahmen, Master-Build und QA abschließen.
2. Dataset-Snapshot validieren und einen neuen Descriptor unter
   `Training/datasets/` anlegen.
3. Dataset-Tag in `Training/run_registry.json` aufnehmen.
4. Den Benchmark mit explizitem `DATASET_TAG` und neuem `EXPERIMENT_TAG`
   starten.
5. Vergleich und Diagramme aus genau diesem verschachtelten Experimentpfad
   erzeugen.
6. Erst nach Annahme des Ergebnisses den Experimentdatensatz im Registry
   ergänzen.

## Verifikation

- Struktur-, Registry- und rekursive Run-Discovery-Smoke-Tests bestanden.
- Alle sieben neuen SLURM-Dateien bestehen `bash -n`.
- Transformer-, MLP-, GRU-, Residual-v2- und Ablations-Smoke-Tests bestanden.
- Die zwölf historischen `final_clean_v1`-Runs werden weiterhin vollständig
  verglichen und zur Diagrammerzeugung geladen.
