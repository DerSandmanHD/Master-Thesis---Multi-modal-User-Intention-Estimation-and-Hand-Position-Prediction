# Training configurations

- `models/` enthält die vier vollständigen Modellkonfigurationen für den
  gemeinsamen Benchmark:
  `transformer_v1.json`, `mlp_v1.json`, `gru_v1.json` und
  `residual_transformer_v2.json`.
- `ablations/` enthält Residual-v2-Varianten, bei denen jeweils eine
  Featuregruppe entfernt wird.

Ein Trainingslauf kopiert seine tatsächlich verwendete und um Laufkontext
ergänzte Konfiguration als `config.json` in das Run-Verzeichnis. Deshalb
werden abgeschlossene Runs nie durch eine spätere Änderung dieser Vorlagen
umdefiniert.

Die alten flachen Namen `hierarchical_*.json` wurden in `models/` eindeutig
umbenannt. Historische `config.json`-Snapshots in bestehenden Runs bleiben
unverändert und sind weiterhin die maßgebliche Beschreibung dieser Läufe.
