# Plattformvergleich der finalen Sensor+CLIP-Modelllatenz

Checkpoint SHA-256:
`c9de5f091b1230bd0117a99a3fbbd69ae2c28ac67353fc5e644bec56bf73967b`

Alle Plattformen verwenden dasselbe reale Testfenster mit SHA-256
`70e9fbf36bf67182bc50b5d719e8e5325559b66ef60fe82e25ef9ad011ccdf6d`,
Batchgröße 1, 100 Warm-ups und 1.000 synchronisierte Messungen.

| Plattform | Forward Median / P95 | Offline-Fenster Median / P95 | ≤ 33,3 ms |
|---|---:|---:|---:|
| Mac CPU | **1,112 / 1,223 ms** | **1,118 / 1,208 ms** | 100 % |
| Mac MPS | 2,602 / 2,851 ms | 2,755 / 2,937 ms | 100 % |
| Uni `login3` CPU | 2,395 / 2,792 ms | 2,464 / 2,562 ms | 100 % |
| TCML Compute CPU | 1,730 / 1,893 ms | 1,780 / 1,974 ms | 100 % |
| TCML Compute CUDA (RTX 2080 Ti) | 1,585 / 1,611 ms | 1,732 / 1,759 ms | 100 % |

`login3` besitzt keine für PyTorch verfügbare GPU; der angeforderte CUDA-Lauf
ist deshalb als `unavailable` statt als fehlende Messung protokolliert.

Diese Tabelle misst den temporalen Fusions-Transformer mit bereits
vorliegenden CLIP-Features. Der getrennte, exakt pixelgleiche
`../clip_vit_b32_openai_5hz/`-Bericht misst das RGB-Frontend: median
`50,52 ms` (Mac CPU), `40,78 ms` (Mac MPS), `74,39 ms` (TCML CPU) und
`25,94 ms` (TCML CUDA) bei einem 5-Hz-Budget von 200 ms. Die visuelle
Berechnung läuft daher nicht an jedem 30-Hz-Sensortakt; ihr letztes kausales
Embedding wird gehalten.

`summary.json`, `latency_summary.csv`, `latency_samples.csv`, die sechs
Plattform-JSONs und die PNG-/PDF-Abbildungen enthalten alle Rohwerte und
Provenienznachweise.
