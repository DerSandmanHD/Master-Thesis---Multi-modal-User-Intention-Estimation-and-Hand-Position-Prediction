# Frozen-CLIP-Latenz (`ViT-B/32`, 5 Hz)

Status: vollständig, vier Plattform-/Device-Kombinationen

TCML-SLURM-Job: `2204145`

Pro Messung wurden 100 Warm-ups und 1.000 synchronisierte Wiederholungen mit
Batchgröße 1 ausgeführt. Der visuelle 5-Hz-Takt ergibt ein Budget von 200 ms.

| Plattform | Encoder Median | RGB→Embedding Median | RGB→Embedding P95 | ≤ 200 ms |
|---|---:|---:|---:|---:|
| Mac CPU | 35,830 ms | 50,521 ms | 53,309 ms | 100 % |
| Mac MPS | 19,984 ms | 40,784 ms | 45,194 ms | 100 % |
| TCML CPU (`tcml-node39`) | 46,072 ms | 74,393 ms | 76,106 ms | 100 % |
| TCML CUDA (RTX 2080 Ti) | **3,607 ms** | **25,943 ms** | **26,477 ms** | 100 % |

## Vergleichbarkeit

Alle vier Läufe verwendeten dieselben, vor dem Laden verifizierten OpenAI-
CLIP-Gewichte und dasselbe gespeicherte RGB-Pixelarray. Dadurch fließen keine
plattformabhängigen Unterschiede des macOS-/Linux-Videodecoders in den
Encodervergleich ein.

- Gewichte SHA-256:
  `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- RGB-Fixture SHA-256:
  `aeb0c8a9e62ab36f494f02a1d248cb8e315643223948c3a27db3e6175ca995cd`
- dekodierte RGB-Pixel SHA-256:
  `758350289ecae269d84f1f618e9adbdfcc38afc7e86679e372958e3a78740a83`
- Quellvideo SHA-256:
  `8fd29091b022a530b4f8f185fd01339bc71e8ad6ea6cda79039f08874d85c722`

Die Werte messen den visuellen Frontend-Aufwand getrennt vom temporalen
Residual-Transformer. Beim Training werden gecachte Embeddings verwendet; für
eine Live-Pipeline muss der RGB→Embedding-Aufwand am 5-Hz-Takt zusätzlich zur
30-Hz-Modelllatenz berücksichtigt werden.

Maschinenlesbare Quellen: `summary.json`, `clip_latency_summary.csv`, die vier
Plattform-JSONs und `figures/01_clip_latency.{png,pdf}`.
