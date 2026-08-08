# Frozen-CLIP-Embeddingcache (`n214`)

SLURM-Job: `2204123`

Der lokale OpenAI-CLIP-Encoder `ViT-B/32` wurde frozen bei 5 Hz auf allen 214
ausgewählten RGB-Sequenzen ausgeführt. Es wurden 36.874 normalisierte
512-D-Embeddings erzeugt; 214/214 Sequenzen sind vollständig und das Manifest
enthält keine Fehler. Der Cache belegt auf BeeGFS 35 MB (35.841.664 Byte).

| Eigenschaft | Wert |
|---|---:|
| Sequenzen | 214/214 |
| Embeddings | 36.874 |
| gemessene Extraktionszeit (Summe) | 2.733,94 s |
| Embeddings pro Sekunde | 13,49 |
| Cache-Fehler | 0 |
| Encoderparameter (frozen) | 151.277.313 |

Encoder-Fingerprint:
`19cce9aeb3ff9e60a82b1cd864258e3c27f4e9dc9a87e9965f62013dab1bab20`

Gewichte SHA-256:
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

## Leakage-freie Projektion

Die feste PCA reduziert 512 auf 32 Dimensionen. Sie wurde nur mit 28.996
Embeddings aus 170 Trainingssequenzen gefittet. Validation-Teilnehmer Atilla,
Ermal und Vanessa sowie Testteilnehmer Edu, Jona und Mona sind explizit
ausgeschlossen. Die 32 Komponenten erklären zusammen 78,71 % der Varianz.
Der abschließende Dataset-/Modell-Smoke-Test meldete für Sensor+CLIP 100,00 %
kausale Abdeckung, null Zuordnungen aus der Zukunft und gültige Modellshapes
(124 Rohfeatures, 248 Features nach Normalisierung und Missing-Masken).

- Projektion:
  `Training/visual_projections/dataset_v2_20260802_n214_5d136a34/clip_vit_b32_openai_5hz_pca32.npz`
- Projektion SHA-256:
  `2c1b4b1824d9f956c6e1a0bbf10f98cc4015c986e44605e0a783316147eef3a2`
- Cache-Manifest SHA-256:
  `c08d356d34f19d6b40847cac3b0f1856b5649c0bc46cc856708611045a23b2a8`

Die einzelnen Cachedateien bleiben wegen Datenmenge und Datenschutz unter
`Data_collection/visual_embeddings/clip_vit_b32_openai_5hz/` und werden nicht
in Git versioniert. Das getrackte Manifest enthält ihren jeweiligen SHA-256-
Hash und bildet damit den reproduzierbaren Index. Falls die abgeleiteten
Bildfeatures gelöscht werden müssen, ist genau dieses Cacheverzeichnis der
Löschpfad; Rohvideos werden davon nicht berührt.
