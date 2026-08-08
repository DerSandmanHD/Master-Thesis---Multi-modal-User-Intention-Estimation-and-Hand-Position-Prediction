# Literatur- und Experimentdesign-Matrix

Stand: 8. August 2026

Diese Notiz hält die Entscheidungen für Hyperparametersuche, Ablationen,
RGB-/CLIP-Integration und qualitative Visualisierung **vor** den neuen
Experimenten fest. Berücksichtigt wurden ausschließlich die Originalpaper
beziehungsweise die offiziellen Proceedings-Versionen.

## 1. Vergleichsmatrix

| Arbeit | Visueller Eingang / Encoder | Zeitliches Sampling | Training und Fusion | Relevante Ablationen / Darstellungen | Konsequenz für diese Arbeit |
|---|---|---|---|---|---|
| [CLIP (Radford et al., ICML 2021)](https://proceedings.mlr.press/v139/radford21a.html) | Einzelbilder; ResNet- und ViT-Familien, unter anderem ViT-B/32 | Kein Videomodell und daher keine eigene zeitliche Samplingrate | Bild- und Textencoder werden kontrastiv vortrainiert; Transfer wird unter anderem zero-shot und mit linearem Klassifikator auf festem Feature-Raum untersucht | Skalierung von Encoder/Compute, zero-shot gegen lineare Probes und Robustheitsanalysen | Als erste kontrollierte RGB-Erweiterung ViT-B/32 lokal und **frozen** verwenden. Nur der Projektions-/Fusionskopf wird auf n214 gelernt; dadurch bleibt der Vergleich bei 214 Sequenzen realistisch und reproduzierbar. |
| [EgoVLP (Lin et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/31fb284a0aaaad837d2930a610cd5e50-Paper-Conference.pdf) | TimeSformer für Video, DistilBERT für Text; Dual-Encoder | 4 Frames pro Clip beim Pretraining | EgoNCE kontrastiv; Encoder werden auf EgoClip vortrainiert und je nach Downstream-Aufgabe transferiert | Clip-Zuschnitt, positive/negative Samplingstrategie, EgoNCE gegen InfoNCE; Auswahl auf EgoMCQ | Egocentrische Domäne ist relevant und Timestamp-Ausrichtung muss explizit geprüft werden. Vier bis zehn visuelle Samples pro 2-s-Fenster sind ein plausibler Startbereich; Entwicklung bleibt auf Validation. |
| [LaViLa (Zhao et al., CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Learning_Video_Representations_From_Large_Language_Models_CVPR_2023_paper.pdf) | TimeSformer, räumlich aus CLIP-ViT initialisiert; Dual-Encoder | 4 Frames beim Pretraining, 16 Frames beim Downstream-Finetuning | Berichtet zero-shot, End-to-End-Finetuning und lineare Probe mit frozen Encoder | Rephraser/Narrator/Pseudo-Captions, Samplingverfahren, Encodergröße und Datenmenge | Sparse Frames können starke semantische Features liefern. Für den ersten Versuch werden keine externen Captions oder LLMs benötigt; frozen Embeddings isolieren den Nutzen des RGB-Kontexts. |
| [Anticipative Video Transformer, AVT (Girdhar & Grauman, ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Girdhar_Anticipative_Video_Transformer_ICCV_2021_paper.pdf) | Kausaler temporaler Transformer über Framefeatures; optional ViT-Backbone | Standardmäßig 10 Frames bei 1 FPS; Antizipationszeit typischerweise 1 s | Gemeinsame nächste-Aktion-, zukünftige Feature- und optionale Zwischenaktionsverluste | Backbone/Initialisierung, anticipative Verluste; Attention-Visualisierungen über Frames und Bildregionen | Unsere 1-s-Vorhersage bleibt strikt kausal. Die RGB-Pipeline darf kein Bild nach dem Ende des Beobachtungsfensters verwenden. Auxiliary future losses sind eine spätere, getrennte Ablation und werden nicht in die erste Suche gemischt. |
| [RU-LSTM + Modality Attention (Furnari & Farinella, ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Furnari_What_Would_You_Expect_Anticipating_Egocentric_Actions_With_Rolling-Unrolling_LSTMs_ICCV_2019_paper.pdf) | RGB, Optical Flow und detektierte Objekte; eigener Zweig je Modalität | Fünf Frames pro Snippet; Auswertung über Antizipationszeiten von 2,0 bis 0,25 s | Adaptive Modality Attention über zweigspezifische Vorhersagen | Einzelmodalitäten, Early/Late Fusion, Modality Attention, Sequence-Completion-Pretraining und Rolling-Unrolling | Modalitäten werden einzeln entfernt und stets gegen das unveränderte Full-Modell verglichen. Einfache Konkatenation/Projektion ist die kontrollierte erste CLIP-Fusion; ein learned Gate wäre ein separates Folgeexperiment. |
| [Joint Hand Motion and Interaction Hotspots / OCT (Liu et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.pdf) | TSN-RGB-Features sowie Hand-, Objekt- und globale RoI-/Bildfeatures; Object-Centric Transformer | Epic-Kitchens: 10 Frames bei 4 FPS, 1 s Prognose; EGTEA: 9 Frames bei 6 FPS, 0,5 s Prognose | Gemeinsame Handtrajektorien- und Kontaktpunktvorhersage, stochastische C-VAE-Köpfe | Hand/Objekt/global entfernen, Beobachtungslänge, deterministische gegen stochastische Köpfe, Trainingsdatenmenge; Trajektorien-/Hotspot-Overlays | 5 Hz für gecachte CLIP-Embeddings ergibt zehn visuelle Samples in unserem 2-s-Fenster und liegt nahe am OCT-Protokoll. Die Videoausgabe zeigt GT/Prediction in getrennten Farben und quantifiziert den Fehler. |

## 2. Übernommene Designprinzipien

Aus den Arbeiten werden folgende Prinzipien übernommen:

1. **Kausalität:** Ein Fenster mit Endzeit `t` darf nur RGB-Frames mit
   Timestamp `<= t` enthalten. Der bestehende Vorhersagehorizont bleibt 1 s.
2. **Participant-wise Split:** Alle Auswahlentscheidungen verwenden weiterhin
   Atilla, Ermal und Vanessa als Validation und Edu, Jona und Mona als Test.
3. **Sparse visuelle Features:** Zunächst 5 Hz, also höchstens zehn
   CLIP-Samples pro 60-Frame-/2-s-Fenster.
4. **Frozen Encoder zuerst:** CLIP ViT-B/32 wird nicht auf n214 finetuned.
   Trainiert werden nur Projektion, zeitliche Aggregation und Aufgaben-Köpfe.
5. **Kontrollierte Fusion:** Bestehende Sensormodalitäten bleiben unverändert;
   visuelle Embeddings erhalten eine eigene Projektion und Missing-Maske.
6. **Ein Faktor pro Ablation:** Keine Ablation erhält eigene nachträgliche
   Hyperparameter. So misst das Delta den Modalitätsbeitrag statt Retuning.
7. **Mehrere Seeds:** Abschließende Vergleiche verwenden 42, 43 und 44 und
   berichten Mittelwert plus Populationsstandardabweichung.

## 3. Vorab festgelegtes Hyperparameter-Suchprotokoll

### 3.1 Ziel und Auswahlregel

Die Suche verwendet ausschließlich Validation-Metriken:

1. primär: Validation-Intention-Macro-F1 maximieren;
2. Kandidaten innerhalb von `0,005` des besten F1 gelten als statistisch
   praktisch gleichwertige F1-Gruppe;
3. innerhalb dieser Gruppe: Validation-Pose-MAE minimieren;
4. danach: Validation-Receiving-Hand-Macro-F1 maximieren;
5. danach: weniger trainierbare Parameter bevorzugen.

Das ist eine vorab definierte Pareto-nahe, lexikografische Auswahl. Es wird
kein gewichteter Score nach Sichtung der Ergebnisse erfunden.

### 3.2 Stufen

- **Stufe A:** 24 reproduzierbare Random-Search-Trials, Such-Seed 20260808,
  Trainings-Seed 42, maximal 20 Epochen, Early-Stopping-Patience 7. Keine
  Testauswertung.
- **Stufe B:** Die besten drei unterschiedlichen Konfigurationen aus Stufe A
  mit Seeds 42, 43 und 44. Weiterhin keine Testauswertung.
- **Stufe C:** Auswahl nach der obigen Regel über Mittelwerte der drei Seeds.
- **Stufe D:** Die exakt festgeschriebene Gewinnerkonfiguration wird einmal als
  finales Experiment mit Seeds 42, 43 und 44 auf dem unveränderten Testsplit
  ausgewertet. Danach findet kein Nachjustieren statt.

### 3.3 Suchraum

| Parameter | Verteilung/Kandidaten |
|---|---|
| Learning Rate | log-uniform `1e-5` bis `1e-3` |
| Weight Decay | `0`, `1e-5`, `1e-4`, `1e-3` |
| Dropout | `0,05`, `0,15`, `0,30` |
| `d_model` | `32`, `64`, `128` |
| Attention Heads | `2`, `4`, `8`; muss `d_model` teilen |
| Transformer-Layer | `1`, `2`, `3` |
| Feedforward-Dimension | `64`, `128`, `256`; mindestens `d_model` |
| Batchgröße | `16`, `32`, `64` |
| Orientierungs-Lossgewicht | `0,10`, `0,25`, `0,50` |
| Receiving-Hand-Lossgewicht | `0,5`, `1,0`, `2,0` |

Fensterlänge, Stride, Horizont, Datensatz, Splits und die übrigen Lossgewichte
bleiben konstant. Damit vergleicht Stufe A Modell-/Optimierungseinstellungen
und nicht gleichzeitig verschiedene Stichprobenmengen.

## 4. Vorab festgelegtes Ablationsprotokoll

### 4.1 Bestehende Sensormodalitäten

Verglichen werden:

- Full Residual v2 aus `benchmark_v2`;
- `no_gaze`;
- `no_hands`;
- `no_objects`;
- `no_vio`.

Alle Varianten verwenden Dataset `dataset_v2_20260802_n214_5d136a34`, Seeds
42/43/44 und zunächst die ursprünglichen Residual-v2-Hyperparameter. Berichtet
werden:

- Intention-Macro-F1 und Delta zum seed-gepaarten Full-Modell;
- Receiving-Hand-Macro-F1 und Delta;
- Pose-MAE in cm und Delta;
- trainierbare Parameter und Forward-Latenz;
- Mittelwert und Populationsstandardabweichung.

`no_hands` entfernt Handfeatures als **Eingang**, nicht die Handreferenz als
Trainingsziel der Poseaufgabe.

### 4.2 RGB-/CLIP-Vergleich

Nach Implementierung werden ohne Testblick festgelegt und verglichen:

- `sensor_full`: bestehendes Residual-Modell;
- `clip_only`: ausschließlich CLIP plus Zeit-/Missing-Maske;
- `sensor_plus_clip`: bestehende Features plus CLIP;
- `sensor_plus_random_frozen`: dimensionsgleiche, deterministische frozen
  Zufallsfeatures als Sanity Check.

Alle drei visuellen Varianten werden im Screening mit Seeds 42, 43 und 44
ausschließlich auf Validation verglichen. Danach wird genau eine Architektur
eingefroren und einmalig mit denselben drei Seeds auf Test ausgewertet.

## 5. Festgelegtes CLIP-Protokoll

- Encoder: OpenAI CLIP `ViT-B/32`, lokal ausgeführt, keine externe Bild-API.
- Preprocessing: offizielle Encodertransformation; Versions- und Gewichts-Hash
  werden gespeichert.
- Sampling: 5 Hz aus dem RGB-Stream, kausal über Device-Timestamps.
- Cache: pro Sequenz Timestamps, L2-normalisierte 512-D-Embeddings in
  `float16`, Validitätsmaske und Provenienz-JSON.
- Zuordnung zum 30-Hz-Takt: letztes verfügbares Embedding (`timestamp <= t`),
  maximale erlaubte Alterung explizit protokollieren; sonst Missing-Maske.
- Projektion: feste PCA von 512 auf 32 Dimensionen, ausschließlich auf
  Trainingsteilnehmern angepasst; die nachfolgende Modellprojektion bleibt
  lernbar.
- Keine PCA über den Gesamtdatensatz; Validation- und Testteilnehmer sind beim
  Fit explizit ausgeschlossen und werden in der Projektionsmetadatei gelistet.
- Cache-Fingerprint umfasst Sequenzliste, Video-Hash, Encodername,
  Gewichts-Hash, Paketversion, Preprocessing und Samplingrate.

## 6. Datenschutz und Speicher

RGB-Aufnahmen können Gesichter, private Umgebungen, Bildschirminhalte und
weitere identifizierende Merkmale enthalten. CLIP-Embeddings sind abgeleitete
Daten und werden nicht als anonym angenommen. Daher gilt:

- Verarbeitung nur lokal beziehungsweise auf dem bestehenden TCML-Speicher;
- keine Übertragung von Frames an externe APIs;
- Zugriff und Aufbewahrung entsprechen den Regeln der ursprünglichen
  Einwilligung und des Forschungsprojekts;
- Rohframes werden nicht zusätzlich dupliziert, wenn VRS/MP4 bereits vorliegt;
- gecachte Embeddings erhalten dieselben Zugriffsbeschränkungen wie Rohdaten;
- qualitative Videos werden vor Weitergabe auf unbeabsichtigte Personen,
  Bildschirme und andere sensible Inhalte geprüft;
- Veröffentlichungen zeigen nur freigegebene, repräsentative Ausschnitte;
- Speicherbedarf, Hashes und Löschpfad des Caches werden im Cache-Manifest
  dokumentiert.

## 7. Für die Ergebnisdarstellung übernommene Formate

- RU-LSTM: Leistung über verschiedene Antizipationszeiten und explizite
  Modalitätsablationen;
- AVT: kausale zeitliche Aufmerksamkeit und Best-Epoch-/Loss-Ablationen;
- OCT: farblich getrennte GT-/Prediction-Trajektorien sowie Erfolgs- und
  Fehlerbeispiele;
- EgoVLP/LaViLa: tabellarische Vergleiche der Pretraining-/Transfer-Varianten;
- diese Arbeit zusätzlich: Mittelwert ± SD über Seeds, Konfusionsmatrizen,
  Per-Class-Metriken, Latenz-CDF und Pareto-Plot F1 gegen Posefehler.

## 8. Grenzen der Übertragbarkeit

Die zitierten Arbeiten verwenden wesentlich größere Datensätze und andere
Zielgrößen. Ihre exakten Learning Rates oder Modellgrößen sind daher keine
direkten Optima für n214. Übernommen werden robuste Versuchsprinzipien
(kausales Sampling, sparse Frames, frozen Transfer, Einzelablationen), während
der konkrete Suchraum bewusst um den vorhandenen Residual-v2-Ausgangspunkt
zentriert ist.
