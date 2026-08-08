# Experiment-Roadmap für die Masterarbeit

Stand: 8. August 2026

Diese Datei verfolgt die noch offenen Experimente und Artefakte für die
Masterarbeit. Sie unterscheidet zwischen bereits vorhandener Infrastruktur,
noch auszuführenden Experimenten und neu zu entwickelnden Werkzeugen.

## 1. Aktueller, reproduzierbarer Ausgangspunkt

- [x] Datasetstand eingefroren:
  `dataset_v2_20260802_n214_5d136a34`
- [x] 214 saubere Sequenzen ausgewählt
- [x] Participant-wise Split:
  170 Training / 21 Validation / 23 Test
- [x] Fenster:
  15.189 Training / 1.978 Validation / 2.199 Test
- [x] Vier Modellfamilien mit Seeds 42, 43 und 44 trainiert
- [x] Alle zwölf Benchmark-Läufe erfolgreich abgeschlossen
- [x] Residual v2 ist die aktuelle Hauptmethode:
  - Intention Macro-F1: `0,8579 ± 0,0015`
  - Posefehler: `14,62 ± 0,13 cm`
- [x] Alter `n156`-Benchmark ist weiterhin vollständig archiviert
- [x] Neue `n214`-Runs und Vergleichsberichte vollständig vom Cluster lokal
  spiegeln
- [x] Benchmark-Registry nach der Ergebnissicherung von `scheduled` auf
  `completed` setzen
- [x] Zentrale maschinenlesbare Ergebniszusammenfassung für `n214` erstellen

Verbindlicher Grundsatz: Hyperparameter und Modellvarianten werden nur anhand
des Validation-Splits ausgewählt. Der Testsplit wird erst nach Abschluss einer
Entscheidungsstufe ausgewertet und nie zum Nachjustieren verwendet.

## 2. Bereits vorhandene Graphen

Unter `Training/evaluation/generated/figures/` existieren bereits:

- [x] Train- und Validation-Loss pro Modell
- [x] Validation-Intention-Macro-F1 pro Modell
- [x] Validation-Posefehler pro Modell
- [x] Validation-Macro-F1 als Mittelwert und Standardabweichung über Seeds

Diese Graphen wurden aus dem alten `final_clean_v1`-/`n156`-Benchmark erzeugt.
Das Skript liegt unter:

```text
Training/evaluation/generate_training_diagrams.py
```

Noch offen:

- [x] Generator auf die verschachtelte `n214`-Runstruktur anwenden/anpassen
- [x] Alle bestehenden vier Graphen für `n214` neu erzeugen
- [x] Test-Macro-F1 der vier Modelle mit Fehlerbalken darstellen
- [x] Posefehler der vier Modelle mit Fehlerbalken darstellen
- [x] Accuracy und Macro-F1 gemeinsam, aber klar getrennt berichten
- [x] Konfusionsmatrizen für `continue`, `fetch` und `handover` erzeugen
- [x] Per-Class Precision, Recall, F1 und Support darstellen
- [x] Receiving-hand-Konfusionsmatrix für Residual v2 erzeugen
- [x] Lernkurven des Residual-Modells inklusive Best-Epoch-Markierung darstellen
- [x] Vergleich `n156` gegen `n214` erstellen
- [x] Alle finalen Abbildungen als PNG und PDF exportieren
- [x] Jede Abbildung mit Dataset-Tag, Split, Metrikdefinition und Seeds versehen

Abnahmekriterium: Alle Zahlen in den Abbildungen müssen maschinell aus den
jeweiligen `metrics.json`-Dateien stammen; keine manuell übertragenen Werte.

## 3. Hyperparametersuche für Residual v2

### 3.1 Bedeutung

Hyperparameter sind Einstellungen, die nicht vom Modell gelernt werden,
sondern vor dem Training festgelegt werden. Eine Hyperparametersuche trainiert
mehrere Varianten und vergleicht sie ausschließlich auf dem Validation-Split.

### 3.2 Momentane Hyperparameter

Quelle: `Training/configs/models/residual_transformer_v2.json`

#### Daten und zeitliches Fenster

| Hyperparameter | Aktueller Wert |
|---|---:|
| Fensterlänge | 60 Frames (ca. 2 s) |
| Stride | 10 Frames |
| Vorhersagehorizont | 1,0 s |
| minimale beobachtete Feature-Quote | 0,05 |
| maximaler Timestamp-Sprung | 0,2 s |

#### Architektur

| Hyperparameter | Aktueller Wert |
|---|---:|
| `d_model` | 64 |
| Attention Heads | 4 |
| Transformer-Layer | 2 |
| Feedforward-Dimension | 128 |
| Dropout | 0,15 |

#### Optimierung

| Hyperparameter | Aktueller Wert |
|---|---:|
| Epochen (Maximum) | 20 |
| Batchgröße | 32 |
| Learning Rate | 0,0003 |
| Weight Decay | 0,0001 |
| Gradient Clipping | 1,0 |
| Early-Stopping-Patience | 7 |
| Assistance-Lossgewicht | 1,0 |
| Assistance-Type-Lossgewicht | 1,0 |
| Receiving-Hand-Lossgewicht | 1,0 |
| Pose-Lossgewicht | 1,0 |
| Orientierungs-Lossgewicht | 0,25 |

Die Live-Parameter `smoothing_window=3`, `minimum_confidence=0.65` und
`minimum_stable_predictions=2` sind keine Trainingshyperparameter. Sie werden
später separat auf Validation-/Replay-Daten kalibriert.

### 3.3 Suchprotokoll

- [x] Literatur und vergleichbare Arbeiten nach verwendeten Suchräumen prüfen
- [x] Suchziel vor Beginn festlegen
  - primär: Validation-Intention-Macro-F1 maximieren
  - zusätzlich: Validation-Pose-MAE minimieren
  - Receiving-Hand-Macro-F1 als Nebenmetrik protokollieren
- [x] Entscheiden, ob eine Pareto-Auswahl oder ein vorab definierter
  kombinierter Score verwendet wird
- [x] Reproduzierbares Suchskript erstellen (bevorzugt Random Search oder
  Optuna statt vollständigem Grid Search)
- [x] Jeden Trial mit Konfiguration, Seed, Git-Commit, Dataset-Tag und Laufzeit
  speichern
- [x] Resume-Funktion für unterbrochene Clusterläufe vorsehen

Vorgeschlagener erster Suchraum:

| Hyperparameter | Kandidaten/Suchraum |
|---|---|
| Learning Rate | log-uniform `1e-5` bis `1e-3` |
| Weight Decay | `0`, `1e-5`, `1e-4`, `1e-3` |
| Dropout | `0,05`, `0,15`, `0,30` |
| `d_model` | `32`, `64`, `128` |
| Attention Heads | `2`, `4`, `8` (muss `d_model` teilen) |
| Transformer-Layer | `1`, `2`, `3` |
| Feedforward-Dimension | `64`, `128`, `256` |
| Batchgröße | `16`, `32`, `64` |
| Orientierungs-Lossgewicht | `0,10`, `0,25`, `0,50` |
| Receiving-Hand-Lossgewicht | `0,5`, `1,0`, `2,0` |

Empfohlenes Vorgehen:

- [ ] Stufe A: 20–30 Trials mit einem festen Such-Seed auf identischem Split
- [ ] Stufe B: beste 3–5 Konfigurationen mit Seeds 42, 43 und 44 bestätigen
- [ ] Stufe C: Konfiguration anhand Validation auswählen
- [ ] Stufe D: genau eine finale Testauswertung durchführen
- [ ] Suchergebnisse als Parallel-Coordinates-Plot und
  Hyperparameter-vs.-Metrik-Graphen darstellen
- [ ] Alte Baseline gegen das getunte Modell mit identischen Seeds vergleichen

Abnahmekriterium: Keine Auswahlentscheidung darf auf Testmetriken basieren.

## 4. Video-Visualisierung von Ground Truth und Modellvorhersage

Ziel ist ein annotiertes Video, das synchron zeigt:

- RGB-Kamerabild
- Ground-Truth-Intention
- Modellwahrscheinlichkeiten für `continue`, `fetch`, `handover`
- vorhergesagte Intention
- Ground-Truth-Empfangshand (`left`/`right`)
- vorhergesagte Empfangshand und Wahrscheinlichkeit
- Ground-Truth-Handpose/-position
- vorhergesagte Handpose/-position
- optional Positionsfehler in Zentimetern und Orientierungsfehler in Grad

Geplantes Layout:

```text
+---------------------------------------------------------------+
| GT: HANDOVER | Pred: HANDOVER | C: .03 F: .08 H: .89         |
| GT hand: right | Pred hand: right (.94) | error: 8.2 cm       |
+---------------------------------------------------------------+
|                                                               |
|                       RGB-Kamerabild                           |
|        Ground Truth: grüner Marker / Trajektorie              |
|        Prediction:  roter Marker / Trajektorie                |
|                                                               |
+---------------------------------------------------------------+
```

Umsetzung:

- [ ] 2–3 repräsentative Testsequenzen auswählen
- [x] Modellvorhersagen über `export_checkpoint_predictions.py` oder Replay
  exportieren
- [x] RGB-Frames und Modellfenster über Device-Timestamps synchronisieren
- [x] Prüfen, ob 3D-Handposen zuverlässig in das RGB-Bild projiziert werden
  können
- [ ] Falls Projektion möglich: Kameraextrinsics/-intrinsics und VIO-Pose
  verwenden
- [x] Falls Projektion nicht belastbar ist: 3D-Ground-Truth und Prediction in
  einem separaten Robot-Frame-Inset darstellen
- [x] Overlay-Skript mit OpenCV implementieren
- [x] Wahrscheinlichkeiten als Balken statt nur als Text darstellen
- [x] Farblegende und klare Kennzeichnung von Ground Truth/Prediction ergänzen
- [ ] Annotiertes MP4 sowie einzelne Thesis-Abbildungen exportieren
- [ ] Erfolgs- und Fehlerbeispiel zeigen

Abnahmekriterium: Die Synchronisation muss durch Timestamp-Prüfungen belegt
werden; eine visuell plausible, aber zeitlich ungesicherte Überlagerung reicht
nicht aus.

## 5. RGB-/CLIP-Embeddings

Ziel: Prüfen, ob visuelle Kontextfeatures aus RGB-Frames die bestehende
multimodale Baseline verbessern.

### 5.1 Literatur und Design

- [x] 4–6 Arbeiten zu egocentric vision, multimodaler Intentionserkennung und
  CLIP-/Vision-Embeddings auswählen
- [x] Pro Paper dokumentieren:
  - verwendeter Bildencoder
  - Samplingrate
  - Frozen vs. Fine-Tuning
  - zeitliche Aggregation
  - Fusionsmethode
  - Ablationen
  - Metriken und Graphen
- [x] Datenschutz- und Speicherfolgen der RGB-Verarbeitung dokumentieren

### 5.2 Technische Umsetzung

- [x] CLIP-Variante festlegen, zunächst kleiner frozen Encoder
- [x] RGB-Frames kausal und timestamp-synchron aus VRS/MP4 extrahieren
- [x] Samplingrate festlegen, zunächst 2–5 Hz
- [ ] Embeddings einmalig berechnen und pro Sequenz cachen
- [x] Hashes von Encoder, Gewichten, Preprocessing und Embeddingdateien sichern
- [x] Dimension reduzieren/projizieren, ohne Validation/Test-Leakage
- [x] Embeddings kausal auf den 30-Hz-Modelltakt übertragen
- [x] Missing-Embedding-Maske ergänzen
- [x] Datenloader und Featureprovenienz erweitern
- [x] Smoke-Test für Zeitabgleich, Shapes und Missing Frames erstellen

### 5.3 Vergleichsexperimente

- [ ] Bestehende multimodale Residual-v2-Baseline
- [ ] CLIP-only Baseline
- [ ] bestehende Features + CLIP
- [ ] bestehende Features + zufällige/frozen Kontrollfeatures als Sanity Check
- [ ] jede finale Variante mit Seeds 42, 43 und 44 trainieren
- [ ] Intention-F1, Receiving-Hand-F1, Pose-MAE, Parameterzahl und Latenz
  vergleichen
- [ ] CLIP-spezifische Ablation (`with CLIP` vs. `without CLIP`) ergänzen

Abnahmekriterium: Testverbesserungen werden erst berichtet, nachdem Architektur
und Hyperparameter anhand Validation festgelegt wurden.

## 6. Ablationsstudie

Die Infrastruktur ist bereits vorhanden:

```text
Training/configs/ablations/residual_v2_no_gaze.json
Training/configs/ablations/residual_v2_no_hands.json
Training/configs/ablations/residual_v2_no_objects.json
Training/configs/ablations/residual_v2_no_vio.json
Training/jobs/ablate_modalities_residual_v2.sbatch
```

Bereits umgesetzt:

- [x] `no_gaze`
- [x] `no_hands`
- [x] `no_objects`
- [x] `no_vio`
- [x] Ablations-Smoke-Test
- [x] identische Architektur, Splits und Trainingshyperparameter vorgesehen

Noch offen:

- [x] Ablationsabschnitte verwandter Papers lesen und Vergleichsmatrix anlegen
- [x] Ablationsprotokoll vor dem Start festschreiben
- [x] Vier vorhandene Varianten auf `n214` mit Seeds 42, 43, 44 ausführen
- [x] vollständige Residual-v2-Baseline aus `benchmark_v2` wiederverwenden
- [x] Mittelwert und Standardabweichung berechnen
- [x] Delta zur vollständigen Baseline berichten
- [x] Balkendiagramme für Intention-F1, Hand-F1 und Pose-MAE erstellen
- [ ] Auswirkungen auf Parameterzahl und Latenz berichten
- [ ] Nach CLIP-Integration `no_clip`/`with_clip` ergänzen

Wichtige Interpretation: `no_hands` entfernt Handfeatures aus dem Encoder, aber
Ground-Truth-Handreferenzen für die Pose-Loss-Berechnung bleiben als
Trainingsziel notwendig. Dies muss im Methodenteil explizit erklärt werden.

## 7. Latenzanalyse

Es müssen drei unterschiedliche Größen getrennt werden:

1. reine Modell-Forward-Latenz
2. vollständige Offline-Pipeline-Latenz pro Fenster
3. Live-End-to-End-Latenz von Sensoreingang bis Ausgabe

### 7.1 Messprotokoll

- [x] Ein gemeinsames Benchmarkskript für Batchgröße 1 erstellen
- [x] dasselbe Modell, denselben Checkpoint und dieselben Eingabefenster nutzen
- [x] mindestens 100 Warm-up- und 1.000 Messdurchläufe verwenden
- [x] CUDA/MPS vor und nach jeder Messung korrekt synchronisieren
- [x] Median, Mittelwert, Standardabweichung, p95 und p99 berichten
- [x] Durchsatz, Peak Memory und Modellladezeit ergänzen
- [x] Hardware, Betriebssystem, PyTorch-Version und Device protokollieren

### 7.2 Plattformen

- [ ] Mac CPU
- [ ] Mac MPS/Apple GPU
- [ ] verfügbarer Uni-Rechner CPU
- [ ] verfügbarer Uni-Rechner GPU, falls vorhanden
- [ ] TCML-Compute-Node CPU
- [ ] TCML-Compute-Node GPU über SLURM

Der TCML-Cluster ist für Offline-Inferenz- und Modelllatenz geeignet. Ein echter
Aria-USB-Livestream ist dort voraussichtlich nicht sinnvoll, weil die Brille am
Compute-Node nicht physisch angeschlossen ist. Live-End-to-End wird daher auf
dem Mac gemessen; der Cluster erhält reproduzierbare Offlinefenster.

### 7.3 Live-Latenz

Vorhandene Grundlage:

- `aria_live_inference.py` protokolliert Pipeline-Zeitstempel
- `analyze_live_validation.py` aggregiert Latenzen
- Replay und Batch-Replay besitzen bereits Latenzfelder

Noch offen:

- [ ] neue Live-Sitzung mit Eventmarkern aufnehmen
- [ ] Capture-/Callback-, Feature-, Inferenz-, Quality- und Outputzeiten trennen
- [ ] mindestens drei Sitzungen pro relevanter Plattform durchführen
- [ ] Latenzverteilung als Box-/Violinplot und CDF darstellen
- [ ] Anteil unter einer vorab definierten Echtzeitgrenze berichten
- [ ] Einschränkung der Device-/Host-Uhrsynchronisation dokumentieren

## 8. Empfohlene Reihenfolge

### Phase A – Ergebnisse sichern und Baseline vervollständigen

- [x] `n214`-Runs und Reports lokal spiegeln
- [x] Registry und zentralen Trainingsbericht aktualisieren
- [x] vorhandene Graphen für `n214` neu erzeugen
- [x] Literatur-/Ablationsmatrix erstellen

### Phase B – Hyperparametersuche

- [x] Suchskript und Clusterjob implementieren
- [ ] 20–30 Suchtrials durchführen
- [ ] Top-Konfigurationen mit drei Seeds bestätigen
- [ ] getunte Baseline festschreiben

### Phase C – Ablationen

- [x] vorhandene vier Modalitätsablationen ausführen
- [x] Ergebnisse aggregieren und visualisieren

### Phase D – CLIP

- [x] Embeddingpipeline implementieren
- [ ] CLIP-only und multimodal+CLIP vergleichen
- [ ] Entscheidung über finale Architektur treffen

### Phase E – qualitative und Laufzeitauswertung

- [x] Video-Overlay zunächst mit bestehendem Modell prototypisieren
- [ ] finales Video mit ausgewähltem Modell erzeugen
- [ ] Latenzmessung auf Mac, Uni-Hardware und TCML durchführen

### Phase F – Thesis-Artefakte

- [ ] finale Tabellen und Graphen versionieren
- [ ] Methoden- und Ergebnisteil aktualisieren
- [ ] Limitationen dokumentieren
- [ ] alle finalen Zahlen gegen maschinenlesbare Reports prüfen

## 9. Unmittelbar nächste Aufgaben

1. [x] Neue Benchmarkreports vom Cluster lokal spiegeln
2. [x] `n214`-Graphen mit dem bestehenden Generator erzeugen
3. [x] Hyperparameter-Suchprotokoll als reproduzierbaren Arbeitsstand festlegen
4. [x] Suchskript und SLURM-Arrayjob implementieren
5. [x] Literaturmatrix für CLIP und Ablationen erstellen
