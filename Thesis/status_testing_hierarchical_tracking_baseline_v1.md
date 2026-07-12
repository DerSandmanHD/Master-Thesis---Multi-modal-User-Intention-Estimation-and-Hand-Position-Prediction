# Status: Hierarchical Tracking Baseline v1

Stand: 12. Juli 2026

## Einordnung

Dieses Dokument beschreibt die erste methodisch konsistente hierarchische
Tracking-Baseline der Masterarbeit. Der fruehere Flat/Object-Lauf bleibt
separat in `Thesis/status_testing_baseline.md` als Legacy-Vorstudie erhalten.

Die aktuelle Baseline entspricht der offiziellen Aufgabenstellung:

1. `continue` gegen `assistance needed`
2. bei Assistenz: `fetch` gegen `handover`
3. nur bei `handover`: zukuenftige Position und Orientierung der Empfangshand

ArUco-Objektmarker werden als geometrischer Szenenkontext verwendet. Die
Objekt-ID ist keine Zielklasse des Transformers.

## Reproduzierbarkeit

- Code-Commit des Trainingslaufs: `04c3156`
- Config: `Training/configs/hierarchical_baseline_v1.json`
- Cluster-Run:
  `Training/runs/hierarchical_baseline_20260712_101448`
- Seed: 42
- Device: CUDA

Der Code-Stand kann mit Git rekonstruiert werden:

```bash
git checkout 04c3156
```

Fuer die dauerhafte Kennzeichnung sollte der Commit nach Ablage dieser
Dokumentation zusaetzlich mit einem annotierten Git-Tag versehen werden.

## Datenstand

- 159 Sequenzen im QA-Manifest
- 153 als Trainingskandidaten markiert
- 127 aktive Master-Datensaetze
- `Tarik_22_20260704_145150` ausgeschlossen: AprilTag 0 vollstaendig verdeckt
- `Jola_4_20260624_131548` ausgeschlossen: manuelle Exclusion bei niedrigem
  Handover-Handtracking
- veraltete Jola-Masterdatei aus dem aktiven Masterordner entfernt

### Phasen

| Zeitbereich | Verwendung |
|---|---|
| `START -> SECOND` | Target `continue` |
| `SECOND -> DONE` | Target `fetch` |
| `DONE -> THIRD` | `transition`, nur Sensorkontext und nie Fenster-Target |
| `THIRD -> Aufnahmeende` | Target `handover` und Pose-Auxiliary-Task |

Die Transition bleibt als kontinuierliche Historie erhalten. Dadurch werden
keine zeitlich getrennten Zeilen kuenstlich zusammengeschoben und fruehe
Handover-Fenster koennen reale Beobachtungen vor `THIRD` verwenden.

### Statischer Roboteranker

AprilTag 0 definiert den Roboterframe. Seine robuste Weltpose wird aus allen
Frames mit gleichzeitig gueltigem Tag und SLAM geschaetzt. Translation-
Ausreisser werden entfernt und Rotationen gemittelt. Wenn der Tag spaeter
verdeckt ist, wird der Roboterframe mit der aktuellen SLAM-Pose fortgefuehrt.

Pro Masterzeile dokumentieren:

- `robot_frame_valid`: robot-relative Transformation ist verfuegbar
- `robot_anchor_interpolated`: Tag 0 war verdeckt und der statische Anker wurde
  ueber SLAM fortgefuehrt

## Participant-wise Split

| Split | Teilnehmer | Fenster |
|---|---|---:|
| Train | Berat, Felix, Jan, Jola, Julienco, Leys, Maria, Melih, Suthan, Tarik, Urim | 7.509 |
| Validation | Atilla, Ermal, Vanessa | 2.387 |
| Test | Edu, Jona, Mona | 2.412 |

Keine Person kommt in mehreren Splits vor.

Verworfene Fenster:

| Grund | Train | Validation | Test |
|---|---:|---:|---:|
| echter Timestamp-Sprung | 0 | 0 | 0 |
| unlabeled Transition als Endpunkt | 505 | 168 | 156 |

## Pose-Target-Audit

Das Pose-Target ist die robot-relative Position und Quaternion der annotierten
Empfangshand genau eine Sekunde nach dem Fensterendpunkt.

### Nach statischer Ankerfortfuehrung

| Split | Handover-Fenster | gueltig | ungueltig | gueltige Quote |
|---|---:|---:|---:|---:|
| Train | 1.357 | 1.082 | 275 | 79,73 % |
| Validation | 397 | 293 | 104 | 73,80 % |
| Test | 323 | 237 | 86 | 73,37 % |

Ursachen im Testsplit:

| Ursache | Anzahl |
|---|---:|
| `future_after_recording_end` | 75 |
| `future_hand_tracking_invalid` | 9 |
| `future_hand_and_robot_frame_invalid` | 2 |
| gueltig | 237 |

Vor der statischen Ankerfortfuehrung waren im Test nur 94 von 323 Targets
gueltig. Die Korrektur gewann alle 143 reinen Tag-0-Ausfaelle zurueck und
steigerte die Quote von 29,10 % auf 73,37 %.

Die 75 Targets nach dem Aufnahmeende sind keine technischen Fehler. Fuer
Fenster in der letzten Sekunde existiert der feste Zeitpunkt `t+1 s` nicht.

## Modell

`HierarchicalGatedMultimodalTransformer` besitzt:

- Temporal-Transformer fuer Entwicklungen ueber das Beobachtungsfenster
- Channel-Transformer fuer Beziehungen zwischen Sensorkanaelen
- lernbares Gate zur Fusion beider Repraesentationen
- Assistance-Head: `continue` gegen `assistance`
- bedingten Type-Head: `fetch` gegen `handover`
- Pose-Head: `x, y, z, qx, qy, qz, qw`

Modellparameter:

- Beobachtungsfenster: 60 Samples, ungefaehr 2 Sekunden
- Stride: 10 Samples
- Zukunftshorizont: 1 Sekunde
- `d_model`: 64
- Attention Heads: 4
- Transformer-Layer pro Turm: 2
- Feedforward-Dimension: 128
- Dropout: 0,15

Der Pose-Loss wird ausschliesslich auf gueltigen Handover-Fenstern berechnet.

## Training

- AdamW
- Lernrate: 0,0003
- Weight Decay: 0,0001
- Batchgroesse: 32
- maximal 20 Epochen
- Early-Stopping-Patience: 7
- Checkpoint-Auswahl nach Validation-Intent-Macro-F1

Der beste Checkpoint stammt aus Epoche 2:

| Metrik | Wert |
|---|---:|
| Validation Intent Macro-F1 | 0,8713 |
| Validation Assistance Macro-F1 | 0,9452 |
| Validation Fetch/Handover Macro-F1 | 0,8848 |
| Validation Position MAE | 14,92 cm |

Early Stopping wurde nach Epoche 9 ausgeloest.

## Testergebnisse

### Gesamte Intention

- Accuracy: 0,8765
- Macro-F1: 0,8203
- Samples: 2.412

| Klasse | F1 | Support |
|---|---:|---:|
| `continue` | 0,9286 | 1.721 |
| `fetch` | 0,6841 | 368 |
| `handover` | 0,8480 | 323 |

Confusion Matrix, Zeilen = Ground Truth, Spalten = Vorhersage:

```text
[[1547, 139,  35],
 [  54, 274,  40],
 [  10,  20, 293]]
```

### Hierarchiestufe 1: Assistenzbedarf

- Accuracy: 0,9013
- Macro-F1: 0,8845
- `continue` F1: 0,9286
- `assistance` F1: 0,8405

```text
[[1547, 174],
 [  64, 627]]
```

### Hierarchiestufe 2: Fetch gegen Handover

Die zweite Stufe wird auf echten Assistenzfenstern ausgewertet.

- Accuracy: 0,8770
- Macro-F1: 0,8769
- `fetch` F1: 0,8794
- `handover` F1: 0,8744

```text
[[310,  58],
 [ 27, 296]]
```

### Zukunftshandpose

- gueltige Samples: 237
- Position MAE: 18,81 cm
- Position RMSE: 21,99 cm
- mittlerer Orientierungsfehler: 63,82 Grad

Gegenueber der vorherigen Version ohne statische Ankerfortfuehrung:

| Metrik | vorher | Baseline v1 | Aenderung |
|---|---:|---:|---:|
| gueltige Pose-Samples | 94 | 237 | +152 % |
| Position MAE | 22,46 cm | 18,81 cm | -16 % |
| Position RMSE | 27,54 cm | 21,99 cm | -20 % |
| Orientierung | 79,72 Grad | 63,82 Grad | -20 % |

### Naive Pose-Vergleichsbaselines

Alle Verfahren wurden auf denselben 237 gueltigen Testtargets ausgewertet. Der
Trainingsmittelwert wurde ausschliesslich aus den 1.082 gueltigen Targets des
Trainingssplits berechnet. Die konstante Geschwindigkeit wurde per linearer
Regression ueber die letzten 0,5 Sekunden geschaetzt; die Orientierung wurde
wie bei `last_observation` unveraendert fortgeschrieben.

| Verfahren | Position MAE | Position RMSE | Orientierung |
|---|---:|---:|---:|
| Trainingsmittelwert | 19,35 cm | 21,75 cm | 86,33 Grad |
| Last Observation | **14,92 cm** | 22,62 cm | **42,71 Grad** |
| konstante Geschwindigkeit | 35,16 cm | 54,77 cm | 42,71 Grad |
| Transformer | 18,81 cm | **21,99 cm** | 63,82 Grad |

`last_observation` konnte bei 236 von 237 Fenstern direkt berechnet werden;
ein Fenster verwendete den Trainingsmittelwert als Fallback. Ohne diesen
Fallback betragen MAE 14,95 cm, RMSE 22,66 cm und Orientierung 42,13 Grad.

Der Transformer verbessert den MAE gegenueber dem Trainingsmittelwert nur um
0,54 cm und hat einen um 0,24 cm hoeheren RMSE. Gegenueber Last Observation
ist sein MAE um 3,89 cm und sein Orientierungsfehler um 21,11 Grad hoeher. Nur
beim RMSE liegt der Transformer um 0,62 cm vorne. Die einfache lineare
Geschwindigkeitsextrapolation ist fuer den einsekundigen Horizont deutlich
instabiler, wahrscheinlich weil Trackingrauschen und nicht-konstante
Handbewegung direkt extrapoliert werden.

Last Observation und konstante Geschwindigkeit sind als
Oracle-Receiving-Hand-Baselines einzuordnen: Sie verwenden die annotierte
Empfangshand zur Auswahl des linken oder rechten Wrists. Der Transformer sieht
dagegen beide Haende, aber keine `receiving_hand_id`. Der Vergleich isoliert
damit die Bewegungsprognose bei bereits bekannter Hand, ist aber kein
vollstaendig input-identischer Modellvergleich. Als naechstes ist eine
nicht-orakelhafte Handauswahl oder ein explizites, fuer alle Modelle gleiches
Receiving-Hand-Signal erforderlich.

Der vollstaendige maschinenlesbare Bericht liegt unter
`Training/experiments/hierarchical_tracking_baseline_v1/pose_baselines.json`.

#### Detailanalyse der Pose-Baselines

- Last Observation hat bei 157 von 237 Testfenstern einen kleineren
  Positionsfehler als der Trainingsmittelwert.
- Unter den drei naiven Verfahren gewinnt Last Observation bei 132 Fenstern,
  der Trainingsmittelwert bei 77 und konstante Geschwindigkeit bei 28.
- Der Last-Observation-Positionsfehler hat einen Median von 7,07 cm, aber ein
  95-%-Quantil von 48,86 cm und ein Maximum von 68,97 cm. Der Mittelwert wird
  damit von einer relevanten Ausreissergruppe beeinflusst.
- Im ersten Viertel der Handover-Fenster pro Sequenz betraegt der
  Last-Observation-MAE 26,39 cm. Im zweiten und dritten Viertel faellt er auf
  9,33 beziehungsweise 8,98 cm. Die Zukunftsprognose ist somit vor allem zu
  Beginn der Handbewegung relevant; spaetere Fenster sind oft bereits nahezu
  statisch.
- Der Fehler der konstanten Geschwindigkeit korreliert stark mit der
  geschaetzten Geschwindigkeit (`r = 0,836`). Bei weniger als 0,1 m/s liegt
  ihr MAE bei 13,15 cm, oberhalb 1 m/s bei 181,18 cm. Einsekundige
  Extrapolation verstaerkt daher Trackingrauschen und kurzzeitige Bewegung.

Die Handseiten sind nicht ausgewogen ueber Personen und Splits verteilt:

| Split | rechte Targets | linke Targets | Quelle der linken Targets |
|---|---:|---:|---|
| Train | 891 | 191 | ausschliesslich Tarik |
| Validation | 293 | 0 | keine |
| Test | 167 | 70 | ausschliesslich Mona |

Beim Test-Trainingsmittelwert liegt der Orientierungsfehler fuer Monas linke
Hand bei 175,08 Grad, gegenueber 49,13 Grad fuer rechte Haende. Dies deutet auf
unterschiedliche linke und rechte Wrist-Orientierungsmoden hin. Die aktuelle
Validation kann linke Haende ueberhaupt nicht bewerten. Participant-wise
Trennung bleibt notwendig, reicht bei nur zwei Personen mit linken Aufnahmen
aber nicht fuer eine belastbare Drei-Wege-Abdeckung von Train, Validation und
Test.

### Gate

- Temporal: 0,8792
- Channel: 0,1208

Das Modell nutzt weiterhin ueberwiegend die zeitliche Repraesentation.

## Bewertung

Die Baseline zeigt eine gute participant-held-out Intentionserkennung. Die
zweite Hierarchiestufe trennt Fetch und Handover ausgeglichen. Fetch bleibt in
der zusammengesetzten Drei-Klassen-Entscheidung die schwaechste Klasse.

Die statische Ankerfortfuehrung verbessert Verfuegbarkeit und Genauigkeit der
Pose-Targets erheblich. Ein Position MAE von 18,81 cm und ein
Orientierungsfehler von 63,82 Grad reichen dennoch nicht fuer eine praezise
Roboteruebergabe. Zudem zeigt die Oracle-Last-Observation-Baseline, dass der
aktuelle Transformer die zeitliche Pose-Dynamik noch nicht ueberzeugend nutzt.

Der Testsplit wurde waehrend der Pipeline-Entwicklung bereits mehrfach
ausgewertet. Weitere Hyperparameterentscheidungen duerfen deshalb nicht anhand
dieses Testresultats getroffen werden. Fuer belastbarere finale Aussagen sind
participant-wise Cross-Validation oder ein neu eingefrorener finaler Split
erforderlich.

## Naechste Schritte

1. nicht-orakelhafte Handauswahl gegen explizites Receiving-Hand-Signal
   vergleichen
2. Fixed-Horizon `t+1 s` gegen robuste finale Uebergabepose vergleichen
3. Gaze-Ablation und verschiedene Beobachtungslaengen
4. einfache MLP- sowie GRU/LSTM-Intentionsbaselines
5. participant-wise Cross-Validation ueber mehrere Folds und Seeds
6. verbleibende MPS- und Review-Faelle abschliessen
7. erst danach komplexere Trajektorienmodelle oder VQ-VAE untersuchen
