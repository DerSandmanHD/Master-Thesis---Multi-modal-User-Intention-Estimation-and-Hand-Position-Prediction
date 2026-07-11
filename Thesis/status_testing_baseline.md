# Status: Testing Baseline

Stand: 11. Juli 2026

> **Legacy-Baseline:** Dieser Lauf bleibt als technische Vorstudie erhalten.
> Er verwendete eine flache Drei-Klassen-Klassifikation, einen zusaetzlichen
> Objekt-ID-Kopf und Pose-Targets fuer Fetch und Handover. Nach dem Abgleich mit
> der offiziellen Thesis-Aufgabe wurde die aktuelle Pipeline auf eine explizite
> Assistenzhierarchie ohne Objektklassifikationskopf und mit Handover-only-Pose
> umgestellt. Die hier dokumentierten Ergebnisse sind daher keine finalen
> Thesis-Ergebnisse und werden nicht geloescht oder ueberschrieben.

## Ziel

Dieses Dokument beschreibt die erste funktionsfaehige Trainingsbaseline fuer die
multimodale Schaetzung von Nutzerintention, Zielobjekt und zukuenftiger Handpose.
Es dokumentiert den verwendeten Datenstand, den participant-wise Split, das
Modell, die Trainingskonfiguration, die Ergebnisse und die bekannten Grenzen.

## Datenstand

Die Dataset-QA wurde auf dem Cluster mit den manuell korrigierten Timestamps
ausgefuehrt:

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/dataset_qa.py \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.reviewed.json
```

Stand nach der Korrektur der Handover-Definition und dem Master-Rebuild:

- 159 Sequenzen insgesamt
- 156 zunaechst fuer Training vorgesehene Sequenzen
- 128 erfolgreich erzeugte Master-Datensaetze
- `Tarik_22_20260704_145150` ausgeschlossen, da AprilTag 0 vollstaendig
  verdeckt ist und keine robot-relativen Koordinaten berechnet werden koennen
- 17 Sequenzen mit fehlenden oder unvollstaendigen MPS-Ausgaben
- weitere Sequenzen mit offenen Tracking- oder Annotationswarnungen

### Phasendefinition

Die urspruengliche Implementierung interpretierte `DONE -> THIRD` faelschlich
als Handover und beendete den Master-Datensatz bei `THIRD`. Nach Klaerung des
Versuchsablaufs gilt:

| Zeitbereich | Bedeutung | Trainingslabel |
|---|---|---|
| `START -> SECOND` | Aufgabe fortsetzen | `continue` |
| `SECOND -> DONE` | Objekt fixieren beziehungsweise holen | `fetch` |
| `DONE -> THIRD` | Warte- und Uebergangsphase | nicht gelabelt |
| `THIRD -> Aufnahmeende` | Hand zum Roboter ausstrecken | `handover` |

Die QA prueft Handtracking seit dieser Korrektur im Bereich `THIRD ->
Aufnahmeende`.

## Train-, Validation- und Test-Split

Der Split erfolgt vollstaendig auf Teilnehmerebene. Keine Person kommt in mehr
als einem Split vor. Dadurch wird verhindert, dass personenspezifische
Bewegungsmuster aus dem Training in Validation oder Test wiederkehren.

Verwendete Konfiguration:
`Training/configs/participant_split_v1.json`

| Split | Teilnehmer | Sequenzen | Fenster |
|---|---|---:|---:|
| Train | Berat, Felix, Jan, Jola, Julienco, Leys, Maria, Melih, Suthan, Tarik, Urim | 78 | 7.631 |
| Validation | Atilla, Ermal, Vanessa | 25 | 2.385 |
| Test | Edu, Jona, Mona | 25 | 2.414 |

Validation wird fuer Checkpoint-Auswahl und Early Stopping verwendet. Der
Testsplit wird erst nach dem Laden des besten Validation-Checkpoints
ausgewertet.

## Eingabefeatures

Die Baseline verwendet noch keine RGB- oder CLIP-Embeddings. Eingaben sind die
numerischen, zeitlich synchronisierten Features aus den Master-CSVs:

- Blickrichtung, Blicktiefe und Blickvaliditaet
- Position, Orientierung und Trackingkonfidenz beider Haende
- SLAM-Bewegung und SLAM-Qualitaet
- ArUco-Objektpositionen relativ zum Roboter
- Blickwinkel und Blickdistanz zu den Objektmarkern
- AprilTag 0 als Roboteranker
- Validitaets- und Beobachtungsmasken fuer fehlende Messwerte

Fehlende kontinuierliche Markerwerte werden als `NaN` behandelt. Zugehoerige
`*_valid`-Features erhalten den Wert `0`. Die Normalisierung wird nur auf dem
Trainingssplit angepasst. Fuer jedes Eingabefeature wird zusaetzlich eine
Beobachtungsmaske an das Modell uebergeben.

### Windowing

- Fensterlaenge: 60 Samples, ungefaehr 2 Sekunden bei 30 Hz
- Stride: 10 Samples, ungefaehr 0,33 Sekunden
- Intention und Zielobjekt beziehen sich auf das Fensterende
- Handpose-Ziel: empfangendes Handgelenk 1 Sekunde in der Zukunft

Bekannte Einschraenkung: Fenster duerfen zukuenftig nicht unbemerkt ueber die
ausgelassene Zeitluecke `DONE -> THIRD` reichen. Eine explizite Pruefung auf
Timestamp-Spruenge ist als naechste Datenpipeline-Korrektur vorgesehen.

## Modell

Verwendet wird `GatedMultimodalTransformer` aus `Training/model.py`.

Das Modell besitzt zwei parallele Repraesentationstuerme:

1. Ein Temporal Transformer modelliert die Entwicklung der Features ueber das
   Zeitfenster.
2. Ein Channel Transformer modelliert Beziehungen zwischen Sensorkanaelen.

Ein lernbares Gate fusioniert beide Repraesentationen. Danach folgen drei
Ausgabekoepfe:

- Intention: `continue`, `fetch`, `handover`
- Zielobjekt: ArUco-IDs 6 bis 14
- zukuenftige Handpose: Position `x, y, z` und Quaternion `qx, qy, qz, qw`

Wichtige Modellparameter:

- `d_model`: 64
- Attention Heads: 4
- Transformer Layers pro Turm: 2
- Feedforward-Dimension: 128
- Dropout: 0,15

## Training

Cluster-Run:

```text
Training/runs/first_test_20260711_155123
```

Trainingsparameter:

- Device: CUDA
- Seed: 42
- Batchgroesse: 32
- Optimizer: AdamW
- Lernrate: 0,0003
- Weight Decay: 0,0001
- maximal 10 Epochen
- Early-Stopping-Patience: 5
- Gradient Clipping: 1,0

Der Multi-Task-Loss kombiniert:

```text
1,0 * Intentions-Loss
+ 0,5 * Objekt-Loss
+ 1,0 * Pose-Loss
```

Der Pose-Loss besteht aus Smooth-L1-Positionsverlust und einem gewichteten
Quaternion-Orientierungsverlust. Der beste Checkpoint wird derzeit
ausschliesslich anhand des Validation-Macro-F1 der Intention ausgewaehlt.

## Trainingsverlauf

| Epoche | Train-Loss | Validation Intent Macro-F1 | Validation Objekt-Accuracy | Validation Position MAE |
|---:|---:|---:|---:|---:|
| 1 | 1,3037 | 0,8444 | 0,3132 | 21,22 cm |
| 2 | 0,6723 | 0,8470 | 0,3187 | 17,47 cm |
| 3 | 0,4558 | 0,8383 | 0,3119 | 16,88 cm |
| 4 | 0,3525 | **0,8536** | 0,3140 | 18,17 cm |
| 5 | 0,2716 | 0,8530 | 0,3048 | 17,06 cm |
| 6 | 0,2242 | 0,8477 | 0,3392 | 17,29 cm |
| 7 | 0,1817 | 0,8521 | 0,3367 | 18,05 cm |
| 8 | 0,1475 | 0,8525 | **0,3451** | 17,88 cm |
| 9 | 0,1274 | 0,8413 | 0,3342 | 18,34 cm |

Early Stopping wurde nach Epoche 9 ausgeloest. Fuer die Testauswertung wurde der
Checkpoint aus Epoche 4 geladen, da dort der beste Validation-Intentions-F1
erreicht wurde.

## Testergebnisse

### Intention

- Accuracy: 0,8028
- Macro-F1: 0,7444
- Samples: 2.414

| Klasse | F1 | Support laut Confusion Matrix |
|---|---:|---:|
| `continue` | 0,8774 | 1.721 |
| `fetch` | 0,6096 | 368 |
| `handover` | 0,7463 | 325 |

Confusion Matrix, Zeilen = Ground Truth und Spalten = Vorhersage:

```text
[[1378, 275,  68],
 [  31, 310,  27],
 [  11,  64, 250]]
```

Die Klasse `fetch` ist aktuell am schwierigsten. Die hohe Anzahl von
`continue`-Fenstern zeigt zudem eine deutliche Klassenunwucht, weshalb
Macro-F1 aussagekraeftiger als Accuracy ist.

### Zielobjekt

- Accuracy: 0,2912
- Macro-F1: 0,2120
- Samples: 2.414
- Zufallsniveau bei neun gleichwahrscheinlichen Klassen: etwa 0,111

| ArUco-ID | F1 | Support |
|---:|---:|---:|
| 6 | 0,1236 | 108 |
| 7 | 0,4860 | 285 |
| 8 | 0,3516 | 473 |
| 9 | 0,0000 | 0 |
| 10 | 0,3146 | 385 |
| 11 | 0,1151 | 330 |
| 12 | 0,4101 | 571 |
| 13 | 0,0000 | 95 |
| 14 | 0,1067 | 167 |

ArUco-ID 9 kommt im Testsplit nicht vor. ID 13 besitzt Testbeispiele, wird aber
nie korrekt vorhergesagt. Der participant-wise Split ist leakage-sicher, aber
nicht ausreichend nach Objektklassen ausbalanciert.

### Zukuenftige Handpose

- gueltige Samples: 310
- Position MAE: 21,54 cm
- Position RMSE: 25,21 cm
- mittlerer Orientierungsfehler: 69,97 Grad

Die geringe Zahl gueltiger Pose-Targets und der hohe Orientierungsfehler zeigen,
dass die Handpose-Vorhersage noch nicht ausreichend stabil ist.

### Gate-Verteilung

- Temporal Transformer: 0,9245
- Channel Transformer: 0,0755

Das Modell verwendet in dieser Baseline fast ausschliesslich die zeitliche
Repraesentation. Der Channel-Turm traegt nur wenig zur Fusion bei.

## Interpretation

Die Baseline zeigt, dass participant-uebergreifende Intentionsschaetzung mit den
vorhandenen Trackingfeatures grundsaetzlich funktioniert. Der Unterschied
zwischen bestem Validation-F1 von 0,8536 und Test-F1 von 0,7444 weist jedoch auf
einen Generalisierungsunterschied zwischen den Personengruppen hin.

Der kontinuierlich sinkende Trainingsverlust bei stagnierender Validation-
Leistung zeigt beginnendes Overfitting. Early Stopping reagiert darauf korrekt.
Objekterkennung und Handpose sind deutlich schwaecher als die
Intentionsklassifikation und benoetigen gesonderte Daten- und Modellanalysen.

## Bekannte Einschraenkungen

1. Die Legacy-Pipeline entfernte `DONE -> THIRD` und konnte dadurch Fenster aus
   zeitlich nicht benachbarten Zeilen zusammensetzen.
2. Objektklassen sind zwischen den Teilnehmer-Splits unausgewogen.
3. ArUco-ID 9 fehlt vollstaendig im Testsplit.
4. Fuer die Pose-Auswertung stehen nur 310 gueltige Testfenster zur Verfuegung.
5. Der beste Checkpoint wird nur anhand der Intention ausgewaehlt, nicht anhand
   eines kombinierten Multi-Task-Kriteriums.
6. Der Channel-Turm wird vom Gate nur schwach genutzt.
7. RGB-Bilder oder visuelle Embeddings sind noch nicht Teil der Baseline.
8. Noch fehlende MPS-Daten und offene Reviews begrenzen den finalen Datenstand.

## Priorisierte naechste Schritte

1. `DONE -> THIRD` als kontinuierlichen, ungelabelten Sensorkontext erhalten;
   unlabeled Endpunkte und echte Timestamp-Spruenge im Windowing verwerfen.
2. Objekt- und Pose-Target-Verteilungen pro Teilnehmer analysieren.
3. Teilnehmer-Split unter Beibehaltung der Leakage-Sicherheit besser nach
   Objektklassen und Anzahl gueltiger Pose-Targets ausbalancieren.
4. Klassengewichte oder Sampling fuer den Objekt-Loss evaluieren.
5. Einfache MLP- sowie GRU/LSTM-Baselines implementieren.
6. Modalitaetsablationen fuer Blick, Hand, SLAM und Objekttracking durchfuehren.
7. Fehlende MPS-Daten ergaenzen und offene Handover-Reviews abschliessen.
8. Erst danach visuelle RGB- beziehungsweise CLIP-Embeddings als zusaetzliche
   Modalitaet evaluieren.

Der Testsplit dieser Baseline darf nicht fuer Hyperparameterentscheidungen
verwendet werden. Weitere Modellentscheidungen erfolgen anhand des
Validation-Splits; der finale Test dient der abschliessenden Bewertung.
