# Trainingspipeline der vier Modellvarianten

## Zweck und Geltungsbereich

Die Trainingspipeline löst zwei gekoppelte Aufgaben:

1. Aus einem zurückliegenden multimodalen Zeitfenster wird die aktuelle Intention als `continue`, `fetch` oder `handover` klassifiziert.
2. Für `handover` wird zusätzlich die Pose des empfangenden Handgelenks nach dem konfigurierten Horizont von `1.0 s` vorhergesagt.

Die Intentionsklasse gehört zum Endpunkt des Fensters. Nur das Pose-Target stammt aus den bereits im Master-CSV vorberechneten Spalten `future_1s_*`. Die Implementierung erzeugt selbst keine zeitliche Verschiebung.

Diese Zusammenfassung basiert auf [model.py](model.py), [data.py](data.py), [train.py](train.py), [train_residual.py](train_residual.py), [metrics.py](metrics.py) und den vier Konfigurationen unter [configs](configs/).

## Gemeinsame Eingabedaten

### Features

`select_feature_columns()` in `data.py` verwendet das Profil `multimodal_robot_frame_v1`. Es wählt aus dem CSV-Header folgende Gruppen in fester Reihenfolge:

| Gruppe | Verwendete Größen |
|---|---|
| Gaze | Gültigkeit, Yaw, Pitch, Tiefe, Ursprung und Blickrichtung im Roboterkoordinatensystem |
| Hände | Tracking-Konfidenz und Gültigkeit beider Hände sowie Position und Quaternion beider Handgelenke im Roboterkoordinatensystem |
| VIO/SLAM | Lineare und angulare Geschwindigkeit sowie SLAM-Qualität |
| Roboterbezug | Sichtbarkeit von AprilTag 0, Gültigkeit des Roboterframes und Kennzeichen für interpolierten Anker |
| Objektmarker | Für ArUco 6 bis 14: Position im Roboterframe, Blickwinkel, Blickdistanz und Gültigkeit |

Das vollständige Kandidatenprofil umfasst `F = 92` Rohfeatures. Die genaue Zahl ist jedoch **nicht allein durch die JSON-Konfiguration garantiert**: Es werden nur Spalten gewählt, die im Header des ersten ausgewählten Master-CSVs existieren; mindestens 20 werden verlangt. Fehlen gewählte Spalten in einem späteren CSV, setzt `load_record()` `*_valid`-Spalten auf `0`, andere Spalten auf `NaN`.

Für jedes Rohfeature wird nach der Normalisierung ein eigener Beobachtungskanal angehängt. Die Modelle erhalten daher:

`x: [B, W, 2F]`

Mit dem vollständigen Profil und `W = 60` ist dies konkret `[B, 60, 184]`. Die 60 Einträge sind Zeilen beziehungsweise Abtastpunkte; ihre exakte Dauer ist im Trainingscode nicht festgeschrieben und daher nicht eindeutig bestimmbar.

### Fehlende und ungültige Werte

`fit_normalizer()` berechnet Mittelwert und Standardabweichung ausschließlich aus endlichen Werten der Trainingssequenzen. Features ohne Trainingsbeobachtung erhalten `mean = 0`, Features ohne ausreichende Streuung `std = 1`.

`Normalizer.transform()` führt anschließend zwei Operationen aus:

- Endliche Werte werden z-normalisiert.
- Nicht endliche Werte werden im normalisierten Kanal durch `0` ersetzt.
- Parallel wird `1` für beobachtet und `0` für fehlend angehängt.

Das Modell kann somit zwischen einem tatsächlichen normalisierten Nullwert und einem ersetzten fehlenden Wert unterscheiden. Semantische Gültigkeitsfelder wie `gaze_valid` oder `hand_left_valid` bleiben zusätzlich normale Eingabefeatures.

```text
Rohwerte [T, F]
   ├─ z-Normalisierung; NaN/Inf → 0
   └─ finite-Maske              → 0/1
                 │
                 └──────────────→ Modellwerte [T, 2F]
```

## Gemeinsame hierarchische Ausgabe

Alle vier Modelle zerlegen die Intention in zwei binäre Entscheidungen:

```text
assistance_logits [B, 2]
   ├─ Klasse 0: continue  ─────────────────────────→ Intention 0
   └─ Klasse 1: assistance
          │
          └─ assistance_type_logits [B, 2]
                 ├─ Klasse 0: fetch    ────────────→ Intention 1
                 └─ Klasse 1: handover ────────────→ Intention 2
```

Im Training wird der zweite Kopf nur an Ground-Truth-Samples mit `fetch` oder `handover` optimiert. Bei der Auswertung entscheidet zuerst `argmax(assistance_logits)`. Nur bei vorhergesagter Assistance wird `argmax(assistance_type_logits) + 1` eingesetzt. Die Implementierung in `run_epoch()` multipliziert hierfür keine Wahrscheinlichkeiten.

## 1. Transformer v1

**Implementierung:** `HierarchicalGatedMultimodalTransformer` in `model.py`  
**Konfiguration:** [transformer_v1.json](configs/models/transformer_v1.json)
**Trainer:** `train.py`

Transformer v1 klassifiziert die Intention und sagt für Handover eine absolute zukünftige Handpose voraus. Er verarbeitet dasselbe Fenster parallel entlang der Zeit- und Featureachse:

```text
x [B, 60, 2F]
   ├─ zeitlicher Pfad: Linear(2F→64), CLS, Positionsembedding
   │                    → 2 Transformer-Layer → temporal [B, 64]
   └─ Kanalpfad: Transpose [B, 2F, 60], Linear(60→64), CLS,
                 Kanal-Embedding → 2 Transformer-Layer → channel [B, 64]
                               │
                Softmax-Gate [B, 2]
                               │
       concat(g_t·temporal, g_c·channel) → fused [B, 128]
```

Beide Encoder verwenden `d_model=64`, vier Attention-Heads, Feedforward-Dimension 128, zwei Layer, GELU, `dropout=0.15` und Pre-Normalization. Der zeitliche Pfad besitzt bei vollständigem Profil 61 Tokens, der Kanalpfad 185 Tokens. Die beiden CLS-Zustände bilden die Repräsentationen. Ein gelerntes Gate gewichtet sie pro Sample; die gewichteten Zustände werden konkateniert und normalisiert.

Aus `fused [B,128]` entstehen:

- `assistance_logits [B,2]`
- `assistance_type_logits [B,2]`
- `pose [B,7]`: drei absolute Positionswerte und vier als `(qx,qy,qz,qw)` normalisierte Quaternionwerte
- `gate [B,2]` zur Auswertung des mittleren Zeit-/Kanalgewichts

Der Pose-Kopf ist `Linear(128→64) → GELU → Dropout → Linear(64→7)`. Seine Loss wird ausschließlich auf gültigen Handover-Targets berechnet.

**Für kausales Live-Inferencing:** Das Modell benötigt nur ein am aktuellen Zeitpunkt endendes Fenster und verwendet damit keine zukünftigen Eingabezeilen. Innerhalb dieses Fensters gibt es allerdings **keine Attention-Causal-Maske**; alle 60 bereits beobachteten Zeitschritte dürfen miteinander interagieren. Vorteile sind der direkte Zugriff auf langfristige Muster und die zusätzliche Modellierung von Beziehungen zwischen Featurekanälen. Nachteile sind das starre 60er-Fenster, die Aufwärmphase bis zu einem vollständigen Fenster und der Rechenaufwand zweier Transformer-Pfade. Encoderzustände werden von der aktuellen Implementierung nicht zwischen Fenstern wiederverwendet.

## 2. MLP

**Implementierung:** `HierarchicalWindowMLP` in `model.py`  
**Konfiguration:** [mlp_v1.json](configs/models/mlp_v1.json)
**Trainer:** `train.py`

Das MLP ist die nicht-rekurrente Vergleichsbasis. Es flacht das vollständige Fenster ab:

`[B,60,2F] → [B,60·2F]`

Beim vollständigen Profil wird `[B,60,184]` somit zu `[B,11040]`. Der Encoder aus der Konfiguration lautet:

`Linear(11040→16) → LayerNorm → GELU → Dropout → Linear(16→128) → LayerNorm → GELU → Dropout`

Die resultierende Repräsentation ist `[B,128]`. Zwei lineare Klassifikationsköpfe erzeugen jeweils `[B,2]`; der absolute Pose-Kopf ist `Linear(128→128) → GELU → Dropout → Linear(128→7)` mit anschließender Quaternion-Normalisierung.

Missing-Data-Behandlung, Targets, Hierarchie und Loss sind identisch zu Transformer v1. Das MLP besitzt weder Attention-Zustände noch einen rekurrenten Hidden State; seine einzige latente Fensterrepräsentation ist der 128-dimensionale Encoderoutput.

**Für kausales Live-Inferencing:** Bei Verwendung eines nach hinten gerichteten Fensters ist auch das MLP kausal auf Fensterebene. Es ist architektonisch einfach und ohne sequentielle Rekurrenz ausführbar. Dafür ist jede Zeitposition fest an bestimmte Eingangsgewichte gebunden; zeitliche Struktur wird nur implizit aus dem abgeflachten Vektor gelernt. Das Modell verlangt immer die vollständige, korrekt geordnete Fensterform und bietet keine Zustandswiederverwendung.

## 3. GRU

**Implementierung:** `HierarchicalGRU` in `model.py`  
**Konfiguration:** [gru_v1.json](configs/models/gru_v1.json)
**Trainer:** `train.py`

Die GRU verarbeitet `[B,60,2F]` chronologisch mit einer unidirektionalen, zweilagigen `nn.GRU`. Die Konfiguration setzt `hidden_size=112`, `num_layers=2` und `dropout=0.15`; der Dropout der GRU liegt zwischen den beiden Schichten.

Die Sequenzausgaben werden nicht weiterverwendet. Aus `hidden [2,B,112]` nimmt das Modell den Zustand der letzten GRU-Schicht `hidden[-1] [B,112]` und normalisiert ihn mit `LayerNorm`. Diese Repräsentation speist:

- zwei lineare Köpfe mit je `[B,2]`,
- einen absoluten Pose-Kopf `Linear(112→112) → GELU → Dropout → Linear(112→7)`.

Die Quaternion des Outputs `[B,7]` wird wie bei Transformer v1 normalisiert. Datenbehandlung und Training der hierarchischen Köpfe sind gleich.

**Für kausales Live-Inferencing:** Die unidirektionale GRU entspricht der zeitlichen Richtung einer Live-Anwendung und verdichtet die Vergangenheit in einen letzten Hidden State. Die aktuelle `forward()`-Methode übernimmt jedoch keinen externen Hidden State und berechnet deshalb jedes überlappende 60er-Fenster vollständig neu. Gegenüber den parallel arbeitenden Alternativen ist die Berechnung innerhalb eines Fensters sequentiell; außerdem muss der 112-dimensionale Endzustand alle relevanten Fensterinformationen tragen.

## 4. Residual Transformer v2

**Implementierung:** `HierarchicalResidualPoseTransformer` in `model.py`  
**Konfiguration:** [residual_transformer_v2.json](configs/models/residual_transformer_v2.json)
**Trainer:** `train_residual.py`

Residual Transformer v2 übernimmt den identischen dualen Transformer-Encoder, das Gate, die 128-dimensionale Fusion und die beiden Intentionsköpfe von Transformer v1. Der Unterschied liegt in der Hand- und Posevorhersage.

### Empfangshand und Referenzpose

`data.include_hand_references=true` ist verpflichtend. Für jedes Fenster sucht `WindowDataset` getrennt für links und rechts die **letzte gültige Handpose innerhalb desselben Fensters**. Daraus entstehen:

- `hand_reference_pose [B,2,7]`
- `hand_reference_valid [B,2]`
- Alter der Referenzen `[B,2]` in Sekunden
- Empfangshand-Target: links `0`, rechts `1`, unbekannt `-1`

Fehlt eine Referenz, bleibt als Platzhalter Position `0` und Identitätsquaternion `(0,0,0,1)` bestehen; `hand_reference_valid` ist dann `false`. Das Referenzalter wird gespeichert, aber weder dem Modell übergeben noch im Loss verwendet.

Ein zusätzlicher linearer Kopf erzeugt `receiving_hand_logits [B,2]`. Er wird nur bei Handover-Samples mit bekannter Empfangshand trainiert.

### Residuale Pose

Die Softmax-Wahrscheinlichkeiten der Empfangshand werden an `fused` angehängt:

`[B,128] + [B,2] → [B,130] → Linear(130→64) → GELU → Dropout → Linear(64→7)`

Der Output wird in `position_delta [B,3]` und `quaternion_delta [B,4]` geteilt. Die Quaternionänderung wird normalisiert. Dieselbe gelernte Änderung wird auf beide Handreferenzen angewandt:

```text
candidate_position[h]  = reference_position[h] + position_delta
candidate_quaternion[h] = reference_quaternion[h] ⊗ quaternion_delta

pose_candidates [B, 2, 7], h ∈ {links, rechts}
```

Der letzte lineare Layer wird so initialisiert, dass zunächst Positionsänderung `0` und Quaternionänderung `(0,0,0,1)` ausgegeben werden. Das Modell startet damit bei der letzten beobachteten Pose. Es lernt keinen separaten Residualvektor pro Hand.

### Oracle-Pose und End-to-End-Pose

| Variante | Gewählte Kandidatenpose | Bedeutung |
|---|---|---|
| Oracle | Auswahl mit der tatsächlichen Empfangshand | Isoliert die Qualität der residualen Posevorhersage bei korrekter Handreferenz |
| End-to-End | Auswahl mit `argmax(receiving_hand_logits)` | Bezieht Fehler der Handklassifikation und die Gültigkeit der vorhergesagten Referenz ein |
| Last Observation Oracle | Unveränderte letzte Pose der tatsächlichen Hand | Baseline ohne gelerntes Residuum |

Der Pose-Loss wird mit der **Oracle-Auswahl** berechnet und erfordert ein gültiges Future-Target, eine bekannte Empfangshand sowie eine gültige Referenz der tatsächlichen Hand (`residual_pose_valid`). Bei der End-to-End-Auswertung kann eine falsche Handwahl daher direkt zu einer falschen Pose oder geringerer gültiger Abdeckung führen.

`run_epoch()` berichtet zusätzlich Posemetriken nach Handover-Fortschritt und Empfangshand sowie die Abdeckung gültiger Future-Targets und Referenzen.

**Für kausales Live-Inferencing:** Die letzte Pose innerhalb des zurückliegenden Fensters ist live verfügbar; der Ansatz benötigt keine zukünftige Referenz. Die Vorhersage ist räumlich an eine reale, zuletzt beobachtete Handpose gekoppelt und muss nur deren Änderung lernen. Dafür hängt sie zusätzlich von zuverlässigem Handtracking und der Empfangshandklassifikation ab. Bezüglich Fenster, Attention und fehlender Zustandswiederverwendung gelten dieselben Einschränkungen wie bei Transformer v1.

## Gemeinsame Trainingspipeline

```text
Manifest + *_master.csv
        ↓
Sequenzen laden und Features auswählen
        ↓
teilnehmerbasierter Train/Validation/Test-Split
        ↓
Normalizer nur auf Train fitten; Werte + Masken erzeugen
        ↓
Fenster [60, 2F] mit Stride 10 bilden
        ↓
DataLoader → Forward Pass → Multi-Task-Loss
        ↓
Backpropagation + AdamW + Gradient Clipping
        ↓
Validation → Checkpoints / Early Stopping
        ↓
einmalige Testauswertung beider Checkpoints
```

### 1. Laden und Filtern

`prepare_data()` sucht `*_master.csv` im konfigurierten `Data_collection/master_datasets`. Der strikte Manifestfilter akzeptiert nur Sequenzen mit:

- `include_in_training=true`,
- `status=valid`,
- `next_action=ready_for_master_merge`,
- `master_csv_exists=true`.

Bei `strict=true` führen nicht gelistete Master-CSVs oder berechtigte Manifestzeilen ohne CSV zu einem Fehler. Labels werden durch `INTENTION_TO_ID` auf `transition=-1`, `continue=0`, `fetch=1`, `handover=2` abgebildet. Teilnehmernamen werden mit `strip()`, `casefold()` und anschließender Großschreibung kanonisiert; Groß-/Kleinschreibung erzeugt daher keine getrennten Teilnehmer.

### 2. Teilnehmerbasierter Split

`split_records()` trennt vollständige Teilnehmer, nicht einzelne Fenster. In allen vier Konfigurationen sind fest vorgegeben:

| Split | Teilnehmer |
|---|---|
| Validation | Atilla, Ermal, Vanessa |
| Test | Edu, Jona, Mona |
| Train | alle übrigen ausgewählten Teilnehmer |

Die Fraktionen `0.2/0.2` werden nur verwendet, wenn keine expliziten Teilnehmerlisten gesetzt sind. Train, Validation und Test müssen jeweils mindestens eine Sequenz enthalten.

### 3. Fenster und Targets

`WindowDataset` bildet Fenster der Länge 60 mit Stride 10. Ein Fenster wird verworfen, wenn:

- sein Endpunkt `transition` ist,
- innerhalb des Fensters eine Zeitlücke größer als `0.2 s` vorkommt,
- im Mittel weniger als 5 % der Featurewerte beobachtet sind.

`transition` darf als Kontext innerhalb eines gültigen Fensters auftreten, aber nie dessen Target sein.

Jedes Sample enthält mindestens:

- `features [60,2F]`,
- skalares Intention-Target,
- zukünftige Pose `[7]`,
- Pose-Gültigkeit,
- Sequenz, Teilnehmer und Zeitstempel.

Das Future-Pose-Target wird nur als gültig behandelt, wenn die vorberechnete Gültigkeit gesetzt ist, alle sieben Werte endlich sind, die Quaternion eine Norm größer `1e-6` hat und die Intention `handover` ist. Gültige Target-Quaternionen werden normalisiert.

### 4. Batches und Forward Pass

Die Konfigurationen verwenden Batchgröße 32 und zwei DataLoader-Worker. Nur der Trainingsloader mischt Samples. CUDA aktiviert zusätzlich `pin_memory`; Worker bleiben persistent. Das letzte Batch kann kleiner als 32 sein.

`train.py` ruft für Transformer v1, MLP und GRU `model(batch["features"])` auf. `train_residual.py` übergibt zusätzlich `batch["hand_reference_pose"]`. Alle Tensoren werden vorher auf das gewählte Gerät verschoben (`cuda`, `mps` oder `cpu` bei automatischer Auswahl).

### 5. Loss

Für Transformer v1, MLP und GRU gilt:

`L = L_assistance + L_fetch/handover + L_pose`

Residual v2 ergänzt:

`L = L_assistance + L_fetch/handover + L_receiving_hand + L_pose`

Alle äußeren Gewichte stehen in den aktuellen Konfigurationen auf `1.0`. Die Klassifikationsverluste sind Cross-Entropy-Losses mit aus den Trainingsfenstern invers bestimmten und auf Mittelwert 1 normalisierten Klassengewichten. Enthält eine benötigte Klassenverteilung eine Null, verwendet `class_weights()` stattdessen einheitliche Gewichte.

Der Poseanteil lautet:

`L_pose = SmoothL1(position) + 0.25 · mean(1 - |q_pred · q_target|)`

Der Absolutbetrag macht den Orientierungsterm gegenüber den äquivalenten Vorzeichen `q` und `-q` invariant. Bei den drei v1-Vergleichsmodellen wird er auf der absoluten Pose berechnet, bei Residual v2 auf der Oracle-Kandidatenpose.

### 6. Optimierung und Early Stopping

Pro Trainingsbatch erfolgt:

1. Gradienten mit `zero_grad(set_to_none=True)` löschen.
2. Forward Pass und Multi-Task-Loss berechnen.
3. `backward()` ausführen.
4. Gesamtnorm der Gradienten auf `1.0` begrenzen.
5. AdamW-Schritt mit Lernrate `3e-4` und Weight Decay `1e-4`.

Ein Learning-Rate-Scheduler ist nicht implementiert. Es werden höchstens 20 Epochen trainiert. Early Stopping greift nach sieben aufeinanderfolgenden Epochen, in denen sich **weder** der Intentions- noch der Pose-Checkpoint verbessert.

### 7. Checkpoints und Test

Da Klassifikation und Pose unterschiedliche Optima erreichen können, werden zwei Zustände separat gespeichert:

| Trainer | Intentions-Checkpoint | Pose-Checkpoint |
|---|---|---|
| `train.py` | `best_model.pt`, maximales Validation-Intention-Macro-F1 | `best_pose_model.pt`, minimale Validation-Positions-MAE |
| `train_residual.py` | `best_intention_model.pt`, maximales Validation-Intention-Macro-F1 | `best_pose_model.pt`, minimale Validation-Oracle-Positions-MAE |

Jede Verbesserung eines der beiden Kriterien setzt den Early-Stopping-Zähler zurück. Bei Residual v2 ist die Trennung besonders wichtig: Der Intentionskopf kann früh sein bestes Generalisierungsniveau erreichen, während Empfangshand- und Residualpose-Köpfe noch lernen. Da alle Köpfe einen gemeinsamen Encoder besitzen, repräsentieren die Dateien zwei verschiedene Snapshots des Gesamtmodells.

Nach dem Training werden beide gespeicherten Zustände geladen und jeweils genau auf dem Testsplit ausgewertet. `classification_metrics()` liefert Accuracy, Konfusionsmatrix, Support, Klassen-F1, Macro-F1 und ein nur über vorhandene Klassen gemitteltes Macro-F1. `pose_metrics()` berichtet den euklidischen Positionsfehler als Mean/RMSE in Zentimetern und den vorzeicheninvarianten mittleren Quaternion-Winkelfehler in Grad.

## Gespeicherte Artefakte

Jeder neue Runordner enthält:

- `config.json`: effektiv verwendete Konfiguration einschließlich aufgelöstem Datenpfad und eventuellen CLI-Overrides,
- `data_metadata.json`: Featurelisten, Normalisierungsparameter, Split, Manifest-Fingerprint, Fensterzahlen, verworfene Fenster und Klassenverteilungen,
- zwei PyTorch-Checkpoints mit Modellzustand, Modelltyp, Modellkonfiguration, Eingabedimension, Fensterlänge, Epoche und Auswahlmetrik,
- `metrics.json`: gesamte Train-/Validation-Historie, Checkpoint-Metadaten und Testergebnisse beider Checkpoints.

## Kompakter Modellvergleich

| Modell | Fensterrepräsentation | Poseausgabe | Zusätzliche Ausgabe | Wesentlicher Live-Trade-off |
|---|---|---|---|---|
| Transformer v1 | Zeit-CLS `[B,64]` + Kanal-CLS `[B,64]`, gegatet zu `[B,128]` | Absolute Future-Pose `[B,7]` | Gate `[B,2]` | Globale Zeit-/Kanalbeziehungen, aber zwei feste Transformer-Pfade |
| MLP | Abgeflachtes Fenster → `[B,128]` | Absolute Future-Pose `[B,7]` | keine | Einfach, aber zeitliche Struktur nur implizit und keine Zustandsnutzung |
| GRU | Letzter Hidden State `[B,112]` | Absolute Future-Pose `[B,7]` | keine | Unidirektional und zeitlich passend, aber im Code pro Fenster neu berechnet |
| Residual Transformer v2 | Wie Transformer v1 | Zwei referenzbasierte Kandidaten `[B,2,7]` | Empfangshand `[B,2]`, Residuen, Gate | Räumlich verankert, aber abhängig von Handreferenz und korrekter Handwahl |

## Eindeutigkeitsgrenzen

- `W=60`, Stride 10 und der Posehorizont `1.0 s` sind eindeutig konfiguriert; die reale Zeitdauer eines 60-Zeilen-Fensters ist ohne garantierte Abtastrate nicht eindeutig.
- `F` wird dynamisch aus dem ersten CSV-Header bestimmt. Bei vollständig vorhandenem Profil gilt `F=92` und Modelleingabe `2F=184`; ein reduziertes kompatibles Schema kann weniger Kanäle ergeben.
- ArUco-IDs 6 bis 14 sind als Markerfeatures definiert. Welche realen Gegenstände diesen IDs zugeordnet sind, ist in den untersuchten Trainingsdateien und Konfigurationen nicht festgelegt.
ç
