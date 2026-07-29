# Technisches Audit der Trainings- und Live-Inferenzpipeline

**Auditstand:** 28. Juli 2026  
**Geprüfter Git-Stand:** `eef6d2eebc421e744d9f48bf9f8cc4b3b3dad0ae`  
**Gegenstand:** Datenerfassung, Master-Dataset-Erstellung, Datenfilterung, Training, Evaluation, Replay- und Aria-Live-Inferenz  
**Ausdrücklich nicht Gegenstand:** Franka-Panda-, ROS-, MoveIt-, Controller- oder sonstige reale Roboteransteuerung

## Einstufung der Befunde

| Einstufung | Bedeutung in diesem Bericht |
|---|---|
| **Bestätigt korrekt** | Direkt im Code und, soweit möglich, durch Artefakte oder lokale Tests belegt. |
| **Plausibel, aber nicht vollständig validiert** | Technisch nachvollziehbar, jedoch nicht mit ausreichenden Daten oder End-to-End-Messungen belegt. |
| **Verbesserungsbedarf** | Kein zwingender Implementierungsfehler, aber ein relevantes Risiko für Aussagekraft, Robustheit oder Live-Betrieb. |
| **Kritischer Fehler** | Verhindert derzeit eine belastbare Reproduktion, eine korrekte Interpretation oder einen verantwortbaren nachgelagerten Einsatz. |
| **Nicht anhand der vorhandenen Informationen bestimmbar** | Der benötigte Datensatz, Zeitstempel, Versuchsaufbau oder externe Kalibrierwert fehlt im Repository. |

---

## 1. Executive Summary

Die Kernimplementierung der finalen Trainingsläufe ist intern weitgehend konsistent. Die zwölf mit `final_clean_v1` gekennzeichneten Läufe – vier Architekturen mit jeweils drei Seeds – verwenden denselben participant-basierten Split, dieselben 156 Sequenzen, dieselben 92 Rohfeatures, dieselbe Erweiterung auf 184 Modellfeatures und identische Fensterzahlen. Die gespeicherten Checkpoints passen zu Modelltyp, Eingabedimension und ausgewählten Epochen. Transformer, MLP, GRU und Residual Transformer v2 werden mit einer nachvollziehbaren hierarchischen Loss-Funktion trainiert.

Das wichtigste semantische Ergebnis ist jedoch: Das Modell sagt die **am Ende des aktuellen Fensters annotierte Aktivitätsphase** voraus. Es prognostiziert nicht, welche Intention in einer definierten Zukunft eintreten wird. Nur die Empfangshandpose ist als Ziel um `1.0 s` in die Zukunft verschoben. Die Bezeichnungen `continue`, `fetch` und `handover` sind daher im aktuellen Versuchsprotokoll brauchbare Phasenlabels, aber noch keine unmittelbar ausführbaren Roboterkommandos.

Die Intentionsmetriken sind für einen ersten Vergleich auf drei nicht im Training enthaltenen Testpersonen plausibel: GRU und Residual Transformer v2 erreichen im Mittel etwa `0.863` beziehungsweise `0.862` Test-Macro-F1. Diese Werte belegen eine Erkennung der geskripteten Phasen unter den aufgezeichneten Bedingungen. Sie belegen weder allgemeine menschliche Intentionserkennung noch Sicherheit, Kalibrierung, geringe Fehlalarmraten oder Eignung zur autonomen Roboterauslösung.

Die Poseergebnisse sind nicht robotertauglich. Der beste Residual-v2-Posecheckpoint erreicht über drei Seeds im Mittel etwa `14.55 cm` End-to-End-Positionsfehler; der einfache Last-Observation-Vergleich liegt bei etwa `14.88 cm`. Die Verbesserung beträgt damit im Mittel nur ungefähr `0.33 cm` und ist nicht über alle Seeds stabil. Beim aktuell für die Live-Demo verwendeten Seed 44 ist der Residual-Posecheckpoint in der Position sogar schlechter als diese Baseline. Orientierungsfehler um etwa `38–42°` bleiben groß. Außerdem ist das als `robot` bezeichnete Koordinatensystem das AprilTag-0-Referenzsystem und nicht nachweislich das reale Panda-Basissystem.

Zwischen Training und Live-Inferenz stimmen Artefaktprüfung, Feature-Reihenfolge, Normalisierung, Fenstergröße und Tensorform überein. Nicht identisch sind jedoch Sensorfusion und Zeitverhalten: Offline werden Messungen mit kleinen beidseitigen Nearest-Neighbor-Toleranzen fusioniert; live werden nur vergangene Messungen verwendet, teilweise mit deutlich größeren Toleranzen und langem Marker-Carry-Forward. Die vorhandenen Live-Logs zeigen zudem eine deutliche Verschiebung einzelner Validitätskanäle gegenüber den Trainingsdaten, insbesondere bei Gaze und Markern. Der neue `InputQualityGate` ist als konservatives Abstentionsprinzip sinnvoll, wurde in den vorhandenen Logs aber noch nicht mit der aktuellen Ausgabestruktur End-to-End belegt.

Die beobachtete Verzögerung ist überwiegend erklärbar. Gemessen wurden rund `2.37 s` einmaliger Fenster-Warm-up, eine Vorhersage ungefähr alle `0.33–0.40 s`, etwa `3.8 ms` mediane Modellinferenz und in einem beobachteten Klassenwechsel ungefähr `0.7 s` zusätzliche Verzögerung bis zur stabilen Entscheidung. Eine vollständige Capture-to-Decision-Latenz kann aus den vorhandenen Zeitstempeln nicht ermittelt werden.

### Gesamturteil in Kurzform

| Frage | Urteil |
|---|---|
| Ist die Kern-Daten- und Trainingslogik technisch nachvollziehbar? | **Ja, mit dokumentierten Einschränkungen.** |
| Sind die finalen Trainingsartefakte untereinander konsistent? | **Ja.** |
| Ist der final verwendete Datensatz aus dem aktuellen Repository exakt reproduzierbar? | **Nein. Kritische Provenienzlücke.** |
| Ist die aktuelle Live-Demo ohne Roboterausgabe grundsätzlich plausibel? | **Ja, als betreuter Shadow-/Inference-only-Test nach erneuter Validierung des aktuellen Quality Gates.** |
| Ist das System bereit, eine reale Roboteraktion auszulösen? | **Nein.** |

---

## 2. Untersuchter Projektumfang

### 2.1 Tatsächlich beteiligte Komponenten

| Ablauf | Zentrale Dateien und Funktionen |
|---|---|
| Rohdaten und Gaze | `Code/extract_multimodal_data.py`; Extraktion aus VRS und Kalibrierung |
| Sprach-/Phasenzeitpunkte | `Code/speech_recognition_demo.py`, `Code/review_timestamps_video.py`, `Code/apply_manual_reviews.py`, gemeinsame Parser in `Code/annotation_utils.py` |
| Marker | `Code/detect_tags.py` |
| Hand- und VIO-Daten | Project-Aria-MPS-Ausgaben; Einlesen und Zusammenführung in `Code/build_master_dataset.py` |
| Qualitätsprüfung/Manifest | `Code/dataset_qa.py` |
| Master-Datasets | `Code/build_master_dataset.py`, Stapelsteuerung in `Code/build_master_dataset_batch.py` |
| Trainingsdaten | `Training/data.py`, insbesondere `manifest_filtered_master_files`, `prepare_data`, `WindowDataset` |
| Modelle | `Training/model.py` |
| Transformer v1, MLP, GRU | `Training/train.py` |
| Residual Transformer v2 | `Training/train_residual.py` |
| Metriken | `Training/metrics.py` |
| Laufvergleich | `Training/compare_final_runs.py`, Bericht unter `Training/reports/` |
| Diagramme | `Training/evaluation/generate_training_diagrams.py` |
| Posebaselines/Exporte | `Training/evaluate_pose_baselines.py`, `Training/export_checkpoint_predictions.py` |
| Gemeinsame Online-Inferenz | `Training/online_inference.py` |
| Replay | `Training/replay_stream_inference.py` |
| Aria-Live-Stream | `Training/aria_live_inference.py`, insbesondere `LiveFeatureAssembler` |
| Qualitäts-, Ziel- und Zustandslogik | `Training/live_decision.py`: `InputQualityGate`, `GazeTargetSelector`, `PerceptionWorkflow` |

### 2.2 Nicht als aktive Pipeline bewertet

Dateien unter `archive/` und `references/` werden von den aktuellen Trainings- und Live-Einstiegspunkten nicht importiert. Ältere Verzeichnisse unter `Training/runs/` und `Training/runs/run_cluster/Ohne Titel/` bleiben als historische Artefakte erhalten, sind aber wegen abweichender Datensätze, Schemas oder fehlender finaler Fingerprints nicht mit `final_clean_v1` gleichzusetzen.

`Training/export_checkpoint_predictions.py` unterstützt trotz des allgemeinen Dateinamens nur den Transformerpfad. Das beeinflusst das Training nicht, kann aber bei späteren Auswertungen irreführen.

`Training/evaluation/generate_training_diagrams.py` sucht die finalen Läufe rekursiv, erzwingt alle vier Modelle und Seeds 42/43/44 und hat die vorhandenen CSV-/PNG-/PDF-Artefakte für zwölf Läufe erzeugt. `Training/compare_final_runs.py` sucht mit seinem Defaultpfad dagegen nur direkt unter `Training/runs/`. Die heute unter `Training/runs/run_cluster/Ohne Titel/` liegenden finalen Runs werden ohne ein explizites `--runs-root` nicht gefunden.

### 2.3 Durchgeführte sichere Prüfungen

Es wurde kein vollständiges Training gestartet und kein Roboterpfad ausgeführt. Die Prüfungen waren read-only oder verwendeten temporäre Ausgabeverzeichnisse:

- Syntaxprüfung aller 434 gefundenen Python-Dateien: keine Syntaxfehler.
- `Training/smoke_test.py`: Transformer, MLP und GRU erfolgreich.
- `Training/residual_smoke_test.py`: Residual-v2-Daten- und Trainingspfad erfolgreich.
- `Training/live_decision_smoke_test.py`: bestehende Quality-/Workflow-Tests erfolgreich.
- `tests/integration/static_robot_anchor_smoke.py`: vorhandener Transformations-Smoke-Test erfolgreich.
- `Training/pose_baselines_smoke_test.py` und `Training/export_predictions_smoke_test.py`: erfolgreich.
- Alle 24 finalen Checkpointdateien – Intentions- und Posecheckpoint für zwölf Läufe – konnten geladen und gegen Modelltyp, Eingabedimension und in `metrics.json` dokumentierte Auswahl geprüft werden.
- Die vorhandenen finalen `metrics.json`, `data_metadata.json`, Vergleichsberichte, Debug- und Live-Ausgaben wurden separat ausgewertet.

### 2.4 Grenzen des Audits

**Nicht anhand der vorhandenen Informationen bestimmbar:** Der vollständige Master-Dataset-Stand, mit dem auf dem Cluster trainiert wurde, ist lokal nicht vorhanden. Im aktuellen `Data_collection/master_datasets/` liegt nur ein Master-CSV, während die finalen Artefakte 156 ausgewählte Sequenzen dokumentieren. Deshalb konnten die finalen Rohzeilen nicht vollständig neu eingelesen und der Datensatz nicht bytegenau reproduziert werden.

**Nicht anhand der vorhandenen Informationen bestimmbar:** Physische Markergrößen, die externe Aria-Kalibrierung, eine Marker-0-zu-Panda-Basis-Kalibrierung und Ground-Truth-Zeitstempel für die reale Capture-to-Host-Latenz sind nicht im geprüften Bestand nachweisbar.

---

## 3. Rekonstruierter Datenfluss

### 3.1 End-to-End-Datenfluss

```text
Aria-VRS + Kalibrierung
        │
        ├── Gaze/Device-Zeitachse ───────────────┐
        ├── RGB ──> AprilTag/Aruco-Erkennung ────┤
        ├── Audio ──> Zeitpunkte ──> Review ─────┤
        └── MPS: Handpose + SLAM/VIO ────────────┤
                                                 v
                                  Code/dataset_qa.py
                                  Manifest + Ausschlüsse
                                                 │
                                                 v
                          Code/build_master_dataset.py
           Zeitfusion, Labels, Tag-0-Koordinaten, Future Targets
                                                 │
                              *_master.csv + *_master_report.json
                                                 │
                                                 v
                                Training/data.py
            Manifestfilter -> Teilnehmer-Split -> Normalisierung
                -> 60er-Fenster -> Missing-Data-Masken
                                                 │
                    ┌────────────┴────────────┐
                    v                         v
             Training/train.py      Training/train_residual.py
                    │                         │
             Checkpoints, metrics.json, data_metadata.json
                    └────────────┬────────────┘
                                 v
             Offline-Test / Replay / OnlineInferenceEngine
                                 │
                     Aria LiveFeatureAssembler
                                 │
               Raw -> Glättung -> Stable -> Quality Gate
                                 │
             Inference-only Ausgabe; keine Roboterbefehle
```

### 3.2 Labelerzeugung

`Code/build_master_dataset.py::add_intention_labels` verwendet die vier annotierten Zeitpunkte in folgender Reihenfolge:

```text
START             SECOND               DONE             THIRD
  │ continue         │ fetch              │ transition      │ handover
  ├──────────────────┼────────────────────┼─────────────────┼────────>
```

- `continue`: `START <= t < SECOND`
- `fetch`: `SECOND <= t <= DONE`
- `transition`: `DONE < t < THIRD`
- `handover`: `t >= THIRD`

In `Training/data.py` gilt `transition = -1`, `continue = 0`, `fetch = 1`, `handover = 2`.

**Bestätigt korrekt:** Ein Fenster darf Transition-Zeilen als historischen Kontext enthalten, sein Endpunkt darf aber nicht `transition` sein. `WindowDataset` verwirft einen solchen Endpunkt. Dadurch werden genau drei Modellklassen erzeugt.

**Verbesserungsbedarf:** Die Phasengrenzen stammen aus einem stark strukturierten Versuchsprotokoll. Das Modell kann phasen- oder bewegungsspezifische Muster lernen, ohne damit eine freie, vor einer Handlung entstehende Nutzerintention zu erkennen.

### 3.3 Zeitfusion und Koordinaten

Die Gaze-Zeitachse bildet die Master-Timeline. Handdaten, SLAM/VIO und Marker werden in `Code/build_master_dataset.py::nearest_merge` mit Standardtoleranzen von `12 ms`, `5 ms` beziehungsweise `20 ms` zugeordnet. `pandas.merge_asof(..., direction="nearest")` darf dabei sowohl einen vergangenen als auch einen leicht zukünftigen Sensorwert wählen.

`Code/build_master_dataset.py::estimate_static_world_robot` schätzt aus allen passenden AprilTag-0-/SLAM-Beobachtungen einer Sequenz eine robuste statische Transformation. Diese wird anschließend zur Umrechnung von Gaze, Händen und Objekten in das als `robot` bezeichnete Tag-0-Referenzsystem verwendet.

**Bestätigt korrekt:** Rotationen werden durchgehend als Quaternion `x, y, z, w` verarbeitet. `scipy.spatial.transform.Rotation` verwendet dieselbe Reihenfolge; die Implementierung normalisiert das Vorzeichen mit `w >= 0`, und Modellmetriken verwenden den betragsmäßigen Quaternion-Dot-Product.

**Plausibel, aber nicht vollständig validiert:** Die algebraische Transformationskette ist durch Code und Smoke Test nachvollziehbar.

**Kritischer Fehler für einen späteren Robotereinsatz:** Das `robot`-Koordinatensystem ist im vorhandenen Code das AprilTag-0-System. Eine vermessene Transformation zur tatsächlichen Franka-Panda-Basis ist nicht implementiert oder artefaktseitig belegt. Der Name darf daher nicht als Nachweis eines Roboterbasissystems interpretiert werden.

**Verbesserungsbedarf:** Die Offline-Schätzung verwendet Beobachtungen aus der gesamten Sequenz. Dadurch können spätere Tag-Beobachtungen die Koordinaten früherer Frames beeinflussen. Für reine Offline-Phasenklassifikation ist dies eine begrenzte Form nichtkausaler Vorverarbeitung; zur exakten Live-Parität müsste die Offline-Aufbereitung ebenfalls nur bis zum jeweiligen Zeitpunkt verfügbare Anker verwenden oder der Livepfad einen bereits stabil kalibrierten, eingefrorenen Anker erhalten.

### 3.4 Featurevektor

Das Profil `multimodal_robot_frame_v1` in `Training/data.py::_candidate_features` umfasst bei vollständigem Schema 92 Rohfeatures:

| Gruppe | Anzahl | Inhalt |
|---|---:|---|
| Gaze | 10 | Validität, Yaw/Pitch/Depth, Ursprung und Richtung im Tag-0-System |
| Hände | 18 | zwei Trackingkonfidenzen, zwei Validitäten, je Hand Position `xyz` und Quaternion `xyzw` |
| VIO/SLAM | 7 | lineare und Winkelgeschwindigkeit sowie Quality Score |
| Referenzmarker | 3 | `apriltag_0_valid`, `robot_frame_valid`, `robot_anchor_interpolated` |
| Objektmarker 6–14 | 54 | je Marker `xyz`, Gazewinkel, Distanz, Validität |
| **Summe** | **92** | |

Der `Normalizer` hängt für jedes Rohfeature einen Beobachtungskanal an. Die Modelle erhalten deshalb:

```text
[Batch, 60, 92 Rohfeatures + 92 Beobachtungsmasken]
                         =
                    [B, 60, 184]
```

**Bestätigt korrekt:** Nicht endliche Rohwerte werden nach der z-Normalisierung auf `0` gesetzt; die zusätzliche Maske unterscheidet einen fehlenden Wert von einem tatsächlich am Trainingsmittel liegenden Wert.

---

## 4. Daten- und Splitvalidierung

### 4.1 Manifestfilter und Ausschlüsse

Alle vier finalen Konfigurationen verwenden:

- `allowed_statuses: ["valid"]`
- `allowed_next_actions: ["ready_for_master_merge"]`
- `strict: true`

`Training/data.py::manifest_filtered_master_files` prüft das Manifest, gruppiert Teilnehmernamen über `canonical_participant`, protokolliert Ausschlussgründe und bildet einen Fingerprint aus den ausgewählten Sequenz-IDs.

**Bestätigt korrekt:** Sequenzen mit `valid_with_warnings` wurden in den finalen Artefakten nicht stillschweigend akzeptiert. Der Metadatensatz dokumentiert unter anderem 32 entsprechende Ausschlüsse. Warnungen wurden für dieses Training somit nicht einfach ignoriert.

**Bestätigt korrekt:** `David` und `david` werden mit `casefold().capitalize()` beide als `David` gruppiert. Die Sequenz `david_7_...` liegt deshalb im selben Trainings-Teilnehmer wie die `David_...`-Sequenzen. Der Teilnehmername `Test` wird als normaler Personenname behandelt; er hat keine technische Test-Sonderbedeutung.

### 4.2 Split und Datenleckage

Die expliziten Teilnehmer aus allen vier Konfigurationen sind:

- Validation: `Atilla`, `Ermal`, `Vanessa`
- Test: `Edu`, `Jona`, `Mona`
- Training: alle übrigen ausgewählten Teilnehmer

Die finalen Metadaten enthalten:

| Split | Teilnehmer | Sequenzen | Fenster |
|---|---:|---:|---:|
| Train | 17 | 116 | 10.958 |
| Validation | 3 | 19 | 1.780 |
| Test | 3 | 21 | 1.981 |

**Bestätigt korrekt:** Die kanonisierten Teilnehmermengen sind disjunkt. Es wurde kein Teilnehmer über mehrere Splits verteilt. Dies verhindert das wichtigste offensichtliche Identitätsleck zwischen überlappenden Fenstern.

**Bestätigt korrekt:** Der Normalizer wird in `Training/data.py::fit_normalizer` ausschließlich aus den Trainingssequenzen und ausschließlich aus endlichen Werten geschätzt. Validation und Test verwenden anschließend denselben gespeicherten Normalizer.

**Bestätigt korrekt:** Target-, Label-, Teilnehmer- und Zeitspalten sind nicht Teil der 92 Eingabefeatures. Ein direktes Einlesen des Zukunftstargets als Modellfeature wurde nicht gefunden.

**Bestätigt:** `target_object_id` ist weder Eingabefeature noch neuronales Ausgabeziel. Die spätere Live-Objektwahl erfolgt separat über `GazeTargetSelector`; die Intentionsmetriken bewerten keine korrekte Objektidentität.

**Verbesserungsbedarf:** Die Fenster eines Videos überlappen bei `window_size=60` und `stride=10` um 50 Frames beziehungsweise 83,3 %. Das ist innerhalb des Trainingssplits zulässig, führt aber dazu, dass Fenstermetriken keine unabhängigen Stichproben sind. Konfidenzintervalle und Ereignisstatistiken sollten auf Sequenz- oder Teilnehmerebene berechnet werden.

### 4.3 Fenster, Zeitlücken und Transition

`Training/data.py::WindowDataset`:

- erzeugt 60-Frame-Fenster,
- schreitet um 10 Frames weiter,
- setzt das Intentionsziel auf das Label am Fensterendpunkt,
- verwirft Endpunkte mit `transition`,
- verwirft Fenster mit einer Zeitlücke über `0.2 s`,
- verwirft Fenster mit weniger als `5 %` beobachteten Rohfeatureeinträgen.

In den finalen Metadaten wurden für Train/Validation/Test `877/150/136` Transition-Endpunkte verworfen; Zeitlücken- und Low-Observation-Ausschlüsse waren jeweils `0`.

**Bestätigt korrekt:** Ein einzelnes Fenster überschreitet keine Sequenzgrenze.

**Verbesserungsbedarf:** 60 Frames entsprechen nicht zwingend einer konstanten Zeitdauer, weil weder beim Masterbau noch live auf eine feste Rate resampelt wird und kein Delta-Time-Feature enthalten ist. Das Modell lernt Framepositionen, nicht eine explizite physikalische Zeitachse.

**Verbesserungsbedarf:** Der globale Mindestbeobachtungsanteil von `0.05` ist sehr niedrig und modality-unabhängig. Ein Fenster kann diesen Wert durch wenige gut beobachtete Gruppen erreichen, obwohl eine semantisch zentrale Modalität fehlt. Die Live-Qualitätsprüfung ist strenger, aber diese strengeren Bedingungen sind nicht Teil der Trainings-/Testselektion.

### 4.4 Klassenverteilung

| Split | continue | fetch | handover |
|---|---:|---:|---:|
| Train | 7.245 (66,1 %) | 1.694 (15,5 %) | 2.019 (18,4 %) |
| Validation | 1.198 (67,3 %) | 273 (15,3 %) | 309 (17,4 %) |
| Test | 1.396 (70,5 %) | 314 (15,9 %) | 271 (13,7 %) |

**Bestätigt korrekt:** `train.py::class_weights` und `train_residual.py::class_weights` bilden normalisierte inverse Gewichte getrennt für Assistance, Assistance Type und bei v2 für die Empfangshand. Die Gewichte werden nur aus dem Trainingssplit abgeleitet.

**Verbesserungsbedarf:** Die dominant häufige `continue`-Klasse sowie stark unterschiedliche Sequenzzahlen pro Trainingsperson können Modelle in Richtung häufig vorkommender Personen und Phasen verzerren. Der Loader sampelt Fenster, nicht Teilnehmer oder Ereignisse balanciert.

### 4.5 Schema- und Provenienzprüfung

Die zwölf finalen Läufe dokumentieren übereinstimmend:

- 198 Manifestzeilen,
- 183 gefundene Masterdateien,
- 156 ausgewählte Sequenzen,
- 27 nach Manifestfilter ausgeschlossene Sequenzen,
- Fingerprint `457a80f15423fe3e3853081e3a0d863248ec337dd7412cde65bc8ee56ff3049d`,
- exakt dieselbe Featureliste und denselben Split.

Der aktuelle Repositorybestand entspricht diesem Datenzustand jedoch nicht:

- Das aktuelle `Data_collection/dataset_manifest.csv` enthält ebenfalls 198 Zeilen, aber nur 109 erfüllen derzeit den strikten Filter.
- 47 der früher ausgewählten 156 IDs sind im aktuellen Manifest nicht mehr entsprechend freigegeben.
- Das aktuelle Manifest meldet 127 vorhandene Master-CSV-Dateien, lokal ist aber nur eine vorhanden.
- `Data_collection/dataset_qa_report.json` und das Manifest stammen erkennbar aus unterschiedlichen Bestandsständen.
- Der lokal vorhandene Master von `Jona_6_...` besitzt ein älteres Schema und wird vom strikten Replaypfad zu Recht abgelehnt.

Der `sequence_fingerprint` wird nur über sortierte Sequenz-IDs gebildet. Er ändert sich nicht, wenn der Inhalt eines Master-CSV unter derselben ID neu gebaut oder korrigiert wird.

**Kritischer Fehler:** Der final verwendete Datensatz ist aus dem aktuellen Repository nicht bytegenau rekonstruierbar. Es fehlen ein eingefrorener Manifest-Snapshot, Inhalts-Hashes der 156 Masterdateien, eine Builder-/Schemasignatur und ein eindeutiger Verweis auf den Code-/Containerstand. Dieser Befund bedeutet nicht, dass die gespeicherten finalen Metriken intern falsch sind; er verhindert aber eine unabhängige Reproduktion und eine belastbare Zuordnung zu den heute sichtbaren Rohdaten.

**Plausibel, aber nicht vollständig validiert:** `singularity/aria.recipe` pinnt die wesentlichen Clusterpakete, darunter Python 3.10, Torch 2.4.1, NumPy 2.2.6, SciPy 1.15.3 und Pandas 2.3.3. Das ist eine gute Grundlage. Der verwendete Image-Digest wird jedoch nicht in den Trainingsartefakten gespeichert; `Training/requirements.txt` enthält nur breite Untergrenzen. Der exakte Laufzeitcontainer eines finalen Runs ist daher nicht artefaktseitig beweisbar.

**Verbesserungsbedarf:** `Code/build_master_dataset_batch.py` überspringt existierende Masterdateien standardmäßig, ohne deren Schema- oder Builderversion zu prüfen. Dadurch können alte und neue Masterschemas unbemerkt gemischt werden.

**Verbesserungsbedarf:** `Training/data.py::select_feature_columns` bestimmt die Featureauswahl anhand des Headers der ersten ausgewählten Masterdatei. Später fehlende Spalten werden mit `NaN` beziehungsweise bei `*_valid` mit `0` ergänzt. Ein unvollständiges erstes Masterfile könnte dadurch das Featureset aller Sequenzen verkleinern. In den finalen Artefakten sind zwar 92 Features dokumentiert, die allgemeine Implementierung sollte dennoch gegen ein kanonisches Schema prüfen.

### 4.6 Missing Data und Sensormodalitäten

**Bestätigt korrekt:** Offline fehlende numerische Werte bleiben bis zur Normalisierung `NaN`; die beobachtet-Maske wird daraus korrekt abgeleitet. Pose-Targets benötigen explizite Validität, endliche sieben Werte und ein normierbares Quaternion.

**Verbesserungsbedarf:** Live wird bei fehlender Hand teilweise `tracking_confidence = -1` als endlicher Wert gesetzt, während ein vollständig fehlendes Offline-Match typischerweise `NaN` und damit `unobserved` ergibt. Das erzeugt eine andere Kombination aus Wert und Missing-Mask.

**Verbesserungsbedarf:** Offline darf Nearest-Neighbor-Fusion einen geringfügig zukünftigen Messwert verwenden. Live nutzt `latest_before_item` und ist damit kausal. Diese Differenz ist klein, aber für eine behauptete exakte Replay-/Live-Parität relevant.

### 4.7 Future-Pose und Empfangshand

`Code/build_master_dataset.py::add_future_targets` sucht die Handpose bei `t + 1.0 s` mit einer Toleranz von `12 ms`. `add_receiving_hand_target` wählt anhand der sequenzweiten manuellen Annotation `left` oder `right` das Zielgelenk.

Für Residual v2 sucht `WindowDataset` innerhalb des aktuellen 60er-Fensters die jeweils letzte gültige linke und rechte Handpose. Diese Suche ist kausal. Ein Pose-Loss ist nur gültig, wenn:

- der Endpunkt `handover` ist,
- ein gültiges Future-Target existiert,
- eine bekannte Empfangshand vorliegt und
- für diese Hand eine gültige Referenz im Fenster existiert.

Finale v2-Zählungen:

| Split | linke Hand | rechte Hand | gültige Residual-Poseziele |
|---|---:|---:|---:|
| Train | 651 | 1.368 | 1.617 |
| Validation | 0 | 309 | 247 |
| Test | 57 | 214 | 202 |

**Bestätigt korrekt:** Oracle- und End-to-End-Auswertung sind im v2-Code getrennt. Oracle wählt den Posekandidaten mit der Ground-Truth-Hand; End-to-End wählt ihn mit der vorhergesagten Hand.

**Verbesserungsbedarf:** Im Validationssplit gibt es kein einziges linkes Handover-Fenster. Modellselektion und Early Stopping validieren die linke Empfangshand daher nicht. Die hohe Handklassifikationsleistung ist außerdem durch Überlappung und das Verhältnis 57:214 im Test eingeschränkt aussagekräftig.

**Nicht anhand der vorhandenen Informationen bestimmbar:** Ob die manuell annotierte Empfangshand und das Ziel bei `t + 1 s` für jede finale Cluster-Masterdatei inhaltlich korrekt sind, kann ohne die 156 verwendeten CSV-Inhalte und eine unabhängige Annotation nicht bestätigt werden.

---

## 5. Bewertung der vier Modelle

### 5.1 Gemeinsame Aufgabe und hierarchische Ausgabe

Alle Modelle erhalten `[B, 60, 184]` und erzeugen mindestens:

- `assistance_logits [B, 2]`: `continue` gegen `assistance`,
- `assistance_type_logits [B, 2]`: innerhalb Assistance `fetch` gegen `handover`,
- eine Handposeausgabe.

Die Offline-Klassenentscheidung in `Training/metrics.py` beziehungsweise den Trainingsschleifen lautet:

```text
assistance = 0  -> continue
assistance = 1  -> assistance_type = 0 -> fetch
assistance = 1  -> assistance_type = 1 -> handover
```

Der Assistance-Loss gilt für alle Fenster; Assistance Type nur für Ground-Truth-`fetch` und -`handover`; der Pose-Loss nur für gültige `handover`-Ziele.

**Bestätigt korrekt:** Die Maskierung der beiden bedingten Loss-Anteile entspricht der beschriebenen Hierarchie.

### 5.2 Gemeinsame Trainingsschleife

`Training/train.py::run_epoch` und `Training/train_residual.py::run_epoch` verwenden Batches der Größe 32. Der Trainingsloader wird gemischt, Validation und Test nicht; konfiguriert sind zwei Worker. Pro Trainingsbatch erfolgt:

```text
Batch auf Gerät
 -> Forward Pass
 -> maskierte Teil-Losses
 -> gewichtete Summe
 -> zero_grad
 -> backward
 -> Gradient Clipping auf Norm 1,0
 -> AdamW-Schritt
```

Alle vier Konfigurationen verwenden AdamW mit Lernrate `3e-4`, Weight Decay `1e-4`, höchstens 20 Epochen und Early-Stopping-Patience 7. Ein Lernratenscheduler ist nicht implementiert. Die Poseorientierung geht mit Faktor `0.25` in den Pose-Loss ein. Die Gesamtgewichte von Assistance, Assistance Type, Pose und – bei v2 – Empfangshand stehen jeweils auf `1.0`.

Der Positions-Loss ist Smooth L1 über `xyz`. Der Orientierungs-Loss ist `1 - |q_pred · q_target|`; der Betrag macht ihn invariant gegenüber den äquivalenten Quaterniondarstellungen `q` und `-q`. Bei Residual v2 wird dieser Loss auf dem mit der Ground-Truth-Hand gewählten Oracle-Kandidaten berechnet.

Aus den finalen Trainingszählungen resultieren ungefähr folgende Cross-Entropy-Gewichte:

| Kopf | Trainingszählungen | Gewichte |
|---|---|---|
| Assistance: continue/assistance | 7.245 / 3.713 | 0,678 / 1,322 |
| Assistance Type: fetch/handover | 1.694 / 2.019 | 1,088 / 0,912 |
| Empfangshand v2: links/rechts | 651 / 1.368 | 1,355 / 0,645 |

**Bestätigt korrekt:** Early Stopping wird zurückgesetzt, wenn sich entweder Intentions-Macro-F1 oder Poseposition verbessert. Transformer v1, MLP und GRU speichern `best_model.pt` und `best_pose_model.pt`; Residual v2 speichert `best_intention_model.pt` und `best_pose_model.pt`. Beide Stände werden anschließend separat auf dem Testsplit ausgewertet.

**Bestätigt korrekt:** Jeder Lauf speichert die aufgelöste `config.json`, `data_metadata.json` mit Split/Featureliste/Normalizer, beide Modellcheckpoints und `metrics.json` mit Historie und Testergebnissen.

**Verbesserungsbedarf:** Python, NumPy, Torch und CUDA werden geseedet, deterministische CUDA-Algorithmen werden jedoch nicht erzwungen. Außerdem werden Optimizerzustand und letzter Trainingszustand nicht gespeichert; die Checkpoints sind für Inferenz und Evaluation geeignet, aber nicht für eine exakte Trainingsfortsetzung.

**Kleine metrische Einschränkung:** Die Epoch-Losses sind Mittelwerte der Batchmittelwerte. Der kleinere letzte Batch erhält damit dasselbe Gewicht wie ein voller Batch. Dies verändert das Training nicht, kann den berichteten Epoch-Loss aber geringfügig gegenüber einem streng samplegewichteten Mittel verschieben.

### 5.3 Transformer v1

Klasse: `Training/model.py::HierarchicalGatedMultimodalTransformer`  
Konfiguration: `Training/configs/hierarchical_baseline_v1.json`

Der Transformer verwendet zwei parallele Repräsentationen:

1. **Temporaler Pfad:** jedes der 60 Zeitframes wird von 184 auf `d_model=64` projiziert; ein lernbarer CLS-Token und 61 Positionsvektoren werden hinzugefügt; zwei Transformer-Encoderlagen mit vier Köpfen und Feedforward-Dimension 128 liefern den temporalen CLS-State `[B, 64]`.
2. **Kanalpfad:** die Eingabe wird zu `[B, 184, 60]` transponiert; jeder Featurekanal wird über seine 60 Werte auf 64 Dimensionen projiziert; CLS- und Kanalidentitätsvektoren sowie zwei Encoderlagen liefern einen Kanal-CLS-State `[B, 64]`.

Ein Softmax-Gate mit zwei Gewichten skaliert beide States. Die Verkettung ergibt `[B, 128]`. Daraus entstehen Assistance-, Type- und absoluter 7D-Posekopf. Das Quaternion des Posekopfs wird normalisiert.

Parameterzahl der finalen Läufe: 183.629.

**Bestätigt korrekt:** Der Transformer enthält keinen Dreiecks-Attention-Mask. Für eine Endpunktvorhersage auf einem ausschließlich vergangenen/aktuellen Fenster ist das innerhalb des Fensters dennoch kausal nutzbar: Jeder Token darf andere bereits beobachtete Fensterframes sehen.

**Vorteil für Live:** paralleles Modell, explizite Zeit- und Kanalmuster, kurze gemessene Inferenz.  
**Nachteil:** feste Fensterlänge und gelernte Framepositionen; empfindlich gegen Samplingratenverschiebung; der Offline-Anker und Nearest-Merge sind nicht exakt live-kausal.

### 5.4 MLP

Klasse: `Training/model.py::HierarchicalWindowMLP`  
Konfiguration: `Training/configs/hierarchical_mlp_v1.json`

Das MLP flacht `[B, 60, 184]` zu `[B, 11.040]` ab. Es verwendet die konfigurierten Hidden-Dimensionen `[16, 128]`; jede Stufe besteht aus Linear, LayerNorm, GELU und Dropout. Der abschließende State `[B, 128]` speist dieselben drei Köpfe wie Transformer v1.

Parameterzahl: 197.051.

**Vorteil für Live:** einfache, parallele und schnelle Baseline.  
**Nachteil:** Zeitpositionen sind nur durch ihre feste Stelle im abgeflachten Vektor repräsentiert; keine rekurrente Zustandsstruktur und keine explizite zeitliche oder kanalweise Gewichtsteilung. Das Modell reagiert besonders empfindlich auf zeitliche Verschiebung oder andere Frameraten.

### 5.5 GRU

Klasse: `Training/model.py::HierarchicalGRU`  
Konfiguration: `Training/configs/hierarchical_gru_v1.json`

Die GRU ist unidirektional, besitzt zwei Lagen, `hidden_size=112` und Dropout `0.15` zwischen den Lagen. Der letzte Hidden State der obersten Lage `hidden[-1]` wird normalisiert und als `[B, 112]` an die Assistance-, Type- und absoluten Poseköpfe übergeben.

Parameterzahl: 190.187.

**Vorteil für Live:** zeitlich gerichtete Verarbeitung ohne Blick auf zukünftige Frames; bestes mittleres Intentionsresultat im finalen Vergleich.  
**Nachteil:** Die aktuelle Online-Implementierung berechnet trotzdem immer das gesamte 60er-Fenster neu und nutzt keinen persistenten GRU-State. Der theoretische Streamingvorteil wird daher noch nicht ausgeschöpft. Auch die GRU bleibt von uneinheitlicher Framezeit betroffen.

### 5.6 Residual Transformer v2

Klasse: `Training/model.py::HierarchicalResidualPoseTransformer`  
Konfiguration: `Training/configs/hierarchical_residual_v2.json`  
Training: `Training/train_residual.py`

Der Encoder und der `[B, 128]`-Fusionsstate entsprechen Transformer v1. Zusätzlich entstehen:

- `receiving_hand_logits [B, 2]` für links/rechts,
- eine Softmax-Handwahrscheinlichkeit `[B, 2]`,
- ein gemeinsames Positionsresiduum `[B, 3]`,
- ein normiertes Quaternionresiduum `[B, 4]`,
- zwei Posekandidaten `[B, 2, 7]`.

Die Handwahrscheinlichkeit wird an den Fusionsstate angehängt und dem Residual-Posekopf übergeben. Dasselbe Positions- und Rotationsresiduum wird auf die letzte gültige linke beziehungsweise rechte Handreferenz angewandt. Für Quaternionen gilt im Code:

```text
candidate_orientation = reference_orientation * predicted_delta_orientation
```

Der finale Layer startet mit Nullgewichten und einer Identitätsrotation. Damit beginnt das Modell näherungsweise als Last-Observation-Modell.

Parameterzahl: 184.015.

**Oracle-Pose:** Auswahl des Kandidaten mit der Ground-Truth-Empfangshand.  
**End-to-End-Pose:** Auswahl mit der vorhergesagten Empfangshand; dies entspricht eher dem späteren Einsatz.

**Bestätigt korrekt:** Der Pose-Loss trainiert Oracle-basiert nur den Kandidaten der wahren Hand. Die Evaluation speichert Oracle- und End-to-End-Metriken getrennt.

**Bestätigt korrekt:** Intentions- und Posecheckpoint werden getrennt gespeichert. Der beste Intentionscheckpoint maximiert Validation-Macro-F1; der beste Posecheckpoint minimiert die Oracle-Positionsmetrik. Dadurch wird verhindert, dass eine Verbesserung der Pose einen guten Intentionsstand überschreibt oder umgekehrt.

**Verbesserungsbedarf:** Der beste Posecheckpoint wird anhand der Oracle-Position ausgewählt, nicht anhand der End-to-End-Pose, Orientierung, Target-Coverage oder einer kombinierten Einsatzmetrik. Die Handreferenzalter werden zwar berechnet und ausgegeben, aber weder als Modellinput noch als Loss-/Qualitätskriterium verwendet.

### 5.7 Architekturvergleich

| Modell | Repräsentation | Poseart | Kerneigenschaft für Live |
|---|---|---|---|
| Transformer v1 | temporaler CLS + Kanal-CLS, gegatet, 128D | absolute 7D-Pose | starke multimodale Fusion, feste Framepositionen |
| MLP | abgeflachtes Fenster, 128D | absolute 7D-Pose | einfach/schnell, geringste zeitliche Induktionsannahme |
| GRU | letzter Hidden State, 112D | absolute 7D-Pose | kausal gerichtete Sequenzmodellierung |
| Residual v2 | Transformer-Fusion, 128D + Handwahrscheinlichkeit | Delta relativ zu letzter Handpose | beste Poseidee, explizite Empfangshand |

---

## 6. Bewertung der Trainings- und Testergebnisse

### 6.1 Vergleichbarkeit der finalen Läufe

Für diese Bewertung wurden nur die zwölf Verzeichnisse `final_clean_v1_*_seed{42,43,44}` unter `Training/runs/run_cluster/Ohne Titel/` als finaler Vergleich verwendet. Der vorhandene Bericht `Training/reports/final_clean_v1_comparison.json` wurde gegen die einzelnen Artefakte geprüft.

**Bestätigt korrekt:** Alle zwölf finalen Läufe besitzen denselben Sequenzfingerprint, Split, Fensterbestand, Featurevektor und Normalizer. Die `.err`-Dateien sind leer; in den geprüften Metriken wurden keine NaN-/Infinity-Werte gefunden.

**Bestätigt korrekt:** `Training/final_clean_v1_residual_v2_seed44` entspricht bytegleich dem zugehörigen Clusterlauf hinsichtlich Konfiguration, Metadaten, Metriken und Checkpoint-Hashes. Die Live-Demo lädt somit das dokumentierte Residual-v2-Seed-44-Artefakt.

### 6.2 Intentionsmetriken über Seeds

Mittelwerte und Stichprobenstandardabweichung der Test-Macro-F1:

| Modell | Validation Macro-F1, Mittel | Test Macro-F1 | continue F1 | fetch F1 | handover F1 |
|---|---:|---:|---:|---:|---:|
| Transformer v1 | 0,9168 | 0,8515 ± 0,0071 | 0,9379 | 0,7769 | 0,8397 |
| MLP | 0,9284 | 0,8324 ± 0,0287 | 0,9118 | 0,7638 | 0,8216 |
| GRU | 0,9251 | **0,8629 ± 0,0154** | 0,9401 | **0,7915** | 0,8570 |
| Residual v2 | 0,9224 | 0,8620 ± 0,0141 | 0,9371 | 0,7713 | **0,8776** |

Einzelne Test-Macro-F1-Werte:

| Modell | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Transformer v1 | 0,8456 | 0,8496 | 0,8593 |
| MLP | 0,8444 | 0,8532 | 0,7996 |
| GRU | 0,8633 | 0,8781 | 0,8472 |
| Residual v2 | 0,8783 | 0,8534 | 0,8544 |

Die folgenden Konfusionsmatrizen sind über die drei Seeds aufsummiert. Da jeder Seed dieselben 1.981 Testfenster sieht, sind die insgesamt 5.943 Einträge pro Modell **keine unabhängigen zusätzlichen Samples**, sondern dienen nur zum Vergleich der Fehlerrichtungen.

| Modell | Diagonale C/F/H | C→F | C→H | F→C | F→H | H→C | H→F |
|---|---|---:|---:|---:|---:|---:|---:|
| Transformer v1 | 3.730 / 796 / 790 | 289 | 169 | 34 | 112 | 1 | 22 |
| MLP | 3.535 / 833 / 801 | 398 | 255 | 26 | 83 | 2 | 10 |
| GRU | 3.755 / 807 / 796 | 276 | 157 | 42 | 93 | 3 | 14 |
| Residual v2 | 3.746 / 828 / 765 | 343 | 99 | 47 | 67 | 14 | 34 |

Dabei steht `C` für continue, `F` für fetch und `H` für handover. Das MLP erzeugt die meisten falschen Assistance-Ausgaben aus `continue`. Residual v2 trennt `fetch` und `handover` vergleichsweise gut, verwechselt aber mehr echte Handover-Fenster mit `continue` oder `fetch` als GRU.

**Belegt:** GRU erzielt den höchsten mittleren Test-Macro-F1-Wert; Residual v2 liegt praktisch gleichauf und hat den höchsten mittleren Handover-F1. Fetch ist bei drei der vier Modellfamilien die schwächste Klasse.

**Plausible Schlussfolgerung:** Die zeitlich strukturierten Modelle GRU und Transformer sind für dieses Phasenmuster geeigneter als das MLP. Der Unterschied zwischen GRU und Residual v2 ist mit nur drei Seeds und einem fixen Testsplit nicht als statistisch gesichert anzusehen.

**Nicht getestet:** Generalisierung auf freie, nicht geskriptete Handlungen, andere Räume, andere Markeranordnungen, längere Ruhephasen, Gegenbeispiele ohne Assistenzwunsch oder andere Nutzergruppen.

### 6.3 Ausgewähltes Live-Modell

Für Residual v2 Seed 44, bester Intentionscheckpoint:

- Accuracy: `0,8910`
- Macro-F1: `0,8544`
- F1 continue/fetch/handover: `0,9296 / 0,7500 / 0,8836`
- Konfusionsmatrix, Zeilen = Ground Truth, Spalten = Vorhersage:

|  | continue | fetch | handover |
|---|---:|---:|---:|
| continue | 1.255 | 117 | 24 |
| fetch | 35 | 267 | 12 |
| handover | 14 | 14 | 243 |

**Bestätigt korrekt:** Diese Matrix ergibt die in `metrics.json` gespeicherten Supports und Metriken.

**Verbesserungsbedarf:** 141 von 1.396 `continue`-Fenstern werden als Assistance eingestuft. Für eine spätere Aktion ist diese Fehlerart wichtiger als der aggregierte Macro-F1 und muss als Fehlalarmrate pro Zeit beziehungsweise pro neutralem Ereignis gemessen werden.

### 6.4 Overfitting und Checkpointwahl

Die besten Intentions-Epochen lauten:

- Transformer: 8, 2, 3
- MLP: 2, 8, 16
- GRU: 20, 6, 3
- Residual v2: 2, 1, 2

Über die drei Seeds gemittelte **Gesamt-Losses** aus den gespeicherten Historien:

| Modell | Train Epoche 1 | Val Epoche 1 | Train letzte Epoche | Val letzte Epoche |
|---|---:|---:|---:|---:|
| Transformer v1 | 0,567 | 0,260 | 0,058 | 0,653 |
| MLP | 0,675 | 0,233 | 0,108 | 0,361 |
| GRU | 0,535 | 0,230 | 0,039 | 0,517 |
| Residual v2 | 0,648 | 0,226 | 0,074 | 0,547 |

Die letzte Epoche ist wegen Early Stopping nicht bei jedem Seed dieselbe. Außerdem enthält der v2-Gesamt-Loss zusätzlich den Empfangshand-Loss; die absoluten v2-Losswerte sind daher nicht direkt mit den drei anderen Familien vergleichbar. Innerhalb jeder Familie ist der gegensätzliche Verlauf dennoch deutlich.

**Bestätigt korrekt:** Ein bester Checkpoint in Epoche 1 oder 2 ist technisch nicht automatisch zu früh. Er bedeutet, dass die gewählte Validation-Metrik dort am besten war. Die Trainingsschleife lädt für den Test den gespeicherten besten Checkpoint und nicht einfach den letzten Epochestand.

**Verbesserungsbedarf:** Mehrere Historien zeigen sehr hohe Train-F1-Werte und weiter sinkenden Train-Loss, während Validation-Loss oder -F1 anschließend schlechter werden. Das ist eine klare Überanpassungstendenz. Der frühe Checkpoint begrenzt deren Auswirkung, löst aber nicht die Ursachen.

**Verbesserungsbedarf:** Validation ist nur drei Personen groß. Dass MLP im Mittel den höchsten Validation-Macro-F1, aber den niedrigsten Test-Macro-F1 erreicht, zeigt eine relevante Auswahlunsicherheit. Teilnehmerbasierte Cross-Validation oder mehrere feste Participant-Folds wären belastbarer.

### 6.5 Pose, Oracle und Last-Observation-Baseline

Vergleich der jeweils besten Posecheckpoints über drei Seeds auf denselben 202 gültigen Testfenstern:

| Modell/Poseart | Position, Mittel ± SD | Orientierung, Mittel ± SD |
|---|---:|---:|
| Transformer v1, absolut | 18,63 ± 1,54 cm | 50,25 ± 6,14° |
| MLP, absolut | 18,39 ± 0,25 cm | 53,48 ± 0,71° |
| GRU, absolut | 16,75 ± 2,04 cm | 47,65 ± 7,15° |
| Residual v2, End-to-End | **14,55 ± 1,17 cm** | **40,41 ± 2,18°** |

**Belegt:** Der residuale Ansatz ist im Mittel besser als die drei direkt absolut regressierenden Köpfe. Dieser Vergleich allein zeigt aber noch nicht, dass v2 tatsächlich zukünftige Bewegung lernt, weil auch die starke Last-Observation-Referenz Bestandteil seiner Ausgabe ist.

Für Residual v2, jeweils bester Posecheckpoint:

| Seed | End-to-End Position | End-to-End Orientierung |
|---|---:|---:|
| 42 | 14,97 cm | 41,81° |
| 43 | 13,23 cm | 37,90° |
| 44 | 15,45 cm | 41,52° |
| **Mittel ± SD** | **14,55 ± 1,17 cm** | ungefähr 40° |

Last-Observation-Baseline im Test:

- Position: `14,88 cm`
- Orientierung: `43,06°`

**Wichtige Einordnung:** Diese Metrik heißt im Artefakt `last_observation_oracle`: Sie verwendet die Ground-Truth-Empfangshand, nicht die vorhergesagte Hand. Der Vergleich zur End-to-End-v2-Pose ist dadurch konservativ, aber semantisch nicht vollständig symmetrisch. Zusätzlich sollte eine echte End-to-End-Baseline „vorhergesagte Hand + deren letzte Beobachtung“ berichtet werden.

Für den live verwendeten Seed 44:

- bester Intentionscheckpoint, End-to-End: `15,28 cm / 44,00°`
- bester Posecheckpoint, End-to-End: `15,45 cm / 41,52°`
- bester Posecheckpoint, Oracle: `15,52 cm / 40,35°`

**Belegt:** Residual v2 verbessert die Position im Drei-Seed-Mittel nur um etwa `0,33 cm` gegenüber der Last-Observation-Baseline. Seed 43 verbessert sie deutlich, Seed 42 und 44 dagegen nicht. Die Poseverbesserung ist damit nicht seed-stabil.

**Belegt:** Frühe Handover-Fenster im 0–25-%-Fortschrittsbin profitieren bei allen drei v2-Seeds: Gegenüber `24,14 cm` Last Observation erreichen Seed 42/43/44 etwa `20,01 / 17,13 / 19,48 cm` bei jeweils 74 Fenstern. Das ist für eine frühe Schätzung interessant. In späteren Bins gewinnt die Baseline häufig wieder; außerdem basieren die Bins auf Fensterindex statt kalibrierter Zeit oder körperlicher Bewegungsphase.

**Verbesserungsbedarf:** Nur 202 Testfenster besitzen ein gültiges Residual-Poseziel. Im letzten 75–100-%-Bin sind nur sechs Samples vorhanden, weil nahe dem Sequenzende häufig kein Ziel bei `+1 s` mehr existiert.

**Verbesserungsbedarf:** Die Metrik `position_mae_cm` in `Training/metrics.py::pose_metrics` ist tatsächlich der Mittelwert der euklidischen 3D-Distanz, kein komponentenweiser Mean Absolute Error. Die Zahl ist rechnerisch brauchbar, aber der Name sollte präzisiert werden.

**Kritischer Fehler für Roboterfreigabe:** Positionsfehler um 13–15 cm und Orientierungsfehler um etwa 40° sind ohne weitere Zieldefinition, Unsicherheit und geometrische Validierung nicht ausreichend für eine direkte Übergabepose. Das vorhergesagte Ziel ist außerdem die künftige Aria-Handgelenkpose, nicht eine validierte Greiferzielpose.

### 6.6 Empfangshand

Die besten Intentionscheckpoints von Residual v2 erreichen im Test im Mittel ungefähr:

- Accuracy: `0,9766`
- Macro-F1: `0,9633`

**Belegt:** Oracle- und End-to-End-Posewerte liegen häufig nahe beieinander, was zur hohen Handklassifikationsrate passt.

**Einschränkung:** Der Test enthält nur 57 linke gegenüber 214 rechten Handover-Fenstern; Validation enthält keine linken. Aufgrund stark überlappender Fenster ist dies kein Nachweis einer robusten Handwahl auf unabhängigen Handover-Ereignissen.

### 6.7 Belastbarkeit der Ergebnisse

Die vorhandenen Metriken belegen:

- Lernen der drei geskripteten Phasen auf drei gehaltenen Testpersonen,
- einen sinnvollen Architekturvergleich unter identischem Datensplit,
- eine hohe fensterweise Empfangshandklassifikation,
- einen begrenzten, seed-abhängigen Vorteil der Residual-Pose gegenüber absoluten Poseköpfen.

Sie belegen nicht:

- kalibrierte Wahrscheinlichkeiten,
- Ereignis- oder Onset-Latenz,
- Fehlalarme pro Minute/Stunde,
- stabile Live-Intentionsmetriken nach Glättung und Quality Gate,
- Generalisierung außerhalb des Erfassungsprotokolls,
- korrekte Objektwahl für Fetch,
- sichere oder ausreichend genaue Roboterausführung.

---

## 7. Konsistenz zwischen Training, Replay und Live-Inferenz

### 7.1 Artefaktladen

`Training/replay_stream_inference.py::load_artifacts` und `Training/online_inference.py::OnlineInferenceEngine` prüfen Modelltyp, Feature-Reihenfolge, Normalizer, Fenstergröße und Checkpointmetadaten. Der Livepfad verwendet für Seed 44 den besten Intentionscheckpoint für Intention und den getrennten besten Posecheckpoint für Empfangshand/Pose.

**Bestätigt:** Der aktuelle Replay-/Online-Deploymentpfad ist bewusst auf `hierarchical_residual_pose_transformer_v2` beschränkt. MLP, GRU und Transformer v1 sind Vergleichsmodelle und können nicht versehentlich als vollständiges Residual-Live-Artefakt geladen werden.

**Bestätigt korrekt:** Die geladenen Modelle erhalten dieselbe Form `[1, 60, 184]` wie im Training.

**Bestätigt korrekt:** Replay und Online-Inferenz verwenden die im Artefakt gespeicherten Mittelwerte, Standardabweichungen und Feature-Reihenfolge, nicht neu berechnete Live-Werte.

**Bestätigt korrekt:** Bei einer Zeitlücke über `0.2 s` werden Onlinefenster und Entscheidungsfilter zurückgesetzt. Die erste Vorhersage entsteht nach 60 Frames, danach standardmäßig alle zehn neuen Frames.

### 7.2 Paritätsmatrix

| Aspekt | Training | Replay | Aria Live | Bewertung |
|---|---|---|---|---|
| Featureliste/-reihenfolge | Artefakt | exakt geprüft | über Engine exakt geprüft | **Bestätigt korrekt** |
| Normalisierung/Missing-Maske | Train-Normalizer | identisch | identisch | **Bestätigt korrekt** |
| Fenster/Stride | 60/10 | 60/10 | 60/10 | **Bestätigt korrekt** |
| Zeitdauer des Fensters | native Gaze-Rate | native CSV-Rate | native Streamrate | **Verbesserungsbedarf** |
| Sensorzuordnung | nearest, ± Toleranz | bereits im Master | nur Vergangenheit, andere Toleranzen | **Verbesserungsbedarf** |
| statischer Tag-0-Anker | gesamte Sequenz | bereits im Master | laufend aus bisherigen Samples | **Verbesserungsbedarf** |
| fehlende Handkonfidenz | typischerweise NaN | aus Master | teilweise `-1` und beobachtet | **Verbesserungsbedarf** |
| Raw-Entscheidung | hierarchischer Argmax | hierarchischer Argmax | hierarchischer Argmax | **Bestätigt korrekt** |
| Stable-Kandidat | nicht vorhanden | Argmax gemeinsamer Klassenwahrscheinlichkeit | gleich wie Replay | **Plausibel, aber abweichend von Raw** |
| Input Quality Gate | nicht angewandt | nicht angewandt | angewandt | **Noch nicht offline evaluiert** |
| Ziel-/Workflowlogik | nicht vorhanden | nicht vorhanden | Live-only | **Noch nicht gegen Labels evaluiert** |

### 7.3 Gemeinsame Klassenwahrscheinlichkeiten und Hierarchie

Replay/Online bilden:

```text
P(continue) = P(no assistance)
P(fetch)    = P(assistance) * P(fetch | assistance)
P(handover) = P(assistance) * P(handover | assistance)
```

Die Raw-Klasse wird wie beim Training hierarchisch bestimmt. Der Glättungsfilter wählt dagegen den Argmax der gemittelten gemeinsamen Drei-Klassen-Wahrscheinlichkeiten.

**Verbesserungsbedarf:** Beide Regeln können mathematisch unterschiedliche Klassen liefern. Beispiel: `P(assistance)=0,51` und `P(fetch|assistance)=0,51` ergibt hierarchisch `fetch`, aber gemeinsam `continue` (`0,49` gegenüber `0,2601`). Eine einheitlich definierte Entscheidungsregel sollte für Offline-, Raw- und Stable-Metriken verwendet und dokumentiert werden.

### 7.4 TemporalDecisionFilter

Standardwerte:

- Glättungsfenster: 3 Vorhersagen
- Mindestkonfidenz: 0,65
- Mindestzahl stabiler Vorhersagen: 2

**Tatsächlicher Fehler – Verbesserungsbedarf:** `TemporalDecisionFilter.update` erhöht `candidate_count`, auch wenn die geglättete Konfidenz noch unter 0,65 liegt. Zwei gleichartige, aber unsichere Kandidaten können deshalb bewirken, dass bereits die erste später ausreichend sichere Vorhersage als stabil gilt. Die Implementierung fordert zwei wiederholte Argmax-Kandidaten, nicht zwei wiederholte **konfidente** Kandidaten. Die vorhandenen Smoke Tests decken diesen Randfall nicht ab.

### 7.5 Live-Sensorfusion und Staleness

Die Default-Toleranzen unterscheiden sich:

| Modalität | Masterbau | Aria Live |
|---|---:|---:|
| Hand | nearest ±12 ms | letzte vergangene Probe, max. 50 ms |
| VIO | nearest ±5 ms | letzte vergangene Probe, max. 10 ms |
| Marker | nearest ±20 ms | kausales Carry-Forward, max. 500 ms |

**Verbesserungsbedarf:** Ein Objektmarker kann live bis zu 500 ms alt sein und dennoch als gültig in den Vektor eingehen; ein explizites Alter wird dem Modell nicht übergeben. Bei bewegten Objekten ist dies besonders relevant.

**Verbesserungsbedarf:** Sobald der Live-Anker die Mindestzahl von acht Samples erreicht, wird `robot_frame_valid=1` gesetzt. Es existiert kein maximales Alter der letzten Tag-0-Beobachtung und kein expliziter Ankerunsicherheitswert. Ein verschwundener Tag, VIO-Drift oder eine nachträgliche Verschiebung des Referenzmarkers invalidiert die Koordinaten nicht automatisch.

**Verbesserungsbedarf:** Der Live-Anker wird fortlaufend aus bis zu 300 Samples aktualisiert. Bereits gepufferte Featureframes wurden möglicherweise mit einem älteren Anker berechnet. Während der Konvergenz kann ein 60er-Fenster daher leicht unterschiedliche Tag-0-Koordinatensystemschätzungen mischen.

Die aktuelle `InputQualityGate` fordert:

- ein vollständiges 60-Frame-Qualitätsfenster,
- mindestens 80 % gültige Gaze,
- keine zusammenhängende Gazelücke über 500 ms,
- 100 % gültige Tag-0-Referenz,
- für `handover` mindestens 50 % Abdeckung der besser beobachteten Hand und aktuell wenigstens eine gültige Hand.

**Verbesserungsbedarf:** Die Handover-Prüfung verwendet die besser beobachtete der beiden Hände, nicht zwingend die anschließend vom Posecheckpoint vorhergesagte Hand. Deshalb kann eine Handover-Entscheidung die Qualitätsprüfung passieren, die Poseausgabe aber wegen fehlender Referenz der vorhergesagten Seite ausfallen. VIO-Quality, Objektalter und Alter/Unsicherheit des statischen Ankers sind ebenfalls keine Gate-Kriterien.

### 7.6 Vorhandene Live-Logs

Die vorhandenen Dateien `Training/Outputs/live_features_debug.jsonl` und `Training/Outputs/aria_live_with_object.jsonl` enthalten 334 Featureframes und 28 Vorhersagen. Ihre Ausgabestruktur enthält noch nicht die aktuellen Felder `decision_intention`, `input_quality_ok` und `input_quality`; sie belegt daher einen älteren Lauf vor oder ohne die aktuelle Quality-Gate-Protokollierung.

Vergleich der beobachteten Validitätsraten mit den im finalen Normalizer gespeicherten Trainingsmitteln:

| Kanal | Trainingsmittel | Live-Log |
|---|---:|---:|
| `gaze_valid` | 0,968 | 0,392 |
| linke Hand gültig | 0,939 | 0,982 |
| rechte Hand gültig | 0,947 | 0,775 |
| `apriltag_0_valid` | 0,632 | 0,000 |
| `robot_frame_valid` | 0,993 | 1,000 |
| `robot_anchor_interpolated` | 0,364 | 1,000 |
| ArUco 6 gültig | 0,465 | 1,000 |
| ArUco 14 gültig | 0,662 | 0,985 |
| ArUco 7–13 gültig | ungefähr 0,60–0,74 | 0,000 |

**Verbesserungsbedarf:** Dies ist eine deutliche Live-/Train-Verteilungsverschiebung. Besonders auffällig ist, dass Tag 0 nie frame-synchron gültig war, das abgeleitete Referenzsystem aber durchgehend als gültig markiert wurde.

Ein read-only diagnostischer Gegenversuch auf diesem einzelnen Log ergab:

- Original: 7 Raw-`continue`, 21 Raw-`handover`
- Gaze künstlich vollständig ungültig: 28 Raw-`handover`
- Handkanäle künstlich ungültig: in diesem kurzen Lauf keine Änderung der Raw-Verteilung

**Plausible technische Schlussfolgerung:** Das Modell reagiert in diesem Lauf stark auf Gaze-Ausfall; fehlende Gaze liegt weit außerhalb des üblichen Trainingsmusters. Dies passt zu den beobachteten Fehlvorhersagen bei geschlossenen Augen.

**Keine wissenschaftliche Kausalaussage:** Es handelt sich nur um einen Gegenversuch auf einem kurzen, nicht kontrolliert gelabelten Lauf. Daraus folgt nicht, dass Hände oder Objektmarker generell irrelevant sind.

Wendet man die aktuelle `InputQualityGate`-Logik nachträglich auf die alten Frames an, würden nur die ersten sieben der 28 Vorhersagezeitpunkte passieren; die folgenden 21 würden wegen zu geringer Gazeabdeckung beziehungsweise zu langer Gazelücke als unzureichende Eingabe blockiert.

**Plausibel, aber nicht vollständig validiert:** Eine Gaze-Anforderung ist als konservative OOD-/Qualitätsregel fachlich sinnvoll, wenn die Ausgabe ausdrücklich `insufficient_input` lautet. Sie wäre ein Workaround, wenn sie als Beweis interpretiert würde, dass jede reale Intention zwingend sichtbare Gaze braucht. Das Modell wurde mit sehr hoher Gazeverfügbarkeit trainiert; daher sollte es bei stark fehlender Gaze abstainieren, bis Dropout-Robustheit gezielt trainiert und evaluiert wurde.

### 7.7 Replay

Der strikte Replaypfad lehnt den lokal vorhandenen älteren `Jona_6`-Master wegen fehlender aktueller Robot-Frame-Felder ab. Das ist korrekt und verhindert eine stillschweigende Verwendung eines inkompatiblen Schemas.

Mit dem diagnostischen Schalter für fehlende Features ließ sich die Mechanik testen, das Ergebnis ist wegen des alten Schemas aber keine gültige Modellevaluation. Die vorhandene Datei `Training/Outputs/Jona_6_replay_predictions.csv` zeigt auf nur einer Sequenz gute Phasewerte, ist jedoch weder ein unabhängiger Test noch ein Ersatz für den finalen Testsplit.

**Verbesserungsbedarf:** Replay wertet Raw-Accuracy aus, aber nicht den Stable-Intent, das Quality Gate, die Zielselektion oder den Workflow. Damit ist Replay derzeit kein vollständiger Emulator des realen Live-Entscheidungspfads.

---

## 8. Latenzanalyse

### 8.1 Gemessene und aus Code ableitbare Komponenten

| Komponente | Beobachtung | Einordnung |
|---|---|---|
| Anker-Warm-up | mindestens 8 passende Tag-0-Samples | einmalig; Dauer abhängig von RGB/Erkennung |
| Fenster-Warm-up | 60 gültige Modellframes | einmalig |
| Gemessene erste Fensterfüllung | ca. `2,37 s` im vorhandenen Debuglauf | einmalig |
| Vorhersageintervall | Median ca. `366,7 ms`, Mittel ca. `400 ms`; min. `333`, max. `633 ms` | laufende Quantisierung durch Stride 10 und Streamrate |
| Intentionsinferenz | Median ca. `3,8 ms`, 95. Perzentil ungefähr `9 ms`; erster Call ca. `118 ms` | Modelllaufzeit, erster Call mit Warm-up |
| Glättung | Mittel über 3 Vorhersagen | absichtlich |
| Stabilität | 2 gleiche Kandidaten | absichtlich, mit beschriebenem Zähler-Randfall |
| Beobachteter Raw-zu-Stable-Wechsel | ungefähr `0,7 s` | absichtliche Entscheidungsverzögerung plus Vorhersageintervall |
| Workflowbestätigung | nochmals 2 bestätigende Updates | zusätzliche Live-Zustandsverzögerung |
| Fetch-Zielfixation | standardmäßig `1,0 s` | absichtlich für Objektwahl; kann parallel zur Intentionsbildung laufen |

Im vorhandenen Live-Log beträgt der Medianabstand zwischen Featureframes etwa `33,3 ms`; Mittel und Maximum liegen bei etwa `39,9 ms` beziehungsweise `133,3 ms`. Das konkrete 60er-Fenster umfasst ungefähr `2,37 s`, nicht exakt 60 mal eine feste Sollperiode.

### 8.2 Was normal und was vermeidbar ist

**Ja, eine gewisse Verzögerung ist mit der aktuellen Architektur normal.**

Unvermeidbar sind:

- physische Sensorbelichtung und Geräteübertragung,
- mindestens eine kleine Verarbeitungslatenz für Dekodierung, Synchronisation, Features und Inferenz,
- das Warten auf erste ausreichende Evidenz nach Beginn einer Handlung.

Durch die aktuelle Konfiguration absichtlich oder strukturell erzeugt werden:

- einmalig das Füllen des 60er-Fensters,
- eine neue Modellentscheidung nur alle zehn Featureframes,
- Glättung über drei Vorhersagen,
- Stabilitätsanforderung über zwei Kandidaten,
- gegebenenfalls zwei Workflowbestätigungen und eine Sekunde Zielfixation.

Reduzierbar sind vor allem:

- der Stride von 10,
- die Länge des Glättungsfensters und die Anzahl bestätigender Schritte,
- erster Torch-Forward-Warm-up,
- blockierende beziehungsweise flushende Debugausgabe,
- ineffiziente Decoder-/Markerverarbeitung,
- gegebenenfalls die Fensterlänge – allerdings nur nach erneutem Training oder einer ausdrücklich getesteten Architekturvariante.

**Wichtig:** Ein bereits gefülltes trailing window verursacht im laufenden Zustand nicht automatisch weitere zwei Sekunden Verzögerung. Es liefert Kontext bis zum aktuellen Frame. Die laufende Reaktionszeit wird vor allem durch das Entstehen klassifizierbarer Bewegung, den 10-Frame-Stride und die Entscheidungsstabilisierung bestimmt.

### 8.3 Nicht messbare Komponenten

**Nicht anhand der vorhandenen Informationen bestimmbar:**

- Capture-to-Host-Empfangslatenz pro Modalität,
- Geräte- zu Host-Uhrversatz,
- RGB-Decoderzeit,
- Callback-Warte- und Queue-Zeit,
- Marker-Detektionsdauer,
- genaue Dauer der Featureassemblierung,
- Print-/Datei-I/O-Anteil,
- tatsächliche Event-Onset-zu-Raw- und Event-Onset-zu-Stable-Latenz.

Die vorhandenen Logs enthalten Gerätesensorzeit und Modelllaufzeit, aber keine vollständige Kette synchronisierter Host-Monotonic-Zeitstempel.

### 8.4 Empfohlene spätere Instrumentierung

Für jeden Sensor und jede Vorhersage sollten ohne Roboterintegration folgende Zeitpunkte protokolliert werden:

```text
device capture timestamp
        -> host callback receive
        -> decode begin/end
        -> marker detection end
        -> selected sensor sample timestamps + ages
        -> feature frame assembled/enqueued
        -> inference begin/end
        -> raw prediction
        -> smoothed candidate
        -> stable decision
        -> quality/workflow result
        -> output/log completion
```

Zusätzlich sind ein Geräte-/Host-Uhrmodell, Run-ID, Git-Commit, Artefakt-Hashes und ein eindeutiger Featureframe-Identifier nötig. Danach sollten Median, 95. und 99. Perzentil für jede Stufe sowie Event-Onset-zu-Decision berichtet werden.

### 8.5 Realistisches Ziel

„Keine Verzögerung“ ist nicht realistisch und wäre bei menschlichen, zeitlich entstehenden Absichten auch semantisch nicht eindeutig. Sinnvoll ist ein messbares Ziel, zum Beispiel:

- definierter maximaler p95-Wert von gelabeltem Intentions-Onset bis Stable Decision,
- getrennte Messung für `fetch` und `handover`,
- gleichzeitig begrenzte Fehlalarmrate in neutralen/OOD-Phasen,
- dokumentierter Trade-off zwischen schneller Reaktion und stabiler Ausgabe.

Eine bloße Reduktion von Glättung und Stabilität kann die Ausgabe schneller, aber deutlich sprunghafter und unsicherer machen.

---

## 9. Plausibilität für einen späteren Robotereinsatz

### 9.1 Semantik der Intentionen

**Plausibel, aber nicht vollständig validiert:** Die drei Zustände sind als Eingaben in eine spätere übergeordnete Zustandsmaschine geeignet:

- `continue`: keine neue Assistenzaktion,
- `fetch`: Nutzer befindet sich in der annotierten Fetch-/Fixationsphase,
- `handover`: Nutzer befindet sich in der annotierten Übergabephase.

Sie sind nicht eindeutig genug, um allein eine Roboteraktion auszulösen:

- `continue` sagt nicht, welche laufende Aktion fortgesetzt werden soll.
- `fetch` enthält keine durch das Modell validierte Objekt-ID, keine Greifpose und keine Freigabe.
- `handover` enthält eine geschätzte Hand und Wrist-Pose, aber keine validierte Greiferpose, Erreichbarkeit oder Nutzerbestätigung.

**Kritischer Fehler für direkte Ausführung:** Eine einzelne Klassenentscheidung darf nicht als Aktionskommando verwendet werden. Die aktuelle Liveimplementierung tut dies erfreulicherweise nicht; `external_action_requested` ist fest `False`.

### 9.2 Fetch-Ziel

`Training/live_decision.py::GazeTargetSelector` wählt außerhalb des neuronalen Modells anhand von Gazewinkel, Mindestabstand zum zweitbesten Kandidaten, zeitlicher Kontinuität, mindestens zehn Samples und etwa einer Sekunde Fixation einen ArUco-Marker.

**Plausibel, aber nicht vollständig validiert:** Dies kann eine Objektidentität für einen betreuten Demonstrator liefern.

**Verbesserungsbedarf:** Die Zielselektion wurde nicht gegen `target_object_id` auf einem gehaltenen Datensatz ausgewertet. Das Training selbst sagt keine Objektklasse vorher. Der Selector liefert außerdem keine validierte Greifpose oder Information darüber, ob das Objekt noch an dieser Position liegt.

### 9.3 Handover-Ziel

**Plausibel, aber nicht vollständig validiert:** Die Empfangshandklassifikation kann als ein Signal in einer späteren Übergabelogik dienen.

**Kritischer Fehler für direkte Ausführung:** Die Future-Wrist-Pose ist keine sicher definierte Endeffektorpose. Es fehlen unter anderem Greiferabstand und -orientierung relativ zur Hand, Nutzerbewegungsunsicherheit, dynamische Aktualisierung, Workspace-/Kollisionsprüfung und eine physische Tag-0-zu-Basis-Kalibrierung.

### 9.4 Erforderliche spätere Freigabeschichten

Zwischen Modell und realem Roboter wären mindestens erforderlich:

1. Datenqualitäts- und OOD-Abstention,
2. kalibrierte Unsicherheit und zeitliche Bestätigung,
3. semantische Zustandsmaschine mit zulässigen Übergängen,
4. explizite Nutzerfreigabe beziehungsweise Abbruchmöglichkeit,
5. frische und unsicherheitsbewertete Koordinatentransformation,
6. Objekt- und Zielposevalidierung,
7. Erreichbarkeits-, Workspace- und Kollisionsprüfung,
8. Geschwindigkeits-, Kraft- und Abstandslimits,
9. Watchdog, unabhängiger Stopppfad und sichere Rückfallzustände,
10. Protokollierung jeder Freigabeentscheidung.

Diese Punkte sind Anforderungen für eine spätere Phase und wurden in diesem Audit nicht implementiert.

### 9.5 Fehlende Tests vor Hardware

- kontrollierte, gelabelte Live-Shadow-Tests mit mehreren unbekannten Personen,
- lange neutrale Läufe und Fehlalarmrate pro Zeit,
- kontrollierte Gaze-, Hand-, Marker- und VIO-Ausfälle,
- freie statt ausschließlich geskripteter Handlungsketten,
- bewegte oder verdeckte Objekte,
- Target-Selector-Accuracy und Ambiguität,
- zeitliche Onset-/Stable-Metriken,
- physische Transformationskalibrierung mit statischen Prüfkörpern,
- Posefehler im später tatsächlich benötigten Basis-/Greiferkoordinatensystem,
- Simulation und danach Hardware-in-the-Loop mit blockierter Aktuation,
- erst anschließend begrenzte, überwachte Versuche mit unabhängiger Sicherheitsfreigabe.

---

## 10. Gefundene Fehler und Risiken

| ID | Einstufung | Befund | Evidenz/Ort |
|---|---|---|---|
| F-01 | **Kritischer Fehler** | Finaler Datensatz nicht inhaltsgenau reproduzierbar; ID-Fingerprint erkennt geänderte CSV-Inhalte nicht. | `Training/data.py::manifest_filtered_master_files`; finale `data_metadata.json`; aktuelles Manifest/Masterverzeichnis |
| F-02 | **Kritischer Fehler für Robotereinsatz** | `robot` ist Tag-0-Frame, nicht belegtes Panda-Basisframe. | `Code/build_master_dataset.py::estimate_static_world_robot`, `add_coordinate_transforms` |
| F-03 | **Kritischer Fehler für Robotereinsatz** | Posefehler und Zielsemantik reichen nicht für direkte Endeffektorsteuerung. | finale `metrics.json`; `Training/metrics.py::pose_metrics` |
| F-04 | **Verbesserungsbedarf** | Modell klassifiziert aktuellen Fensterendpunkt, keine zukünftige Intention. | `Training/data.py::WindowDataset.__getitem__` |
| F-05 | **Verbesserungsbedarf** | Offline-/Live-Sensorfusion und Toleranzen sind nicht identisch. | `Code/build_master_dataset.py::nearest_merge`; `Training/aria_live_inference.py::latest_before_item` |
| F-06 | **Verbesserungsbedarf** | Live-Anker bleibt ohne explizite Frische/Unsicherheit gültig und kann innerhalb eines Fensters driften. | `Training/aria_live_inference.py::LiveFeatureAssembler` |
| F-07 | **Verbesserungsbedarf** | Vorhandener Live-Lauf weist starke Gaze-/Marker-Verteilungsverschiebung auf. | `Training/Outputs/*.jsonl`; finaler Normalizer |
| F-08 | **Verbesserungsbedarf, tatsächlicher Logikfehler** | Stabilitätszähler zählt auch unkonfidente Kandidaten. | `Training/replay_stream_inference.py::TemporalDecisionFilter.update` |
| F-09 | **Verbesserungsbedarf** | Raw-Hierarchie und Stable-Joint-Argmax können unterschiedliche Klassenregeln anwenden. | `hierarchical_intention_id`, `joint_intention_probabilities`, `TemporalDecisionFilter` |
| F-10 | **Verbesserungsbedarf** | Quality Gate ist im aktuellen Code sinnvoll, aber in vorhandenen Logs nicht End-to-End belegt und im Replay nicht enthalten. | `Training/live_decision.py::InputQualityGate`; Logschema |
| F-11 | **Verbesserungsbedarf** | Markerwerte können live bis zu 500 ms alt und trotzdem gültig sein; Alter fehlt als Feature. | `Training/aria_live_inference.py` Default `--marker-tolerance-ms` |
| F-12 | **Verbesserungsbedarf** | Validationssplit enthält keine linke Empfangshand. | finale `data_metadata.json` |
| F-13 | **Verbesserungsbedarf** | Posecheckpoint wird nach Oracle-Position statt End-to-End-Einsatzmetrik gewählt. | `Training/train_residual.py` Checkpointauswahl |
| F-14 | **Verbesserungsbedarf** | Niedrige globale 5-%-Observation-Schwelle schützt nicht pro Modalität. | alle vier Konfigurationen; `WindowDataset` |
| F-15 | **Verbesserungsbedarf** | Starke Fensterüberlappung und nur drei Testpersonen begrenzen statistische Aussagekraft. | `window_size=60`, `stride=10`; Splitmetadaten |
| F-16 | **Verbesserungsbedarf** | Batch-Masterbau überspringt alte Dateien ohne Schema-/Builderprüfung. | `Code/build_master_dataset_batch.py` |
| F-17 | **Verbesserungsbedarf** | Featureprofil kann durch den ersten unvollständigen Header still verkleinert werden. | `Training/data.py::select_feature_columns`, `prepare_data` |
| F-18 | **Verbesserungsbedarf** | `position_mae_cm` ist euklidischer Distanzmittelwert, nicht klassischer MAE. | `Training/metrics.py::pose_metrics` |
| F-19 | **Nicht bestimmbar** | Vollständige Sensor-/Streaminglatenz fehlt mangels Host-/Device-Zeitkette. | bestehende Live-Logs |
| F-20 | **Nicht bestimmbar** | Richtigkeit aller finalen Annotationen, Markergrößen und externen Kalibrierungen kann lokal nicht erneut geprüft werden. | fehlender finaler Master-Snapshot/externe Messwerte |
| F-21 | **Verbesserungsbedarf** | Last-Observation-Baseline verwendet die Ground-Truth-Hand und ist keine reine End-to-End-Baseline. | `Training/train_residual.py::run_epoch`, `last_observation_oracle` |
| F-22 | **Verbesserungsbedarf** | Paketversionen sind im Clusterrezept gepinnt, aber Containerdigest und Runtimeversionen fehlen in den einzelnen Runartefakten. | `singularity/aria.recipe`, `Training/requirements.txt`, finale Runverzeichnisse |
| F-23 | **Verbesserungsbedarf** | Das Diagrammskript findet verschachtelte finale Runs, das Vergleichsskript mit Defaultpfad dagegen nicht. | `Training/evaluation/generate_training_diagrams.py::discover_runs`, `Training/compare_final_runs.py::main` |
| F-24 | **Verbesserungsbedarf** | Handover-Quality prüft die besser beobachtete Hand statt der vorhergesagten Hand; Ankerfrische und VIO-Quality fehlen. | `Training/live_decision.py::InputQualityGate`, `Training/online_inference.py` |

---

## 11. Priorisierte Verbesserungsvorschläge

### P0 – vor weiterer belastbarer Demoaussage oder jeder Robotervorbereitung

| Maßnahme | Abnahmekriterium |
|---|---|
| **Finalen Datensatz einfrieren und reproduzierbar machen.** Manifest, manuelle Annotationen, Builderversion, vollständige Schema-ID und SHA-256 jedes Master-CSV gemeinsam versionieren oder als unveränderliches Dataset-Manifest speichern. | Ein sauberer Neubau reproduziert die erwarteten 156 Sequenzen, Splits, Features und Fenster; jede Inhaltsänderung ändert den Dataset-Fingerprint. |
| **Aktuellen Livepfad erneut als Inference-only testen.** Ein kontrollierter Lauf muss die neuen Felder für Modell-, Quality- und Decision-Intent loggen. | Gazeausfall führt nachvollziehbar zu `insufficient_input`, nicht zu `handover`; keine externe Aktion; klare Phasenzeitpunkte vorhanden. |
| **Live-/Offline-Featuresemantik angleichen oder explizit neu trainieren.** Kausale Synchronisation, Sensoralter, feste/erfasste Zeitbasis, Hand-Missing-Semantik und Ankerfrische müssen definiert werden. | Dieselbe aufgezeichnete Sequenz ergibt aus kausalem Offline-Featurebau und Live-Replay innerhalb definierter Toleranzen dieselben Featureframes und Vorhersagen. |
| **Vor jedem späteren Robotertest Tag-0 und reale Basis sauber trennen.** Noch keine Steuerung implementieren; zunächst Kalibrier- und Unsicherheitsanforderung spezifizieren. | Nachweisbare Transformation mit Fehlerbudget; veralteter/verdeckter Anker führt zur Abstention. |

### P1 – als nächste technische Arbeit

| Maßnahme | Ziel |
|---|---|
| Stabilitätsfilter korrigieren und Raw-/Stable-Klassenregel vereinheitlichen. | `minimum_stable_predictions` zählt nur ausreichend konfidente, aufeinanderfolgende Kandidaten; gleiche Semantik offline/live. |
| Replay um Quality Gate, Stable- und Workflow-Metriken erweitern. | Raw-, Stable-, Blocked- und Event-Metriken auf denselben Daten vergleichen. |
| Latenzinstrumentierung ergänzen und vorwärmen. | Capture-to-Raw/Stable Median und p95 statt subjektiver Verzögerung; erster Forward vor Demo vorgewärmt. |
| Participant-Folds oder Group-Cross-Validation verwenden. | Stabilität gegenüber Teilnehmerwahl; Testset nicht wiederholt zur Architekturentscheidung verwenden. |
| Ereignisbasierte Evaluation ergänzen. | Onset-Latenz, Fehlalarme pro neutraler Minute, Mindestdauer, Klassenwechsel und Calibration/ECE. |
| Links/rechts in Validation und Test ausgewogen abdecken. | Beide Hände werden auf unabhängigen Handover-Ereignissen validiert. |
| Poseziel und Baseline neu definieren. | Residuum muss robuste Verbesserung gegenüber Last Observation liefern; Handgelenk- und spätere Greiferzielsemantik klar trennen. |
| Target Selector gegen Objektlabels prüfen. | Accuracy, Ambiguität, Zeit bis Fixation und Verhalten bei keinem/mehreren Zielen. |
| Modality-Dropout/OOD-Training und kontrollierte Ablationen. | Fehlende Gaze/Hand/Marker führen zu kalibrierter Unsicherheit statt überkonfidenter Ersatzklasse. |

### P2 – sinnvolle spätere Verbesserung

- Kanonisches Schema unabhängig vom ersten Masterheader erzwingen.
- Alte Masterdateien bei Builder-/Schemamismatch automatisch als veraltet markieren.
- Dataset-, Git-, Container- und Checkpoint-Hashes in jedes Artefakt schreiben.
- Participant-/Sequenz-balanciertes Sampling untersuchen.
- Feste zeitliche Resamplingrate oder Delta-Time-/Sensor-Age-Features vergleichen.
- Metriknamen präzisieren und Konfidenzintervalle auf Ereignis-/Teilnehmerebene ergänzen.
- Export- und Vergleichsskripte für alle vier Modellfamilien vereinheitlichen.
- `compare_final_runs.py` rekursiv suchen lassen oder den kanonischen Runs-Root eindeutig konfigurieren.
- Historische Läufe klar von finalen Läufen trennen und generierte `__pycache__`-/`.DS_Store`-Dateien aus Auswertungsverzeichnissen ausschließen.

---

## 12. Empfohlene nächste Schritte

1. **Datensatzstand sichern:** Zuerst den exakten Cluster-Datenstand wiederherstellen oder neu und deterministisch bauen. Danach Inhalts-Hashes, Manifest und Annotationen als einen Audit-Snapshot speichern.
2. **Aktuellen Live-Code kontrolliert validieren:** Mehrere kurze, gelabelte Inference-only-Läufe mit neutral, fetch, handover sowie gezieltem Gaze-/Hand-/Marker-Ausfall durchführen. Dabei die aktuelle Quality-Ausgabe mitschreiben.
3. **Offline-/Live-Paritätstest bauen:** Eine aufgezeichnete Sensorsequenz einmal durch den kausalen Featureassembler und einmal durch den Replaypfad schicken; pro Feature und Zeitstempel vergleichen.
4. **Entscheidungslogik bereinigen:** Konfidenz-/Stabilitätszähler und einheitliche Hierarchie definieren; anschließend Stable- statt nur Raw-Metriken berichten.
5. **Latenz messbar machen:** Vollständige Zeitstempelkette instrumentieren und erst danach Stride, Glättung und Bestätigungsdauer anhand des Speed/False-Trigger-Trade-offs optimieren.
6. **Evaluation verbreitern:** Participant-Folds, freie negative Szenarien, Ereignismetriken, Kalibrierung und robuste Hand-/Objektabdeckung.
7. **Pose separat weiterentwickeln:** Last-Observation als verpflichtende Baseline beibehalten; nur ein Modell übernehmen, das über Seeds und Ereignisse stabil besser ist. Das spätere Roboterziel muss getrennt von der Wrist-Prognose spezifiziert werden.
8. **Erst nach Betreuerfreigabe Robotervorbereitung planen:** Kalibrierung, Sicherheitsarchitektur, Simulation und Hardware-in-the-Loop als eigene Phase; keine direkte Verbindung des heutigen Klassifikators zu Aktuation.

---

## 13. Abschließendes Urteil

### Ist die Datenpipeline technisch konsistent?

**Plausibel, aber nicht vollständig validiert.** Der Code bildet eine nachvollziehbare Kette von Annotation, Sensorfusion, Tag-0-Transformation, Future-Target, Manifestfilter, Normalisierung und Fensterung. Labels, Klassen-IDs und Quaternionreihenfolge sind konsistent. Der konkrete finale Datenstand ist jedoch wegen fehlender Inhaltsprovenienz nicht aus dem aktuellen Repository reproduzierbar.

### Ist das Training korrekt implementiert?

**Bestätigt korrekt für die definierte Aufgabe, mit Verbesserungsbedarf.** Forward Pass, bedingte Loss-Masken, Klassengewichte, AdamW, Gradient Clipping, Early Stopping und getrennte Checkpoints sind technisch stimmig. Die definierte Aufgabe ist aktuelle Phasenklassifikation plus zukünftige Wrist-Pose, nicht zukünftige Intentionsprognose. Validationsabdeckung, Zeitrepräsentation und Posecheckpoint-Kriterium sollten verbessert werden.

### Sind die aktuellen Evaluationsergebnisse belastbar?

**Plausibel für den internen Modellvergleich, nicht belastbar für allgemeine oder sicherheitsbezogene Aussagen.** Die zwölf finalen Läufe sind untereinander sauber vergleichbar. Drei Seeds und drei feste Testpersonen reichen für einen ersten Architekturvergleich, aber nicht für robuste Generalisierung, statistische Sicherheit oder Robotereignung. Überlappende Fenster erhöhen den nominellen Support.

### Funktioniert die Live-Inferenz entsprechend der Trainingslogik?

**Teilweise.** Feature-Reihenfolge, Normalizer, Modellformen, Raw-Hierarchie und Fenstermechanik stimmen. Sensorzeitfusion, Missing-Semantik, Framerate und Ankerbildung sind nicht exakt gleich. Stable-Entscheidung und Quality-/Workflowlogik existieren nur live beziehungsweise im Replay teilweise und sind noch nicht gegen den finalen Testbestand ausgewertet.

### Ist die beobachtete Latenz erklärbar?

**Ja, größtenteils.** Warm-up, 10-Frame-Vorhersageintervall und Stabilisierung erklären Sekunden beim Start und mehrere hundert Millisekunden im laufenden Wechsel. Die Modellinferenz selbst ist mit wenigen Millisekunden nicht der Hauptfaktor. Die vollständige Sensor-to-Decision-Latenz bleibt ohne zusätzliche Zeitstempel unbestimmbar.

### Ist das System für eine betreute Live-Demonstration ohne Roboter plausibel?

**Ja, bedingt.** Als klar gekennzeichnete Inference-only-/Shadow-Demo ist es plausibel, sofern der aktuelle Quality-Gate-Pfad vorher mit kontrollierten Ausfällen erneut geprüft wird, Unsicherheit sichtbar bleibt und keine robuste allgemeine Intentionserkennung behauptet wird.

### Ist es bereits für eine reale Roboteransteuerung geeignet?

**Nein.** Es fehlen belastbare Datenprovenienz, Live-Parität, kalibrierte Unsicherheit, Ereignis-/Fehlalarmmetriken, validierte Objektwahl, genaue Pose, reales Basisframe, Frische-/Transformationsunsicherheit und die komplette unabhängige Sicherheits- und Freigabeschicht.

### Was fehlt bis zu einem verantwortbaren ersten Robotertest?

Vor einem ersten begrenzten Robotertest müssen mindestens die P0-Punkte abgeschlossen sein, gefolgt von kontrollierter Live-Shadow-Evaluation, vollständiger Latenzmessung, einer nachweisbaren Koordinatenkalibrierung, validierter Objekt-/Zielpose, Simulation beziehungsweise Hardware-in-the-Loop ohne Aktuation und einer unabhängigen Sicherheitsfreigabe. Die heutige Modellentscheidung darf dabei nur ein Eingangssignal einer späteren Zustands- und Sicherheitslogik sein, niemals der direkte Aktionsauslöser.
