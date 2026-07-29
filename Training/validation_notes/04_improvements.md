# Verbesserungsprotokoll zum technischen Audit

**Arbeitsabschnitt:** ohne Hardware und ohne neues Training umsetzbare P0-/P1- sowie zugehörige P2-Maßnahmen

**Auditgrundlage:** `Training/project_validation.md`

**Umfang:** Replay-/Online-/Live-Entscheidung, Quality Gate, Diagnostik, Run-Werkzeuge und Provenienz zukünftiger Runs; keine Änderung vorhandener Trainingsdaten oder Checkpoints und keine Roboterintegration

| Audit-ID | Änderung | Dateien | Tests | Status | verbleibendes Risiko |
|---|---|---|---|---|---|
| F-08 | Unkonfidente geglättete Kandidaten setzen Kandidat und Stabilitätszähler auf null. Nur unmittelbar aufeinanderfolgende Kandidaten ab `minimum_confidence` zählen für `minimum_stable_predictions`. | `Training/replay_stream_inference.py`, `Training/inference_decision_smoke_test.py` | Konfidenzreset, unterbrochene Kandidatenserie, erneuter Aufbau bis zur stabilen Ausgabe | Umgesetzt und getestet | Glättungsfenster, Schwelle und Mindestanzahl sind weiterhin heuristische, noch nicht auf Event-Latenz und Fehlalarmrate kalibrierte Werte. |
| F-09 | `raw_intention` und `stable_intention` verwenden denselben Argmax über die gemeinsame Drei-Klassen-Verteilung. Die vorherige hierarchische Entscheidung bleibt als Diagnosefeld `hierarchical_raw_intention` erhalten. | `Training/replay_stream_inference.py`, `Training/online_inference.py`, `Training/inference_decision_smoke_test.py` | Hierarchie-/Joint-Gegenbeispiel, Raw-/Stable-Parität, Online-Engine-Dummy, echtes Online-Artefakt, kurzer Replaylauf | Umgesetzt und getestet | Historische Trainings-/Testmetriken und bestehende Logdateien verwenden weiterhin die damalige hierarchische Entscheidung. Die metrische Auswirkung der neuen Deploymentregel muss später auf dem vollständigen finalen Testbestand ausgewertet werden. |
| F-10 | Replay und Aria Live verwenden dieselbe `InputQualityGate`- und Actionability-Funktion. Replay führt das Gate über jeden kausalen Quellframe und gibt Raw, Stable, Quality und Actionable getrennt aus. | `Training/live_decision.py`, `Training/replay_stream_inference.py`, `Training/aria_live_inference.py`, beide Entscheidungs-Smoke-Tests | Direkte Replay-/Live-Gate-Parität, Freigabe bei guter Qualität, Gazeausfall, Modulimport und kurzer echter Replaylauf | Umgesetzt und getestet | Schwellen sind konservative Defaults und noch nicht ereignisbasiert kalibriert. |
| F-24 | Bei stabilem `handover` werden Abdeckung, aktuelle Gültigkeit und Alter ausschließlich für `predicted_receiving_hand` geprüft. Eine gut beobachtete andere Hand ersetzt sie nicht; zusätzlich werden VIO und Ankerfrische geprüft. | `Training/live_decision.py`, `Training/aria_live_inference.py`, `Training/replay_stream_inference.py`, `Training/live_decision_smoke_test.py` | Links/rechts jeweils gültig und jeweils durch die andere Hand nicht kompensierbar; fehlende/alte Hand; Continue/Fetch ohne Handzwang; VIO-/Ankeralter | Umgesetzt und getestet | Quality-Schwellen und Ankerunsicherheit sind noch nicht empirisch kalibriert. |
| F-05 | Fehlende Live-Handkonfidenz ist nun wie offline `NaN`/unbeobachtet. Replay rekonstruiert Sensoralter kausal; vom Offline-Nearest-Merge gewählte Zukunftssamples werden nicht als vergangene Livewerte ausgegeben. | `Training/aria_live_inference.py`, `Training/replay_stream_inference.py`, `Training/inference_decision_smoke_test.py` | Fehlende linke/rechte Hand, `NaN`-Semantik, kausaler Lookup, positive/negative Merge-Offsets, Fensterreset bei Zeitlücke | Teilweise umgesetzt und getestet | Der bestehende Offline-Masterbau bleibt beidseitiger Nearest-Neighbor. Exakte Featureparität erfordert einen neuen kausalen Dataset-Build und anschließend neues Training; dies wurde nicht ausgeführt. |
| F-06 | Der AprilTag-0-Anker wird nach den erforderlichen Startsamples eingefroren und kann damit nicht innerhalb des Modellfensters wandern. Tatsächliche Folgesichtungen aktualisieren sein Alter; ein alter/unbestimmbarer Anker blockiert Actionable. | `Training/aria_live_inference.py`, `Training/live_decision.py`, beide Entscheidungs-Smoke-Tests | eingefrorener Anker, frischer/veralteter/unbestimmbarer Anker; Alter über vollständiges Qualitätsfenster | Umgesetzt und getestet | Der Schätzfehler und langfristige VIO-Drift sind nicht quantifiziert; Tag-0 ist weiterhin kein kalibriertes Panda-Basisframe. |
| F-11 | Objektmarker erhalten Altersdiagnostik. Fetch wird bei aktuell sichtbaren, aber veralteten Markern blockiert; der Gaze-Target-Selector ignoriert alte Marker. | `Training/aria_live_inference.py`, `Training/live_decision.py`, `Training/live_decision_smoke_test.py` | frischer/veralteter Marker, Fetch-Blockade, Continue ohne unnötigen Markerzwang, Selector-Status `stale_visible_objects` | Umgesetzt und getestet | Die Defaultgrenze von `250 ms` ist konservativ und noch nicht empirisch kalibriert; das neuronale Modell kann innerhalb seiner Eingabe weiterhin den bis zur Streamtoleranz getragenen Markerwert sehen. |
| F-19 | Live-Ausgaben enthalten monotone Host-Zeitstempel für Empfang, RGB-/Markerverarbeitung, Featureaufbau, Inferenz, Raw, Stable, Quality, Workflow und Ausgabe. Beide Modelle werden beim Start einmal vorgewärmt. | `Training/aria_live_inference.py`, `Training/online_inference.py`, `Training/replay_stream_inference.py`, `Training/inference_decision_smoke_test.py` | Vorhandensein der Phasenzeitstempel, genau ein Warm-up pro Modell, Replay-Latenzstatistik | Umgesetzt und getestet | Device- und Hostuhr sind nicht automatisch synchronisiert; echte Capture-to-Host-Latenz benötigt weiterhin eine bekannte Uhrabbildung und gelabelte Event-Onsets. |
| F-10 (Auswertung) | Replay berichtet Raw-, Stable- und Actionable-Coverage, bedingte sowie End-to-End-Accuracy, Konfusionen, Quality-Blockaden und Ablehnungsgründe; optional als JSON. | `Training/replay_stream_inference.py`, `Training/inference_decision_smoke_test.py` | künstlicher Raw-/Stable-/Actionable-Fall und kurzer echter Replaylauf | Umgesetzt und getestet | Ereignisbasierte Onset-Latenz, Fehlalarme pro Minute und Kalibrierungsmetriken fehlen weiterhin. |
| F-23 | Eine gemeinsame rekursive Run-Suche findet direkte und unter `run_cluster` verschachtelte Runs; Mehrdeutigkeiten werden abgelehnt. Vergleich und Export verwenden dieselbe Logik. | `Training/run_discovery.py`, `Training/compare_final_runs.py`, `Training/export_checkpoint_predictions.py`, `.sbatch`, `Training/run_discovery_smoke_test.py` | direkte/verschachtelte Runs, unvollständige Runs, eindeutige Auflösung, Namenskollision | Umgesetzt und getestet | Gleiche Run-Basisnamen an mehreren Stellen erfordern bewusst einen expliziten Pfad. |
| F-18 | Neue Metriken verwenden additiv `position_mean_euclidean_error_cm`, `position_root_mean_square_euclidean_error_cm` und eine Definition. `position_mae_cm`/`position_rmse_cm` bleiben als ausdrücklich gekennzeichnete Legacy-Aliase erhalten. | `Training/metrics.py`, Trainings-/Vergleichs-/Export-/Baseline-Ausgaben, Smoke Tests | Identität von neuem Schlüssel und Legacy-Alias, bestehende Referenzprüfung | Umgesetzt und getestet | Historische Artefakte tragen nur die alten Schlüssel; deren Zahlenwerte wurden nicht umgedeutet oder verändert. |
| F-01/F-22 | Zukünftige Runs speichern SHA-256 je ausgewähltem Master-CSV, Inhalts- und Schemafingerprint, Manifest-Snapshot, Builder-Dateihashes, Git-Zustand, Laufzeit- und verfügbare Containerinformationen. Checkpoints erhalten eine kompakte Provenienzreferenz; der Run-Vergleich bevorzugt den Inhaltsfingerprint. | `Training/data.py`, `Training/train.py`, `Training/train_residual.py`, `Training/compare_final_runs.py`, `Training/smoke_test.py`, `Training/residual_smoke_test.py` | sechs gefilterte synthetische Masters, SHA-Längen/Inhaltsfingerprint, Manifestkopie, separate Provenienzdatei, Checkpointreferenz, Residual- und Standardtraining-Smoke-Test, historischer Vergleichsfallback | Für zukünftige Runs umgesetzt und getestet | Historische Runs werden dadurch nicht nachträglich reproduzierbar. Ein Containerdigest ist nur vorhanden, wenn die Laufzeit ihn als `CONTAINER_IMAGE_DIGEST` bereitstellt. |
| V-01 | Der Clusterbestand wurde in einen separaten lokalen Snapshot kopiert und gegen Manifest, Modellfeatures und den im Artefakt gespeicherten Split geprüft. Alle Dateien erhalten SHA-256-Fingerprints. | `Training/validate_dataset_snapshot.py`, `Training/dataset_snapshot_smoke_test.py` | gültiger synthetischer Snapshot, absichtlich fehlendes Feature, vollständiger realer Snapshot | Umgesetzt und getestet | Der Snapshot liegt wegen seiner Größe unter dem von Git ausgeschlossenen `Data_collection/`; seine Integrität wird über `snapshot_validation.json` statt über Git garantiert. |
| V-02 | Batch-Replay liest den erwarteten Split direkt aus dem Deploymentartefakt, verlangt jede Master-CSV und aggregiert Raw, Stable, Actionable, F1, Quality-Gründe, Inferenzzeit, Pose und angenäherte Label-Onset-Zeiten. | `Training/batch_replay_validation.py`, `Training/batch_replay_validation_smoke_test.py`, `Training/replay_stream_inference.py` | fehlende Splitdatei, CSV-Parsing, Aggregation sowie vollständiger Replay aller 21 Testsequenzen | Umgesetzt und getestet | Das strikte Frische-Gate ist mit historischen beidseitigen Nearest-Merges nicht kausal auswertbar; die Roh- und Stable-Klassifikation bleibt auswertbar. |
| V-03 | Ein roboterfreies Liveprotokoll, ein separater monotonic Event-Marker und eine Auswertung für Raw/Stable/Quality/Actionable, Sensoralter und Latenz wurden ergänzt. | `Training/live_validation_protocol.md`, `Training/live_event_marker.py`, `Training/analyze_live_validation.py`, `Training/live_validation_smoke_test.py` | Startkommando, bewertete/unbewertete Szenarien, Quality-Blockade, Eventlatenz und Host-interne Pipelinezeit | Werkzeug umgesetzt und getestet; reale Aufnahme offen | Die reale Wiederholung benötigt eine angeschlossene Aria und manuelle Gesten. Capture-to-Host bleibt ohne Device-/Host-Uhrabbildung unbestimmbar. |

## F-08: Konfidente Stabilitätsserie

### Ursprüngliches Problem

`TemporalDecisionFilter.update` erhöhte `candidate_count` allein bei gleichem Argmax. Die Konfidenz wurde erst bei der finalen Stable-Prüfung berücksichtigt. Dadurch konnten zwei unkonfidente `continue`-Kandidaten und eine einzige folgende konfidente `continue`-Vorhersage unmittelbar eine stabile Ausgabe erzeugen.

Das Verhalten wurde vor der Änderung reproduziert:

```text
low-1: uncertain, candidate_count=1
low-2: uncertain, candidate_count=2
high-1: continue,  candidate_count=3
```

### Gewählte Regel

Nach der Glättung wird zuerst der gemeinsame Drei-Klassen-Argmax und dessen Konfidenz bestimmt.

- Liegt die Konfidenz unter `minimum_confidence`, werden `candidate=None` und `candidate_count=0` gesetzt.
- Ein ausreichend konfidenter Kandidat beginnt anschließend wieder bei Zähler 1.
- Nur derselbe, unmittelbar folgende und ebenfalls ausreichend konfidente Kandidat erhöht den Zähler.
- Eine Stable-Ausgabe entsteht erst ab `minimum_stable_predictions`.

Die Wahrscheinlichkeits-Historie des Glättungsfensters wird bei niedriger Konfidenz nicht gelöscht. Nur die Serie stabiler Kandidaten wird zurückgesetzt.

### Implementierungsumfang

- `Training/replay_stream_inference.py::TemporalDecisionFilter.update`
- Keine Änderung an Glättungsfenster, Defaultschwelle oder Mindestanzahl.
- Keine Änderung an Modell, Checkpoint oder Trainingskonfiguration.

### Rückwärtskompatibilität

Signatur und Rückgabeform von `TemporalDecisionFilter.update` bleiben unverändert. Beabsichtigt geändert ist ausschließlich das Freigabeverhalten nach unkonfidenten Vorhersagen: Die stabile Ausgabe kann dadurch später erfolgen als zuvor.

## F-09: Gemeinsame Raw-/Stable-Entscheidungsregel

### Ursprüngliches Problem

Replay und Online bestimmten `raw_intention` über getrennte Argmax-Entscheidungen der beiden hierarchischen Köpfe. Der Stable-Filter verwendete dagegen den Argmax der gemeinsamen Drei-Klassen-Verteilung:

```text
P(continue) = P(no assistance)
P(fetch)    = P(assistance) * P(fetch | assistance)
P(handover) = P(assistance) * P(handover | assistance)
```

Beide Regeln können unterschiedliche Klassen liefern. Das im Audit beschriebene Beispiel wurde reproduziert:

```text
P(assistance)=0,51
P(fetch | assistance)=0,51

hierarchische Entscheidung: fetch
gemeinsamer Argmax:         continue
```

### Gewählte Regel

Für zukünftige Replay- und Live-Ausgaben gilt:

- `raw_intention`: Argmax der aktuellen gemeinsamen Drei-Klassen-Verteilung.
- `raw_confidence`: gemeinsame Wahrscheinlichkeit der Raw-Klasse.
- `stable_intention`: Argmax derselben, zeitlich geglätteten Verteilung nach F-08.
- `stable_confidence`: geglättete Wahrscheinlichkeit des Stable-Kandidaten.
- `hierarchical_raw_intention`: bisherige hierarchische Entscheidung, nur zur Diagnose.

Die gemeinsame Funktion `intention_id_from_probabilities` wird sowohl für Raw als auch innerhalb des Stable-Filters verwendet.

### Implementierungsumfang

- `Training/replay_stream_inference.py`
  - gemeinsame Argmax-Funktion,
  - Replay-Raw-Entscheidung,
  - diagnostisches Feld.
- `Training/online_inference.py`
  - dieselbe Raw-Entscheidung,
  - dasselbe diagnostische Feld.
- `Training/aria_live_inference.py` musste nicht verändert werden, weil es die Ausgabe von `OnlineInferenceEngine` übernimmt.

### Rückwärtskompatibilität

Alle bisherigen Ausgabeattribute bleiben vorhanden. `hierarchical_raw_intention` kommt additiv hinzu. Die Bedeutung von `raw_intention` ändert sich in den Fällen, in denen hierarchischer und gemeinsamer Argmax voneinander abweichen; dies ist die beabsichtigte Korrektur von F-09. Bestehende CSV-/JSONL-Dateien und `metrics.json`-Artefakte wurden nicht verändert.

## F-10: Gemeinsames Quality Gate und Actionability

### Ursprüngliches Problem

Aria Live wandte `InputQualityGate` nach der stabilisierten Modellentscheidung an, Replay dagegen nicht. Damit konnten aus derselben Sequenz zwar Raw- und Stable-Ausgaben verglichen werden, aber nicht, ob die Eingabequalität eine Ausgabe tatsächlich freigeben würde.

### Gewählte Gate-Semantik

`Training/live_decision.py::evaluate_actionability` ist nun die gemeinsame nachgelagerte Freigabefunktion für Live und Replay:

- `raw_intention` und `stable_intention` werden nicht verändert.
- Das vollständige kausale Qualitätsfenster besitzt dieselbe Länge wie das Modellfenster.
- Replay speist jeden Quellframe in zeitlicher Reihenfolge in `InputQualityGate`, einschließlich der Frames zwischen zwei Vorhersage-Endpunkten.
- Zeitlücken werden weiterhin in `InputQualityGate.push_frame` erkannt und setzen dessen Fenster zurück.
- Bei `input_quality_ok=true` entspricht `actionable_intention` dem `stable_intention`.
- Bei `input_quality_ok=false` ist `actionable_intention=insufficient_input`.
- `input_quality_reasons` enthält die maschinenlesbaren Ablehnungsgründe; `input_quality` enthält zusätzlich Abdeckungs- und Fensterdiagnostik.

Die Defaultwerte `0,80` Gazeabdeckung, `500 ms` maximale zusammenhängende Gazelücke, `0,50` Handover-Handabdeckung sowie die Altersgrenzen für Hand (`50 ms`), VIO (`10 ms`), Anker (`500 ms`) und Objektmarker (`250 ms`) sind gemeinsame Konstanten. Replay und Live reichen dieselbe Semantik an `InputQualityGate` weiter.

### Ausgabefelder und Rückwärtskompatibilität

Live-JSONL und Replay-Zeilen unterscheiden ausdrücklich:

- `raw_intention`
- `stable_intention`
- `input_quality_ok`
- `input_quality_reasons`
- `actionable_intention`

Das bisherige Live-Feld `decision_intention` bleibt als Alias von `actionable_intention` erhalten. Das vorhandene verschachtelte Feld `input_quality` bleibt ebenfalls erhalten. Replay ergänzt die neuen Felder additiv; Listen und Dictionaries werden beim CSV-Schreiben als gültiges JSON serialisiert. Bestehende Checkpoints, historische Logs und Metrikartefakte werden nicht verändert.

Die Replayzusammenfassung enthält inzwischen Raw-, Stable-, Quality- und Actionable-Fenstermetriken. Workflow- und ereignisbasierte Metriken bleiben bewusst offen, weil dafür verlässliche Ereignis-Onsets beziehungsweise ein gelabeltes Liveprotokoll benötigt werden.

## F-24: Qualität der vorhergesagten Empfangshand

### Ursprüngliches Problem

Das Gate verwendete bei `handover` die höhere Abdeckung beider Hände und verlangte aktuell lediglich irgendeine sichtbare Hand. Dadurch konnte beispielsweise eine vorhergesagte linke Empfangshand trotz fehlender linker Daten durch eine gut beobachtete rechte Hand freigegeben werden.

### Gewählte Regel

`OnlineInferenceEngine` und Replay bestimmen `predicted_receiving_hand` bereits vor der nachgelagerten Qualitätsentscheidung. Für ein stabiles `handover` gilt nun:

- Nur `left` oder `right` ist eine gültige Handvorhersage.
- Abdeckung und aktuelle Gültigkeit werden ausschließlich für diese vorhergesagte Seite geprüft.
- Die andere Hand kann weder fehlende Abdeckung noch den aktuell fehlenden Handwert kompensieren.
- Ohne gültige Handvorhersage wird keine actionable Handover-Ausgabe freigegeben.
- Für `continue`, `fetch` und `uncertain` ist keine vorhergesagte Empfangshand erforderlich.

Die neuen Ablehnungsgründe lauten:

- `handover_predicted_hand_unavailable`
- `handover_predicted_hand_coverage_too_low`
- `handover_predicted_hand_missing_currently`

Die früheren generischen Handover-Gründe werden dadurch für neue Ausgaben absichtlich ersetzt. Das Quality-Diagnoseobjekt enthält zusätzlich `predicted_receiving_hand`, `predicted_hand_coverage` und `predicted_hand_valid_currently`.

## Sensor-, Marker- und Ankerfrische

Nach Erreichen von `minimum_anchor_samples` wird die robuste Tag-0-Transformation für die laufende Session eingefroren. Weitere echte Sichtungen aktualisieren `last_anchor_observation_ns`, verändern aber nicht das Koordinatensystem eines bereits laufenden Modellfensters.

Jeder Quality-Frame speichert neben den bisherigen Validitätsflags optionale Alterswerte. Die Freigabe verwendet folgende maschinenlesbare Gründe:

- `vio_age_unavailable`, `vio_data_too_old`
- `robot_anchor_age_unavailable`, `robot_anchor_too_old`
- `handover_predicted_hand_age_unavailable`, `handover_predicted_hand_too_old`
- `fetch_marker_age_unavailable`, `fetch_visible_markers_too_old`

VIO und Anker werden über das vollständige Modell-/Qualitätsfenster geprüft. Für Handover zählt das Handalter nur an Frames, an denen die vorhergesagte Seite als gültig markiert ist. Für Fetch wird das Alter der aktuell als sichtbar geführten Objektmarker geprüft. Der Target-Selector verwendet dieselbe Markergrenze und liefert `stale_visible_objects`, wenn ausschließlich zu alte Kandidaten vorliegen.

Replay leitet Alter aus den optionalen `*_time_offset_ms`- und `*_timestamp_ns`-Spalten des Masters ab. Ein positiver Nearest-Merge-Offset bezeichnet einen aus Sicht des aktuellen Frames zukünftigen Messwert und wird daher als `unavailable` behandelt. Dies macht die bestehende Offline-/Live-Differenz sichtbar; es erfindet keinen vergangenen Ersatzwert.

## Missing-Value-Semantik

Bei vollständig fehlender linker oder rechter Live-Hand ist `hand_*_tracking_confidence` nun `NaN` und `hand_*_valid=0`. Damit erzeugt der `Normalizer` wie offline den Wert null plus Beobachtungsmaske null. Der frühere endliche Ersatzwert `-1` entfällt. Gültigkeitsflags bleiben explizit null; fehlende kontinuierliche Werte bleiben unbeobachtet.

Nicht geändert wurde der historische Masterbau mit `merge_asof(direction="nearest")`. Eine vollständige kausale Offline-/Live-Parität kann ohne Neuaufbau der Masters und erneutes Training nicht hergestellt werden.

## Latenz- und Ausgabelogging

Live-JSONL enthält weiterhin Raw, Stable, Quality und Actionable sowie jetzt:

- `sensor_ages_ms` und `sensor_timestamps`
- `intention_inference_ms` und `pose_inference_ms`
- `pipeline_timestamps` mit Device-Capture sowie monotonen Hostzeitpunkten für Callback, RGB-/Markerverarbeitung, Featureaufbau, Inferenzbeginn/-ende, Raw, Stable, Quality, Workflow und Ausgabe
- `anchor_diagnostics` einschließlich Ankeralter und Frische

Die beiden Checkpoints werden beim Aufbau von `OnlineInferenceEngine` je einmal mit einem Dummyfenster vorgewärmt. Die Warm-up-Dauern werden beim Start ausgegeben. Damit wird die einmalig deutlich langsamere erste Torch-Ausführung aus dem ersten echten Modellfenster herausgenommen.

Die Zeitstempel erlauben Host-interne Teilzeiten. Eine direkte Subtraktion von Device- und Hostzeit ist ohne bekannte Uhrenabbildung nicht zulässig.

## Replay-Auswertung

`build_replay_summary` trennt drei Ebenen:

1. Raw-Klasse
2. Stable-Klasse, wobei `uncertain` als Abstention zählt
3. Actionable-Klasse, wobei `uncertain` und `insufficient_input` nicht als freigegebene Klasse zählen

Für jede Ebene werden Coverage, bedingte Accuracy, End-to-End-Accuracy und Konfusionen gespeichert. Zusätzlich werden angenommene/abgelehnte Qualitätsfenster, Ablehnungsgründe, Inferenzzeiten und Posefehler berichtet. `--summary-json` schreibt diese Daten maschinenlesbar; `--output-csv` enthält pro Fenster alle vier Entscheidungsebenen und Diagnoseobjekte.

## Run-Suche, Metriknamen und Provenienz

`Training/run_discovery.py` ist die gemeinsame rekursive Auflösung für Vergleich und Export. Nur Verzeichnisse mit den jeweils erforderlichen Artefakten gelten als Run. Ein mehrfach vorhandener Basisname wird nicht willkürlich gewählt, sondern mit einem Fehler abgelehnt.

`pose_metrics` ergänzt die eindeutigen Schlüssel `position_mean_euclidean_error_cm`, `position_root_mean_square_euclidean_error_cm` und `position_error_definition`. Die historischen Schlüssel `position_mae_cm` und `position_rmse_cm` bleiben mit identischen Zahlenwerten als Legacy-Aliase erhalten, damit bestehende `metrics.json`, Checkpoints und Auswertungen lesbar bleiben.

Für neu gestartete Runs erzeugt `prepare_data` einen inhaltsbasierten Provenienzblock. `save_data_metadata` schreibt:

- `data_metadata.json` mit Provenienz,
- `dataset_provenance.json`,
- bei Manifestfilterung `dataset_manifest_snapshot.csv`.

Der Datensatzfingerprint hängt von den SHA-256-Werten aller tatsächlich ausgewählten Master-CSVs, dem Manifest-Snapshot und dem Schemaprint ab. Zusätzlich werden Git-Commit/Dirty-State, Builder-Dateihashes, Python-/Bibliotheks-/CUDA-Versionen und verfügbare Container-/SLURM-Variablen dokumentiert. Checkpoints enthalten nur die kompakte Referenz aus Dataset-, Schema-, Builder- und Git-Fingerprint.

## Ausgeführte kurze Tests

Ausgeführt mit dem lokalen Conda-Interpreter und deaktivierter automatischer Bytecodeerzeugung:

| Test | Ergebnis |
|---|---|
| `Training/inference_decision_smoke_test.py` | Bestanden: F-08/F-09, Replay-/Live-Gate-Parität, kausale Altersrekonstruktion, Zeitlückenreset, Missing-Hand-`NaN`, Phasenzeitstempel, zweifaches Modell-Warm-up und geschichtete Replayzusammenfassung. |
| `Training/live_decision_smoke_test.py` | Bestanden: Gaze-, linke/rechte Hand-, VIO-, Anker- und Markerfälle; alte vorhergesagte Hand; Target-Selector und bestehender Workflow. |
| `Training/run_discovery_smoke_test.py` | Bestanden: direkte und verschachtelte Runs, unvollständiger Run sowie sichere Ablehnung eines mehrdeutigen Namens. |
| `Training/compare_final_runs.py` auf `Training/runs` mit Ausgaben unter `/tmp` | Bestanden: alle zwölf unter `run_cluster/Ohne Titel` verschachtelten finalen Runs wurden gefunden und mit den historischen Metrikschlüsseln validiert. |
| `Training/smoke_test.py` | Bestanden: Transformer v1, MLP und GRU; additive Metriknamen; SHA-256-Provenienz und Manifest-Snapshot. |
| `Training/residual_smoke_test.py` | Bestanden: Residual v2 einschließlich zwei kurzer synthetischer Epochen, beider Checkpoints, neuer Provenienz und Metrikdefinition. |
| `Training/export_predictions_smoke_test.py` | Bestanden: Vorhersageexport und Referenzvergleich bleiben mit dem Legacy-Metrikalias kompatibel. |
| `Training/pose_baselines_smoke_test.py` | Bestanden: Pose-Baseline-Auswertung bleibt funktionsfähig. |
| AST-Syntaxprüfung aller 29 Pythonmodule direkt unter `Training/` | Bestanden. |
| Seed-44-Replay mit lokalem `Jona_6`-Master, CPU, fünf Vorhersagen | Bestanden: Raw/Stable wurden berechnet; Actionable wurde wegen fehlendem Robot-Frame und nicht kausal belegbaren Alterswerten des alten Masters nachvollziehbar blockiert. CSV und Summary-JSON unter `/tmp` wurden anschließend eingelesen und ihre verschachtelten Diagnosefelder validiert. |
| `Training/dataset_snapshot_smoke_test.py` | Bestanden: gültiger Snapshot sowie absichtlich fehlendes Modellfeature. |
| `Training/batch_replay_validation_smoke_test.py` | Bestanden: Artefaktsplit, fehlende Masterdatei, CSV-Typisierung, F1-/Quality- und Onset-Aggregation. |
| `Training/live_validation_smoke_test.py` | Bestanden: Eventmarker-Semantik, bewertete und unbewertete Intervalle, Quality-Blockade und Host-Latenz. |
| Vollständiger Seed-44-Testsplit-Replay | Bestanden: alle 21 Sequenzen, 2.117 Vorhersagen; Detailbericht in `Training/validation_notes/05_deployment_validation.md`. |
| `aria_live_inference.py --check-only --device cpu` | Bestanden: 92 Features, Fensterlänge 60 und beide Checkpoints geladen/vorgewärmt; erwartungsgemäß war bei der lokalen Prüfung kein USB-Gerät verbunden. |

## Offene Punkte nach diesem Arbeitsabschnitt

- Die neue Joint-Argmax-Regel muss später auf den vollständigen finalen Testsequenzen gegen die historische hierarchische Regel ausgewertet werden.
- Für eine belastbare Stable-/Actionable-Auswertung werden aktuelle finale Master-CSVs mit vollständigem Schema und gelabelte Ereignis-Onsets benötigt. Der lokal vorhandene alte Master ist dafür nicht ausreichend.
- Konfidenz, Glättungsfenster, Stabilitätsanzahl sowie die Alters- und Coveragegrenzen benötigen eine kontrollierte ereignisbasierte Kalibrierung.
- Exakte kausale Offline-/Live-Featureparität erfordert neu gebaute Daten und erneutes Training. Beides blieb gemäß Aufgabenbegrenzung unangetastet.
- Historische Runs erhalten nicht rückwirkend die neue Inhaltsprovenienz; dafür müsste der exakte damalige Cluster-Datensatz wiederhergestellt werden.
- Participant-Cross-Validation, neue Daten, Label-/Architekturänderungen und eine ausgeglichene Links-/Rechts-Evaluation bleiben spätere Trainingsarbeiten.
- Die reale Koordinatenkalibrierung, Simulation/HIL, Sicherheitsarchitektur und jede Roboterschnittstelle bleiben vollständig außerhalb dieses Codes.
- Es wurde kein Training auf realen Projektdaten und kein langer Trainingslauf gestartet; nur die vorhandenen kurzen synthetischen Smoke Tests liefen. Hardware und Roboterkommunikation wurden nicht verwendet.
