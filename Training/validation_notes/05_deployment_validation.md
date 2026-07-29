# Deployment-Validierung ohne Roboter

**Stand:** 29. Juli 2026  
**Modell:** `final_clean_v1_residual_v2_seed44`  
**Umfang:** Dataset-Snapshot, vollständiger Testsplit-Replay und Vorbereitung
eines kontrollierten Aria-Livetests; keine Roboterintegration

## Ergebnis in Kürze

Der finale Datenbestand ist vollständig und stimmt exakt mit dem Split des
Deploymentartefakts überein. Die korrigierte Raw-Entscheidung reproduziert die
historische Testleistung nahezu exakt. Die zeitliche Stabilisierung erhöht die
Trefferquote unter den ausgegebenen Entscheidungen, enthält sich aber bei etwa
12,5 Prozent der gelabelten Fenster.

Das strikte Live-Quality-Gate kann anhand der historischen Master-CSVs nicht
gültig als Actionable-Test ausgewertet werden. Die Masters wurden mit einem
beidseitigen Nearest-Neighbor-Merge gebaut und können deshalb pro Fenster
minimal zukünftige Sensorwerte enthalten. Diese Werte werden im kausalen
Replay korrekt als Altersangabe `unavailable` behandelt. Das Gate zu lockern
oder Zukunftswerte als vergangen umzudeuten wäre keine valide Lösung.

## 1. Eingefrorener Datenbestand

Der Clusterbestand wurde nach
`Data_collection/final_dataset_snapshot_20260729/` kopiert. Dieser Ordner ist
lokal und wegen der Datengröße von Git ausgeschlossen. Seine Integrität wird
durch `snapshot_validation.json` abgesichert.

| Prüfung | Ergebnis |
|---|---:|
| vorhandene Master-CSVs | 183 |
| vom strikten Manifestfilter zugelassen | 156 |
| vom Artefakt erwartete Trainsequenzen | 116/116 vorhanden |
| vom Artefakt erwartete Validationsequenzen | 19/19 vorhanden |
| vom Artefakt erwartete Testsequenzen | 21/21 vorhanden |
| Schema-/Identitäts-/Duplikatfehler | 0 |
| Manifest-SHA-256 | `d4c0a4a7fe1866a43888ab966ec43d57c4e91d01fbb090e9d9e2d5596487ac7b` |
| Sequenzfingerprint | `457a80f15423fe3e3853081e3a0d863248ec337dd7412cde65bc8ee56ff3049d` |

Die 156 Sequenzen entsprechen genau den im Seed-44-Artefakt gespeicherten
Splits. Damit ist ausgeschlossen, dass der folgende Replay versehentlich
zusätzliche, ausgeschlossene oder falsche Teilnehmersequenzen verwendet.

## 2. Vollständiger Testsplit-Replay

`Training/batch_replay_validation.py` spielte alle 21 Testsequenzen von Edu,
Jona und Mona kausal durch beide Seed-44-Checkpoints. Ausgewertet wurden 2.117
Vorhersagefenster, davon 1.981 mit einem der drei Ziel-Labels; Transitionen
werden nicht als Klassenziel gewertet.

### Intentionsklassifikation

| Ebene | Coverage | Accuracy bedingt | Accuracy Ende-zu-Ende | Macro-F1 bedingt | Macro-F1 Ende-zu-Ende |
|---|---:|---:|---:|---:|---:|
| Raw | 100,00 % | 89,05 % | 89,05 % | 85,35 % | 85,35 % |
| Stable | 87,48 % | 94,00 % | 82,23 % | 91,28 % | 83,18 % |
| Actionable | 0,00 % | nicht bestimmbar | 0,00 % | nicht bestimmbar | 0,00 % |

Raw verwendet jetzt den gemeinsamen Drei-Klassen-Argmax. Das historische
Seed-44-Testergebnis aus `metrics.json` beträgt 89,10 Prozent Accuracy und
85,44 Prozent Macro-F1. Die Abweichung ist minimal und durch die korrigierte
Entscheidungsregel erklärbar; sie ist kein Hinweis auf eine beschädigte
Inferenzpipeline.

Raw-F1 nach Klasse:

| Klasse | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| continue | 1.396 | 96,03 % | 90,04 % | 92,94 % |
| fetch | 314 | 67,00 % | 84,71 % | 74,82 % |
| handover | 271 | 87,64 % | 88,93 % | 88,28 % |

Fetch bleibt damit die schwächste Klasse, vor allem wegen 117
Continue-Fenstern, die Raw als Fetch einstuft. Die Stabilisierung reduziert
Fehlentscheidungen, erkauft dies jedoch durch `uncertain`-Ausgaben.

### Angenäherte Erkennung nach Label-Onset

Die Messung beginnt am ersten ausgewerteten Replay-Endpunkt mit einem neuen
Label. Sie ist daher keine Capture-to-Decision-Latenz und kann wegen
`stride=10` um bis zu etwa 333 ms vom echten Annotationsbeginn abweichen.

| Stable-Klasse | erkannte Segmente | Median | p95 |
|---|---:|---:|---:|
| continue | 21/21 | 333 ms | 2.000 ms |
| fetch | 20/21 | 167 ms | 3.767 ms |
| handover | 20/21 | 667 ms | 1.367 ms |

Die hohen Fetch-Ausreißer zeigen, dass eine niedrige reine Modelllaufzeit nicht
mit sofortiger stabiler Ereigniserkennung gleichgesetzt werden darf.

### Inferenz- und Posewerte

| Größe | Ergebnis |
|---|---:|
| Intentionsinferenz CPU, Mittel | 1,24 ms |
| Intentionsinferenz CPU, p95 | 1,46 ms |
| Poseinferenz CPU, Mittel | 1,20 ms |
| Poseinferenz CPU, p95 | 1,42 ms |
| Replay-Poseposition, 139 freigegebene Stable-Handover-Fenster | 13,60 cm |
| Replay-Orientierung, dieselben Fenster | 37,48° |

Die Posewerte sind nicht direkt mit den 202 Handover-Targets der historischen
Checkpointauswertung vergleichbar: Der Replay ruft den Posecheckpoint nur bei
`stable_intention=handover` auf und bewertet dadurch eine andere Teilmenge.

## 3. Warum Actionable offline vollständig blockiert

Alle 2.117 Replayfenster enthalten den Grund `vio_age_unavailable`. Die
benötigte Spalte ist vorhanden; die Ursache ist ihre kausale Interpretation.
Bei `Jona_6` liegen beispielsweise 689 von 1.033 gültigen
`slam_time_offset_ms` minimal in der Zukunft und nur 344 in der Vergangenheit.
Da das Gate ein vollständiges 60er-Fenster prüft, enthält praktisch jedes
Fenster mindestens einen nicht kausal belegbaren Wert.

Zusätzlich treten auf:

- `robot_anchor_too_old`: 1.107 Fenster
- `robot_anchor_age_unavailable`: 489 Fenster
- `fetch_marker_age_unavailable`: 205 Fenster
- Gazeabdeckung beziehungsweise Gazelücke: 95 Fenster
- handover-spezifische Handalters-/Abdeckungsgründe: einzelne Fenster

Diese Zahlen belegen nicht, dass Live-VIO oder der Live-Anker schlecht sind.
Sie belegen, dass die historischen Nearest-Merge-Masters nicht alle
Zeitinformationen enthalten, die für eine nachträgliche strikte kausale
Frischeprüfung nötig wären. Raw und Stable bleiben valide Offline-
Modellauswertungen; Actionable muss mit einem frischen Live-Log geprüft werden.

## 4. Kontrollierter Live-Test

`Training/live_validation_protocol.md` beschreibt den noch auszuführenden
Hardwareversuch. `Training/live_event_marker.py` setzt auf derselben Maschine
monotone Start-/Endmarken. `Training/analyze_live_validation.py` wertet
anschließend aus:

- Raw, Stable und Actionable getrennt,
- Quality-Akzeptanz und maschinenlesbare Gründe,
- falsche actionable Assistenz während Continue,
- Zeit vom manuell markierten Onset bis Raw/Stable/Actionable,
- Hand-, VIO-, Anker- und Markeralter,
- reine Modelllaufzeit und Host-interne Callback-to-Output-Zeit.

Dieser Teil ist vorbereitet und per Smoke Test geprüft, kann ohne
angeschlossene Aria und reale Gesten aber nicht seriös vorweggenommen werden.
Der lokale `--check-only`-Lauf lud SDK, 92 Modellfeatures, Fensterlänge 60 und
beide Checkpoints erfolgreich; bei dieser Prüfung war kein USB-Gerät
angeschlossen.

## 5. Urteil und nächste Freigabe

- **Datensatz:** vollständig, manifestkonform und zum Artefaktsplit passend.
- **Modell-/Replay-Pfad:** konsistent; die Raw-Metrik reproduziert das
  gespeicherte Testergebnis nahezu exakt.
- **Stabilisierung:** technisch wirksam, aber mit messbarer Abstention und
  Ereignisverzögerung.
- **Quality Gate:** sicherheitsgerichtet implementiert, mit historischen
  Nearest-Merge-Masters jedoch nicht als Actionable-Metrik bewertbar.
- **Neues Training:** für diese Code- und Validierungsänderungen nicht nötig.
- **Betreute Demo ohne Roboter:** plausibel, sobald der kontrollierte Livetest
  wiederholt besteht.
- **Realer Robotereinsatz:** weiterhin nicht freigegeben und nicht Teil dieses
  Arbeitsschritts.

Thresholds werden nicht anhand eines einzelnen erfolgreichen Livetests
verändert. Erst die gelabelten Wiederholungen aus dem Liveprotokoll liefern
eine Grundlage, um Stabilität, Fehlalarme und Latenz gegeneinander
abzuwägen.
