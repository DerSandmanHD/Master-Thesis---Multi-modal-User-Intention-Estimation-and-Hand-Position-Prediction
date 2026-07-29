# Kontrollierte Live-Validierung ohne Roboter

Dieses Protokoll prüft die reale Aria-Live-Pipeline reproduzierbar. Es erzeugt
nur Vorhersagen und Annotationen. Es enthält keine Robotersteuerung und fordert
keine externe Aktion an.

## Ziel und Voraussetzungen

Verwendet werden der finale Residual-v2-Intentionscheckpoint und der separate
Posecheckpoint aus `Training/final_clean_v1_residual_v2_seed44`. Für jeden
Versuch werden neue Dateinamen benutzt, weil die Live-Inferenz JSONL-Dateien
absichtlich anhängt.

Vor dem Versuch:

1. Aria über USB verbinden und `aria_conda` aktivieren.
2. AprilTag 0 am statischen Referenzort sichtbar halten, bis
   `anchor_ready=true` gemeldet wird.
3. Objektmarker 6 bis 14 wie bei der Datenerfassung positionieren.
4. Keine Datei eines älteren Versuchs wiederverwenden.
5. Keine Roboter- oder ROS-Komponente starten.

## Aufzeichnung

Terminal A startet die Inferenz:

```bash
python Training/aria_live_inference.py \
  --artifacts-dir Training/final_clean_v1_residual_v2_seed44 \
  --profile-name profile9 \
  --interface usb \
  --hand-tolerance-ms 50 \
  --vio-tolerance-ms 10 \
  --marker-tolerance-ms 500 \
  --print-mode all \
  --output-jsonl Training/Outputs/live_validation_01_predictions.jsonl \
  --debug-features-jsonl Training/Outputs/live_validation_01_features.jsonl
```

Terminal B zeichnet Ereignisgrenzen auf derselben Maschine auf:

```bash
python Training/live_event_marker.py \
  --output-jsonl Training/Outputs/live_validation_01_events.jsonl
```

Erst annotieren, wenn `anchor_ready=true`, `buffer_frames=60` und mehrere
Vorhersagen ausgegeben wurden. `start` wird unmittelbar vor Beginn der
Bewegung eingegeben, `end` nach mindestens zehn Sekunden stabiler Ausführung.
Zwischen bewerteten Intentionen mindestens fünf Sekunden neutral bleiben.

Beispiel:

```text
start neutral_01 continue true
end
start fetch_combined_01 fetch true
end
start neutral_02 continue true
end
start handover_combined_01 handover true
end
```

Jedes bewertete Szenario sollte mindestens dreimal wiederholt werden:

| Szenario | Markerkommando | Ausführung |
|---|---|---|
| Neutral | `continue true` | aktuelle Tätigkeit ohne Hilfegeste fortsetzen |
| Fetch kombiniert | `fetch true` | Zielobjekt ansehen und wie in der Aufnahme darauf zeigen |
| Handover kombiniert | `handover true` | Empfangshand wie in der Aufnahme zum Übergabebereich führen und Zielkontext ansehen |
| Gaze-Ausfall | `unscored false` | Augen deutlich länger als einen kurzen Blink schließen |
| Fetch nur Gaze | `unscored any` | Objekt ansehen, Hand neutral lassen |
| Fetch nur Hand | `unscored any` | zeigen, aber bewusst woanders hinsehen |
| Handover nur Hand | `unscored any` | Empfangshand ausstrecken, Blick abwenden |

Die unimodalen Szenarien sind absichtlich `unscored`: Aus den aktuellen Labels
folgt keine eindeutige Sollklasse für künstlich widersprüchliche Modalitäten.
Sie dienen zur qualitativen Prüfung der gelernten Abhängigkeiten. Der
Gaze-Ausfall prüft dagegen eindeutig, ob das Quality Gate eine actionable
Ausgabe blockiert, ohne Raw und Stable zu verwerfen.

## Auswertung

Nach `end` für das letzte Szenario zuerst den Event-Marker mit `quit` und dann
die Live-Inferenz mit `Ctrl-C` beenden. Anschließend:

```bash
python Training/analyze_live_validation.py \
  --predictions-jsonl Training/Outputs/live_validation_01_predictions.jsonl \
  --events-jsonl Training/Outputs/live_validation_01_events.jsonl \
  --output-json Training/evaluation/deployment_validation_runs/live_validation_01.json
```

Der Bericht trennt Raw, Stable und Actionable, Quality-Ablehnungen,
Sensoralter, Modelllaufzeit und die Host-interne Pipelinezeit. Die
Event-Onset-Latenz wird vom manuell markierten `start` bis zur ersten
erwarteten Ausgabe gemessen. Sie enthält daher auch die Reaktionsungenauigkeit
beim Drücken der Enter-Taste.

Eine Capture-to-Host-Latenz wird bewusst nicht durch Subtraktion von Device-
und Hostzeit berechnet: Dafür fehlt derzeit eine validierte Abbildung der
beiden Uhren. Beurteilbar sind dagegen Inferenzzeit, Callback-to-Output-Zeit
und die absichtlich durch Glättung und Stabilitätsprüfung erzeugte
Entscheidungsverzögerung.

## Akzeptanz vor einer betreuten Demo

- Keine actionable Fetch-/Handover-Ausgabe während neutraler Intervalle.
- Gaze-Ausfall führt nach Ablauf der Fensterabdeckung nachvollziehbar zu
  `insufficient_input`; Raw und Stable bleiben im Log sichtbar.
- Kombinierte Fetch- und Handover-Szenarien werden wiederholt erkannt.
- Handover wird nur bei gültiger vorhergesagter Empfangshand freigegeben.
- Sensoralter und Quality-Gründe sind vollständig protokolliert.
- Die beobachtete Event-Latenz wird berichtet, nicht nur die reine
  Modelllaufzeit.

Grenzwerte sollten erst nach mindestens drei vollständigen Wiederholungen pro
Szenario angepasst werden. Einzelne erfolgreiche oder fehlgeschlagene Gesten
sind keine belastbare Grundlage für Threshold-Tuning.
