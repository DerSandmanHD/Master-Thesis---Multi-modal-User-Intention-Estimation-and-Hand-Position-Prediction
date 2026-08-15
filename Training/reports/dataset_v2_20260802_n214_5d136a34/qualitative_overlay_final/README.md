# Qualitative Vorhersage-Overlays des finalen Modells

> **Historisch und ungültig für finale Thesis-v2-Abbildungen.** Diese Overlays
> stammen aus `dataset_v2_20260802_n214_5d136a34`, vor dem kausalen Master-
> Rebuild, der korrigierten VRS-`DEVICE_TIME`-CLIP-Synchronisation und den
> hashgebundenen Prediction-/Final-Test-/Video-Sidecars. Die untenstehenden
> Werte dürfen nur als historische Beschreibung erhalten bleiben. Sie sind
> keine Ergebnisse für den aktiven geplanten Stand
> `dataset_v3_causal_20260815_n214_5d136a34` und müssen nach dessen Training
> neu erzeugt werden.

Historischer Checkpoint: damaliges Sensor+CLIP-Modell, Seed 42

Split: Test (`Edu`, `Jona`, `Mona`)

Export: 2.199 Fenster aus 23 Sequenzen

Automatisch ausgewählt wurden:

| Sequenz | Rolle | Fenster-Accuracy | Posefenster / Pose-MAE |
|---|---|---:|---:|
| `Edu_5_20260604_170944` | Erfolgsbeispiel | 0,989 | 12 / 15,84 cm |
| `Jona_7_20260616_182214` | Fehlerbeispiel | 0,663 | 9 / 8,47 cm |
| `Mona_3_20260624_123548` | Medianbeispiel | 0,924 | 10 / 16,09 cm |

Jedes H.264-MP4 enthält RGB, Ground Truth, vorhergesagte Intention,
Wahrscheinlichkeitsbalken, Empfangshand und ein separates XY-Inset für
Ground-Truth- und vorhergesagte zukünftige Handposition im Robot-Frame. Eine
3D-Projektion in das RGB-Bild wird bewusst nicht behauptet, weil keine
vollständig validierte zeitabhängige Kameraprojektion vorlag.

Die Zuordnung ist kausal: Für jeden Videoframe wird nur die jüngste
Vorhersage mit `prediction_time <= frame_time` verwendet. Der maschinenlesbare
`overlay_report.json` bestätigt streng steigende Prediction-Timestamps,
`future_prediction_matches = 0` für alle drei Videos und damit eine gültige
Synchronisation. Die Videos bleiben wegen ihrer Größe lokale, reproduzierbare
Artefakte; die Einzelbilder und der Report können versioniert werden.
