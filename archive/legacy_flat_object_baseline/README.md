# Legacy Flat/Object Baseline

Dieses Archiv dokumentiert die erste Trainingsbaseline vor der Umstellung auf
die hierarchische Thesis-Aufgabe.

Enthalten sind:

- `first_test.json`: ursprüngliche Konfiguration mit zufälligem
  Participant-Split
- `participant_split_v1.json`: später festgelegter Teilnehmer-Split
- `first_test.sbatch`: stillgelegter Cluster-Job

Die damalige Baseline sagte zusätzlich eine Objekt-ID als Modellziel voraus.
Das entspricht nicht der aktuellen Forschungsaufgabe. Objektmarker werden im
aktuellen Modell als Szenenkontext verwendet; die Modellziele sind Intention
und die zukünftige Handover-Handpose.

Die Ergebnisse bleiben in `Thesis/status_testing_baseline.md` dokumentiert.
Für neue Läufe sind `Training/configs/hierarchical_baseline_v1.json` oder
`Training/configs/hierarchical_residual_v2.json` zu verwenden.
