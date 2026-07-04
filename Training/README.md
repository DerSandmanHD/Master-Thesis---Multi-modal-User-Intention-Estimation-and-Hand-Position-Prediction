# Multimodales Training

Dieser Ordner enthält den ersten reproduzierbaren End-to-End-Trainingslauf für
die Aria-Master-Datensätze. Das Modell hat drei Ausgaben:

1. Intention: continue, fetch, handover
2. Zielobjekt: ArUco-ID 6 bis 14
3. Handpose: Position und Quaternion des empfangenden Handgelenks eine Sekunde
   in der Zukunft, im Koordinatensystem des Robotermarkers (AprilTag 0)

## Architektur

model.py ist eine moderne, an GTN angelehnte Zwei-Turm-Architektur. Ein Turm
modelliert zeitliche Abhängigkeiten, der zweite Abhängigkeiten zwischen
Sensorkanälen. Ein lernbares Gate fusioniert beide Repräsentationen. Im
Gegensatz zum unveränderten Code unter GTN/ unterstützt diese Variante fehlende
Messwerte, participant-wise Splits und Multi-Task-Ausgaben.

Das alte GTN ist eine sinnvolle Architekturidee und Vergleichsbaseline, aber
nicht unverändert als finales Modell geeignet: Es erwartet MAT-Dateien, flacht
alle Tokens ab, besitzt keine robuste Missing-Data-Behandlung und kann nur
klassifizieren. Ob die angepasste GTN-Variante das finale Modell wird, muss über
participant-held-out Ergebnisse gegen einfachere Baselines entschieden werden.

## Datenvoraussetzung

Die Eingabe sind aktuelle Dateien unter:

    Data_collection/master_datasets/*_master.csv

Sie müssen mit semantischen Annotationen neu erzeugt worden sein, damit
target_object_id und future_1s_receiving_wrist_* vorhanden sind. Labels und
zukünftige Zielwerte werden niemals als Eingabefeatures verwendet.

Master-Datensätze nach Abschluss der Reviews erzeugen:

    python3 Code/build_master_dataset_batch.py \
      --data-root Data_collection \
      --require-semantic-annotations \
      --overwrite

## Erster Test

Zuerst den Pipeline-Smoke-Test ausführen:

    python3 Training/smoke_test.py

Dann auf dem Cluster:

    sbatch Training/first_test.sbatch

Alternativ interaktiv:

    python3 Training/train.py --config Training/configs/first_test.json

Ergebnisse landen unter Training/runs/:

- best_model.pt: bester Checkpoint nach Validation-Macro-F1
- config.json: tatsächlich verwendete Konfiguration
- data_metadata.json: Features, Normalisierung und participant-wise Split
- metrics.json: Lernverlauf sowie einmalige Testauswertung

Der Test-Split wird erst nach Auswahl des besten Modells ausgewertet. Dadurch
bleibt er von Modellwahl und Early Stopping getrennt.
