# Master-Thesis-Ready checklist status

Stand: 26. August 2026

Arbeitsplan: `/Users/dersandmannhd/Downloads/master_thesis_ready_finale_checkliste.md`

Verbindlicher Scope:

- Primärziel: aktuelle Intention + Receiving-Hand + Receiving-Wrist-Pose bei
  `t + 1 s`;
- sekundär und getrennt: terminale Endpose;
- aktives Dataset: `dataset_v3_causal_20260815_n214_5d136a34`;
- historische v2-Ergebnisse bleiben historisch;
- keine neuen Trainingsjobs und kein Überschreiben eingefrorener Ergebnisse.

Statuslegende: `DONE` = implementiert und verifiziert; `PENDING` = noch
ausstehender externer oder Git-bezogener Abschluss; `SUPERSEDED` = durch
vorhandene Evidenz nicht mehr erforderlich.

## Stufe 0

| Punkt | Status | Ergebnis / Evidenz |
|---|---|---|
| 0.1 Identifier `Test` klären | **DONE** | Am 26.08.2026 direkt als echtes Teilnehmerpseudonym bestätigt. 214 Sequenzen und 25 Teilnehmende bleiben korrekt; kein Rebuild oder Retraining. [`IDENTITY_PROVENANCE_RESOLUTION_20260826.md`](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md) |
| 0.2 Pose-Loss-Scale diagnostizieren | **DONE** | Finale Train-Position-Errors 8,954/9,278/9,203 cm; 14–15-cm-Trigger nicht erfüllt. [`pose_learning_diagnosis.json`](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/pose_learning_diagnosis_v1/pose_learning_diagnosis.json) |
| Optionaler `normalized_smooth_l1`-Run | **SUPERSEDED** | Lernkurven begründen keinen neuen Sensitivity-Run; keiner wurde gestartet. |

## Stufe 1

| Punkt | Status | Ergebnis / Evidenz |
|---|---|---|
| 1.1 `Success@τ` | **DONE** | Kaskade für 5/10/15/20 cm in `grouped_metrics.py`, Tests und autoritativem Summary. |
| 1.2 Drei Receiving-Hand-LOPO-Sichten | **DONE** | Fixed/all 0,6011; supported/all 0,9579; mixed-hand 0,8723. [`group_cv_summary.json`](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/summary_v2/group_cv_summary.json) |
| 1.3 Participant–Hand-Confounding | **DONE** | Cramérs V 0,7620, Participant-Majority 0,8505 und sieben Mixed-Hand-Teilnehmende im finalen Bericht/Thesis-Entwurf. |
| 1.4 Triviale Intention-Baselines | **DONE** | Majority, kausale Elapsed-Time-Logistik, Last-Sensorframe-Logistik; train-only Fit, keine Zukunftsinformation. |
| 1.5 Persistence/CV im Pose-Hauptvergleich | **DONE** | Fair-common Vergleich auf 213/217 t+1-Zielen; Learned verbessert Persistence im MAE nicht konsistent. |
| 1.6 Autoritativer v3-Summary | **DONE** | Job 2246329; 48 Seed-Zeilen; Fingerprint `ec078d5ed0d1eda3c2b009b92b3575da57f45cd2a3bbaa2ceca1154544184b9c`. |
| 1.7 Qualitative Pipeline | **DONE** | Job 2246134; drei Device-Time-Sidecars, Stills und MP4s; keine Zukunftsmatches. |
| 1.8 Empirische Samplingrate | **DONE** | Median Δt 0,033333 s / 30,0003 Hz; 60 Samples = median 1,966667 s. |
| 1.9 Offline vs. Deployment | **DONE** | Finaler Summary und Thesis-Entwurf trennen Offline-/Grouped-Metriken von raw/stable/actionable Replay-Metriken. |
| 1.10 Live-Toleranzen | **DONE** | 50/10/500 ms live vs. 12/5/20 ms offline als Deployment-Shift dokumentiert. |

## Stufe 2

| Punkt | Status | Ergebnis / Evidenz |
|---|---|---|
| 2.1 Abstract auf v3 | **PENDING** | V3-Absatz ist in [`Thesis/experiment_results_n214.md`](Thesis/experiment_results_n214.md) übertragungsfertig; die eigentliche Abstract-Quelldatei liegt nicht im Repository. |
| 2.2 Methodik mit Code synchronisieren | **DONE** | Markerrollen/-größen, `Rotation.mean()`, Anchor-Fallback, Flag-Semantik, `closed_loop_trajectory.csv`, Gaze-Ursprung und Toleranzen korrigiert. |
| 2.3 Splitgrößen | **DONE** | 170/21/23 Sequenzen, 19/3/3 Teilnehmende und 15.189/1.978/2.199 Fenster dokumentiert. |
| 2.4 Fixed Test vs. LOPO | **DONE** | Getrennte Tabellen und Interpretation; keine gemeinsame Unsicherheitsaussage. |
| 2.5 He et al. | **DONE** | Primärquelle und Abgrenzung im transferfähigen Ergebnistext ergänzt. |
| 2.6 Implementierungsgenaue Claims | **DONE** | Aktuelle Intention statt Future Intention; Objekt-Featuremodalität statt Object Tokens. |
| 2.7 Kausalitätsclaim | **DONE** | „Causal temporal observation alignment with offline static robot-frame calibration.“ |
| 2.8 Limitationen | **DONE** | Offline-Kalibrierung, Confounding, Testgröße, Single-Hand-Gruppen, Objektmodell und Sensoralter enthalten. |

## Stufe 3

| Punkt | Status | Ergebnis / Evidenz |
|---|---|---|
| 3.1 Hauptmodellvergleich | **DONE** | Majority, Elapsed Time, Last-Frame Logistic, MLP, GRU, Transformer, Residual; Accuracy, Macro-F1 und Klassen-F1. |
| 3.2 Architekturablationen | **DONE** | Hierarchical/Flat, Current/Simple/Modality Gate und Pose-Loss-Ablation. |
| 3.3 Modalitätsablationen | **DONE** | Full, no gaze/hands/objects/VIO. |
| 3.4 Visual Context | **DONE** | Sensor, Random Control, corrected CLIP Current Gate und CLIP Modality Gate. |
| 3.5 Generalisation | **DONE** | Fixed Test, vollständiges 25-Fold-LOPO und drei Hand-Sichten. |
| 3.6 Pose-Tabelle | **DONE** | Persistence, CV, Learned/GT-Hand und Learned/predicted hand samt RMSE, Orientierung, Coverage und Kohorte. |
| 3.7 System-Level Success | **DONE** | Vollständige Kaskadenstufen und `Success@5/10/15/20 cm`. |
| 3.8 Confusion Matrices | **DONE** | Intention und Receiving-Hand für den validation-selektierten ausführbaren Seed-44-Checkpoint. |
| 3.9 Participant-Level | **DONE** | Participantmetriken und Verteilung liegen im `summary_v2`-CSV/JSON; Mixed-Hand-Ergebnis ist im Haupttext. |
| 3.10 Qualitative Beispiele | **DONE** | Good: Jona_7; typical: Edu_3; failure: Mona_6; Stills, Videos und maschinenlesbare Fallwerte. |

## Stufe 4

| Punkt | Status | Ergebnis / Evidenz |
|---|---|---|
| 4.1 Vollständige Testsuite | **DONE** | 174 Unit-/Invarianten-Tests, 22/22 Scientific Smokes, 214/214 Master-QA, Summary-/Qualitative-Hashes und 48/48 Training-Freezes bestanden. [`FINAL_VERIFICATION_20260826.md`](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/FINAL_VERIFICATION_20260826.md) |
| 4.2 Finaler autoritativer Report | **DONE** | [`final_summary/`](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/) vollständig und lokal hashverifiziert. |
| 4.3 Finaler Commit | **DONE** | Der autorisierte Abschluss-Commit enthält Code, Tests, kompakte v3-Reports und diesen Status; fremde Arbeitsordner und ignorierte MP4s bleiben außen vor. |
| 4.4 Merge nach `main` | **DONE** | Der autorisierte Fast-Forward-Merge nach `main` ist Bestandteil dieses Abschlussvorgangs. |
| 4.5 Git-Tag | **PENDING** | Erst nach Merge und expliziter Freigabe. |
| 4.6 Reproduzierbarkeit | **PENDING** | Dataset-/Report-/Config-/Checkpoint-Provenienz sowie Commit und Merge sind gesichert; nur der optionale finale Git-Tag fehlt. |

## Verbleibende Entscheidungen

1. Abstract-Absatz später in die außerhalb dieses Repositories liegende
   Thesisquelle übertragen.
2. Einen finalen Git-Tag bei Bedarf separat freigeben.
