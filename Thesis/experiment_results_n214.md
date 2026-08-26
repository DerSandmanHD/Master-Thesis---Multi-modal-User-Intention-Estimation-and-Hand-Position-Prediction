# Methoden- und Ergebnisentwurf: autoritative kausale v3-Auswertung

Stand: 26. August 2026

Dieses Dokument ist die direkt in die Thesis übertragbare Ergebnisquelle für
den aktuellen Experimentstand. Es verwendet ausschließlich das aktive Dataset
`dataset_v3_causal_20260815_n214_5d136a34`. Historische v2-Ergebnisse sind
nicht Bestandteil der folgenden Zahlen und dürfen nicht als v3 interpretiert
werden.

## 1. Geltungsbereich und Provenienz

Das Primärziel ist die Schätzung der **aktuellen** Assistenzintention
(`continue`, `fetch`, `handover`) aus einer multimodalen Beobachtungshistorie
sowie, für Handover, die Vorhersage der Receiving-Hand und ihrer
6-DoF-Wrist-Pose bei `t + 1 s`. Die terminale Endpose ist ein separates,
sekundäres Experiment und wird in Abschnitt 10 getrennt berichtet.

Alle neuronalen Hauptzahlen stammen aus einem validation-selektierten
`best_intention`-Checkpoint pro Seed. Angegeben werden Mittelwert und
Stichprobenstandardabweichung über die Seeds 42, 43 und 44. Aggregate sind
keine ausführbaren Checkpoints. Triviale Intention-Baselines sind nachträglich
ergänzte, deskriptive Kontrollen: Sie wurden nur auf Trainingsfenstern
angepasst und nicht anhand des Testsets ausgewählt.

Verbindliche Provenienz:

- Dataset-Content-Fingerprint:
  `339c313d56c7be8564a3035cd8334bad103e15659315c33135e96b95e820245a`
- Source-Content-Fingerprint:
  `b2cfd10991c5638277795aadcabfd322d3e9eb41dffb70696b63dc657f5f22cb`
- Matrix-SHA-256:
  `20ebefa13be76635b2f78139a6a8506dcf1eae6fedd8fdc1c7bdcf4d53e7e6d2`
- Autoritativer Report-Fingerprint:
  `ec078d5ed0d1eda3c2b009b92b3575da57f45cd2a3bbaa2ceca1154544184b9c`
- Checkpoint-kohärente Seed-Tabelle SHA-256:
  `7b975744db612cf1f2cf87f12f56baa94e1a0263125b8ad43401f2c5a3d0d027`

## 2. Datensatz, Splits und Beobachtungsfenster

Das aktive Dataset enthält 214 Sequenzen von 25 Teilnehmenden. Der
eingefrorene Split ist sequenz- und teilnehmendendisjunkt:

| Split | Sequenzen | Fenster | Teilnehmende |
|---|---:|---:|---:|
| Training | 170 | 15.189 | 19 |
| Validation | 21 | 1.978 | 3 |
| Test | 23 | 2.199 | 3 |

`Test` umfasst sechs Trainingssequenzen und wurde am 26.08.2026 vom
Projektverfasser als echtes Teilnehmerpseudonym bestätigt. Damit bleiben 214
Sequenzen und 25 Teilnehmende korrekt; Dataset, Splits und Ergebnisse ändern
sich nicht. Die Auflösung ist im datierten
[Provenienzvermerk](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md)
dokumentiert.

Die empirische mediane Abtastperiode beträgt `0,033333 s`, entsprechend
`30,0003 Hz`; der Interquartilsabstand beträgt `0,000001 s`. Ein akzeptiertes
60-Sample-Fenster überspannt 59 Intervalle und damit im Median `1,966667 s`
(IQR `0,000001 s`). Das Ziel liegt eine Sekunde nach dem Fensterendpunkt.

## 3. Hauptvergleich der Intentionserkennung

| Methode | Test-Accuracy | Macro-F1 | Continue F1 | Fetch F1 | Handover F1 |
|---|---:|---:|---:|---:|---:|
| Majority (`continue`) | 0,7058 | 0,2758 | 0,8275 | 0,0000 | 0,0000 |
| Elapsed time since START, logistisch | 0,7294 | 0,4660 | 0,8742 | 0,0000 | 0,5239 |
| Letzter kausaler Sensorframe, logistisch | 0,8558 | 0,7884 | 0,9156 | 0,7355 | 0,7141 |
| MLP | 0,8715 ± 0,0147 | 0,8225 ± 0,0184 | 0,9210 ± 0,0089 | 0,7746 ± 0,0292 | 0,7719 ± 0,0190 |
| GRU | 0,8564 ± 0,0129 | 0,8027 ± 0,0182 | 0,9119 ± 0,0069 | 0,7139 ± 0,0164 | 0,7824 ± 0,0450 |
| Transformer | 0,8604 ± 0,0092 | 0,8069 ± 0,0141 | 0,9140 ± 0,0041 | 0,7226 ± 0,0274 | 0,7841 ± 0,0332 |
| Residual Transformer, Current Gate | **0,8731 ± 0,0165** | **0,8280 ± 0,0152** | 0,9197 ± 0,0158 | 0,7593 ± 0,0182 | **0,8051 ± 0,0122** |

Das Residual-Hauptmodell erreicht den höchsten mittleren Macro-F1 unter den
vier sensorbasierten neuronalen Hauptarchitekturen. Der Abstand zum MLP ist
mit 0,0056 klein gegenüber der Seed-Streuung und wird daher nicht als
statistisch gesicherte Überlegenheit interpretiert. Gegenüber der kausalen
Last-Frame-Logistik beträgt der deskriptive Abstand 0,0397 Macro-F1. Die
Elapsed-Time-Kontrolle zeigt, dass zeitlicher Versuchsablauf allein einen Teil,
aber nicht annähernd die volle Modellleistung erklärt.

## 4. Architektur-, Modalitäts- und Visual-Ablationen

### 4.1 Architektur und Auxiliary Pose Loss

| Variante | Test-Accuracy | Intent Macro-F1 |
|---|---:|---:|
| Hierarchical + Current Gate (Hauptmodell) | 0,8731 ± 0,0165 | 0,8280 ± 0,0152 |
| Hierarchical + Simple Fusion | 0,8704 ± 0,0133 | 0,8215 ± 0,0163 |
| Hierarchical + Modality Gate | 0,8812 ± 0,0105 | 0,8315 ± 0,0075 |
| Flat Intention Head | 0,8666 ± 0,0117 | 0,8193 ± 0,0148 |
| Ohne Future-Pose-Auxiliary-Loss | 0,8707 ± 0,0090 | 0,8267 ± 0,0104 |

Die beobachteten Unterschiede sind klein und überlappen mit der Seed-Streuung.
Der Current-Gate-Residual bleibt das vorab festgelegte Primärmodell; der
Testsplit wird nicht nachträglich zur Auswahl einer anderen Fusion verwendet.

### 4.2 Sensorablationen

| Variante | Test-Accuracy | Intent Macro-F1 | Δ Macro-F1 zum Hauptmodell |
|---|---:|---:|---:|
| Volles Hauptmodell | 0,8731 ± 0,0165 | 0,8280 ± 0,0152 | 0,0000 |
| Ohne Gaze | 0,8584 ± 0,0051 | 0,7995 ± 0,0145 | −0,0286 |
| Ohne Hände | 0,8152 ± 0,0250 | 0,7344 ± 0,0301 | −0,0937 |
| Ohne Objektfeatures | 0,8898 ± 0,0118 | 0,8465 ± 0,0152 | +0,0184 |
| Ohne VIO | 0,8543 ± 0,0123 | 0,7935 ± 0,0173 | −0,0345 |

Handfeatures liefern den größten beobachteten positiven Beitrag. Das bessere
Testergebnis ohne Objektfeatures ist ausschließlich deskriptiv; es darf nach
Sichtung des Tests nicht zur nachträglichen Modellwahl verwendet werden. Die
Objektinformation besteht aus einer eigenen Modalität geometrischer und
gaze-relationaler Features, nicht aus expliziten Transformer-Object-Tokens.

### 4.3 Visueller Kontext

| Variante | Test-Accuracy | Intent Macro-F1 |
|---|---:|---:|
| Sensor, Current Gate | 0,8731 ± 0,0165 | 0,8280 ± 0,0152 |
| Sensor + frozen Random Control | 0,8719 ± 0,0150 | 0,8135 ± 0,0113 |
| Sensor + korrigiertes CLIP, Current Gate | 0,9006 ± 0,0098 | 0,8465 ± 0,0111 |
| Sensor + korrigiertes CLIP, Modality Gate | 0,9065 ± 0,0069 | 0,8630 ± 0,0107 |

Diese Zeilen sind die vollständig vorgegebenen Matrixvarianten. Der höchste
beobachtete Testwert wird berichtet, aber nicht post hoc zum neuen Primärmodell
erklärt. Das eingefrorene Primärziel und dessen Systemkaskade bleiben an
`residual_current_gate` gebunden.

## 5. Fixed Test, LOPO und Receiving-Hand-Interpretation

Auf dem festen Testsplit erreicht das Residual-Hauptmodell einen
Receiving-Hand-Macro-F1 von `0,9477 ± 0,0132` und eine Accuracy von
`0,9583 ± 0,0100`. Diese Fenstermetriken über drei Test-Teilnehmende sind von
der participant-balancierten LOPO-Auswertung zu trennen:

| LOPO-Metrik, Seed 42 | Participant-balancierter Schätzer |
|---|---:|
| Intention Accuracy | 0,8629 |
| Intention Macro-F1 | 0,8152 |
| Hand, feste zwei Klassen / alle 25 Teilnehmenden | 0,6011 |
| Hand, nur unterstützte Klassen / alle 25 Teilnehmenden | 0,9579 |
| Hand, feste zwei Klassen / 7 Mixed-Hand-Teilnehmende | **0,8723** |

Die feste Zwei-Klassen-Sicht bestraft Single-Hand-Participants durch eine
nicht unterstützte Klasse; die Supported-Class-Sicht misst dagegen nicht die
Left-vs-Right-Diskrimination bei diesen Personen. Der Mixed-Hand-Wert ist
daher die wichtigste echte Hand-Diskriminationsanalyse, ersetzt die beiden
anderen transparenten Sichten aber nicht.

Participant und Receiving Hand sind stark gekoppelt: Cramérs V beträgt
`0,7620`, ein Participant-Majority-Hand-Prädiktor erreicht `0,8505`, und 72 %
der Teilnehmenden enthalten nur eine Handklasse. Hohe Handklassifikation kann
somit teilweise participant-spezifische Präferenz oder Bewegung widerspiegeln.

## 6. Primäre Receiving-Wrist-Pose bei t + 1 s

Die drei Oracle-Hand-Methoden werden auf exakt derselben fairen Kohorte von
213/217 gültigen t+1-Zielen verglichen. Die End-to-End-Zeile verwendet die
vorhergesagte Receiving-Hand und hat daher eine kleinere ausführbare Kohorte;
sie ist nicht paarweise direkt mit den Oracle-Zeilen gleichzusetzen.

| Methode / Handkontext | Position MAE | Position RMSE | Orientation Error | Samples | Coverage |
|---|---:|---:|---:|---:|---:|
| Persistence, Ground-Truth-Hand | **14,804 cm** | 22,308 cm | 42,049° | 213 | 0,9816 |
| Constant Velocity, Ground-Truth-Hand | 33,490 cm | 47,923 cm | 42,049° | 213 | 0,9816 |
| Learned, Ground-Truth-Hand | 14,903 ± 0,308 cm | **20,067 ± 0,335 cm** | **41,173 ± 0,900°** | 213 | 0,9816 |
| Learned, vorhergesagte Hand | 16,545 ± 0,308 cm | 21,297 ± 0,315 cm | 48,046 ± 1,683° | 181 | 0,8341 |

Das gelernte Modell verbessert die Last-Observation-Persistence beim mittleren
Positionsfehler nicht konsistent (`+0,099 cm` statt einer Verbesserung), hat
aber auf derselben Kohorte einen niedrigeren RMSE und geringfügig niedrigeren
Orientierungsfehler. Constant Velocity ist deutlich schlechter. Die
Receiving-Hand-Fehler erhöhen im ausführbaren End-to-End-Fall den Posefehler
und reduzieren die Coverage.

## 7. Systemkaskade Intention → Hand → t+1-Pose

Die Kaskade nutzt dieselben 217 Ground-Truth-Handover-Fenster mit gültigem
t+1-Receiving-Wrist-Ziel. `Success@τ` verlangt gleichzeitig eine korrekte
Handover-Intention, die korrekte Receiving-Hand und einen strikt kleineren
Positionsfehler als τ.

| Systemstufe, Residual-Hauptmodell | Erfolgsrate |
|---|---:|
| Handover korrekt | 0,8986 ± 0,0092 |
| + Receiving-Hand korrekt | 0,8848 ± 0,0092 |
| + Pose < 20 cm | 0,6406 ± 0,0092 |
| + Pose < 15 cm | 0,5899 ± 0,0046 |
| + Pose < 10 cm | 0,5207 ± 0,0122 |
| + Pose < 5 cm | 0,3318 ± 0,0997 |

Die größte zusätzliche Reduktion entsteht in diesem Schwellenbereich durch
die Positionsgenauigkeit, nicht durch die Handklassifikation.

## 8. Konfusionsmatrizen des repräsentativen Checkpoints

Für eine einzelne, ausführbare Darstellung wird der ausschließlich über
Validation gewählte Residual-Checkpoint von Seed 44 verwendet (Epoche 10,
SHA-256
`9019ecb1d67ceae34fb6f47f17bc4cd8ce5661b7f889c0d0678d87ee8cdf3551`).
Die Seed-Wahl hat keine Testmetrik gelesen. Zeilen sind Ground Truth, Spalten
sind Vorhersagen. Die Matrizen werden nicht über Seeds summiert, damit jedes
Testfenster nur einmal erscheint.

Intention (`continue`, `fetch`, `handover`):

| Ground Truth \ Prediction | Continue | Fetch | Handover |
|---|---:|---:|---:|
| Continue | 1.303 | 153 | 96 |
| Fetch | 26 | 308 | 9 |
| Handover | 9 | 27 | 268 |

Receiving Hand (`left`, `right`; nur Handover-Fenster):

| Ground Truth \ Prediction | Left | Right |
|---|---:|---:|
| Left | 74 | 16 |
| Right | 0 | 214 |

Dieser Einzelcheckpoint erreicht `0,8116` Intention-Macro-F1 und `0,9332`
Receiving-Hand-Macro-F1. Für den Hauptvergleich bleiben die Drei-Seed-Werte
aus den vorherigen Abschnitten maßgeblich.

## 9. Pose-Loss-Diagnose und qualitative Evidenz

Die finalen Train-Position-Errors der drei Primärläufe liegen bei 8,954,
9,278 und 9,203 cm. Damit ist der in der Checkliste definierte
14–15-cm-Underfitting-Trigger nicht erfüllt. Bei Seeds 42 und 43 liegt das
Validation-Pose-Optimum nach dem `best_intention`-Checkpoint; bei Seed 44
fallen beide Optima zusammen. Ein neuer `normalized_smooth_l1`-Sensitivity-Run
ist durch die vorhandenen Lernkurven nicht begründet und wird nicht gestartet.

Die qualitative Pipeline verwendet denselben validation-selektierten
Seed-44-Checkpoint und Project-Aria-Device-Time. Für jede Sequenz wurde der
einzige terminale VRS-RGB-Record ohne MP4-Gegenstück explizit ausgeschlossen;
es gibt kein Zukunfts-Matching und kein pauschales Abschneiden per
`min(len(mp4), len(vrs))`.

| Fall | Sequenz | Intention | Hand | Positionsfehler |
|---|---|---|---|---:|
| Good | `Jona_7_20260616_182214` | Handover korrekt | Right korrekt | 0,778 cm |
| Typical | `Edu_3_20260604_170622` | Handover korrekt | Right korrekt | 8,437 cm |
| Failure | `Mona_6_20260624_123930` | Handover → Fetch | Left → Right | 54,784 cm |

Zu allen drei Fällen existieren ein Still und ein H.264-Overlay-Video. Die
Overlays zeigen keine behauptete 3D-Projektion in das RGB-Bild, sondern eine
separate Robot-Frame-Darstellung.

## 10. Separates sekundäres Endpose-Experiment

Die terminale Endpose ist **nicht** das primäre t+1-Ziel. Im strikt
zukunftsgerichteten Teil des sekundären Experiments beträgt der mittlere
Positionsfehler `19,678 ± 0,982 cm` mit Ground-Truth-Hand und
`19,808 ± 0,520 cm` end-to-end; Persistence erreicht `18,251 cm` auf der
gemeinsamen Kohorte von 201 Samples. Auch hier verbessert das gelernte Modell
Persistence nicht. Ergebnisse mit bereits teilweise beobachteter Zielpose
sind nur diagnostisch und werden nicht mit dem Pure-Future-Ergebnis gepoolt.

## 11. Methodische Claims und Offline-/Deployment-Trennung

Die korrekte Aufgabenbeschreibung lautet:

> Current assistive intention estimation from a temporal history of
> multimodal observations, jointly with receiving-hand classification and
> 6-DoF receiving-wrist prediction one second into the future.

Die zeitlichen Sensorbeobachtungen werden kausal rückwärts ausgerichtet. Die
statische Transformation vom World- in den Robot-Frame wird jedoch offline
über die vollständige Sequenz geschätzt. Der zulässige Claim lautet daher
„causal temporal observation alignment with offline static robot-frame
calibration“, nicht „fully online causal pipeline“.

Die Offline-Dataset-Erzeugung nutzt strengere Altersgrenzen als die
Live-Inferenz. Diese Deployment-Verschiebung wird als Limitation behandelt.
Alle Zahlen in diesem Dokument sind Offline-Fenster-, gruppierte oder
retrospektive Modellmetriken. Ohne checkpoint-gebundenen Replay-Report dürfen
sie nicht als `raw`, `stable` oder `actionable` Deployment-Performance
bezeichnet werden.

Die genaue Implementierung von Markern, chordaler Rotationsmittelung mittels
`scipy.spatial.transform.Rotation.mean()`, Anchor-Fallback,
`robot_anchor_interpolated`, `closed_loop_trajectory.csv` und dem fest
kalibrierten Gaze-Ursprung ist in
[`raw_data_and_processing.md`](raw_data_and_processing.md) beschrieben.

## 12. Related Work: He et al.

He, Zhang und Stienen (2025),
[“Gaze-Guided 3D Hand Motion Prediction for Detecting Intent in Egocentric
Grasping Tasks”](https://arxiv.org/abs/2504.01024), prognostizieren aus
historischer Handbewegung, Gaze und Objektkontext zukünftige Sequenzen beider
3D-Handposen bzw. Gelenkpositionen. Sie kombinieren einen VQ-VAE mit einem
autoregressiven generativen Transformer und zeigen insbesondere bei kurzer
Historie einen Nutzen von Gaze.

Die vorliegende Arbeit bearbeitet eine andere, komplementäre Aufgabe: Sie
schätzt die aktuelle assistive Intention in drei Klassen, klassifiziert die
Receiving-Hand und prognostiziert deren 6-DoF-Wrist-Pose im Robot-Frame bei
einem festen Horizont von einer Sekunde. Zusätzlich werden einfache
kinematische Baselines und eine vollständige Intention→Hand→Pose-Kaskade
ausgewertet. Der direkte Vergleich unterstreicht, dass komplexe
Handbewegungsmodelle stets gegen Persistence und Constant Velocity geprüft
werden sollten.

## 13. Limitationen

- Der feste Testsplit enthält nur drei Teilnehmende. Drei Trainingsseeds
  quantifizieren Optimierungsstreuung, nicht Unsicherheit über neue Personen;
  LOPO liefert deshalb separate participant-balancierte Evidenz.
- Participant–Hand-Confounding und viele Single-Hand-Participants begrenzen
  die Interpretation der Handklassifikation.
- Die Robot-Frame-Kalibrierung ist statisch und sequenzweit offline, obwohl
  die zeitliche Beobachtungsausrichtung kausal ist.
- Objektkontext ist eine Featuremodalität, keine Menge expliziter Object
  Tokens mit Cross-Attention.
- Die Live-Inferenz akzeptiert teilweise ältere Sensorwerte als der
  Offline-Build und kann dadurch einem Distribution Shift unterliegen.
- Das gelernte t+1-Posemodell schlägt Persistence beim mittleren
  Positionsfehler nicht konsistent; Receiving-Hand-Fehler verringern zudem
  die End-to-End-Coverage.
- Fixed-Test-, LOPO-, Offline- und Deployment-Metriken sind unterschiedliche
  Evidenzebenen und dürfen nicht zusammengelegt werden.

## 14. Abstract-fertiger Ergebnisabsatz

> On the active causally aligned dataset of 214 sequences from 25
> participants, the residual multimodal model achieved an intention macro-F1
> of 0.828 ± 0.015 and a receiving-hand macro-F1 of 0.948 ± 0.013 on the
> participant-disjoint fixed test split. Participant-balanced leave-one-out
> evaluation yielded an intention macro-F1 of 0.815; receiving-hand macro-F1
> was 0.872 among the seven participants with both hand classes. For the
> primary one-second receiving-wrist target, the learned predictor achieved
> 14.903 ± 0.308 cm using the ground-truth receiving hand, but did not
> consistently improve upon last-observation persistence at 14.804 cm.
> End-to-end predicted-hand pose error was 16.545 ± 0.308 cm with 0.834
> coverage. The complete cascade reached Success@20 cm of 0.641 ± 0.009 and
> Success@10 cm of 0.521 ± 0.012. These findings support multimodal intention
> and receiving-hand estimation while identifying future wrist-pose
> generalisation as the main remaining limitation.

Eine eigentliche Abstract-Quelldatei ist in diesem Repository nicht vorhanden;
dieser Absatz ist daher die übertragbare Quelle.

## 15. Reproduzierbare Artefakte

- Autoritative Zusammenfassung:
  [`FINAL_MATRIX_SUMMARY.md`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/FINAL_MATRIX_SUMMARY.md)
- Maschinenlesbare Seed- und Aggregattabellen:
  [`final_seed_results.json`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/final_seed_results.json) und
  [`final_seed_aggregates.json`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/final_seed_aggregates.json)
- Summary-Manifest:
  [`summary_artifact_manifest.json`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/summary_artifact_manifest.json)
- Group-CV-Auswertung:
  [`group_cv_summary.json`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/summary_v2/group_cv_summary.json)
- Kausale Intention-Baselines:
  [`INTENTION_BASELINES.md`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/causal_intention_baselines_v1/INTENTION_BASELINES.md)
- Sampling-Audit:
  [`SAMPLING_WINDOW_AUDIT.md`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/sampling_window_audit_v1/SAMPLING_WINDOW_AUDIT.md)
- Pose-Lernkurven-Diagnose:
  [`POSE_LEARNING_DIAGNOSIS.md`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/pose_learning_diagnosis_v1/POSE_LEARNING_DIAGNOSIS.md)
- Qualitative Evidenz:
  [`qualitative_artifact_manifest.json`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/qualitative/qualitative_artifact_manifest.json)
- Identitätsprovenienz:
  [`IDENTITY_PROVENANCE_RESOLUTION_20260826.md`](../Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md)
