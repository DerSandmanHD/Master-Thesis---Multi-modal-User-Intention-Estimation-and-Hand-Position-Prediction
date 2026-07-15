# Rohdaten, Datenformate und Verarbeitungspipeline

## Zweck und Geltungsbereich

Dieses Dokument beschreibt den aktuellen, im Repository implementierten Datenfluss der Masterarbeit **Multi-modal User Intention Estimation and Hand Position Prediction**. Es erklärt:

- welche Daten tatsächlich aufgezeichnet oder durch Project Aria/MPS bereitgestellt werden,
- wie Gaze, Handtracking, SLAM, RGB, Audio sowie AprilTag- und ArUco-Marker strukturiert sind,
- welche Einheiten, Koordinatensysteme, Gültigkeitswerte und Qualitätsmetriken verwendet werden,
- wie die asynchronen Modalitäten zeitlich zusammengeführt werden,
- wie robot-relative Features und zukünftige Handposen berechnet werden,
- welche Spalten im Master-Dataset landen,
- welche dieser Spalten das aktuelle Trainingsskript tatsächlich verwendet,
- welche Informationen derzeit ausdrücklich **nicht** als Modellinput verwendet werden.

Die Beschreibung basiert auf dem produktiven Code in:

- [`Code/extract_multimodal_data.py`](../Code/extract_multimodal_data.py)
- [`Code/detect_tags.py`](../Code/detect_tags.py)
- [`Code/speech_recognition_demo.py`](../Code/speech_recognition_demo.py)
- [`Code/apply_manual_reviews.py`](../Code/apply_manual_reviews.py)
- [`Code/dataset_qa.py`](../Code/dataset_qa.py)
- [`Code/build_master_dataset.py`](../Code/build_master_dataset.py)
- [`Training/data.py`](../Training/data.py)

Stand dieser Dokumentation: **14. Juli 2026**.

## 1. Wichtige Begriffsabgrenzung: Was bedeutet hier Rohdaten?

In der Pipeline werden mehrere Verarbeitungsebenen oft gemeinsam als "Daten" bezeichnet. Für eine wissenschaftlich saubere Beschreibung müssen sie getrennt werden.

| Ebene | Inhalt | Beispiel |
|---|---|---|
| Ebene 0: Aufzeichnung | Von der Aria-Brille gespeicherter VRS-Container mit zeitgestempelten Sensor- und On-Device-Signalen | `Data_vrs/<sequence_id>.vrs` |
| Ebene 1: Primärdaten/Exporte | Aus dem VRS gelesene oder durch MPS bzw. Computer Vision erzeugte Signale | Gaze-CSV, `hand_tracking_results.csv`, `closed_loop_trajectory.csv`, Marker-CSV |
| Ebene 2: Synchronisiertes Master-Dataset | Auf einer gemeinsamen Gaze-Zeitachse zusammengeführte Modalitäten plus Labels und abgeleitete Koordinaten | `<sequence_id>_master.csv` |
| Ebene 3: Trainingseingaben | Ausgewählte, normalisierte Features und zeitliche Fenster | Tensor der Form Zeit x Feature |

Damit ist beispielsweise eine robot-relative Handposition **keine Rohmessung**. Sie ist eine abgeleitete Größe aus Handtracking, SLAM, RGB-Kalibrierung und AprilTag 0.

Auch Gaze und Handtracking sind keine direkten Rohpixel. Sie sind bereits geschätzte Machine-Perception-Ausgaben, die aus den Eye-Tracking- beziehungsweise SLAM-Kamerabildern erzeugt wurden. Der aktuelle Code liest diese Schätzungen und verarbeitet sie weiter.

## 2. Verzeichnis- und Dateistruktur pro Sequenz

Eine Sequenz besitzt eine vollständige ID nach dem Muster:

```text
<Teilnehmer>_<Versuchsnummer>_<YYYYMMDD>_<HHMMSS>
```

Beispiel:

```text
Jona_6_20260616_182111
```

Die produktive Datenstruktur ist logisch wie folgt aufgebaut:

```text
Data_collection/
├── Data_vrs/
│   ├── <sequence_id>.vrs
│   ├── mps_<sequence_id>_vrs/
│   │   ├── hand_tracking/
│   │   │   ├── hand_tracking_results.csv
│   │   │   └── summary.json
│   │   └── slam/
│   │       └── closed_loop_trajectory.csv
│   ├── timestamps_summary.json
│   ├── timestamps_summary.reviewed.json
│   ├── timestamps_debug.json
│   └── debug_audio/
│       └── <sequence_id>.wav
├── Data_mp4/
│   └── <sequence_id>.mp4
├── aruco_poses_<sequence_id>.csv
├── manual_timestamp_review.csv
├── dataset_manifest.csv
├── dataset_qa_report.json
└── master_datasets/
    ├── <sequence_id>_master.csv
    └── <sequence_id>_master_report.json
```

Auf dem Cluster können die Marker-CSVs zusätzlich unter `Data_collection/Aruco_CSV/` liegen und über Symlinks im Wurzelverzeichnis erreichbar sein. Der Master-Builder erwartet standardmäßig den direkten Pfad `Data_collection/aruco_poses_<sequence_id>.csv`.

## 3. Einheiten, Zeitbasis und Transformationsnotation

### 3.1 Verwendete Einheiten

| Suffix oder Feldtyp | Einheit | Bedeutung |
|---|---:|---|
| `*_timestamp_ns` | Nanosekunden | Zeitstempel im Device-Time-Domain |
| `tracking_timestamp_us` | Mikrosekunden | MPS-Zeitstempel im Device-Time-Domain |
| `*_time_offset_ms` | Millisekunden | Differenz zwischen gemergtem Sample und Master-Zeitpunkt |
| `*_x_m`, `*_y_m`, `*_z_m`, `tx_*`, `ty_*`, `tz_*` | Meter | Position oder Translation |
| `*_yaw_rad`, `*_pitch_rad`, `rvec_*_rad` | Radiant | Winkel oder Rodrigues-Rotationsvektor |
| `angular_velocity_*` | Radiant pro Sekunde | Winkelgeschwindigkeit |
| `linear_velocity_*` | Meter pro Sekunde | lineare Geschwindigkeit |
| `gravity_*` | Meter pro Quadratsekunde | Gravitationsvektor |
| `reprojection_error_px` | Pixel | mittlerer geometrischer Projektionsfehler |
| `marker_area_px2` | Quadratpixel | projizierte Markerfläche im Bild |
| Quaternion `qx,qy,qz,qw` | dimensionslos | normierte 3D-Orientierung |

### 3.2 Gemeinsame Zeitbasis

Für die Analyse einer einzelnen Brille wird `DEVICE_TIME` verwendet. Alle relevanten Aria-Sensoren einer Aufnahme teilen diese geräteeigene Zeitbasis. Das ist die Voraussetzung dafür, Gaze, RGB, Audio, Handtracking und SLAM zeitlich zu vergleichen.

Die MPS-Dateien speichern `tracking_timestamp_us` in Mikrosekunden. Der Master-Builder rechnet diese Werte ohne Offset in Nanosekunden um:

```text
timestamp_ns = tracking_timestamp_us * 1000
```

Gaze- und RGB-Zeitstempel werden bereits als Nanosekunden gelesen. Der Gaze-Code reduziert den zugrunde liegenden `timedelta` auf Mikrosekundenauflösung und multipliziert anschließend mit 1000. Die Spalte heißt daher zwar `timestamp_ns`, besitzt in diesem Export praktisch Mikrosekundenauflösung.

Offizielle Hintergrundinformation: [Project Aria Timestamp Definitions](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs).

### 3.3 SE(3)-Notation

Eine starre 3D-Transformation wird als homogene 4x4-Matrix geschrieben:

```text
T_A_B = [ R_A_B  t_A_B ]
        [ 0 0 0      1 ]
```

`T_A_B` transformiert einen Punkt aus Frame `B` in Frame `A`:

```text
p_A = T_A_B * p_B
```

Transformationsketten werden von rechts nach links angewendet:

```text
T_A_C = T_A_B * T_B_C
```

Alle Quaternionen werden in der Reihenfolge **x, y, z, w** verarbeitet. Die Pipeline verwendet `scipy.spatial.transform.Rotation`. Beim Zurückwandeln einer Rotationsmatrix in ein Quaternion wird das Vorzeichen so gewählt, dass `qw >= 0` ist. `q` und `-q` repräsentieren dieselbe Rotation; die Vorzeichenkonvention verhindert unnötige Sprünge.

Offizielle Hintergrundinformation: [Project Aria 3D Coordinate Frame Conventions](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention).

### 3.4 In der Pipeline vorkommende Koordinatenframes

| Frame | Ursprung und Achsen | Verwendung |
|---|---|---|
| CPF | Ursprung nominell zwischen den Augen; x nach links, y nach oben, z in Blickrichtung nach vorne | ursprüngliche Gaze-Winkel und Gaze-Punkt |
| Device | Aria-Gen2-Referenzframe an der Brille; laut Gen2-Kalibrierung x nach rechts, y nach unten, z nach vorne | Handtracking und gemeinsame Sensorreferenz |
| lineare RGB-Kamera | optisches Zentrum; in der linearen OpenCV-Projektion x entlang der Bildbreite nach rechts, y entlang der Bildhöhe nach unten, z vor die Kamera | Markerdetektion und PnP |
| SLAM World | sequenzspezifischer, gravitationsausgerichteter Frame; MPS gibt Gravitation als ungefähr `[0,0,-9.81]` aus | globale Device-, Hand-, Objekt- und Ankerpose innerhalb einer Sequenz |
| AprilTag-0/Robot | Ursprung und Orientierung des gedruckten Markers 0 | gemeinsame robot-relative Repräsentation |
| Wrist | Ursprung am Handgelenk, Orientierung aus der MPS-6-DoF-Handpose | aktuelle und zukünftige Handpose |

Für die Markerobjektpunkte verwendet der Code die Eckreihenfolge oben links, oben rechts, unten rechts, unten links mit x nach rechts über den Marker und y nach oben. Die Marker-z-Achse ergibt sich aus dem rechtshändigen System. Entscheidend für alle Verkettungen sind dennoch die kalibrierten Transformationsmatrizen, nicht eine manuelle Achsenannahme.

Die CPF- und Device-Achsen zeigen seitlich und vertikal in entgegengesetzte Richtungen. Gaze-Werte dürfen deshalb nicht allein durch Umbenennen von CPF nach Device übernommen werden. Die Pipeline wendet dafür ausdrücklich `T_device_cpf` an.

## 4. VRS als ursprünglicher Aufzeichnungscontainer

### 4.1 Inhalt

Eine `.vrs`-Datei ist kein gewöhnliches Video. Sie ist ein zeitlich sortierter Container für mehrere Sensorstreams. In der aktuellen Pipeline werden daraus mindestens folgende Informationen genutzt:

| Signal | Zugriff in der Pipeline | Zweck |
|---|---|---|
| RGB-Kamera | Stream-ID `214-1`, Label `camera-rgb` | Markerdetektion und MP4-Review |
| Eye-Gaze | Stream-ID `373-1` | Master-Zeitachse und Gaze-Features |
| Mikrofon | Stream-Label `mic` | Erkennung der Phasenkommandos |
| Gerätekalibrierung | `get_device_calibration()` | Transformation CPF zu Device und RGB-Kamera zu Device |

Handtracking und Closed-Loop-SLAM werden im Master-Builder nicht direkt aus Rohbildern neu berechnet. Er liest die bereits erzeugten MPS-CSVs.

Die Stream-ID `373-1` ist der im Repository für die vorhandenen Aufnahmen fest konfigurierte Gaze-Stream. Neuere öffentliche Gen2-Tabellen nennen für bestimmte Profile andere gemeinsame Stream-IDs. Die ID sollte daher nicht allein aufgrund einer allgemeinen Dokumentation geändert werden; maßgeblich ist das Stream-Inventar der tatsächlich aufgenommenen VRS-Dateien.

### 4.2 RGB und MP4

Die RGB-Daten bleiben im VRS die geometrisch relevante Quelle. Für Review-Zwecke wird zusätzlich mit `vrs_to_mp4` ein MP4 erzeugt:

```text
Data_collection/Data_mp4/<sequence_id>.mp4
```

Das MP4 dient aktuell:

- der manuellen Kontrolle der gesprochenen Phasenmarker,
- der manuellen Auswahl der Zielobjekt-ID,
- der Annotation der empfangenden Hand,
- der visuellen technischen Review.

**Wichtig:** Das MP4 oder ein visueller Encoder wie CLIP ist im aktuellen Featureprofil `multimodal_robot_frame_v1` noch **kein Eingabefeature des neuronalen Netzes**. RGB wird zur Markerextraktion und zur manuellen Qualitätssicherung genutzt, aber nicht als gelernter Bildfeature-Vektor in das Baseline-Modell eingespeist.

## 5. Gaze-Daten

### 5.1 Herkunft

[`Code/extract_multimodal_data.py`](../Code/extract_multimodal_data.py) liest für jedes Eye-Gaze-Sample:

```python
eye_data = provider.get_eye_gaze_data_by_index(StreamId("373-1"), index)
```

Zusätzlich werden zwei statische Kalibriertransformationen aus dem VRS geladen:

```text
T_device_cpf
T_device_camera_rgb
```

`T_device_cpf` transformiert Gaze-Größen aus dem Central Pupil Frame in den Device-Frame. `T_device_camera_rgb` wird später für Marker verwendet.

### 5.2 Central Pupil Frame (CPF)

Der CPF-Ursprung liegt nominell zwischen den Augen. Der Blick wird zunächst relativ zu diesem Frame beschrieben. In der Project-Aria-Konvention zeigt die CPF-z-Achse nach vorne, x seitlich und y vertikal. Die konkrete Extrinsik `T_device_cpf` aus der Brillenkalibrierung ist für die Umrechnung maßgeblich.

### 5.3 Exportierte Gaze-Spalten

Die Datei lautet:

```text
Data_collection/gaze_<sequence_id>.csv
```

Sie enthält genau folgende Gruppen:

| Spalte | Einheit | Bedeutung |
|---|---:|---|
| `timestamp_ns` | ns | Trackingzeit des Gaze-Samples in Device Time |
| `gaze_valid` | 0/1 | `combined_gaze_valid` aus dem Eye-Gaze-Objekt |
| `gaze_yaw_rad` | rad | kombinierter horizontaler Blickwinkel im CPF |
| `gaze_pitch_rad` | rad | kombinierter vertikaler Blickwinkel im CPF |
| `gaze_depth_m` | m | geschätzte Tiefe des Blickpunkts, sofern endlich und > 0 |
| `gaze_point_cpf_{x,y,z}_m` | m | 3D-Blickpunkt im CPF |
| `gaze_direction_cpf_{x,y,z}` | 1 | normierter Blickrichtungsvektor im CPF |
| `gaze_origin_device_{x,y,z}_m` | m | CPF-Ursprung in Device-Koordinaten |
| `gaze_point_device_{x,y,z}_m` | m | transformierter Blickpunkt im Device-Frame |
| `gaze_direction_device_{x,y,z}` | 1 | transformierter, normierter Blickvektor im Device-Frame |
| `left_gaze_yaw_rad` | rad | horizontaler Winkel des linken Auges |
| `right_gaze_yaw_rad` | rad | horizontaler Winkel des rechten Auges |
| `left_gaze_pitch_rad` | rad | vertikaler Winkel des linken Auges |
| `right_gaze_pitch_rad` | rad | vertikaler Winkel des rechten Auges |

### 5.4 Berechnung des 3D-Blicks

Für ein gültiges Sample werden `yaw`, `pitch` und `depth` ausgelesen. Der CPF-Blickpunkt wird mit der Project-Aria-Hilfsfunktion berechnet:

```text
p_cpf = get_eyegaze_point_at_depth(yaw, pitch, depth_for_direction)
d_cpf = p_cpf / ||p_cpf||
```

Danach erfolgt die Transformation in den Device-Frame:

```text
o_device = translation(T_device_cpf)
p_device = R_device_cpf * p_cpf + o_device
d_device = normalize(R_device_cpf * d_cpf)
```

### 5.5 Sonderfall: Gültige Richtung, aber keine gültige Tiefe

Wenn `combined_gaze_valid=True`, die Tiefe aber nicht endlich oder <= 0 ist, setzt der Code intern für die Richtungsberechnung eine Ersatzdistanz von **1,0 m** ein:

```text
depth_for_direction = 1.0 m
```

In diesem Fall gilt:

- `gaze_depth_m` wird leer/`NaN`,
- der Richtungsvektor bleibt verwendbar,
- `gaze_point_cpf_*` und `gaze_point_device_*` sind ein Punkt in 1 m Entfernung entlang der Blickrichtung und **kein gemessener Fixationspunkt**.

Diese Unterscheidung ist für die Interpretation wichtig. Die Richtung kann valide sein, obwohl keine zuverlässige metrische Blicktiefe vorliegt.

### 5.6 Ungültige Gaze-Samples

Wenn `combined_gaze_valid=False` ist:

- wird `gaze_valid=0` gesetzt,
- alle übrigen Gaze-Features dieser Zeile bleiben leer/`NaN`,
- es wird keine zeitliche Interpolation der Gaze-Werte durchgeführt.

Im Training werden fehlende Werte später maskiert und nicht als echte Nullmessung interpretiert.

### 5.7 Aus Gaze und Objekten abgeleitete Größen

Für jeden sichtbaren Objektmarker `i` wird die Verbindung vom Gaze-Ursprung zum Markerzentrum berechnet:

```text
v_i = p_object_i_device - o_gaze_device
distance_i = ||v_i||
angle_i = arccos(clip(d_gaze_device dot normalize(v_i), -1, 1))
```

Daraus entstehen:

| Spalte | Einheit | Interpretation |
|---|---:|---|
| `aruco_<id>_gaze_angle_rad` | rad | Winkel zwischen Blickrichtung und Richtung zum Markerzentrum; kleiner bedeutet stärkere Ausrichtung |
| `aruco_<id>_gaze_distance_m` | m | euklidische Distanz vom CPF/Gaze-Ursprung zum Markerzentrum |

`gaze_distance_m` ist **nicht** der Abstand zwischen dem geschätzten 3D-Gaze-Punkt und dem Objekt. Es ist der Abstand vom Blickursprung zum Objektmarker. `gaze_angle_rad` ist ebenfalls kein binäres Fixationslabel. Die Pipeline setzt derzeit keinen festen Winkelgrenzwert für "angeschaut".

Offizielle Hintergrundinformation: [Project Aria Eye Gaze Data Format](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_eye_gaze) und [Aria Gen2 VRS Data Loading](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/tutorials/vrs_loading).

## 6. Handtracking-Daten

### 6.1 Herkunft und Datei

Der Master-Builder liest:

```text
Data_collection/Data_vrs/mps_<sequence_id>_vrs/
└── hand_tracking/hand_tracking_results.csv
```

Die Datei ist eine MPS-Handtracking-Ausgabe. Sie enthält pro Trackingzeitpunkt Ergebnisse für linke und rechte Hand im Device-Frame.

Offizielle Formatspezifikation: [Project Aria MPS Hand Tracking](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking).

### 6.2 Zeitstempel und Konfidenz

| Spalte | Einheit | Bedeutung |
|---|---:|---|
| `tracking_timestamp_us` | µs | Zeit des zugrunde liegenden SLAM-Kameraframes in Device Time |
| `left_tracking_confidence` | typischerweise -1 oder [0,1] | Konfidenz für die linke Hand |
| `right_tracking_confidence` | typischerweise -1 oder [0,1] | Konfidenz für die rechte Hand |

MPS verwendet `-1.0`, wenn eine Hand nicht vorhanden oder nicht getrackt ist. Der aktuelle Code definiert eine Hand jedoch allgemein als gültig, sobald:

```text
tracking_confidence > 0
```

Daraus werden die binären Felder erzeugt:

```text
hand_left_valid
hand_right_valid
```

Es wird derzeit kein strengerer Konfidenzgrenzwert wie 0,5 oder 0,8 angewendet. Die kontinuierlichen Konfidenzwerte bleiben zusätzlich als Modellfeatures erhalten.

### 6.3 21 Hand-Landmarks

Für jede Hand werden 21 dreidimensionale Punkte im Device-Frame gespeichert. Jede Landmark besitzt x-, y- und z-Koordinate in Metern:

```text
tx_<side>_landmark_<i>_device
ty_<side>_landmark_<i>_device
tz_<side>_landmark_<i>_device
```

Die offiziellen Landmark-IDs sind:

| ID | Anatomischer Punkt |
|---:|---|
| 0 | Daumenspitze |
| 1 | Zeigefingerspitze |
| 2 | Mittelfingerspitze |
| 3 | Ringfingerspitze |
| 4 | Kleinfingerspitze |
| 5 | Handgelenk |
| 6 | Daumen, intermediäres Gelenk |
| 7 | Daumen, distales Gelenk |
| 8 | Zeigefinger, proximales Gelenk |
| 9 | Zeigefinger, intermediäres Gelenk |
| 10 | Zeigefinger, distales Gelenk |
| 11 | Mittelfinger, proximales Gelenk |
| 12 | Mittelfinger, intermediäres Gelenk |
| 13 | Mittelfinger, distales Gelenk |
| 14 | Ringfinger, proximales Gelenk |
| 15 | Ringfinger, intermediäres Gelenk |
| 16 | Ringfinger, distales Gelenk |
| 17 | Kleinfinger, proximales Gelenk |
| 18 | Kleinfinger, intermediäres Gelenk |
| 19 | Kleinfinger, distales Gelenk |
| 20 | Handflächenzentrum |

Alle 21 Landmarks werden in das Master-Dataset übernommen. Das aktuelle Trainingsfeatureprofil verwendet sie jedoch **nicht direkt**. Für das Baseline-Modell werden nur Handgelenkpose, Trackingkonfidenz und Validitätsmasken ausgewählt.

### 6.4 Vollständige 6-DoF-Handgelenkpose

Für jede Hand enthält MPS eine starre Transformation vom Wrist-/Hand-Frame in den Device-Frame:

```text
T_device_wrist
```

Die zugehörigen Spalten sind:

| Spaltenmuster | Einheit | Bedeutung |
|---|---:|---|
| `tx_<side>_device_wrist` | m | x-Translation Wrist nach Device |
| `ty_<side>_device_wrist` | m | y-Translation Wrist nach Device |
| `tz_<side>_device_wrist` | m | z-Translation Wrist nach Device |
| `qx_<side>_device_wrist` | 1 | Quaternion x |
| `qy_<side>_device_wrist` | 1 | Quaternion y |
| `qz_<side>_device_wrist` | 1 | Quaternion z |
| `qw_<side>_device_wrist` | 1 | Quaternion w |

Nach dem Merge tragen diese Spalten das Präfix `hand_`, beispielsweise:

```text
hand_tx_left_device_wrist
hand_qw_right_device_wrist
```

### 6.5 Handflächen- und Handgelenknormalen

Für jede Seite werden zusätzlich Richtungsvektoren gespeichert:

```text
n{xyz}_<side>_palm_device
n{xyz}_<side>_wrist_device
```

Diese Vektoren beschreiben die Orientierung von Handfläche und Handgelenk im Device-Frame. Sie werden in den Master-Datensatz übernommen, sind aber im aktuellen Trainingsfeatureprofil nicht ausgewählt. Die Orientierung des Pose-Targets stammt stattdessen aus dem Wrist-Quaternion.

Die `n*`-Werte werden als Richtungskomponenten interpretiert. Der aktuelle Master-Builder normalisiert oder validiert ihre Länge nicht zusätzlich. Da sie nicht zum aktiven Featureprofil gehören, beeinflussen sie die aktuelle Baseline nicht.

### 6.6 Behandlung ungültiger Hände

Für jede Seite mit `tracking_confidence <= 0` setzt der Master-Builder alle räumlichen Spalten dieser Seite auf `NaN`. Damit werden alte oder numerisch vorhandene Koordinaten nicht versehentlich als gültige Handpose verwendet.

Es erfolgt **keine Interpolation** fehlender Handtracking-Samples. Beim zeitlichen Merge wird lediglich das nächste vorhandene Sample innerhalb der erlaubten Zeitdifferenz gewählt.

### 6.7 Transformation der Handpose

Aus `T_device_wrist` und der SLAM-Pose `T_world_device` wird die Weltpose berechnet:

```text
T_world_wrist = T_world_device * T_device_wrist
```

Aus der robot-relativen Device-Pose `T_robot_device` wird berechnet:

```text
T_robot_wrist = T_robot_device * T_device_wrist
```

Erzeugte Spaltengruppen pro Seite:

```text
<side>_wrist_world_{x,y,z}_m
<side>_wrist_world_q{x,y,z,w}

<side>_wrist_robot_{x,y,z}_m
<side>_wrist_robot_q{x,y,z,w}
```

Für Training und Future-Pose-Target ist insbesondere die robot-relative Wrist-Pose relevant.

## 7. SLAM- und Bewegungsdaten

### 7.1 Herkunft und Datei

Die Pipeline verwendet die MPS Closed-Loop-Trajektorie:

```text
Data_collection/Data_vrs/mps_<sequence_id>_vrs/
└── slam/closed_loop_trajectory.csv
```

Closed-Loop-SLAM liefert eine nachträglich optimierte, hochfrequente Trajektorie in einem gravitationsausgerichteten, aber pro Aufnahme beliebigen Weltkoordinatensystem.

Offizielle Formatspezifikation: [Project Aria MPS Closed-Loop Trajectory](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory).

### 7.2 Spalten der Closed-Loop-Trajektorie

| Spalte | Einheit | Bedeutung |
|---|---:|---|
| `graph_uid` | String | ID des SLAM-Weltframes |
| `tracking_timestamp_us` | µs | Device-Time-Zeitstempel |
| `utc_timestamp_ns` | ns | UTC-Zeit, falls vorhanden; sonst typischerweise -1 |
| `tx_world_device`, `ty_world_device`, `tz_world_device` | m | Device-Position im Weltframe |
| `qx_world_device`, `qy_world_device`, `qz_world_device`, `qw_world_device` | 1 | Device-Orientierung im Weltframe |
| `device_linear_velocity_{x,y,z}_device` | m/s | lineare Geschwindigkeit, im Device-Frame ausgedrückt |
| `angular_velocity_{x,y,z}_device` | rad/s | Winkelgeschwindigkeit im Device-Frame |
| `gravity_{x,y,z}_world` | m/s² | Gravitationsvektor im Weltframe |
| `quality_score` | [0,1] | SLAM-Qualität; größer bedeutet höhere Qualität |
| `geo_available` | bool/0/1 | Verfügbarkeit geographischer Lokalisierung |
| `{tx,ty,tz}_ecef_device` | m | optionale Position im ECEF-Frame |
| `{qx,qy,qz,qw}_ecef_device` | 1 | optionale Orientierung im ECEF-Frame |

Nach dem Laden erhalten alle Originalspalten das Präfix `slam_`, zum Beispiel:

```text
slam_tx_world_device
slam_quality_score
```

### 7.3 Bedeutung des Weltframes

`T_world_device` ist die Pose der Brille im SLAM-Weltframe:

```text
p_world = T_world_device * p_device
```

Dieser Weltframe ist:

- gravitationsausgerichtet,
- innerhalb einer Sequenz konsistent,
- ohne zusätzliche Registrierung **nicht sequenzübergreifend identisch**,
- nicht automatisch der Roboterbasisframe.

Weltkoordinaten verschiedener Aufnahmen dürfen daher nicht direkt als dieselbe absolute Umgebung interpretiert werden.

### 7.4 Welche SLAM-Größen trainiert werden

Das aktuelle Modell verwendet direkt:

```text
slam_device_linear_velocity_{x,y,z}_device
slam_angular_velocity_{x,y,z}_device
slam_quality_score
```

Die absolute SLAM-Pose wird nicht als direktes Modellfeature gewählt. Sie wird jedoch zwingend für die Umrechnung von Händen, Gaze und Objekten in den statischen Robotermarker-Frame verwendet.

Gravitation, UTC, ECEF und `graph_uid` bleiben im Master-Dataset, werden aber derzeit nicht als neuronale Eingabefeatures verwendet.

### 7.5 Qualitätsbehandlung

Die Spalte `slam_quality_score` wird als kontinuierliches Feature genutzt. Der Master-Builder verwirft ein zeitlich passendes SLAM-Sample derzeit nicht anhand eines festen Quality-Score-Grenzwerts. Die Voraussetzungen für einen Match sind:

- ein vorhandener SLAM-Datensatz,
- ein nächster Zeitstempel innerhalb von 5 ms,
- endliche Posewerte für Transformationen.

Ein niedriger `quality_score` führt also nicht automatisch zu `slam_valid=0`. Das Modell bekommt den Score, um die Qualität selbst berücksichtigen zu können.

## 8. AprilTag- und ArUco-Marker

### 8.1 Markerfamilien, IDs, Rollen und physische Größen

Die aktive Markerextraktion unterstützt zwei Familien:

| Familie | IDs | Kantenlänge | Rolle |
|---|---:|---:|---|
| AprilTag `36h11` | 0 | 0,10 m | Roboteranker |
| AprilTag `36h11` | 1 bis 5 | 0,08 m | Tisch-/Umgebungsanker |
| ArUco `4x4_50` | 6 bis 14 | 0,05 m | Objektmarker/Kandidatenobjekte |

Die Größen sind im Code fest hinterlegt. Eine falsche physische Markergröße skaliert die mit PnP geschätzten Translationen systematisch falsch.

### 8.2 RGB-Vorverarbeitung

Für jeden RGB-Frame aus Stream `214-1` wird:

1. die werkseitige RGB-Kalibrierung geladen,
2. aus dem Mittelwert der beiden Brennweiten eine lineare Kamerakalibrierung gleicher Bildgröße erzeugt,
3. das Fisheye-/verzerrte RGB-Bild in dieses lineare Kameramodell entzerrt,
4. das Bild in Graustufen konvertiert,
5. separat nach AprilTag `36h11` und ArUco `4x4_50` gesucht.

Nach der Entzerrung wird für OpenCV eine Nullverzerrung verwendet, weil die Verzerrung bereits durch die Projektion in das lineare Kameramodell behandelt wurde.

### 8.3 6-DoF-Posenschätzung

Die vier Marker-Ecken werden als quadratische 3D-Punkte in der Marker-z=0-Ebene definiert. Anschließend wird die Pose mit:

```python
cv2.solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)
```

geschätzt. Das Ergebnis ist:

```text
T_camera_marker
```

Die Translation gibt damit den Markerursprung im linearen RGB-Kameraframe an. Die Pose wird nur akzeptiert, wenn:

- `solvePnP` erfolgreich ist und
- `tz_camera_m > 0`, der Marker also vor der Kamera liegt.

Nicht konfigurierte IDs werden verworfen und gezählt.

### 8.4 Marker-CSV-Schema

Die Ausgabe heißt:

```text
Data_collection/aruco_poses_<sequence_id>.csv
```

Jede Zeile repräsentiert genau eine Markerpose in genau einem RGB-Frame:

| Spalte | Einheit | Bedeutung |
|---|---:|---|
| `sequence_id` | String | vollständige Sequenz-ID |
| `frame_index` | Integer | Index des RGB-Frames |
| `timestamp_ns` | ns | `capture_timestamp_ns` des RGB-Frames |
| `marker_family` | String | `apriltag_36h11` oder `aruco_4x4_50` |
| `marker_id` | Integer | erkannte ID |
| `marker_role` | String | `robot`, `table` oder `object` |
| `marker_size_m` | m | verwendete physische Kantenlänge |
| `tx_camera_m`, `ty_camera_m`, `tz_camera_m` | m | Markertranslation im Kameraframe |
| `rvec_x_rad`, `rvec_y_rad`, `rvec_z_rad` | rad | Rodrigues-Rotationsvektor von OpenCV |
| `qx_camera_marker`, `qy_camera_marker`, `qz_camera_marker`, `qw_camera_marker` | 1 | normiertes Quaternion der Markerpose im Kameraframe |
| `reprojection_error_px` | px | RMS-Reprojektionsfehler der vier Ecken |
| `marker_area_px2` | px² | projizierte Fläche des Markerpolygons |

### 8.5 Qualitätsmetriken der Markerdetektion

#### Reprojektionsfehler

Nach PnP werden die bekannten 3D-Ecken mit der geschätzten Pose zurück ins Bild projiziert. Für vier Ecken gilt:

```text
e_reproj = sqrt(mean(||u_projected - u_detected||²))
```

Ein kleiner Wert bedeutet, dass die geschätzte Pose die beobachteten Ecken geometrisch gut erklärt. Ein hoher Wert kann auf Unschärfe, Teilverdeckung, falsche Ecken oder eine instabile Pose hindeuten.

#### Markerfläche

`marker_area_px2` ist der Betrag der Konturfläche der vier detektierten Ecken. Eine kleine Fläche bedeutet typischerweise einen weit entfernten oder stark schräg sichtbaren Marker und damit häufig eine unsicherere Pose.

#### Aktuelle Filterregel

Der produktive Code speichert beide Qualitätsmetriken, setzt aber **keinen festen Grenzwert** für Reprojektionsfehler oder Markerfläche. Eine Pose mit großem Fehler kann daher formal `valid=1` erhalten, solange `solvePnP` erfolgreich ist und z positiv bleibt. Diese Werte sind für QA verfügbar, werden im aktuellen Training aber nicht direkt benutzt.

### 8.6 Zeitliche Markerfeatures im Master-Dataset

Für jeden tatsächlich detektierten Marker wird pro Gaze-Zeitpunkt das nächste RGB-Markersample innerhalb von 20 ms gewählt. Erzeugt werden unter anderem:

```text
<marker>_timestamp_ns
<marker>_tx_camera_m
<marker>_ty_camera_m
<marker>_tz_camera_m
<marker>_q{x,y,z,w}_camera_marker
<marker>_reprojection_error_px
<marker>_marker_area_px2
<marker>_valid
<marker>_time_offset_ms
```

Für ArUco-IDs 6 bis 14 wird immer ein identisches Spaltenschema erzeugt. Wenn ein Objektmarker in einer Sequenz nie erkannt wurde, bleiben seine numerischen Werte `NaN` und `aruco_<id>_valid=0`. Dadurch besitzen alle Sequenzen kompatible Objektfeatures.

### 8.7 Transformation von Objekten

Mit der RGB-Extrinsik gilt:

```text
T_device_object = T_device_camera * T_camera_object
```

Anschließend:

```text
T_world_object = T_world_device * T_device_object
T_robot_object = T_robot_device * T_device_object
```

Für Objektmarker werden erzeugt:

```text
aruco_<id>_device_{x,y,z}_m
aruco_<id>_world_{x,y,z}_m
aruco_<id>_robot_{x,y,z}_m
aruco_<id>_robot_q{x,y,z,w}
```

Das aktuelle Modell verwendet nur robot-relative Objektposition, Gaze-Winkel, Gaze-Distanz und Sichtbarkeitsflag. Objektorientierung, Kamera-Pose, Welt-Pose, Reprojektionsfehler und Fläche sind nicht Teil des aktuellen Modellinputs.

## 9. Roboteranker und robot-relatives Koordinatensystem

### 9.1 AprilTag 0 als Referenz

AprilTag 0 definiert den in der Pipeline sogenannten Robot-Frame. Für einen Zeitpunkt mit gleichzeitig gültigem SLAM und sichtbarem Tag 0 gilt:

```text
T_world_robot_candidate
    = T_world_device
    * T_device_camera
    * T_camera_apriltag0
```

Ohne AprilTag 0 kann die aktuelle Pipeline keine eindeutig robot-relativen Koordinaten initialisieren. Deshalb ist Tag 0 für den Master-Build erforderlich.

### 9.2 Robuste Schätzung eines statischen Ankers

Der Roboteranker soll während einer Sequenz statisch sein. Aus allen zeitlich zusammengeführten Kandidaten wird daher eine einzige robuste Pose `T_world_robot_static` geschätzt.

Verfahren:

1. Alle Kandidatentranslationen werden gesammelt.
2. Der komponentenweise Median bildet ein erstes Zentrum.
3. Für jeden Kandidaten wird die euklidische Distanz zu diesem Zentrum berechnet.
4. Aus den Distanzen werden Median und Median Absolute Deviation (MAD) bestimmt.
5. Die Inlier-Schwelle ist:

```text
threshold = max(
    0.02 m,
    median_distance + 3 * 1.4826 * MAD
)
```

6. Die statische Translation ist der komponentenweise Median der Inlier.
7. Die statische Orientierung ist der Rotationsmittelwert der Inlier über SciPy.

`robot_static_anchor_samples` speichert die Zahl der verwendeten Inlier.

### 9.3 Verwendung bei verdecktem Marker

Wenn der statische Weltanker einmal aus der Sequenz geschätzt werden konnte und für eine spätere Zeile SLAM verfügbar ist, wird die robot-relative Device-Pose auch ohne aktuelle Tag-Sichtbarkeit berechnet:

```text
T_robot_device
    = inverse(T_world_robot_static)
    * T_world_device
```

Dadurch können Hand-, Gaze- und Objektpositionen im Robot-Frame weitergeführt werden, obwohl Tag 0 in einzelnen Frames verdeckt ist.

Die Spalte `robot_anchor_interpolated` ist dabei historisch benannt. `1` bedeutet im aktuellen Code:

- für diese Zeile war keine instantane AprilTag-0-Pose verfügbar,
- der statische Weltanker wurde zusammen mit SLAM verwendet.

Es handelt sich **nicht** um eine lineare oder splinebasierte Interpolation zwischen Markerposen.

### 9.4 Gültigkeitsfelder

| Spalte | Bedeutung |
|---|---|
| `apriltag_0_valid` | Tag 0 besitzt für diesen Master-Zeitpunkt einen Marker-Match innerhalb 20 ms |
| `robot_frame_valid` | `T_robot_device` konnte für diesen Zeitpunkt bestimmt werden |
| `robot_anchor_interpolated` | statischer Anker wurde verwendet, weil die instantane Tag-0-Pose fehlte |
| `robot_static_anchor_samples` | Anzahl robuster Tag-0/SLAM-Inlier der Sequenz |

`robot_frame_valid` kann daher 1 sein, obwohl `apriltag_0_valid=0` ist.

Für `robot_anchor_interpolated` gilt genauer:

- `0`: eine instantane Tag-0-Pose war in der Zeile vorhanden oder wurde als direkter Fallback verwendet,
- `1`: der statische Weltanker wurde mit SLAM verwendet, während die instantane Tag-0-Pose fehlte,
- `NaN`: es konnte für die Zeile weder die entsprechende Ankerverwendung noch ein gültiger Robot-Frame bestimmt werden.

### 9.5 Wissenschaftliche Einschränkung

Der Robot-Frame ist derzeit der Koordinatenrahmen des gedruckten AprilTag 0. Die starre Transformation vom Marker zur tatsächlichen kinematischen Roboterbasis beziehungsweise zum Endeffektor ist noch nicht eingerechnet.

Somit gilt:

```text
Robot-Frame in den CSVs = AprilTag-0-Frame
```

und nicht automatisch:

```text
Robot-Frame in den CSVs = physische Roboterbasis
```

Für reale Roboterplanung muss eine separat vermessene Transformation `T_robot_base_apriltag0` ergänzt werden.

## 10. Audio, Sprachkommandos und zeitliche Labels

### 10.1 Zweck von Audio

Audio wird nicht als kontinuierliche Eingabemodalität des Baseline-Modells verwendet. Es dient zur Ermittlung der vier Phasengrenzen:

```text
START -> SECOND -> DONE -> THIRD
```

### 10.2 Audioextraktion

[`Code/speech_recognition_demo.py`](../Code/speech_recognition_demo.py) liest den Stream mit Label `mic` aus dem VRS.

- Mehrkanal-Audio wird durch Mittelung in Mono umgewandelt.
- Stummgeschaltete Audioblöcke werden als Nullen beibehalten, damit sich die Zeitachse nicht verschiebt.
- Das Signal wird um seinen Mittelwert zentriert und RMS-normalisiert.
- Die Verstärkung ist auf Faktor 50 begrenzt.
- Peaks werden auf 0,98 begrenzt.
- Für Whisper wird auf 16 kHz, Mono, PCM16 resampelt.
- FFmpeg verwendet Hochpass bei 80 Hz, Kompressor und Loudness-Normalisierung.
- Falls die neue WAV-Dauer um mehr als 20 ms von der Roh-WAV abweicht, wird zur ursprünglichen WAV zurückgefallen.

### 10.3 Speech-Event-Erkennung

Die aktive Methode erkennt zunächst Sprachfenster und klassifiziert danach jedes Fenster mit Faster-Whisper.

Standardparameter:

| Parameter | Wert |
|---|---:|
| Silero-VAD Threshold | 0,15 |
| negativer VAD-Threshold | 0,05 |
| minimale Sprache | 80 ms |
| minimale Stille | 250 ms |
| maximale Sprachfensterlänge | 5 s |
| Kontext vor/nach Fenster | 0,3 s |
| minimaler Abstand zwischen Kommandos | 0,4 s |
| lexikalischer Fuzzy-Threshold | 0,78 |

Wenn Silero-VAD nicht verfügbar ist, wird ein energie- und bandpassbasiertes Fallback verwendet.

Die erkannten Texte werden mit Aliaslisten für `START`, `SECOND`, `DONE` und `THIRD` verglichen. Das Timing stammt bewusst vom **Beginn des Sprachereignisses**, nicht vom von Whisper geschätzten Wortzeitstempel.

Automatisch erkannte Events tragen eine nominelle Zeitunsicherheit von 32 ms:

```text
timestamp_source = speech_window_start
timestamp_uncertainty_ms = 32
```

### 10.4 Umrechnung in Device Time

Die absolute Kommandzeit wird gebildet aus:

```text
command_timestamp_ns
    = audio_start_timestamp_ns
    + relative_seconds * 1e9
```

Die Ergebnisse werden in `timestamps_summary.json` gespeichert. Debugdaten, Kandidaten, Warnungen und VAD-Informationen landen in `timestamps_debug.json`.

### 10.5 Manuelle Korrektur

Unsichere oder falsche Kommandzeiten werden mit MP4 und WAV kontrolliert. Die Review-Datei enthält:

```text
sequence_id
decision
auto_start_s, auto_second_s, auto_done_s, auto_third_s
manual_start_s, manual_second_s, manual_done_s, manual_third_s
target_object_id
receiving_hand
annotation_confidence
missing_commands
status
next_action
notes
```

Manuelle Zeiten werden durch [`Code/apply_manual_reviews.py`](../Code/apply_manual_reviews.py) validiert. Es muss eine monotone Reihenfolge mit mindestens 0,4 s Abstand bestehen. Die korrigierte produktive Datei lautet:

```text
Data_collection/Data_vrs/timestamps_summary.reviewed.json
```

### 10.6 Semantische Labels der Master-Zeitachse

Die Gaze-Zeitachse wird ab `START` behalten. Vor `START` liegende Samples werden entfernt.

| Intervall | `intent_label` | `intent_id` | Bedeutung |
|---|---|---:|---|
| `START <= t < SECOND` | `continue` | 0 | aktuelle Tätigkeit wird fortgesetzt |
| `SECOND <= t <= DONE` | `fetch` | 1 | Fixierung/Ausrichtung auf das Zielobjekt |
| `DONE < t < THIRD` | `transition` | -1 | Warte-/Übergangsphase, kein Trainingsziel |
| `t >= THIRD` bis Aufnahmeende | `handover` | 2 | Hand wird für die Übergabe zum Roboter ausgestreckt |

Die Phase `DONE -> THIRD` bleibt im Master-Dataset als Sensorkontext erhalten. Ein Trainingsfenster darf diese Zeilen in seiner Historie enthalten, aber ein Fenster mit `transition` am Endpunkt wird verworfen und nie als Klassifikationsziel verwendet.

Das Ende von `handover` ist das Ende der Aufnahme beziehungsweise der verfügbaren Gaze-Zeitachse. Es gibt keinen separaten fünften End-Trigger.

### 10.7 Semantische Sequenzannotation

Pro Sequenz werden zusätzlich annotiert:

| Feld | Werte | Zweck |
|---|---|---|
| `target_object_id` | 6 bis 14 | ID des tatsächlich angeforderten Objekts |
| `receiving_hand` | `left` oder `right` | Hand, mit der das Objekt empfangen wird |
| `annotation_confidence` | z. B. `certain`, `uncertain` | Vertrauensangabe der Review |
| `decision` | z. B. `exclude`, `manual_fix`, `accept_auto` | Reviewentscheidung |

`target_object_id` ist Ground Truth und wird nicht aus dem kleinsten Gaze-Winkel geschätzt. Der Marker mit kleinstem Blickwinkel ist nur ein beobachtetes Kandidatenfeature.

Im aktuellen hierarchischen Intention-/Pose-Modell wird `target_object_id` nicht als Eingabefeature verwendet. Damit wird ein direktes Durchreichen des Ziel-Labels in das Modell verhindert. Die Objekt-ID bleibt für spätere Zielobjektklassifikation, objektbezogene Evaluation und Datenanalyse relevant.

## 11. Zeitliche Synchronisation der Modalitäten

### 11.1 Master-Zeitachse

Die native Gaze-Zeitachse ist die Referenz des Master-Datasets. Es wird **kein separates gleichförmiges 30-Hz-Raster erzeugt**. Jede gültige oder ungültige Gaze-Zeile ab `START` bildet einen Master-Zeitpunkt.

Die effektive Rate hängt daher vom Eye-Gaze-Stream der jeweiligen Aufnahme ab. Historische Sequenzen liegen ungefähr bei 30 Zeilen/s, dies ist aber keine im Master-Builder erzwungene Resamplingrate.

### 11.2 Nearest-Neighbor-Merge

Alle anderen Modalitäten werden mit `pandas.merge_asof(..., direction="nearest")` auf die Gaze-Zeitachse gemergt.

| Modalität | Quellzeitstempel | Standardtoleranz |
|---|---|---:|
| Handtracking | `tracking_timestamp_us * 1000` | 12 ms |
| Closed-Loop-SLAM | `tracking_timestamp_us * 1000` | 5 ms |
| jeder Marker separat | RGB `capture_timestamp_ns` | 20 ms |

Der zeitlich nächste Wert kann vor oder nach dem Gaze-Zeitpunkt liegen. Liegt kein Sample innerhalb der Toleranz, bleiben die Modalitätswerte leer.

### 11.3 Zeitoffset-Metriken

Für jeden Merge wird die tatsächliche Abweichung gespeichert:

```text
hand_time_offset_ms
slam_time_offset_ms
<marker>_time_offset_ms
```

Allgemein:

```text
time_offset_ms
    = (source_timestamp_ns - master_timestamp_ns) / 1e6
```

Interpretation:

- positiver Offset: das gewählte Quellsample liegt nach dem Gaze-Zeitpunkt,
- negativer Offset: es liegt davor,
- `NaN`: kein Match innerhalb der Toleranz.

### 11.4 Keine allgemeine Interpolation

Die Pipeline interpoliert Hand-, SLAM-, Gaze- oder Objektmessungen beim Merge nicht numerisch. Sie nutzt den nächsten Nachbarn. Die einzige zeitüberbrückende Logik ist der statische Weltanker von AprilTag 0, der zusammen mit SLAM robot-relative Koordinaten auch bei temporär unsichtbarem Tag ermöglicht.

## 12. Aufbau des Master-Datasets

### 12.1 Grundprinzip

Eine Datei:

```text
Data_collection/master_datasets/<sequence_id>_master.csv
```

enthält eine Zeile pro Gaze-Zeitpunkt ab `START`. Die genaue Spaltenzahl ist dynamisch, weil erkannte AprilTags sequenzabhängig sein können und das Schema im Verlauf des Projekts erweitert wurde. Die Objektmarker 6 bis 14 besitzen dagegen immer ein festes Schema.

### 12.2 Spaltengruppen

| Gruppe | Beispiele | Herkunft |
|---|---|---|
| Identität | `sequence_id`, `participant` | Dateiname |
| Zeit | `timestamp_ns`, `time_since_start_s` | Gaze + START |
| Intention | `intent_label`, `intent_id` | Kommandzeiten |
| Semantik | `target_object_id`, `receiving_hand`, `annotation_confidence` | manuelle Annotation |
| Gaze CPF/Device | `gaze_yaw_rad`, `gaze_direction_device_x` | Eye-Gaze + Kalibrierung |
| Hand MPS | `hand_left_tracking_confidence`, Landmarks, Wrist-Pose, Normalen | Handtracking-CSV |
| Hand abgeleitet | `left_wrist_world_x_m`, `left_wrist_robot_x_m` | Hand + SLAM + Tag 0 |
| SLAM | `slam_tx_world_device`, Geschwindigkeiten, `slam_quality_score` | Closed-Loop-Trajektorie |
| Marker Kamera | `aruco_6_tx_camera_m`, Quaternion, Reprojektionsfehler | RGB + PnP |
| Marker abgeleitet | `aruco_6_robot_x_m`, `aruco_6_gaze_angle_rad` | Marker + Kalibrierung + SLAM/Gaze |
| Roboteranker | `robot_frame_valid`, `robot_anchor_interpolated` | Tag 0 + SLAM |
| Zukunftstarget | `future_1s_receiving_wrist_robot_x_m` usw. | zeitverschobene Wrist-Pose |
| Synchronisations-QA | `hand_time_offset_ms`, `slam_time_offset_ms` | Merge |

### 12.3 Präfix `hand_` und `slam_`

Um Namenskonflikte zu vermeiden, werden beim Laden:

- alle originalen Handtracking-Spalten mit `hand_` präfixiert,
- alle originalen SLAM-Spalten mit `slam_` präfixiert.

Die künstlich erzeugten Zeitspalten `hand_timestamp_ns` und `slam_timestamp_ns` erhalten kein doppeltes Präfix.

### 12.4 Fehlende Werte

Ein leeres Feld beziehungsweise `NaN` bedeutet je nach Gruppe:

- Sensor-/MPS-Wert war ungültig,
- kein zeitlich passendes Sample lag innerhalb der Toleranz,
- Marker war nicht sichtbar,
- eine notwendige Transformation konnte nicht berechnet werden,
- Zukunftszeitpunkt lag außerhalb der Aufnahme,
- die empfangende Hand war nicht annotiert.

`NaN` ist daher ein **Missing-Data-Signal** und nicht der Zahlenwert 0.

## 13. Zukünftiges Handpose-Target

### 13.1 Definition

Der Standardvorhersagehorizont ist:

```text
Δt = 1.0 s
```

Für jede Master-Zeile mit Zeit `t` wird eine Zielzeit gebildet:

```text
t_query = t + 1.0 s
```

Gesucht wird die nächstgelegene bereits robot-relative Wrist-Pose innerhalb von 12 ms um `t_query`.

### 13.2 Targets für beide Hände

Zunächst werden beide Seiten erzeugt:

```text
future_1s_left_wrist_robot_{x,y,z}_m
future_1s_left_wrist_robot_q{x,y,z,w}
future_1s_left_wrist_valid

future_1s_right_wrist_robot_{x,y,z}_m
future_1s_right_wrist_robot_q{x,y,z,w}
future_1s_right_wrist_valid
```

Zusätzlich:

| Spalte | Bedeutung |
|---|---|
| `future_target_timestamp_ns` | tatsächlich verwendeter Quellzeitpunkt |
| `future_1s_time_error_ms` | Differenz zwischen Quellzeitpunkt und exakt `t+1s` |

### 13.3 Auswahl der empfangenden Hand

Mit dem Sequenzlabel `receiving_hand` wird eine gemeinsame Zielpose erzeugt:

```text
future_1s_receiving_wrist_robot_{x,y,z}_m
future_1s_receiving_wrist_robot_q{x,y,z,w}
future_1s_receiving_wrist_valid
```

Bei `receiving_hand=left` werden die linken Targets kopiert, bei `right` die rechten. Ohne eindeutige Handannotation bleibt das gemeinsame Target ungültig.

### 13.4 Builder- und Trainingsvalidität

Der Master-Builder setzt die seitenspezifische Zukunftspose als gültig, wenn die robot-relative x-Koordinate nicht leer ist. Der Trainingsloader prüft strenger:

- explizites `future_1s_receiving_wrist_valid > 0`,
- alle sieben Posewerte sind endlich,
- Quaternionnorm ist größer als `1e-6`,
- Quaternion wird normalisiert,
- Pose-Loss wird nur für `intent_id=2` (`handover`) aktiviert.

### 13.5 Warum Targets fehlen können

Ein Future-Pose-Target kann ungültig sein, wenn:

- `t+1s` nach dem Aufnahmeende liegt,
- die empfangende Hand bei `t+1s` nicht getrackt wurde,
- SLAM bei `t+1s` fehlt,
- der statische Robot-Frame nicht bestimmt werden konnte,
- die robot-relative Wrist-Pose nicht endlich ist,
- die empfangende Hand nicht eindeutig annotiert wurde.

Die bereits durchgeführte Pose-Target-Audit trennt diese Fälle unter anderem in `future_after_recording_end`, `future_hand_tracking_invalid`, `future_slam_pose_invalid` und `future_hand_and_robot_frame_invalid`.

### 13.6 Residual-v2-Referenzpose

Für das Residual-v2-Modell wird zusätzlich pro Trainingsfenster und Hand die **letzte gültige robot-relative Wrist-Pose innerhalb des Beobachtungsfensters** gesucht.

Gespeichert werden intern:

```text
hand_reference_pose[side]
hand_reference_valid[side]
hand_reference_age_seconds[side]
```

Ein Residual-Pose-Target ist nur gültig, wenn:

- das Future-Target gültig ist,
- die empfangende Hand links oder rechts bekannt ist,
- für diese Hand eine gültige Referenzpose im Fenster existiert.

Das Modell lernt dann die Veränderung relativ zur letzten Beobachtung statt die vollständige absolute Zielpose von Grund auf neu zu schätzen.

## 14. Tatsächlich verwendete Trainingsfeatures

### 14.1 Featureprofil

Das aktuelle Profil heißt:

```text
multimodal_robot_frame_v1
```

Es enthält 92 Rohfeatures, sofern das aktuelle Master-Schema vollständig vorliegt. Nach Hinzufügen einer Beobachtungsmaske für jedes Feature erhält das Modell 184 Kanäle.

### 14.2 Gaze-Features

```text
gaze_valid
gaze_yaw_rad
gaze_pitch_rad
gaze_depth_m
gaze_origin_robot_{x,y,z}_m
gaze_direction_robot_{x,y,z}
```

Nicht direkt verwendet werden unter anderem CPF-Punkt, Device-Punkt, Welt-Gaze und per-eye yaw/pitch.

### 14.3 Handfeatures

```text
hand_left_tracking_confidence
hand_right_tracking_confidence
hand_left_valid
hand_right_valid

left_wrist_robot_{x,y,z}_m
left_wrist_robot_q{x,y,z,w}
right_wrist_robot_{x,y,z}_m
right_wrist_robot_q{x,y,z,w}
```

Nicht direkt verwendet werden die 42 x 3 Landmark-Koordinaten und die Palm-/Wrist-Normalen.

### 14.4 Bewegungs- und Ankerfeatures

```text
slam_device_linear_velocity_{x,y,z}_device
slam_angular_velocity_{x,y,z}_device
slam_quality_score
apriltag_0_valid
robot_frame_valid
robot_anchor_interpolated
```

### 14.5 Objektfeatures

Für jede ID 6 bis 14:

```text
aruco_<id>_robot_{x,y,z}_m
aruco_<id>_gaze_angle_rad
aruco_<id>_gaze_distance_m
aruco_<id>_valid
```

Das ergibt 6 Features pro Objekt und 54 Objektfeatures insgesamt.

### 14.6 Nicht verwendete Informationen

Folgende Daten sind vorhanden oder ableitbar, werden in der aktuellen Baseline aber nicht direkt trainiert:

| Information | Aktueller Status |
|---|---|
| RGB-Bild oder CLIP-Embedding | nicht im Modellinput |
| Audio-Embedding oder Sprache | nur zur Labelsegmentierung |
| vollständige 21-Hand-Landmarks | im Master, nicht im Featureprofil |
| Palm-/Wrist-Normalen | im Master, nicht im Featureprofil |
| absolute SLAM-Weltpose | nur für Transformationen |
| SLAM-Gravitation/ECEF/UTC | nicht im Modellinput |
| Objektorientierungen | nicht im Modellinput |
| Marker-Reprojektionsfehler und -fläche | QA, nicht im Modellinput |
| Tisch-AprilTags 1 bis 5 | gespeichert, nicht im Modellinput |
| `target_object_id` | Ground Truth/Metadatum, nicht als Eingabe |

Diese Liste ist wichtig für spätere Ablationsstudien. Eine Aussage wie "das Modell verwendet Handtracking" bedeutet aktuell Wrist-Pose plus Konfidenz und Validität, nicht die komplette Handgeometrie.

## 15. Normalisierung und Missing-Data-Behandlung im Training

### 15.1 Fit nur auf Trainingspersonen

Mittelwert und Standardabweichung werden ausschließlich auf den Trainingssequenzen berechnet. Validation und Test werden mit denselben Trainingsstatistiken transformiert. Dadurch gelangen keine Verteilungsinformationen aus Validation/Test in das Training.

Für Feature `j` gilt über alle endlichen Trainingswerte:

```text
z_j = (x_j - mean_train_j) / std_train_j
```

Wenn kein Trainingswert vorhanden ist, werden `mean=0` und `std=1` verwendet. Standardabweichungen kleiner `1e-6` werden ebenfalls auf 1 gesetzt.

### 15.2 Beobachtungsmaske

Vor der Normalisierung wird für jedes Feature gespeichert:

```text
observed_j = isfinite(x_j)
```

Fehlende normalisierte Werte werden anschließend auf 0 gesetzt. Danach werden Werte und Masken konkateniert:

```text
model_input = [normalized_features, observed_mask]
```

Dadurch kann das Modell unterscheiden zwischen:

- einem echten, nach Standardisierung ungefähr nullwertigen Messwert und
- einem fehlenden Messwert, der nur numerisch durch 0 ersetzt wurde.

### 15.3 Fensterbildung

Die aktuellen Baseline-Konfigurationen verwenden:

| Parameter | Wert |
|---|---:|
| `window_size` | 60 Master-Zeilen |
| `stride` | 10 Master-Zeilen |
| `future_horizon_seconds` | 1,0 s |
| `max_timestamp_gap_seconds` | 0,2 s |
| `minimum_observed_fraction` | 0,05 |

Bei ungefähr 30 Gaze-Zeilen/s entsprechen 60 Zeilen ungefähr 2 s Beobachtung. Da nicht auf eine feste Rate resampelt wird, ist die exakte Zeitspanne datenabhängig.

Ein Fenster wird verworfen, wenn:

- sein Endpunkt `transition` und damit ungelabelt ist,
- innerhalb des Fensters eine Zeitlücke > 0,2 s liegt,
- weniger als 5 % aller Feature-Masken beobachtet sind.

Das Ziel der Intention ist immer das Label am letzten Zeitpunkt des Fensters.

## 16. QA-Metriken und technische Gültigkeitsregeln

### 16.1 Sequenzebene im Dataset-Manifest

[`Code/dataset_qa.py`](../Code/dataset_qa.py) erzeugt:

```text
Data_collection/dataset_manifest.csv
Data_collection/dataset_qa_report.json
```

Geprüft werden unter anderem:

- VRS vorhanden,
- Backup-VRS vorhanden,
- MP4 und WAV vorhanden und lesbar,
- MP4-/WAV-Dauern konsistent,
- MPS-Verzeichnis vorhanden,
- Handtracking vorhanden,
- SLAM vorhanden,
- Marker-CSV vorhanden und zeitlich überlappend,
- alle vier Kommandzeitpunkte vorhanden und geordnet,
- Zielobjekt-ID und empfangende Hand annotiert,
- Zielobjektmarker wurde tatsächlich detektiert,
- Master-CSV und Master-Report vorhanden,
- manuelle Review wurde korrekt in die Timestamp-Datei übernommen.

### 16.2 Handover-Handtracking-Coverage

Für alle Handtracking-Zeilen ab `THIRD` bis Aufnahmeende werden gezählt:

```text
left_valid_rows
right_valid_rows
either_valid_rows
```

Die Ratios sind:

```text
left_valid_ratio   = left_valid_rows / rows
right_valid_ratio  = right_valid_rows / rows
either_valid_ratio = either_valid_rows / rows
```

Regeln:

- keine Handover-Zeile oder keine einzige gültige Hand: `missing_handover_hand_tracking`,
- `either_valid_ratio < 0.8`: Warnung `low_handover_hand_tracking`,
- für eine annotierte linke/rechte Empfangshand wird zusätzlich die entsprechende seitenspezifische Coverage gegen 0,8 geprüft.

### 16.3 Medien- und Phasenschwellen

| QA-Regel | Standardwert |
|---|---:|
| minimale Phasenlänge | 0,5 s |
| maximale Sequenzlänge | 180 s |
| maximale MP4-/WAV-Differenz | 0,1 s |
| minimale Handover-Hand-Coverage | 0,8 |

Eine kurze Phase oder lange Sequenz erzeugt eine Review-Warnung. Fehlende technische Kerndaten wie VRS, Handtracking, SLAM oder vollständige Zeitlabels sind blockierende Probleme.

### 16.4 Master-Report pro Sequenz

`<sequence_id>_master_report.json` fasst unter anderem zusammen:

| Metrik | Berechnung |
|---|---|
| `gaze_valid_ratio` | Mittelwert von `gaze_valid` |
| `hand_left_valid_ratio` | Mittelwert von `hand_left_valid`, fehlend als 0 |
| `hand_right_valid_ratio` | Mittelwert von `hand_right_valid`, fehlend als 0 |
| `slam_match_ratio` | Anteil Master-Zeilen mit gemergtem SLAM-Zeitstempel |
| `robot_marker_valid_ratio` | Mittelwert `apriltag_0_valid` |
| `robot_frame_valid_ratio` | Anteil mit berechenbarer robot-relativer Pose |
| `robot_anchor_interpolated_ratio` | Anteil statisch überbrückter Tag-0-Zeilen |
| `robot_static_anchor_samples` | Zahl der Inlier für den statischen Anker |
| `marker_valid_ratios` | Sichtbarkeits-/Matchanteil pro Marker |
| `future_left_wrist_valid_ratio` | Anteil gültiger linker +1-s-Targets |
| `future_right_wrist_valid_ratio` | Anteil gültiger rechter +1-s-Targets |
| `future_receiving_wrist_valid_ratio` | Anteil gültiger +1-s-Targets der annotierten Empfangshand |

Diese Ratios messen Datenverfügbarkeit. Sie sind keine Modellgütemaße wie Accuracy, F1 oder Positionsfehler.

## 17. Datenqualitätsmetriken versus Modellmetriken

Zur Vermeidung von Missverständnissen werden zwei Arten von Metriken verwendet.

### 17.1 Daten- und Sensorqualität

| Metrik | Ebene | Aussage |
|---|---|---|
| Tracking Confidence | Handtracking | Sicherheit der Handdetektion |
| `quality_score` | SLAM | Qualität der Trajektorienschätzung |
| Reprojection Error | Markerpose | geometrische PnP-Konsistenz |
| Marker Area | Markerpose | Bildgröße/Sichtbarkeit des Markers |
| Valid Ratio/Coverage | Sequenz | Anteil verfügbarer Messungen |
| Time Offset | Synchronisation | zeitlicher Abstand zum Master-Sample |
| Future Target Coverage | Target | Anteil nutzbarer +1-s-Posen |

### 17.2 Modellbewertung

Diese Werte entstehen erst beim Training/Evaluieren und gehören nicht zu den Rohdaten:

| Aufgabe | Modellmetrik |
|---|---|
| Intention | Accuracy, Macro-F1, F1 pro Klasse, Confusion Matrix |
| Assistance ja/nein | Accuracy, Macro-F1 |
| Fetch vs. Handover | Accuracy, Macro-F1 |
| empfangende Hand | Accuracy/F1 für links/rechts |
| Handposition | MAE in cm, RMSE in cm |
| Handorientierung | mittlerer Quaternion-Winkelfehler in Grad |

Ein hoher Daten-Valid-Ratio garantiert keine gute Vorhersage, und ein niedriger Modellfehler sagt nicht automatisch, dass die Sensorrohdaten fehlerfrei sind.

## 18. End-to-End-Datenfluss

Die gesamte aktive Pipeline lässt sich wie folgt zusammenfassen:

```text
Aria Gen2 Aufnahme
    |
    v
<sequence_id>.vrs
    |
    +--> Eye-Gaze-Stream ----------------------> gaze_<sequence_id>.csv
    |
    +--> RGB + Kalibrierung + PnP ------------> aruco_poses_<sequence_id>.csv
    |
    +--> Mikrofon + VAD + Whisper ------------> timestamps_summary.json
    |                                              |
    |                                              +--> manuelle Review
    |                                                   |
    |                                                   v
    |                                      timestamps_summary.reviewed.json
    |
    +--> MPS Handtracking --------------------> hand_tracking_results.csv
    |
    +--> MPS Closed-Loop SLAM ----------------> closed_loop_trajectory.csv
                                                   |
                                                   v
                                  build_master_dataset.py
                                                   |
                         nearest timestamp merge + intent labels
                                                   |
                       SE(3)-Transformationen + statischer Tag-0-Anker
                                                   |
                         Gaze-Objekt-Relationen + Future-Pose t+1s
                                                   |
                                                   v
                                  <sequence_id>_master.csv
                                                   |
                                                   v
                                        Training/data.py
                                                   |
                      Featureauswahl + Train-Normalisierung + Missing Masks
                                                   |
                               60-Zeilen-Fenster, Stride 10
                                                   |
                                                   v
                                  Transformer-/Residual-Modell
```

## 19. Reproduzierbare Verarbeitungsschritte

### 19.1 Gaze für eine Sequenz exportieren

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/extract_multimodal_data.py \
  --input-vrs Data_collection/Data_vrs/<sequence_id>.vrs \
  --overwrite
```

### 19.2 Markerposen extrahieren

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/detect_tags.py \
  --input-vrs Data_collection/Data_vrs/<sequence_id>.vrs \
  --overwrite
```

Optional kann ein annotiertes Kontrollvideo über `--output-video` geschrieben werden.

### 19.3 Dataset-QA ausführen

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/dataset_qa.py \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.reviewed.json
```

### 19.4 Master-Dataset für eine Sequenz bauen

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/build_master_dataset.py \
  --sequence-id <sequence_id> \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.reviewed.json \
  --annotations Data_collection/manual_timestamp_review.csv \
  --overwrite
```

### 19.5 Batch-Build auf dem Cluster

```bash
sbatch --export=ALL,OVERWRITE=1 \
  singularity/aria_build_master_dataset.sbatch
```

## 20. Bekannte Einschränkungen und Interpretationsrisiken

### 20.1 Keine direkte physische Roboterbasis-Kalibrierung

Robot-relative Posen beziehen sich auf AprilTag 0. Der feste Offset zur echten Roboterbasis ist noch nicht angewendet.

### 20.2 RGB ist noch keine gelernte Modalität

Obwohl RGB aufgenommen wird, verarbeitet die Baseline keine CLIP- oder CNN-Features. Aussagen über "multimodal" beziehen sich aktuell auf Gaze, Hand/Wrist, Kopfbewegung/SLAM und geometrische Objektinformationen.

### 20.3 Handlandmarks werden noch nicht ausgeschöpft

Die vollständige 21-Punkt-Handgeometrie ist im Master vorhanden, aber nicht im Featureprofil. Das Modell sieht vor allem die 6-DoF-Wrist-Pose.

### 20.4 Keine harte Markerqualitätsfilterung

Reprojektionsfehler und Fläche werden gespeichert, aber nicht zur automatischen Ablehnung von Posen verwendet. Teilverdeckte oder sehr kleine Marker können dadurch verrauschte Posen liefern.

### 20.5 Sichtbarer Marker ist nicht gleich sichtbares Objektzentrum

Die Objektposition entspricht dem Koordinatenursprung des befestigten ArUco-Markers. Sie ist nicht zwingend der geometrische Mittelpunkt oder ideale Greifpunkt des Objekts.

### 20.6 Gaze-Winkel ist kein Fixationslabel

Ein kleiner Winkel zum Marker ist ein kontinuierliches geometrisches Merkmal. Er beweist weder bewusste Fixation noch die Fetch-Intention. Kopfbewegung, Augenbewegung, Markerposition und Messfehler beeinflussen den Wert.

### 20.7 Ersatzdistanz bei fehlender Gaze-Tiefe

Bei gültiger Richtung, aber ungültiger Tiefe wird für den exportierten 3D-Richtungspunkt intern 1 m verwendet. Dieser Punkt darf nicht als gemessene Fixation in 1 m Entfernung interpretiert werden.

### 20.8 Weltframe ist sequenzspezifisch

SLAM-Weltpositionen verschiedener Aufnahmen sind ohne Registrierung nicht direkt vergleichbar. Die Robotermarker-Transformation soll dieses Problem für modellrelevante Positionen reduzieren.

### 20.9 Nearest Neighbor statt Interpolation

Die Modalitäten werden nicht aufwendig kontinuierlich resampelt. Die gespeicherten Offsetspalten müssen bei Synchronisationsanalysen berücksichtigt werden.

### 20.10 Future-Target am Aufnahmeende

Für die letzte Sekunde einer Aufnahme existiert prinzipbedingt kein +1-s-Target. Dies ist keine Trackingstörung, sondern ein Randproblem des festen Vorhersagehorizonts.

### 20.11 Confidence > 0 als Hand-Gültigkeitsregel

Jede positive Trackingkonfidenz gilt derzeit als gültig. Eine Sensitivitätsanalyse mit strengeren Schwellen kann später sinnvoll sein.

### 20.12 SLAM-Qualität wird nicht hart gefiltert

Ein zeitlich passendes Sample mit niedrigem `quality_score` kann weiterhin für Koordinatentransformationen verwendet werden. Der Score wird zwar dem Modell gegeben, schützt die abgeleiteten Targets aber nicht automatisch vor schlechter SLAM-Geometrie.

## 21. Empfohlene Erweiterungen für spätere Versionen

Diese Punkte beschreiben mögliche wissenschaftliche Erweiterungen und sind **nicht** Teil der aktuellen Baseline:

1. Reprojektionsfehler- und Flächengrenzen auf einem separaten Validationssatz kalibrieren.
2. Strengere oder probabilistische Handtracking-Gültigkeit untersuchen.
3. SLAM-Qualitätsgrenze oder Quality-weighted Pose-Loss testen.
4. 21 Handlandmarks oder eine kompakte Hand-Skeleton-Repräsentation als zusätzliche Modalität aufnehmen.
5. Palmennormale explizit für die Handover-Orientierung verwenden.
6. RGB-Features mit CLIP, DINO oder einem Videoencoder ergänzen.
7. Marker-zu-Roboterbasis-Extrinsik physisch vermessen.
8. Objektmarker-zu-Objektzentrum beziehungsweise Marker-zu-Grasp-Pose kalibrieren.
9. Nearest-Neighbor-Merge gegen kontrollierte Interpolation oder Continuous-Time-Modelle vergleichen.
10. Unsicherheiten der Sensoren explizit modellieren statt nur Masken und Scores zu übergeben.

## 22. Kompakte Referenz: Welche Modalität liefert was?

| Modalität | Primärinformation | Frame | Qualitäts-/Validitätssignal | Aktuell im Modell? |
|---|---|---|---|---|
| Gaze | yaw, pitch, depth, Richtung | CPF, Device, Robot | `gaze_valid`, fehlende Tiefe | ja, ausgewählte Felder |
| Handtracking | 21 Landmarks, Wrist-6DoF, Normalen | Device | Tracking Confidence, `hand_*_valid` | Wrist + Confidence + Validität |
| SLAM | Device-Pose und Dynamik | World/Device | `quality_score`, zeitlicher Match | Dynamik + Score; Pose indirekt |
| RGB | egocentrisches Bild | RGB-Kamera | Kalibrierung/Framezeit | nur Marker und Review |
| AprilTag 0 | Roboteranker | RGB-Kamera -> World/Robot | Reprojektionsfehler, Fläche, Validität | Ankerflags + indirekte Transformation |
| AprilTags 1-5 | Tisch-/Umgebungsanker | RGB-Kamera | Reprojektionsfehler, Fläche | nein |
| ArUco 6-14 | Objektpose | Kamera, Device, World, Robot | Reprojektionsfehler, Fläche, Validität | Position + Gaze-Relation + Validität |
| Audio | Kommandos | Device Time | VAD/Whisper-Warnungen, manuelle Review | nur Labels |
| Semantische Review | Zielobjekt und Empfangshand | Sequenzebene | `annotation_confidence` | Handlabel für Pose; Objekt-ID nicht als Input |

## 23. Offizielle Referenzen

- [Project Aria VRS Data Format](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs)
- [Project Aria Timestamp Definitions](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs)
- [Aria Gen2 Device Calibration](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/device/calibration)
- [Project Aria 3D Coordinate Frame Conventions](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention)
- [Project Aria Eye Gaze Data Format](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_eye_gaze)
- [Project Aria MPS Hand Tracking](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking)
- [Project Aria MPS Closed-Loop Trajectory](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory)
- [Aria Gen2 Device-Time Alignment](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/projectariatools/pythontutorials/time-sync)

## 24. Kurzglossar

| Begriff | Bedeutung |
|---|---|
| VRS | zeitgestempelter Multi-Sensor-Aufzeichnungscontainer |
| MPS | Machine Perception Services von Project Aria |
| CPF | Central Pupil Frame zwischen den Augen |
| Device-Frame | Referenzframe der Brille |
| Camera-Frame | lokaler Frame der RGB-Kamera |
| World-Frame | sequenzspezifischer Closed-Loop-SLAM-Frame |
| Robot-Frame | in dieser Pipeline AprilTag-0-Frame |
| Wrist-Frame | Handframe mit Ursprung am Handgelenk |
| SE(3) | starre 3D-Transformation aus Rotation und Translation |
| PnP | Bestimmung einer 3D-Pose aus 3D-zu-2D-Korrespondenzen |
| Reprojektionsfehler | Pixelabweichung zwischen detektierten und zurückprojizierten Markerecken |
| Coverage/Valid Ratio | Anteil technisch verfügbarer Messwerte |
| Future Horizon | Zeitabstand zwischen Beobachtung und Pose-Target, aktuell 1 s |
| Observation Mask | binäres Signal, ob ein Featurewert beobachtet oder fehlend ist |
