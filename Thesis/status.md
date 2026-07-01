# Technische Dokumentation: Entwicklungsstand Framework zur multimodalen Intentionsschätzung

**Dokument aktualisiert:** 1. Juli 2026

**Dokumentierter Entwicklungsstand:** 1. Juli 2026

## 1. System-Architektur & Infrastruktur

Um eine reproduzierbare, plattformunabhängige und isolierte Entwicklungsumgebung zu gewährleisten und Konflikte mit dem globalen System zu vermeiden, wurde eine dedizierte Python-Infrastruktur aufgesetzt.

* **Virtual Environment (Conda):** Es wurde eine virtuelle Umgebung namens `aria_conda` mittels Anaconda eingerichtet. Dies stellt sicher, dass alle hardwarenahen, vorkompilierten C++-Bindings des Meta SDKs exakt mit den Python-Bibliotheken harmonieren.
* **Python-Version:** Verwendung von Python 3.10, um maximale Stabilität mit den Deep-Learning-Bibliotheken (PyTorch) und dem Meta-Aria-Ecosystem zu garantieren.
* **Hardware-Abstraktion:** Die Pipeline nutzt auf Linux-Infrastrukturen die Nvidia CUDA-Beschleunigung für das parallele Deep-Learning-Training. Die lokale Vorverarbeitung unterstützt zudem hardwarebeschleunigte Dekodierung, fällt bei Bedarf jedoch nahtlos auf einen robusten Software-Dekoder (`H.265 SW decoder via xprs`) zurück.
* **TCML-Cluster-Container:** Für reproduzierbare Verarbeitung auf dem TCML-GPU-Cluster wurde unter `singularity/singularity.recipe` eine Singularity-Definition angelegt. Das Image basiert auf Ubuntu 22.04 mit CUDA 12.4.1 und cuDNN und verwendet eine isolierte Python-3.10-Umgebung unter `/opt/aria_env`.
* **Festgeschriebene Kernabhängigkeiten:** Der Container installiert unter anderem PyTorch 2.4.1 mit CUDA 12.4, OpenCV Contrib 4.10, Project Aria Tools 2.1.2, Project Aria MPS 1.2.1, Faster-Whisper 1.2.1 und Ultralytics 8.4.53. Ein integrierter Smoke-Test prüft die zentralen Imports, die Verfügbarkeit von `cv2.aruco`, `ffmpeg` und `vrs_to_mp4` sowie die erkannte CUDA-Laufzeit.
* **Cluster-Ausführung:** Der Container ist für den Aufruf mit `singularity --nv` vorgesehen. Hinweise zur Datenübertragung, zur Anforderung eines GPU-Knotens und zum Start einer beschreibbaren Sandbox sind in `Links and Commands.txt` dokumentiert.

---

## 2. Implementierte Software-Module & Kernfunktionen

### Modul A: Daten-Parsing & Sensor-Streaming (`projectaria_tools`)
* **Funktion:** Zugriff, Aktivierung und Dekapselung des proprietären Meta `.vrs`-Datenformats. Das Skript initialisiert und synchronisiert die hochfrequenten Datenströme der RGB-Weitwinkelkamera (aufgenommen in Profil 10 mit 30 Hz), der Eye-Tracking-Infrarotkameras, des 7-Kanal-Mikrofon-Arrays, der Hand-Tracking-Schnittstelle sowie der IMU- und VIO-Sensoren.
* **Warum:** Aus Datenschutz- und Performancegründen speichert das Aria-System Sensordaten nicht als Standard-Multimediadatei. Ohne dieses Modul wäre ein zeitsynchroner, framegenauer Zugriff auf die Rohdaten für das spätere KI-Training unmöglich.

### Modul B: Automatische Kamera-Kalibrierung & Rektifizierung
* **Funktion:** Automatisches Auslesen der werkseitigen intrinsischen Kameraparameter direkt aus den Metadaten der VRS-Aufnahme. Das Skript extrahiert die Brennweiten (`get_focal_lengths()`) und den Hauptpunkt (`get_principal_point()`), um die Kamera-Matrix $K$ zur Laufzeit mathematisch korrekt aufzubauen.
* **Warum:** Da es sich um Fisheye-Objektive handelt, weist das Bild starke Verzerrungen auf. Da das Meta-SDK die Bilder bereits rektifiziert (verzerrungsfrei) übergibt, wird die Linsenverzerrung im Algorithmus präzise genullt, was für die exakte 3D-Geometrie zwingend erforderlich ist.

### Modul C: Visuelle Extraktion & MP4-Videokonvertierung
* **Funktion:** Extrahieren der kontinuierlichen RGB-Videodaten aus dem VRS-Container und native Umwandlung in das standardisierte `.mp4`-Format. Das Modul berechnet dynamisch die exakte Framerate anhand der Nanosekunden-Zeitstempel der Hardware (z. B. 30 FPS), um das Video in Echtzeitgeschwindigkeit auszugeben. 
* **Aktualisierung des Batch-Skripts:** Das Skript `Code/vrs_to_mp4_all.py` wurde zu einem robusten CLI-Tool erweitert. Es unterstützt nun frei wählbare Eingabe- und Ausgabeordner (`--input-dir`, `--output-dir`), Dry-Runs (`--dry-run`), begrenzte Testläufe (`--limit`), optionales Überschreiben (`--overwrite`), alternative Konvertierungsbefehle (`--tool`) und eine Fehlerzusammenfassung am Ende. Standardmäßig werden nur fehlende MP4-Dateien erzeugt; vorhandene Exporte bleiben unverändert.
* **Warum:** Dies ermöglicht eine schnelle visuelle Inspektion der aufgezeichneten Daten, das reibungslose Teilen von Sequenzen und dient als Grundlage für die überlagerte Visualisierung der 3D-Koordinaten.

### Modul D: Synchronisiertes Audio-Annotation-System (Aria Gen 2 Codec-Handling)
* **Funktion:** Da die Meta Aria Gen 2 komprimierte Audio-Datenblöcke verwendet, mischt das Modul die Kanäle des Mikrofon-Arrays mathematisch zu einer Mono-Spur zusammen und exportiert ein unkomprimiertes 16-Bit-PCM-WAV-Signal (16 kHz).
* **Robuste Audio-Vorverarbeitung:** Da sich bei den bisherigen Aufnahmen gezeigt hat, dass die Sprachkommandos teilweise sehr leise aufgenommen wurden, wurde das Skript `Code/speech_recognition_demo.py` erweitert. Die aus dem VRS-Container extrahierte Audiospur wird nun zuerst per RMS-Normalisierung angehoben. Wenn `ffmpeg` verfügbar ist, wird zusätzlich eine Whisper-optimierte Vorverarbeitung angewendet (`highpass`, Kompressor und Loudness-Normalisierung auf ca. -16 LUFS). Dadurch wird die Erkennung leiser kurzer Befehle deutlich stabiler, ohne dass die Rohdaten verändert werden.
* **Verbesserte Command-Erkennung:** Das normalisierte Signal wird an ein lokales **Faster-Whisper-Modell** übergeben, welches die Spur nach der sequentiellen Trigger-Abfolge (*"Start"*, *"Second"*, *"Done"*, *"Third"*) scannt. Die Erkennung basiert nicht mehr nur auf exakten Worttreffern, sondern nutzt zusätzlich Fuzzy-Matching und Alias-Listen für typische Fehltranskriptionen (z. B. `"star"` für `"start"` oder `"down"` für `"done"`). Die finale Auswahl wird anschließend über die erwartete chronologische Reihenfolge `START -> SECOND -> DONE -> THIRD` stabilisiert.
* **CPU-/GPU-Konfiguration:** Die Ausführungsplattform und numerische Repräsentation von Faster-Whisper sind nun über `--device` und `--compute-type` konfigurierbar. Lokal bleibt `--device cpu --compute-type int8` der Standard; auf einem GPU-Knoten kann die Erkennung beispielsweise mit `--device cuda --compute-type float16` ausgeführt werden. Damit ist dasselbe Skript ohne Codeänderung lokal und auf dem TCML-Cluster nutzbar.
* **Debug- und QA-Ausgaben:** Die finale Datei `timestamps_summary.json` bleibt für nachgelagerte Pipelines kompatibel und enthält nur die final ausgewählten Trigger-Zeitpunkte. Zusätzlich erzeugt das Skript nun `timestamps_debug.json`, in der erkannte Kandidaten, Match-Scores und Warnungen wie fehlende Trigger gespeichert werden. Optional können mit `--keep-debug-audio` die normalisierten WAV-Dateien zur manuellen Kontrolle abgelegt werden.
* Über den Aufruf `get_first_time_ns` wird weiterhin der Hardware-Startzeitpunkt (`TimeDomain.DEVICE_TIME`) abgegriffen und mit den relativen Wort-Zeitstempeln von Whisper verrechnet. Die daraus resultierenden Trigger-Zeitpunkte bleiben somit nanosekundengenau mit den Sensorströmen synchronisiert.
* **Warum:** Für das Supervised Learning wird so eine nanosekundengenaue, automatisierte zeitliche Synchronität zwischen dem gesprochenen Befehl und den physikalischen Augen- und Handbewegungen des Probanden hergestellt.
* **Architektonischer Ausschluss von Feature-Leckagen:** Die extrahierten Audio-Befehle dienen ausschließlich der automatisierten Offline-Phasensegmentierung (Ersatz für manuelles Frame-Labeling). Die Audiospur wird **nicht** als Feature in die finale KI-Dateneinspeisung übernommen. Das neuronale Netz lernt rein auf den physikalischen Geometrie- und Bilddaten.

---

## 3. Struktur der finalen KI-Dateneinspeisung & Feature-Matrix

Ein wichtiger architektonischer Meilenstein ist die Trennung von visueller Kontrolle und mathematischem KI-Training. Die originalen hochfrequenten Sensordaten werden nicht zerschnitten, sondern über Nanosekunden-Zeitstempel in eine Master-Tabelle überführt, über die der PyTorch-`DataLoader` mittels "Sliding Window" iteriert.

### Modul E: Cloud-optimierte Tracking-Extraktion (MPS)
* **Funktion:** Anstelle der reinen On-Device-Berechnung werden die Meta Perception Services (MPS) genutzt, um eine hochpräzise Offline-Optimierung (SLAM) über das gesamte Video laufen zu lassen. 
* **Extraktion der Features:**
  * **Hand-Pose:** Extraktion der geglätteten 3D-Skelett-Landmarks (z. B. `wrist_position`) im Brillen-Koordinatensystem.
  * **SLAM (Brillenpose):** Hochfrequente Welt-Koordinaten ($tx, ty, tz$ und Quaternionen) der Brille im initialisierten Raum.
  * **Eye-Gaze:** Abfrage der Blickrichtungswerte aus dem *Central Pupil Frame (CPF)* über die geräteinterne `TimeQueryOptions.CLOSEST`-Metrik.

### Modul F: Duale 3D-Lokalisierung (AprilTag & ArUco 4x4 Mix)
* **Funktion:** Eine zweigleisige geometrische 3D-Posenschätzung (`cv2.aruco.ArucoDetector`), die auf die unterschiedlichen physischen Größen und Typen der Marker im Versuchsaufbau abgestimmt ist.
* **Kalibrierte Extraktion:** `Code/detect_tags.py` wurde zu einem CLI-Tool umgebaut. Die RGB-Fisheye-Bilder werden vor der Detektion mit der Project-Aria-Kalibrierung in ein lineares Kameramodell rektifiziert. Damit basiert `solvePnP` nicht mehr fälschlich auf unverzerrten Pinhole-Annahmen für ein rohes Fisheye-Bild.
* **Eindeutige Markertrennung:** Infrastrukturmarker werden als `apriltag_36h11` mit den erlaubten IDs 0–5, Objektmarker als `aruco_4x4_50` mit den IDs 6–14 gespeichert. Nicht erlaubte beziehungsweise wahrscheinlich falsch positive IDs werden verworfen und im Laufprotokoll gezählt.
* **Erweitertes Ausgabeformat:** Pro Marker und Bildzeitpunkt werden vollständige Sequenz-ID, Markerfamilie, semantische Rolle, physische Größe, Translation, Rotationsvektor, Quaternion, Reprojektionsfehler und Markerfläche exportiert. Die Ausgabedateien verwenden die vollständige Sequenz-ID, um mehrdeutige Zuordnungen wie `Miro_1` versus `Miro_2` zu verhindern.
* **Der mathematische Ablauf (Sensor Fusion):**
  1. **Infrastruktur-Tracking (AprilTags 36h11):** AprilTag 0 an der Roboterreferenz (100 mm) und die Marker auf der Tischoberfläche (80 mm) werden detektiert, um eine stabile räumliche Referenz aufzubauen.
  2. **Objekt-Tracking (ArUco 4x4):** Die auf dem künstlichen Obst (z. B. Äpfel, Bananen) angebrachten kleinen 50-mm-Marker aus der Familie `DICT_4X4_50` werden parallel detektiert und über die IDs 6–14 eindeutig unterschieden.
  3. **3D-Pose & Export:** Über den PnP-Schätzer (`cv2.solvePnP`) werden Position und Orientierung zunächst relativ zur rektifizierten RGB-Kamera berechnet. Diese Posen werden zusammen mit `timestamp_ns` framegenau exportiert und erst in der Master-Pipeline in Device-, Welt- und Robotermarkerkoordinaten transformiert.

### Modul G: Multimodale Datenfusion (Sensor Alignment)
* **Native Gaze-Extraktion:** `Code/extract_multimodal_data.py` wurde von einem fest verdrahteten Hand-/Gaze-JSON-Prototyp zu einem wiederverwendbaren CLI-Extractor umgebaut. Er exportiert Eye-Gaze mit nativer Abtastrate, Validitätsmaske, Yaw, Pitch, Tiefe, Vergenzwerten sowie Blickpunkt und normalisierter Blickrichtung im CPF- und Device-Koordinatensystem. Die statischen Transformationen `T_Device_CPF` und `T_Device_Camera` werden direkt aus der VRS-Kalibrierung gelesen.
* **Autoritative Handquelle:** Für den Master-Datensatz werden die umfangreicheren MPS-Handtracking-Dateien verwendet. Sie enthalten beide Hände, 21 Landmarken, Wrist-Translation und -Quaternion, Palm-/Wrist-Normalen sowie Konfidenzen. Zeilen mit ungültiger Konfidenz werden explizit maskiert; die von MPS eingetragenen Nullkoordinaten werden nicht als echte Messwerte interpretiert.
* **Master-Dataset-Builder:** Das neue Skript `Code/build_master_dataset.py` verwendet die native Gaze-/RGB-Taktung von ungefähr 30 Hz als Zeitachse. MPS-Handdaten (ca. 60 Hz), SLAM (ca. 1 kHz) und Markerposen (ca. 30 Hz) werden über `pandas.merge_asof` mit konfigurierbaren Millisekunden-Toleranzen auf den jeweils nächsten Hardwarezeitstempel abgebildet.
* **Koordinatentransformationen:** Hand, Gaze und Objekte werden mit den kalibrierten Extrinsiken und der SLAM-Pose aus dem Device- beziehungsweise Kamerasystem in Weltkoordinaten transformiert. Wenn AprilTag 0 sichtbar ist, werden zusätzlich robotermarker-relative Positionen und Orientierungen berechnet. Der noch nicht vermessene physische Offset vom Marker zur realen Roboterbasis wird bewusst nicht angenommen.
* **Trainingsziele:** Jede Zeile erhält eines der drei Intentionslabels `continue`, `fetch` oder `handover`. Zusätzlich werden zukünftige Wrist-Positionen und -Quaternionen beider Hände für einen standardmäßigen Vorhersagehorizont von 1 s erzeugt. Fehlende zukünftige Posen erhalten Validitätsmasken.
* **Gaze-Objekt-Relationen:** Für jeden sichtbaren Objektmarker werden Abstand und Winkel zum Blickstrahl berechnet. Diese Größen können später für den Objektselektionskopf verwendet werden, ohne die noch fehlende Ground-Truth-Objekt-ID vorwegzunehmen.
* **Warum:** Das Ergebnis ist eine flache, zeitlich sortierte Master-Dataset-Matrix mit expliziten Koordinatensystemen, Messmasken und Zielwerten für den PyTorch-Transformer.

### Modul H: Dataset-QA und Manifest-Generierung
* **Funktion:** Zur systematischen Kontrolle des Datensatzes wurde das Skript `Code/dataset_qa.py` eingeführt. Es erstellt aus den aktuell vorhandenen Roh- und Zwischendaten eine Sequenzübersicht in `Data_collection/dataset_manifest.csv` sowie einen aggregierten Bericht in `Data_collection/dataset_qa_report.json`.
* **Validierte Artefakte pro Sequenz:** Das Skript prüft, ob die jeweilige `.vrs`-Datei, die konvertierte `.mp4`-Datei, der passende MPS-Ordner, `hand_tracking_results.csv`, `closed_loop_trajectory.csv`, optionale ArUco-CSV-Dateien sowie ein Eintrag in `timestamps_summary.json` vorhanden sind.
* **Backup-Abgleich:** Zusätzlich wird der Backup-Ordner `BackUp_Videos/` mit `Data_collection/Data_vrs/` abgeglichen. Dadurch wird sichtbar, ob Arbeitsordner und Backup dieselben VRS-Aufnahmen enthalten und ob gleichnamige Dateien Größenunterschiede aufweisen.
* **Trainings-Ausschlusskandidaten:** Sequenzen mit Präfixen wie `Test_*`, `Unknown_*` oder `unknown_*` werden im Manifest als `include_in_training=False` markiert und mit einem Ausschlussgrund versehen. Dadurch können Test- und unklare Aufnahmen im Arbeitsordner bleiben, ohne später versehentlich in das Training einzufließen.
* **MPS-Fortschrittskontrolle:** Fehlende und unvollständige MPS-Ausgaben werden nun explizit im QA-Report zusammengefasst. Das ist besonders nützlich während der laufenden Verarbeitung mit `aria_mps single -i`.
* **Label-QA:** Zusätzlich wird kontrolliert, ob die Trigger `START`, `SECOND`, `DONE` und `THIRD` vollständig vorhanden sind und der erwarteten Reihenfolge `START -> SECOND -> DONE -> THIRD` folgen. Daraus werden Phasendauern für `continue`, `fetch` und `handover` berechnet.
* **Phasenspezifische Hand-QA:** Die reine Existenz einer Handtracking-Datei reicht nicht mehr als Qualitätskriterium. Für das Fenster `DONE -> THIRD` werden Zeilenzahl und gültige Anteile der linken, rechten und mindestens einer Hand berechnet. Sequenzen ohne gültige Hand im Handover werden als `missing_handover_hand_tracking` blockiert; eine Abdeckung unter standardmäßig 80 % erzeugt `low_handover_hand_tracking`.
* **Marker-Zeitbereichsprüfung:** Markerdateien werden nur akzeptiert, wenn ihr `timestamp_ns`-Bereich mit dem Triggerbereich derselben Aufnahme überlappt. Dadurch wurde erkannt, dass die Legacy-Datei `aruco_poses_Miro_1.csv` zeitlich zu `Miro_2` gehört und nicht für `Miro_1` verwendet werden darf.
* **Status- und Handlungsempfehlung:** Für jede Sequenz werden ein `status` und eine `next_action` erzeugt, z. B. `fix_timestamps`, `convert_mp4`, `download_or_process_mps`, `run_aruco_extraction` oder `ready_for_master_merge`. Das Skript verändert keine Rohdaten; es dient ausschließlich als Validierungs- und Fortschrittskontrolle.

---

## 4. Übersicht der Pipeline-Zustände (Phasen-Grenzlogik)

Die kontinuierlichen Datenströme werden anhand der verbalen Meilensteine zeitlich segmentiert. Die Audio-Trigger definieren exakt die Start- und Endpunkte (Grenzen) der jeweiligen Phase für das spätere Supervised Learning.

| Phase / Segment | Start-Trigger | End-Trigger | Aktivität des Probanden | Intentionsklasse (ML-Target) | Zielsetzung für das ML-Modell |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `"start"` | `"second"` | Proband manipuliert, inspiziert oder lagert Objekte frei um (Explorationsphase). | **continue** | Stabilisierung der Baseline; Erkennung von Alltagsbewegungen ohne Roboterbezug (Hintergrundrauschen). |
| **2** | `"second"` | `"done"` | Proband fokussiert visuell ein Zielobjekt nahe des (simulierten) Roboterarms. | **fetch** | *Intention Alignment*: Verknüpfung von Gaze-Vektor und räumlicher Objekt-ID/YOLO-Klasse zur Zielerkennung. |
| **3** | `"done"` | `"third"` | Proband streckt seine offene Hand in Richtung des Roboters aus (Einleitung der Übergabe). | **handover** | Aktivierung des Übergabe-Zustands + *Hand Position Prediction*: Kontinuierliche Trajektorien-Vorhersage des Handgelenks für die Pfadplanung. |

---

## 5. Durchführung der Probanden-Studie & Datenerhebung

Zur Evaluierung des Frameworks und zum späteren Training des Multimodal Transformers wurde eine systematische Datenerhebung im Labor durchgeführt. Der Fokus lag auf der Erzeugung einer variantenreichen, realistischen Interaktionsumgebung bei gleichzeitiger Wahrung einer präzisen mathematischen Kontrollstruktur (Ground Truth).

### 5.1 Kohorte und Datengröße
* **Probandenanzahl:** $N = 12$ Teilnehmer.
* **Sequenzen pro Proband:** Jeweils 8 bis 10 vollständige Videosequenzen.
* **Gesamtdatensatz:** Ca. 96 bis 120 multimodale, synchronisierte Aufnahmesequenzen.
* **Durchschnittliche Sequenzlänge:** Ca. 25–40 Sekunden pro Durchlauf.

### 5.2 Versuchsaufbau und Varianzkontrolle
Als Interaktionsobjekte wurde künstliches Plastik-Obst verwendet. Um ein Überfitten des Modells auf feste Raumkoordinaten zu verhindern, wurden folgende Parameter vor jeder Aufnahme modifiziert:
* Die physische Position des Tisches relativ zum Roboter sowie der Abstand zwischen dem Tisch-AprilTag und der Roboterbasis.
* Die Anzahl, Auswahl und geometrische Startplatzierung der ArUco-Objekte auf der Tischoberfläche (stochastische Verteilung).

### 5.3 Chronologischer Ablauf einer Sequenz
Jeder Durchlauf folgt einer strikten zeitlichen Abfolge. Die verbalen Befehle des Probanden dienen als exakte Zeitstempel für die Phasen-Segmentierung im Post-Processing:

1. **Explorations- und Manipulationsphase (ca. 10–20 Sekunden):** Der Trigger `"start"` öffnet die Sequenz. Der Proband spielt frei mit den Gegenständen (Umlagern, Sortieren). Das Zeitfenster von `"start"` bis `"second"` liefert die Trainingsdaten für die Klasse `continue`.
2. **Fixations- und Pointing-Phase (ca. 2–4 Sekunden):** Nach dem Trigger `"second"` fokussiert der Proband visuell ein spezifisches Zielobjekt nahe des Roboterarms. Das Zeitfenster von `"second"` bis `"done"` generiert die Trainingsdaten für die Klasse `fetch` (Ziel-Fokussierung / Fetching).
3. **Trajektorien- und Übergabephase (ca. 5–10 Sekunden):** Mit dem Trigger `"done"` leitet der Proband die physische Bewegung ein und streckt seine offene Hand flach in Richtung des Roboters aus. Das Zeitfenster von `"done"` bis zum finalen Trigger `"third"` liefert die Ground Truth für die Klasse `handover` und dient dem Regressions-Kopf als Berechnungsfenster für die zukünftige Handposition-Prognose. Mit `"third"` wird die Interaktion erfolgreich abgeschlossen und die Aufnahme beendet.

---

## 6. Aktueller Verarbeitungsstand und QA-Ergebnis

Nach der Erweiterung der Audio-Annotation wurde die automatische Command-Erkennung erneut auf allen damals vorhandenen VRS-Dateien ausgeführt. Da das lokal gecachte `medium.en`-Modell unvollständig war und der vollständige CPU-Batch damit nicht praktikabel abschloss, wurde der vollständige Batchlauf mit `tiny.en` durchgeführt. Die verbesserte Audio-Normalisierung, das Fuzzy-Matching und die Sequenzlogik wurden dabei verwendet.

Anschließend wurde der neu angelegte Backup-Ordner `BackUp_Videos/` mit dem Arbeitsordner `Data_collection/Data_vrs/` abgeglichen. Dabei wurden 94 fehlende VRS-Dateien in den Arbeitsordner kopiert, ohne bestehende Dateien zu überschreiben. Aktuell enthalten beide Ordner dieselben 136 VRS-Dateinamen.

### 6.1 Erzeugte Artefakte
* **Backup der vorherigen Whisper-Ausgabe:** `Data_collection/Data_vrs/timestamps_summary.before_audio_fix.json`
* **Aktualisierte automatische Trigger-Datei:** `Data_collection/Data_vrs/timestamps_summary.json`
* **Debug-Ausgabe mit Kandidaten und Warnungen:** `Data_collection/Data_vrs/timestamps_debug.json`
* **Normalisierte Debug-Audios:** `Data_collection/Data_vrs/debug_audio/` (42 WAV-Dateien)
* **Dataset-Manifest:** `Data_collection/dataset_manifest.csv`
* **QA-Report:** `Data_collection/dataset_qa_report.json`
* **Archivierte Audio-Smoke-Test-Ausgabe:** `audio_smoke_results/job_2151504/`
* **Archivierte Audio-Batch-Ausgaben:** `audio_batch_backups/job_2151515/` und `audio_batch_backups/job_2151518/`
* **VRS-Health-Check-Berichte:** zehn Berichte unter `Test_SLAM/`, davon neun unterschiedliche Julienco-Aufnahmen und ein zusätzlicher Bericht aus dem Einzeltest-Verzeichnis
* **Reproduzierbare Clusterumgebung:** `singularity/singularity.recipe`
* **Kalibrierte Markerposen der Golden Sequence:** `Data_collection/aruco_poses_Jona_6_20260616_182111.csv`
* **Annotiertes Marker-Kontrollvideo:** `Data_collection/aruco_Jona_6_20260616_182111.mp4`
* **Nativer Gaze-Export:** `Data_collection/gaze_Jona_6_20260616_182111.csv`
* **Master-Dataset:** `Data_collection/master_datasets/Jona_6_20260616_182111_master.csv`
* **Master-Dataset-Validierung:** `Data_collection/master_datasets/Jona_6_20260616_182111_master_report.json`

### 6.2 Ergebnis des aktuellen QA-Laufs

| Kennzahl | Wert |
| :--- | ---: |
| Gesamtzahl erkannter Sequenzen | 136 |
| Trainingskandidaten (`include_in_training=True`) | 124 |
| Ausgeschlossene Test-/Unknown-Sequenzen | 12 |
| Aktuell nutzbar (`valid` oder `valid_with_warnings`) | 60 |
| Vollständig validiert (`valid`) | 1 |
| Nutzbar mit Folgearbeit (`valid_with_warnings`) | 59 |
| Status `partial_timestamps` | 24 |
| Sequenzen mit mindestens einem unvollständigen Triggerproblem | 51 |
| Fehlender MPS-Ordner | 0 |
| Unvollständige MPS-Ausgaben | 47 |
| Fehlende Handtracking-Datei | 46 |
| Keine gültige Hand im Handover | 2 |
| Zu geringe Handabdeckung im Handover | 1 |
| Backup/Data_vrs Namensabgleich | 136 / 136, keine fehlenden Dateien |
| Vorhandene MP4-Dateien | 32 |
| Fehlende MP4-Dateien | 104 |
| Fehlende Marker-CSV-Dateien | 131 |
| Marker-CSV mit falschem Zeitbereich | 1 |

### 6.3 Nächste automatisch abgeleitete Arbeitsschritte

| `next_action` | Anzahl Sequenzen | Bedeutung |
| :--- | ---: | :--- |
| `download_or_process_mps` | 47 | Handtracking oder SLAM sind noch fehlend beziehungsweise unvollständig. |
| `fix_timestamps` | 27 | Fehlende Trigger müssen manuell oder halbautomatisch korrigiert werden. |
| `run_aruco_extraction` | 57 | Labels und MPS sind grundsätzlich vorhanden; kalibrierte Markerposen fehlen noch. |
| `review_or_exclude_sequence` | 3 | Handtracking im Handover fehlt oder unterschreitet die Mindestabdeckung. |
| `manual_review` | 1 | Eine ungewöhnlich kurze Phase muss manuell geprüft werden. |
| `ready_for_master_merge` | 1 | Sequenz ist für den nächsten Merge-Schritt vollständig vorbereitet. |

Ein wichtiger Fortschritt gegenüber dem vorherigen Stand ist, dass die QA nun nicht nur Dateiexistenz, sondern die tatsächliche zeitliche Nutzbarkeit der Hand- und Markerdaten bewertet. Der aktuelle Hauptengpass liegt bei 47 unvollständigen MPS-Ausgaben, 27 Sequenzen mit zu korrigierenden Triggern und 57 noch ausstehenden kalibrierten Markerextraktionen.

### 6.4 Durchgeführte Skript-Tests
* `python3 -m py_compile Code/dataset_qa.py Code/vrs_to_mp4_all.py`
* `python3 Code/dataset_qa.py`
* `python3 Code/dataset_qa.py --help`
* `python3 Code/vrs_to_mp4_all.py --help`
* `python3 Code/vrs_to_mp4_all.py --dry-run --limit 5`
* `python3 -m py_compile Code/detect_tags.py Code/extract_multimodal_data.py Code/build_master_dataset.py`
* `conda run -n aria_conda python Code/detect_tags.py --help`
* Marker-Smoke-Test mit drei Frames von `Miro_2_20260604_153451`
* Vollständige Markerextraktion und visuelle Kontrolle für `Jona_6_20260616_182111`
* `conda run -n aria_conda python Code/build_master_dataset.py --sequence-id Jona_6_20260616_182111 --overwrite`
* Struktur-, Zeit-, Label-, Transformations- und Overwrite-Guard-Prüfungen des erzeugten Master-Datensatzes

Der damalige Dry-Run der MP4-Konvertierung erkannte 136 VRS-Dateien, 31 bereits vorhandene MP4-Dateien und 105 noch fehlende MP4-Exporte. Dabei wurde keine echte Konvertierung gestartet. Durch den späteren Export der Golden Sequence liegt der aktuelle Stand bei 32 vorhandenen und 104 fehlenden MP4-Dateien.

### 6.5 Audio-Verarbeitung auf dem TCML-Cluster

Die GPU-Unterstützung des Audio-Skripts wurde für die Batchverarbeitung auf dem TCML-Cluster vorbereitet. Neben der Wahl des Whisper-Modells können Gerät und Rechentyp nun explizit übergeben werden. Dadurch ist insbesondere die Kombination `--device cuda --compute-type float16` auf einem GPU-Knoten möglich, während lokale CPU-Läufe weiterhin mit `int8` durchgeführt werden können.

Die Ergebnisse mehrerer Clusterläufe wurden getrennt vom aktiven Datensatz im Repository archiviert:

| Job | Umfang | Archivierte Dateien | Zweck |
| :--- | ---: | :--- | :--- |
| `2151504` | 10 VRS-Sequenzen | `timestamps_summary.json`, `timestamps_debug.json`, `problem_vrs_list.txt` | Smoke-Test der Verarbeitungskette |
| `2151515` | 136 VRS-Sequenzen | vorheriger Summary-Stand, neuer Summary-Stand und Debug-Ausgabe | Vollständiger Batchlauf mit reproduzierbarem Vorher-/Nachher-Vergleich |
| `2151518` | 136 VRS-Sequenzen | vorheriger Summary-Stand, neuer Summary-Stand und Debug-Ausgabe | Weiterer vollständiger Batchlauf zur Ergebnisprüfung |

Die beiden vollständigen Batchläufe erzeugten unterschiedliche Ergebnisdateien. Die Archive werden deshalb als Laufprotokolle und nicht als austauschbare Kopien behandelt. Die produktiv verwendete Trigger-Datei bleibt `Data_collection/Data_vrs/timestamps_summary.json`; ein archivierter Jobstand darf sie nur nach separater QA ersetzen.

### 6.6 VRS-Health-Checks der SLAM-Testaufnahmen

Für neun unterschiedliche Julienco-Testaufnahmen wurden detaillierte `vrs_health_check.json`-Berichte erzeugt. Ein zehnter Bericht liegt zusätzlich im Verzeichnis `Test_SLAM/mps_test_single/` und dokumentiert den separaten Einzeltest derselben ersten Sequenz. Die Berichte prüfen unter anderem Vollständigkeit, Zeitstempelmonotonie, Drop-Raten und Synchronität der Kamera-, Audio-, IMU-, Gaze-, Hand- und VIO-Ströme.

Alle vier ausgewerteten Aria-Gen-2-Prüfprofile (`Default`, `CI`, `Location` und `Handtracking`) markieren diese Dateien formal als `fail`. Als fehlgeschlagener Check wird jeweils ausschließlich `file_level_checks.streams_match_profile` aufgeführt. Die Detailmeldung nennt zusätzliche beziehungsweise nicht zugeordnete Gen-2-Ströme wie GPS, Time-Domain-Mapping und Battery Data. Das Ergebnis ist daher zunächst als Profilkompatibilitätsproblem des Health-Check-Regelsatzes zu behandeln und nicht automatisch als Nachweis beschädigter Sensordaten. Bei einzelnen Aufnahmen werden zusätzlich Warnungen zur größten Periodenabweichung der SLAM-Kameras ausgegeben.

### 6.7 Repository- und Dokumentationspflege

* Die technische Statusdokumentation und die Problemliste wurden im Verzeichnis `Thesis/` gebündelt; zusätzlich wurde die TCML-Dokumentation als `Thesis/TCML_Documentation_2025-10.28.pdf` aufgenommen.
* Die veralteten Marker-PDFs unter `Marker/` wurden entfernt. Die im Versuchsaufbau verwendeten Markerfamilien und physischen Größen bleiben in Abschnitt 3 dokumentiert.
* `.gitignore` wurde um generierte Python-Dateien, lokale Daten- und Backupordner sowie große Audio-/Videoformate erweitert. Dadurch sollen Rohdaten und abgeleitete Medien nicht versehentlich eingecheckt werden; gezielt versionierte QA- und Ergebnisarchive bleiben davon unberührt.

### 6.8 Golden Sequence und erster Master-Datensatz

Für die End-to-End-Validierung wurde zunächst `Miro_2_20260604_153451` untersucht. Marker, Trigger und SLAM waren zeitlich kompatibel, im gesamten Handover-Fenster von 78 Handtracking-Zeilen war jedoch weder die linke noch die rechte Hand gültig. Die Sequenz wird deshalb nicht für die Handpositionsvorhersage verwendet und von der erweiterten QA als `missing_handover_hand_tracking` markiert.

Als Golden Sequence wurde anschließend `Jona_6_20260616_182111` ausgewählt. Sie besitzt ein 5,44 s langes Handover-Fenster mit 327 Handtracking-Zeilen. Mindestens eine Hand ist in 100 % dieser Zeilen gültig; die linke Hand erreicht 100 %, die rechte Hand 90,52 %. Das annotierte Kontrollvideo bestätigt gleichzeitig sichtbare Hand-, Roboter- und Objektmarker während der Übergabe.

Die kalibrierte Markerextraktion für diese Sequenz verarbeitete 1.096 RGB-Frames und erzeugte 9.550 akzeptierte 6-DoF-Markerposen. Drei nicht erlaubte ArUco-IDs wurden als wahrscheinliche Fehlpositive verworfen. Der mediane Reprojektionsfehler beträgt 0,48 px; 95 % der Posen liegen unter 1,41 px. Im Handover-Fenster sind Roboteranker und Objektmarker in allen ungefähr 164 RGB-Frames vorhanden.

Der daraus erzeugte Master-Datensatz besitzt folgende Eigenschaften:

| Kennzahl | Wert |
| :--- | ---: |
| Zeitpunkte bei ungefähr 30 Hz | 936 |
| Spalten/Features einschließlich Masken und Targets | 576 |
| Dauer | 31,167 s |
| `continue` | 576 Zeilen |
| `fetch` | 196 Zeilen |
| `handover` | 164 Zeilen |
| Gültige Gaze-Samples | 95,19 % |
| Gültige linke Hand | 95,41 % |
| Gültige rechte Hand | 87,82 % |
| SLAM-Matchrate | 96,15 % |
| Sichtbarkeit Roboter-AprilTag 0 | 88,68 % |
| Gültiges zukünftiges linkes Wrist-Target bei +1 s | 84,08 % |
| Gültiges zukünftiges rechtes Wrist-Target bei +1 s | 76,50 % |

Die zeitliche Zuordnung liegt für Handtracking maximal 5,4 ms und für SLAM maximal 0,36 ms vom jeweiligen Master-Zeitpunkt entfernt. Der aus SLAM und AprilTag 0 berechnete Roboteranker schwankt in Weltkoordinaten nur ungefähr 2–3 mm, was die implementierte Transformationskette für diesen ersten Datensatz plausibilisiert.

Noch offene Ground-Truth- und Kalibrierungspunkte:

* **Zielobjekt:** `target_object_id` bleibt aktuell `-1` beziehungsweise unbekannt. Blickwinkel zu allen sichtbaren Objektmarkern sind als Eingabefeatures vorhanden, dürfen aber ohne manuell oder protokollbasiert bestätigte Objekt-ID nicht als Ground Truth behandelt werden.
* **Empfangende Hand:** Zukünftige Posen werden vorerst für beide Hände erzeugt. Welche Hand tatsächlich die Zielhand ist, muss pro Sequenz gelabelt oder über eine klar dokumentierte Regel bestimmt werden.
* **Roboterbasis:** Die robotermarker-relativen Koordinaten verwenden AprilTag 0 als Referenz. Vor einer realen Roboteransteuerung muss noch die starre Transformation vom Marker zur physischen Roboterbasis vermessen und angewendet werden.
