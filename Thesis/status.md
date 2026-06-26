# Technische Dokumentation: Entwicklungsstand Framework zur multimodalen Intentionsschätzung

## 1. System-Architektur & Infrastruktur

Um eine reproduzierbare, plattformunabhängige und isolierte Entwicklungsumgebung zu gewährleisten und Konflikte mit dem globalen System zu vermeiden, wurde eine dedizierte Python-Infrastruktur aufgesetzt.

* **Virtual Environment (Conda):** Es wurde eine virtuelle Umgebung namens `aria_conda` mittels Anaconda eingerichtet. Dies stellt sicher, dass alle hardwarenahen, vorkompilierten C++-Bindings des Meta SDKs exakt mit den Python-Bibliotheken harmonieren.
* **Python-Version:** Verwendung von Python 3.10, um maximale Stabilität mit den Deep-Learning-Bibliotheken (PyTorch) und dem Meta-Aria-Ecosystem zu garantieren.
* **Hardware-Abstraktion:** Die Pipeline nutzt auf Linux-Infrastrukturen die Nvidia CUDA-Beschleunigung für das parallele Deep-Learning-Training. Die lokale Vorverarbeitung unterstützt zudem hardwarebeschleunigte Dekodierung, fällt bei Bedarf jedoch nahtlos auf einen robusten Software-Dekoder (`H.265 SW decoder via xprs`) zurück.

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
* **Funktion:** Da die Meta Aria Gen 2 komprimierte Audio-Datenblöcke verwendet, mischt das Modul die channels des Mikrofon-Arrays mathematisch zu einer Mono-Spur zusammen und exportiert ein unkomprimiertes 16-Bit-PCM-WAV-Signal (16 kHz).
* **Robuste Audio-Vorverarbeitung:** Da sich bei den bisherigen Aufnahmen gezeigt hat, dass die Sprachkommandos teilweise sehr leise aufgenommen wurden, wurde das Skript `Code/speech_recognition_demo.py` erweitert. Die aus dem VRS-Container extrahierte Audiospur wird nun zuerst per RMS-Normalisierung angehoben. Wenn `ffmpeg` verfügbar ist, wird zusätzlich eine Whisper-optimierte Vorverarbeitung angewendet (`highpass`, Kompressor und Loudness-Normalisierung auf ca. -16 LUFS). Dadurch wird die Erkennung leiser kurzer Befehle deutlich stabiler, ohne dass die Rohdaten verändert werden.
* **Verbesserte Command-Erkennung:** Das normalisierte Signal wird an ein lokales **Faster-Whisper-Modell** übergeben, welches die Spur nach der sequentiellen Trigger-Abfolge (*"Start"*, *"Second"*, *"Done"*, *"Third"*) scannt. Die Erkennung basiert nicht mehr nur auf exakten Worttreffern, sondern nutzt zusätzlich Fuzzy-Matching und Alias-Listen für typische Fehltranskriptionen (z. B. `"star"` für `"start"` oder `"down"` für `"done"`). Die finale Auswahl wird anschließend über die erwartete chronologische Reihenfolge `START -> SECOND -> DONE -> THIRD` stabilisiert.
* **Debug- und QA-Ausgaben:** Die finale Datei `timestamps_summary.json` bleibt für nachgelagerte Pipelines kompatibel und enthält nur die final ausgewählten Trigger-Zeitpunkte. Zusätzlich erzeugt das Skript nun `timestamps_debug.json`, in der erkannte Kandidaten, Match-Scores und Warnungen wie fehlende Trigger gespeichert werden. Optional können mit `--keep-debug-audio` die normalisierten WAV-Dateien zur manuellen Kontrolle abgelegt werden.
* Über den Aufruf `get_first_time_ns` wird weiterhin der Hardware-Startzeitpunkt (`TimeDomain.DEVICE_TIME`) abgegriffen und mit den relativen Wort-Zeitstempeln von Whisper verrechnet. Die daraus resultierenden Trigger-Zeitpunkte bleiben somit nanosekundengenau mit den Sensorströmen synchronisiert.
* **Warum:** Für das Supervised Learning wird so eine nanosekundengenaue, automatisierte zeitliche Synchronität zwischen dem gesprochenen Befehl und den physikalischen Augen- und Handbewegungen des Probanden hergestellt.
* **Architektonischer Ausschluss von Feature-Leckagen:** Die extrahierten Audio-Befehle dienen ausschließlich der automatisierten Offline-Phasensegmentierung (Ersatz für manuelles Frame-Labeling). Die Audiospur wird **not** als Feature in die finale KI-Dateneinspeisung übernommen. Das neuronale Netz lernt rein auf den physikalischen Geometrie- und Bilddaten.

---

## 3. Structure der finalen KI-Dateneinspeisung & Feature-Matrix

Ein wichtiger architektonischer Meilenstein ist die Trennung von visueller Kontrolle und mathematischem KI-Training. Die originalen hochfrequenten Sensordaten werden nicht zerschnitten, sondern über Nanosekunden-Zeitstempel in eine Master-Tabelle überführt, über die der PyTorch-`DataLoader` mittels "Sliding Window" iteriert.

### Modul E: Cloud-optimierte Tracking-Extraktion (MPS)
* **Funktion:** Anstelle der reinen On-Device-Berechnung werden die Meta Perception Services (MPS) genutzt, um eine hochpräzise Offline-Optimierung (SLAM) über das gesamte Video laufen zu lassen. 
* **Extraktion der Features:**
  * **Hand-Pose:** Extraktion der geglätteten 3D-Skelett-Landmarks (z. B. `wrist_position`) im Brillen-Koordinatensystem.
  * **SLAM (Brillenpose):** Hochfrequente Welt-Koordinaten ($tx, ty, tz$ und Quaternionen) der Brille im initialisierten Raum.
  * **Eye-Gaze:** Abfrage der Blickrichtungswerte aus dem *Central Pupil Frame (CPF)* über die geräteinterne `TimeQueryOptions.CLOSEST`-Metrik.

### Modul F: Duale 3D-Lokalisierung (AprilTag & ArUco 4x4 Mix)
* **Funktion:** Eine zweigleisige geometrische 3D-Posenschätzung (`cv2.aruco.ArucoDetector`), die auf die unterschiedlichen physischen Größen und Typen der Marker im Versuchsaufbau abgestimmt ist.
* **Der mathematische Ablauf (Sensor Fusion):**
  1. **Infrastruktur-Tracking (AprilTags 36h11):** Die Marker an der Roboterbasis (100 mm) und auf der Tischoberfläche (80 mm) werden detektiert, um die Tischebene und den globalen Nullpunkt stabil zu definieren.
  2. **Objekt-Tracking (ArUco 4x4):** Die auf dem künstlichen Obst (z. B. Äpfel, Bananen) angebrachten kleinen 50 mm Marker aus der Familie `DICT_4X4_50` werden parallel detektiert. Der Roboterarm selbst wird simulativ über einen markanten ArUco-Marker im Video-Feed lokalisiert.
  3. **3D-Pose & Export:** Über den PnP-Schätzer (`cv2.solvePnP`) werden die hochpräzisen 3D-Raumkoordinaten ($X, Y, Z$) relativ zur Brille berechnet. Diese Posen werden zusammen mit dem `timestamp_ns` framegenau in eine strukturierte CSV-Datei exportiert.

### Modul G: Multimodale Datenfusion (Sensor Alignment)
* **Funktion:** Eine asynchrone Zeitreihen-Zusammenführung (`pandas merge_asof`). Da die RGB/ArUco-Kamera, das Eye-Tracking und das MPS-SLAM mit leicht unterschiedlichen Taktungen laufen, synchronisiert dieses Modul alle Datenströme auf Basis des Aufnahme-Zeitstempels (`timestamp_ns`) mit einer maximalen Toleranz im Millisekundenbereich.
* **Warum:** Das Ergebnis ist eine fehlerfreie Master-Dataset-Matrix, die pro Zeile (Frame) alle benötigten multimodalen Features für den PyTorch-Transformer enthält.

### Modul H: Dataset-QA und Manifest-Generierung
* **Funktion:** Zur systematischen Kontrolle des Datensatzes wurde das Skript `Code/dataset_qa.py` eingeführt. Es erstellt aus den aktuell vorhandenen Roh- und Zwischendaten eine Sequenzübersicht in `Data_collection/dataset_manifest.csv` sowie einen aggregierten Bericht in `Data_collection/dataset_qa_report.json`.
* **Validierte Artefakte pro Sequenz:** Das Skript prüft, ob die jeweilige `.vrs`-Datei, die konvertierte `.mp4`-Datei, der passende MPS-Ordner, `hand_tracking_results.csv`, `closed_loop_trajectory.csv`, optionale ArUco-CSV-Dateien sowie ein Eintrag in `timestamps_summary.json` vorhanden sind.
* **Backup-Abgleich:** Zusätzlich wird der Backup-Ordner `BackUp_Videos/` mit `Data_collection/Data_vrs/` abgeglichen. Dadurch wird sichtbar, ob Arbeitsordner und Backup dieselben VRS-Aufnahmen enthalten und ob gleichnamige Dateien Größenunterschiede aufweisen.
* **Trainings-Ausschlusskandidaten:** Sequenzen mit Präfixen wie `Test_*`, `Unknown_*` oder `unknown_*` werden im Manifest als `include_in_training=False` markiert und mit einem Ausschlussgrund versehen. Dadurch können Test- und unklare Aufnahmen im Arbeitsordner bleiben, ohne später versehentlich in das Training einzufließen.
* **MPS-Fortschrittskontrolle:** Fehlende und unvollständige MPS-Ausgaben werden nun explizit im QA-Report zusammengefasst. Das ist besonders nützlich während der laufenden Verarbeitung mit `aria_mps single -i`.
* **Label-QA:** Zusätzlich wird kontrolliert, ob die Trigger `START`, `SECOND`, `DONE` und `THIRD` vollständig vorhanden sind und der erwarteten Reihenfolge `START -> SECOND -> DONE -> THIRD` folgen. Daraus werden Phasendauern für `continue`, `fetch` und `handover` berechnet.
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

### 6.2 Ergebnis des aktuellen QA-Laufs

| Kennzahl | Wert |
| :--- | ---: |
| Gesamtzahl erkannter Sequenzen | 136 |
| Trainingskandidaten (`include_in_training=True`) | 124 |
| Ausgeschlossene Test-/Unknown-Sequenzen | 12 |
| Aktuell nutzbar (`valid` oder `valid_with_warnings`) | 21 |
| Vollständig validiert (`valid`) | 1 |
| Nutzbar mit Folgearbeit (`valid_with_warnings`) | 20 |
| Unvollständige Trigger (`partial_timestamps`) | 18 |
| Fehlender MPS-Ordner | 37 |
| Fehlendes oder unvollständiges Handtracking | 59 |
| Fehlende SLAM-Trajektorie | 1 |
| Backup/Data_vrs Namensabgleich | 136 / 136, keine fehlenden Dateien |
| Vorhandene MP4-Dateien | 31 |
| Fehlende MP4-Dateien laut Dry-Run | 105 |

### 6.3 Nächste automatisch abgeleitete Arbeitsschritte

| `next_action` | Anzahl Sequenzen | Bedeutung |
| :--- | ---: | :--- |
| `download_or_process_mps` | 97 | MPS-Daten, Handtracking oder SLAM sind noch fehlend oder unvollständig. |
| `fix_timestamps` | 18 | Fehlende Trigger müssen manuell oder halbautomatisch korrigiert werden. |
| `run_aruco_extraction` | 14 | Labels und MPS sind grundsätzlich vorhanden; ArUco-Posen fehlen noch. |
| `convert_mp4` | 6 | Sequenzen sind grundsätzlich nutzbar, aber MP4-Exports fehlen noch. |
| `ready_for_master_merge` | 1 | Sequenz ist für den nächsten Merge-Schritt vollständig vorbereitet. |

Ein wichtiger Fortschritt gegenüber dem vorherigen Stand ist, dass nach der verbesserten Audio-Erkennung keine Sequenz mehr wegen `bad_timestamp_order` klassifiziert wird. Durch den Backup-Abgleich ist der VRS-Arbeitsbestand nun vollständig. Der aktuelle Hauptengpass liegt dadurch nicht mehr beim Vorhandensein der Rohdaten, sondern bei der noch laufenden MPS-Verarbeitung sowie bei 18 Sequenzen mit unvollständigen Triggern (`partial_timestamps`), insbesondere fehlendem `DONE` oder `THIRD`.

### 6.4 Durchgeführte Skript-Tests
* `python3 -m py_compile Code/dataset_qa.py Code/vrs_to_mp4_all.py`
* `python3 Code/dataset_qa.py`
* `python3 Code/dataset_qa.py --help`
* `python3 Code/vrs_to_mp4_all.py --help`
* `python3 Code/vrs_to_mp4_all.py --dry-run --limit 5`

Der Dry-Run der MP4-Konvertierung erkannte 136 VRS-Dateien, 31 bereits vorhandene MP4-Dateien und 105 noch fehlende MP4-Exporte. Es wurde dabei keine echte Konvertierung gestartet.
