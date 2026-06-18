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
* **Warum:** Dies ermöglicht eine schnelle visuelle Inspektion der aufgezeichneten Daten, das reibungslose Teilen von Sequenzen und dient als Grundlage für die überlagerte Visualisierung der 3D-Koordinaten.

### Modul D: Synchronisiertes Audio-Annotation-System (Aria Gen 2 Codec-Handling)
* **Funktion:** Da die Meta Aria Gen 2 komprimierte Audio-Datenblöcke verwendet, mischt das Modul die channels des Mikrofon-Arrays mathematisch zu einer Mono-Spur zusammen und exportiert ein unkomprimiertes 16-Bit-PCM-WAV-Signal (16 kHz).
* Dieses Signal wird an ein lokales **Faster-Whisper-Modell** übergeben, welches die Spur nach der exakten sequentiellen Trigger-Abfolge (*"Start"*, *"Second"*, *"Done"*, *"Third"*) scannt. Über den Aufruf `get_first_time_ns` wird der Hardware-Startzeitpunkt (`TimeDomain.DEVICE_TIME`) abgegriffen und mit den relativen Wort-Zeitstempeln von Whisper verrechnet.
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