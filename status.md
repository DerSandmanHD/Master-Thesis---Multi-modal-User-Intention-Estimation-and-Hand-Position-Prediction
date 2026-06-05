# Technische Dokumentation: Entwicklungsstand Framework zur multimodalen Intentionsschätzung

## 1. System-Architektur & Infrastruktur

Um eine reproduzierbare, plattformunabhängige und isolierte Entwicklungsumgebung zu gewährleisten und Konflikte mit dem globalen System zu vermeiden, wurde eine dedizierte Python-Infrastruktur aufgesetzt.

* **Virtual Environment (Conda):** Es wurde eine virtuelle Umgebung namens `aria_conda` mittels Anaconda eingerichtet. Dies stellt sicher, dass alle hardwarenahen, vorkompilierten C++-Bindings des Meta SDKs exakt mit den Python-Bibliotheken harmonieren.
* **Python-Version:** Verwendung von Python 3.10, um maximale Stabilität mit den Deep-Learning-Bibliotheken (PyTorch) und dem Meta-Aria-Ecosystem zu garantieren.
* **Hardware-Abstraktion:** Die Pipeline nutzt auf Linux-Infrastrukturen die Nvidia CUDA-Beschleunigung für das parallele Deep-Learning-Training. Die lokale Vorverarbeitung unterstützt zudem hardwarebeschleunigte Dekodierung, fällt bei Bedarf jedoch nahtlos auf einen robusten Software-Dekoder (`H.265 SW decoder via xprs`) zurück.

---

## 2. Implementierte Software-Module & Kernfunktionen

### Modul A: Daten-Parsing & Sensor-Streaming (`projectaria_tools`)
* **Funktion:** Zugriff, Aktivierung und Dekapselung des proprietären Meta `.vrs`-Datenformats. Das Skript initialisiert und synchronisiert die hochfrequenten Datenströme der RGB-Weitwinkelkamera, der Eye-Tracking-Infrarotkameras, des 7-Kanal-Mikrofon-Arrays, der On-Device-Hand-Tracking-Schnittstelle sowie der IMU- und VIO-Sensoren.
* **Warum:** Aus Datenschutz- und Performancegründen speichert das Aria-System Sensordaten nicht als Standard-Multimediadatei. Ohne dieses Modul wäre ein zeitsynchroner, framegenauer Zugriff auf die Rohdaten für das spätere KI-Training unmöglich.

### Modul B: Automatische Kamera-Kalibrierung & Rektifizierung
* **Funktion:** Automatisches Auslesen der werkseitigen intrinsischen Kameraparameter direkt aus den Metadaten der VRS-Aufnahme. Das Skript extrahiert die Brennweiten (`get_focal_lengths()`) und den Hauptpunkt (`get_principal_point()`), um die Kamera-Matrix $K$ zur Laufzeit mathematisch korrekt aufzubauen.
* **Warum:** Da es sich um Fisheye-Objektive handelt, weist das Bild starke Verzerrungen auf. Für die spätere 3D-Berechnung im Raum muss die genaue optische Geometrie der Linse bekannt sein. Da das Meta-SDK die Bilder bereits rektifiziert (verzerrungsfrei) übergibt, wird die Linsenverzerrung im Algorithmus präzise genullt.

### Modul C: Modernisierte AprilTag-Detektion & 3D-Posenschätzung
* **Funktion:** Implementierung der modernisierten OpenCV-Struktur (`cv2.aruco.ArucoDetector`) zur Erkennung der AprilTag-Familie `tag36h11`. Über den PnP-Algorithmus (`cv2.solvePnP`) wird die 2D-Bildkoordinate des Tags mit seiner realen physischen Größe ($10\text{ cm}$) und der Kamera-Matrix verknüpft.
* **Warum:** Das System benötigt einen fixen Bezugspunkt zur realen Welt. Indem ein AprilTag starr an der Basis des Franka Emika Panda Roboters sowie auf der Arbeitsfläche (Tisch) platziert wird, kann die Brille die millimetergenaue 3D-Position ($X, Y, Z$) und Orientierung der Roboterbasis relativ zum Kopf des Probanden berechnen. Dank der SLAM-Raumkartierung (VIO) der Brille muss diese Kalibrierung zur Transformation der globalen Koordinatensysteme nur **einmalig zu Beginn** (statisch) stattfinden und spart im Post-Processing massiv Rechenleistung.

### Modul D: Visuelle Validierung (Reprojektion)
* **Funktion:** Mathematische Rückprojektion der berechneten 3D-Raumkoordinaten auf die 2D-Bildebene mittels `cv2.projectPoints`. Es zeichnet ein virtuelles 3D-Koordinatenkreuz (Rot=$X$, Grün=$Y$, Blau=$Z$) direkt auf den AprilTag im Videobild.
* **Warum:** Dies dient als visueller "Ground Truth"-Beweis. Sitzt das Koordinatenkreuz in der Videovorschau stabil und im exakten Winkel auf dem physischen Tag, ist die mathematische Korrektheit der gesamten Geometrie-Pipeline und der homogenen Transformationsmatrizen bewiesen.

### Modul E: Synchronisiertes Audio-Annotation-System (Aria Gen 2 Codec-Handling)
* **Funktion:** Da die Meta Aria Gen 2 im standardmäßigen Aufnahmeprofil (`profile8`) komprimierte Audio-Datenblöcke verwendet, liest dieses Modul die Datenblöcke sequenziell als `float32`-Arrays über den SDK-Daten-Provider aus dem VRS-Container aus. Es mischt die Kanäle des Mikrofon-Arrays im Interleaved-Layout mathematisch zu einer Mono-Spur zusammen, normalisiert die Amplituden auf den Bereich $[-1, 1]$ und exportiert ein unkomprimiertes 16-Bit-PCM-WAV-Signal mit einer Ziel-Abtastrate von $16\text{ kHz}$.
* Dieses Signal wird an ein lokales **Faster-Whisper-Modell** übergeben. Das Skript scannt die Spur nach den vom Tutor vorgegebenen Zustands-Triggern (*"Start"*, *"Second"*, *"Third"*, *"Done"*). Über den SDK-Aufruf `get_first_time_ns` wird der exakte Hardware-Startzeitpunkt des Mikrofons im Gerätesystem (`TimeDomain.DEVICE_TIME`) abgegriffen und mit den relativen Wort-Zeitstempeln von Whisper verrechnet.
* **Warum:** Für das überwachte Lernen (Supervised Learning) des Multimodal Transformers wird ein gelabelter Datensatz benötigt. Durch die Verrechnung mit der `DEVICE_TIME` wird eine nanosekundengenaue, automatisierte zeitliche Synchronität zwischen dem gesprochenen Befehl und den physikalischen Augen- und Handbewegungen des Probanden hergestellt, ohne dass Videodateien manuell geschnitten werden müssen.

---

## 3. Struktur der finalen KI-Dateneinspeisung & Feature-Matrix

Ein wichtiger architektonischer Meilenstein ist die Entscheidung gegen das physische Zerschneiden von Videodateien. Die originalen VRS-Dateien bleiben als kontinuierliche Rohdaten erhalten, während der PyTorch-`DataLoader` die generierte `metadata.json` als virtuelles Schnittbuch nutzt, um über ein "Sliding Window" auf die relevanten Sensorfenster zuzugreifen.

### Modul F: Synchronisierte On-Device Tracking-Extraktion (Hand- & Eye-Tracking)
* **Funktion:** Simultanes Auslesen und zeitliches Resampling der On-Device Machine Perception (MP) Datenströme für das Eyegaze-Tracking und das Hand-Pose-Tracking. Das Modul nutzt die hardwarenahe C++-Schnittstelle `get_index_by_time_ns` in Kombination mit der `TimeQueryOptions.CLOSEST`-Metrik, um für jeden Hand-Tracking-Frame den mathematisch am exaktesten passenden Blickvektor im Nanosekundenbereich zuzuordnen.
* **Extraktion der Features:**
  * **Hand-Pose:** Extraktion von 21 3D-Skelett-Landmarks pro Hand im lokalen *Device-Koordinatensystem*, inklusive der gerätegenerierten Konfidenzwerte sowie der präzisen 3D-Vektoren für das Handgelenk (`wrist`) und die Handfläche (`palm`).
  * **Eye-Gaze:** Abfrage der Blickrichtungswerte für Rotation (Yaw, Pitch) und die geschätzte Fixationstiefe (Depth) aus dem *Central Pupil Frame (CPF)*.
* **Warum:** Diese numerischen Features bilden die Eingangsmatrix für den Transformer. Die Blickrichtung dient als multimodaler Prädiktor (Input), während die synchronen 3D-Handgelenkskoordinaten als Ground Truth für die Positionsvorhersage (*Hand Position Prediction*) fungieren.

### Modul G: Hybride 3D-Objekt-Lokalisierung via YOLOv8 & ArUco 5x5 Grid
* **Funktion:** Integration eines tiefenlernbasierten **YOLOv8-Detektionsmodells** gekoppelt mit einer klassischen **ArUco-Marker-Posenschätzung** im 5x5-Grid (`DICT_5X5_50`). Dieser hybride Ansatz realisiert ein redundantes System aus semantischer Objekterkennung und geometrisch präziser 3D-Referenzierung.
* **Der mathematische Ablauf (Sensor Fusion & Alignment):**
  1. **Semantische Erkennung:** YOLOv8 detektiert die Objekte (z. B. Bausteine) im 2D-RGB-Schnittfeld der Brille und liefert eine Bounding-Box mitsamt Objektklasse.
  2. **Intentions-Kopplung:** Der Eye-Gaze-Vektor (Modul F) wird mit den YOLO-Bounding-Boxes verschnitten. Das System bestimmt, welche Objektklasse der Proband fixiert (*Intention Alignment*).
  3. **Geometrische Posenschätzung:** Gleichzeitig isoliert der OpenCV-ArUco-Detektor die Ecken des physisch auf dem Objekt angebrachten 5x5-Markers. Über den PnP-Schätzer (`cv2.solvePnP`) wird die hochpräzise 3D-Raumkoordinate und Orientierung relativ zur Brille berechnet.
  4. **Koordinaten-Transformation:** Die ermittelte 3D-Pose des Objekts wird über die homogenen Transformationsmatrizen (Tisch-Anker aus Modul C) direkt in das Basis-Koordinatensystem des Franka Panda Roboters transformiert.
* **Warum:** Dieser zweigleisige Ansatz kombiniert die Stärken beider Welten: YOLOv8 liefert eine robuste semantische Klassifizierung über weite Distanzen und weite Blickwinkel. Das 5x5-ArUco-Grid sorgt im Nahbereich und während der physischen Manipulation für eine millimetergenaue, driftfreie 3D-Pose für den Roboter, selbst wenn das Object durch die Hand des Nutzers teilweise verdeckt wird.

---

## 4. Übersicht der Pipeline-Zustände (Tutor-Logik)

Die aus Modul E und G gewonnenen Datenströme werden anhand der verbalen Trigger in vier distinkte Phasen für das KI-Modell unterteilt:

| Phase | Verbaler Trigger | Systemzustand / Aktivität | Zielsetzung für das ML-Modell |
| :--- | :--- | :--- | :--- |
| **1** | `"start"` | Proband beginnt die Montageaufgabe (Assembly Task) | Stabilisierung der Baseline, Erkennung des Startmusters |
| **2** | `"second"` | Proband fokussiert das von YOLO detektierte und mit dem ArUco-Marker versehene Zielobjekt | *Intention Alignment*: Verknüpfung von Gaze-Vektor, YOLO-Klasse und Objekt-ID |
| **3** | `"third"` | Proband greift das Objekt und führt Bewegung aus | *Hand Position Prediction*: Trajektorien-Vorhersage des Handgelenks |
| **4** | `"done"` | Übergabephase / Interaktion abgeschlossen | Triggerung der ROS/Deoxys-Zielpose; Roboter beendet Greifvorgang |

---

## 5. Durchführung der Probanden-Studie & Datenerhebung

Zur Evaluierung des Frameworks und zum späteren Training des Multimodal Transformers wurde eine systematische Datenerhebung im Labor durchgeführt. Der Fokus lag hierbei auf der Erzeugung einer variantenreichen, realistischen Interaktionsumgebung bei gleichzeitiger Wahrung einer präzisen mathematischen Kontrollstruktur (Ground Truth).

### 5.1 Kohorte und Datengröße
* **Probandenanzahl:** $N = 9$ Teilnehmer.
* **Sequenzen pro Proband:** Jeweils 4 vollständige Videosequenzen, was einer Gesamtzahl von **36 multimodalen Datensätzen** entspricht.
* **Durchschnittliche Sequenzlänge:** ca. 17–18 Sekunden pro Durchlauf.

### 5.2 Versuchsaufbau und Varianzkontrolle
Als Interaktionsobjekte wurde künstliches Plastik-Obst (z. B. Äpfel, Bananen, Zitronen) verwendet, da dieses eine ideale semantische Erkennung über die vortrainierten YOLOv8-Klassen bietet. 

Um ein Überfitten des Modells auf feste Raumkoordinaten zu verhindern, wurden folgende Parameter **vor jeder Aufnahme** bewusst variiert:
* Die physische Position des Tisches relativ zum Roboter sowie der Abstand zwischen dem Tisch-AprilTag und dem Roboter-Basis-Marker wurden modifiziert (Szenariovarianz).
* Die Anzahl, Auswahl und geometrische Startplatzierung der Objekte auf der Tischoberfläche wurden für jeden Durchlauf randomisiert (stochastische Verteilung).

### 5.3 Chronologischer Ablauf einer Sequenz
Jeder der 36 aufgezeichneten Durchläufe folgte einer strikten zeitlichen und verhaltensbasierten Phaseneinteilung, um die in Abschnitt 4 definierte Tutor-Logik abzubilden:

1. **Explorations- und Manipulationsphase (ca. 0–15 Sekunden):** Nach dem verbalen Trigger `"start"` interagierte der Proband frei mit den Objekten in seiner unmittelbaren Nähe (Inspizieren, Bewegen, Umlagern). Dies dient dem ML-Modell als negatives Trainingssignal (Hintergrundrauschen ohne Intention bezüglich des Roboters).
2. **Fixations- und Pointing-Phase (ca. 15–17 Sekunden):** Der Proband fokussierte visuell ein spezifisches Zielobjekt, welches sich in der Übergabezone nahe des Roboters befand, für 2–3 Sekunden und zeigte explizit mit der Hand darauf (Kopplung von *Eye-Gaze* und *Hand-Pose-Pointing*). In diesem Fenster erfolgte der verbale Trigger `"second"`.
3. **Trajektorien- und Übergabephase (ca. 17–18 Sekunden):** Der Proband leitete die finale Übergbebewegung ein, indem er die Hand mitsamt dem Gelbobjekt geradlinig in Richtung der Roboterbasis ausstreckte (*Hand Position Prediction* unter den Triggern `"third"` und `"done"`).

Die Auswertung dieser 36 extrahierten `.vrs`-Dateien über die Module A bis G liefert nun die zeitlich hochaufgelöste, nanosekundengenaue Feature-Matrix (Blickvektoren, 21 Hand-Skelett-Landmarks, YOLO-Klassen und Sprach-Zeitstempel) für das anschließende KI-Training.