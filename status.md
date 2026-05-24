# Technische Dokumentation: Entwicklungsstand Framework zur multimodalen Intentionsschätzung

## 1. System-Architektur & Infrastruktur

Um eine reproduzierbare und isolierte Entwicklungsumgebung zu gewährleisten und Konflikte mit dem globalen macOS-System zu vermeiden, wurde eine dedizierte Python-Infrastruktur aufgesetzt.

* **Virtual Environment (Conda):** Es wurde eine virtuelle Umgebung namens `aria_conda` mittels Anaconda auf dem Mac Silicon eingerichtet. Dies stellt sicher, dass alle hardwarenahen C++-Bindings des Meta SDKs exakt mit den Python-Bibliotheken harmonieren.
* **Python-Version:** Verwendung von Python 3.10, um maximale Kompatibilität mit den Deep-Learning-Bibliotheken (PyTorch) und den vorkompilierten Meta-Bibliotheken zu garantieren.
* **Hardware-Abstraktion:** Die Pipeline wurde so konfiguriert, dass sie die hardwarebeschleunigte Dekodierung (VideoToolbox HW acceleration) des Mac nutzt, bei inkompatiblen Farbformaten jedoch nahtlos auf einen robusten Software-Dekoder (`H.265 SW decoder via xprs`) zurückgreift.

---

## 2. Implementierte Software-Module & Kernfunktionen

### Modul A: Daten-Parsing & Sensor-Streaming (`projectaria_tools`)
* **Funktion:** Zugriff, Aktivierung und Dekapselung des proprietären Meta `.vrs`-Datenformats. Das Skript initialisiert und synchronisiert die Datenströme der RGB-Kamera (`214-1`), der Eye-Tracking-Kameras (`211-1`/`211-2`), des Mikrofon-Arrays (`231-1`), der Hand-Tracking-Schnittstelle sowie der IMU- und VIO-Sensoren.
* **Warum:** Die Aria-Brille speichert Daten aus Datenschutz- und Performancegründen nicht als Standard-MP4. Ohne dieses Modul wäre ein framegenauer Zugriff auf die synchronisierten Rohdaten für das spätere KI-Training unmöglich.

### Modul B: Automatische Kamera-Kalibrierung & Rektifizierung
* **Funktion:** Automatisches Auslesen der werkseitigen intrinsischen Kameraparameter direkt aus den Metadaten der VRS-Aufnahme. Das Skript extrahiert die Brennweiten (`get_focal_lengths()`) und den Hauptpunkt (`get_principal_point()`), um die Kamera-Matrix $K$ zur Laufzeit mathematisch korrekt aufzubauen.
* **Warum:** Da es sich um Weitwinkel-/Fischaugen-Objektive handelt, weist das Bild starke Verzerrungen auf. Für die spätere 3D-Berechnung im Raum (z. B. wo befindet sich der Roboter relativ zur Brille) muss die genaue optische Geometrie der Linse bekannt sein. Da das Meta-SDK die Bilder bereits rektifiziert (verzerrungsfrei) übergibt, wird die Linsenverzerrung im Algorithmus präzise genullt.

### Modul C: Modernisierte AprilTag-Detektion & 3D-Posenschätzung
* **Funktion:** Implementierung der modernisierten OpenCV-Struktur (`cv2.aruco.ArucoDetector`) zur Erkennung der AprilTag-Familie `tag36h11`. Über den PnP-Algorithmus (`cv2.solvePnP`) wird die 2D-Bildkoordinate des Tags mit seiner realen physikalischen Größe ($10 \text{ cm}$) und der Kamera-Matrix verknüpft.
* **Warum:** Das System benötigt einen fixen Bezugspunkt zur realen Welt. Indem ein AprilTag (z. B. Tag 0) an der Roboterbasis befestigt wird, kann die Brille die millimetergenaue 3D-Position ($X, Y, Z$) und Orientierung des Roboters relativ zum Kopf des Probanden berechnen. Dank der SLAM-Raumkartierung der Brille muss diese Kalibrierung nur **einmalig zu Beginn** (statisch) stattfinden und nicht permanent mitlaufen, was massiv Rechenleistung spart.

### Modul D: Visuelle Validierung (Reprojektion)
* **Funktion:** Mathematische Rückprojektion der berechneten 3D-Koordinaten auf die 2D-Bildebene mittels `cv2.projectPoints`. Es zeichnet ein virtuelles 3D-Koordinatenkreuz (Rot=$X$, Grün=$Y$, Blau=$Z$) direkt auf den AprilTag im Videobild.
* **Warum:** Dies dient als visueller "Ground Truth"-Beweis. Sitzt das Koordinatenkreuz in der Vorschau stabil und im exakten Winkel auf dem physischen Tag, ist die mathematische Korrektheit der gesamten Geometrie-Pipeline bewiesen.

### Modul E: Vollautomatisches Audio-Annotation-System (Post-Processing)
* **Funktion:** Native Extraktion des Audio-Streams (`231-1`) direkt aus der VRS-Datei über Python, Konvertierung in ein NumPy-Array und Übergabe an ein lokales neuronales Sprachmodell (**OpenAI Whisper**). Das Skript scannt die Audiospur nach vordefinierten verbalen Kommandos des Probanden (*"Start"*, *"Intention"*, *"Action"*, *"Stop"*) und extrahiert die exakten Zeitstempel.
* **Warum:** Für das Training des Multimodal Transformers wird ein gelabelter Datensatz benötigt. Das händische Schneiden und Verschlagworten von hunderten Videoschnipseln würde Wochen dauern. Dieses Modul erlaubt eine **berührungslose, automatische Annotation**: Der Proband spricht die Phasen während des Experiments laut aus, und das Skript generiert nach der Aufnahme vollautomatisch eine strukturierte Metadaten-Datei (`.json`).

---

## 3. Struktur der finalen KI-Dateneinspeisung

Ein wichtiger architektonischer Meilenstein war die Entscheidung gegen das physische Zerschneiden von Videodateien. 

* **Der Ansatz:** Die originalen VRS-Dateien (Video, Eye-Tracking, Hand-Gesten) bleiben als kontinuierliche, ungeschnittene Rohdaten im Speicher erhalten. 
* **Die Brücke zur KI:** Der spätere PyTorch-`DataLoader` nutzt die generierte `metadata.json` wie ein virtuelles Schnittbuch. Er greift über die Zeitstempel (`timestamp_vrs_seconds`) sequenziell per "Sliding Window" nur auf die relevanten Sensorfenster zu und versieht sie mit dem passenden Label (z. B. Zustand: *Fokus/Intention auf Objekt*).

**Wissenschaftlicher Vorteil:** Die absolute zeitliche Synchronität zwischen den verschiedenen Sensoren (Bild vs. Eye-Tracking-Vektor) bleibt perfekt erhalten, und das Transformer-Modell kann die fließenden Übergänge zwischen den Absichten des Nutzers weitaus besser erlernen.

### Modul F: Synchronisierte On-Device Tracking-Extraktion (Hand- & Eye-Tracking)
* **Funktion:** Simultanes Auslesen und zeitliches Synchronisieren der On-Device Machine Perception (MP) Datenströme für das Eyegaze-Tracking (`373-1`) und das Hand-Pose-Tracking (`371-1`). Da Sensoren herstellerbedingt mit unterschiedlichen Frequenzen aufnehmen, nutzt das Modul die hardwarenahe C++-Schnittstelle `get_index_by_time_ns` in Kombination mit der `TimeQueryOptions.CLOSEST`-Metrik, um für jeden Hand-Tracking-Frame den mathematisch am exaktesten passenden Blickvektor im Nanosekundenbereich zuzuordnen.
* **Extraktion der Features:**
  * **Hand-Pose:** Extraktion von 21 3D-Skelett-Landmarks (`landmark_positions_device`) pro Hand im *Device-Koordinatensystem*, inklusive der globalen Konfidenzwerte sowie der präzisen 3D-Vektoren für das Handgelenk (`wrist`) und die Handfläche (`palm`).
  * **Eye-Gaze:** Abfrage der Blickrichtungswerte für Rotation (Yaw, Pitch) und die geschätzte Fixationstiefe (Depth) aus dem *Central Pupil Frame (CPF)*.
* **Warum:** Dieses Modul isoliert die kontinuierlichen, numerischen Features aus der VRS-Datei und überführt sie in eine strukturierte Matrix (`.json`). Im Kontext des überwachten Lernens (Supervised Learning) liefert dieser Datenstrom die mathematischen Variablen für das KI-Modell: Die Blickrichtung dient als multimodaler Input (Prädiktor), während die synchronen 3D-Handgelenkskoordinaten als direkter Ground Truth für die Positionsvorhersage (*Hand Position Prediction*) des Transformers fungieren.

---

### Aktueller Meilenstein-Status
Die komplette **Eingabe-, Kalibrierungs-, Lokalisierungs- und Annotations-Pipeline** steht und wurde lokal auf dem System erfolgreich validiert. Damit ist das technische Fundament für den Start der Probanden-Aufnahmen im Labor vollständig gelegt.