# Zusammenfassung: Diskussion zum Datensatz & Augmentation der Masterarbeit

## Kontext

Multimodale Intentionsschätzung (continue / fetch / handover) mit Hand-Position-Prediction als Auxiliary-Task, Datenaufnahme via Aria Meta Gen 2. Aktueller Stand: ~3/4 der Probanden aufgenommen (~80–100 Sequenzen à 25–40 s). Die Datenpipeline (nanosekundengenaue Synchronisation über `DEVICE_TIME`, `merge_asof`-Fusion, duale Marker-Strategie AprilTag/ArUco, Trennung von visueller Kontrolle und Trainings-Matrix) wurde als technisch sehr solide eingeschätzt, teils über Masterarbeits-Niveau.

---

## Problem 1: Audio-Leck — geklärt, kein Problem

Erste Sorge war, dass das Modell nur den Audio-Trigger reproduziert statt echter Intention. **Entkräftet:** Audio dient ausschließlich der Offline-Phasensegmentierung (Ersatz für manuelles Frame-Labeling) und ist keine Modell-Modalität. Das ist eine legitime, gängige Methode, solange Audio nicht ins Feature-Set leckt.

## Problem 2: Strukturelles Leck durch feste Phasenreihenfolge — *das Kernproblem, ungelöst*

Auch ohne Audio bleibt bestehen, dass die Phasen **immer in derselben Reihenfolge und Dauer-Struktur** auftreten (Exploration → Fixation → Handover). Ein Zeitreihen-Transformer kann die **relative Zeitposition im Clip** als Abkürzung lernen, statt echte Intentions-Merkmale aus Gaze und Handbewegung. Da *fetch* ausnahmslos nach Exploration und vor Handover liegt, korreliert die Klasse stark mit der Zeitposition.

Besonders relevant, weil die Aufgabenstellung menschliches Verhalten explizit als *„gradual and ambiguous"* beschreibt — genau diese Ambiguität wurde durch das saubere, fest geordnete Protokoll aus den Daten herausdesignt. Das Modell löst dann ein leichteres Problem als das eigentliche Forschungsproblem.

*Offene Rückfrage, die die Schwere bestimmt:* Wird absolute/relative Zeitposition (Frame-Index, Zeit seit Sequenzbeginn) als Feature eingespeist, oder bekommt der Transformer nur rohe Sensorwerte plus Positional Encoding über das Sliding Window?

## Problem 3: Fehlende saubere Trennung der Intentionsklassen

Aktuell unterscheiden sich *fetch* (Objekt nahe Roboter fixieren) und *handover* (offene Hand ausstrecken) im Setup primär durch die verbal markierte Phase. Risiko: Es fehlen Negativbeispiele — z. B. ein fixiertes Objekt **ohne** folgenden Handover, oder ein Objekt, das der Proband selbst greift (continue trotz fetch-ähnlichem Gaze). Die *continue*-Klasse tritt nie in Situationen auf, die *fast* in fetch übergehen.

## Problem 4: Datensatzgröße

~96–120 Sequenzen final ist für einen von Grund auf trainierten Transformer mit 3 Klassen plus Regressions-Auxiliary-Task **wenig**. Sollte proaktiv adressiert werden (Augmentation, Sliding-Window-Overlap, ggf. Transfer/Pretraining), sonst kommt die Frage in der Verteidigung.

---

## Lösungsvorschläge — was funktioniert und was nicht

### ✅ Funktioniert: Sequenzen abschneiden

Früheres Beenden von Sequenzen (z. B. Fixation **ohne** folgenden Handover) erzeugt **echte** neue Varianz und liefert die fehlenden Negativbeispiele für die Klassentrennung. Legitim, solange physikalisch plausible Teilsequenzen entstehen.

### ❌ Funktioniert NICHT: Sequenzen vertauschen

Die Phasen sind **physikalisch kausal verkettet** — Handposition, Gaze und SLAM-Pose am Ende einer Phase sind der Startzustand der nächsten. Umordnen erzeugt **unphysikalische Sprünge** an den Schnittstellen (Hand „teleportiert", Gaze/Pose springen). Das Modell lernt dann die Schnittkanten zu erkennen statt der Intention — ein Leck wird gegen ein anderes getauscht, und da solche Sprünge in echten Daten nie vorkommen, generalisiert das Modell schlechter.

**Kernpunkt:** Post-hoc-Manipulation am fertigen Clip kann fehlende *Verhaltensvarianz* nicht herstellen — die entsteht nur an der Quelle bei der Aufnahme.

### ✅ Beste Lösung für Problem 2 & 3: Varianz bei den verbleibenden Probanden

Bei den letzten ~1/4 Probanden gezielt einbauen: variable Explorationsdauer, Fixation ohne folgenden Handover, Objekt selbst greifen statt Handover, wo möglich variierte Phasenreihenfolge. Das bricht die Zeit-Korrelation an der Quelle auf.

### ✅ Funktioniert: Spiegeln gegen den Rechts-Arm-Bias

Da ausschließlich der rechte Arm nach rechts ausgestreckt wird, droht das Modell „rechte Bildhälfte + rechter Arm = Handover" zu lernen statt „eine offene Hand wird zum Roboter ausgestreckt". Horizontales Spiegeln ist das richtige, etablierte Augmentationswerkzeug — adressiert nebenbei die Datenknappheit.

**Aber kritisch:** Spiegeln muss **konsistent über alle Modalitäten im 3D-/Feature-Raum** erfolgen, nicht als simpler Bild-Flip, sonst wird die räumliche Kohärenz des `merge_asof`-Setups zerstört. Zu transformieren:

- **RGB-Bild** (horizontaler Flip)
- **3D-Hand-Landmarks** (X-Achse spiegeln **und** links/rechts-Gelenklabels tauschen)
- **Eye-Gaze-Vektor / CPF** (X-Komponente negieren)
- **SLAM-Pose** (Translation-X spiegeln + Quaternionen transformieren — fehlerträchtigste Stelle, da Reflexion die Händigkeit des Koordinatensystems umkehrt)
- **ArUco/AprilTag-3D-Posen** (gleiche X-Reflexion auf solvePnP-Ergebnisse)

**Empfehlung:** Erst 3D-Features extrahieren, **dann** im Feature-Raum spiegeln — nicht das Bild flippen und neu detektieren (sonst muss auch der Hauptpunkt der Kamera-Matrix angepasst werden: `c_x → Bildbreite − c_x`).

### ✅ Fallback wenn Nachjustieren am Setup nicht mehr möglich

Strukturelle Limitation ehrlich benennen: „Der Datensatz folgt einer festen Phasenstruktur; Generalisierung auf frei geordnete Interaktionssequenzen ist Gegenstand zukünftiger Arbeit." In der Ablation explizit testen, ob das Modell ohne Zeit-/Positions-Information ähnlich gut bleibt — bleibt die Performance, ist das ein starkes Argument für echte Intentions-Erkennung; bricht sie ein, ist der Shortcut nachgewiesen (auch ein verwertbares Ergebnis). Ehrlich benannte Limitationen kosten keine Punkte, übersehene schon.

---

## Wichtigste Take-aways für das Betreuer-Gespräch

1. Audio-Annotation ist methodisch sauber — kein Handlungsbedarf.
2. Das zentrale Risiko ist die **feste Phasenstruktur**, die das Modell zu einer Zeit-Positions-Abkürzung verleitet und die geforderte Verhaltens-Ambiguität aus den Daten entfernt.
3. **Vertauschen ist keine Lösung** (unphysikalische Sprünge), **Abschneiden und vor allem Varianz bei den letzten Probanden** schon.
4. **Spiegeln ist richtig** gegen den Arm-Bias, aber nur geometrisch konsistent über alle Modalitäten.
5. Datensatzgröße und strukturelle Limitation proaktiv in der Arbeit adressieren.