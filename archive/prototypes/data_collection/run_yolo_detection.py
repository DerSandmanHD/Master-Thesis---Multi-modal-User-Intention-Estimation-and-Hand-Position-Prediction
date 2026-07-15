import os
import cv2
from ultralytics import YOLO

def detect_objects_with_yolo(image_path, output_path="yolo_output.jpg"):
    # Pruefen, ob das Eingabebild existiert
    if not os.path.exists(image_path):
        print(f"Fehler: Bild nicht gefunden: {image_path}")
        return

    print(f"Lade Bild: {image_path}...")
    img = cv2.imread(image_path)

    # Vortrainiertes YOLOv8 Nano Modell laden (wird bei Bedarf automatisch heruntergeladen)
    # Fuer deine finale Arbeit wird hier der Pfad zu deinen Lego-Gewichten eingetragen
    print("Lade YOLOv8 Modell...")
    model = YOLO("yolov8n.pt")

    # Objekterkennung ausfuehren
    print("Starte Objekterkennung (Inferenz)...")
    results = model(img, conf=0.25)  # Konfidenz-Schwellenwert von 25%

    print("\nGefundene Objekte und Koordinaten:")
    print("-" * 60)

    # Ergebnisse verarbeiten (Es gibt nur ein Ergebnis-Objekt, da wir nur ein Bild uebergeben)
    result = results[0]

    # Iteration ueber alle erkannten Bounding-Boxes
    for box in result.boxes:
        # 2D-Pixelkoordinaten der Box extrahieren (links oben, rechts unten)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Klasse (ID) und Konfidenzwert (Wahrscheinlichkeit) auslesen
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = box.conf[0].item()

        # Mittelpunkt (Zentrum) der Bounding-Box berechnen (Wichtig fuer dein Greif-Skript!)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        print(f"Objekt: '{class_name}' | Konfidenz: {confidence:.2f}")
        print(f"   -> Box-Ecken (Pixel): [{int(x1)}, {int(y1)}] bis [{int(x2)}, {int(y2)}]")
        print(f"   -> Pixel-Zentrum: [{center_x:.1f}, {center_y:.1f}]")
        print("-" * 60)

    # Das annotierte Bild (mit eingezeichneten Boxen und Labels) generieren
    annotated_img = result.plot()

    # Ergebnis abspeichern
    cv2.imwrite(output_path, annotated_img)
    print(f"Kontrollbild erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    # Pfad zu einem Testbild definieren (z. B. ein extrahierter Frame deiner Aria-Brille)
    test_image = "kalibrierung_kontrolle2.jpg"
    
    # Zum Testen erstellen wir ein leeres Bild, falls du gerade kein Testbild im Ordner hast
    if not os.path.exists(test_image):
        import numpy as np
        # Erstellt ein einfaches graues Dummy-Bild (640x480)
        dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.imwrite(test_image, dummy_img)
        print(f"Hinweis: Dummy-Bild '{test_image}' erstellt, da kein Bild existierte.")

    detect_objects_with_yolo(test_image)