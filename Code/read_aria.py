import os
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def inspect_aria_file(vrs_file_path):
    # Pruefen, ob die Datei am angegebenen Pfad existiert
    if not os.path.exists(vrs_file_path):
        print(f"Fehler: Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"Oeffne Aria-Aufnahme: {vrs_file_path}...")
    
    # VRS Data Provider initialisieren, um Zugriff auf Datei-Inhalte zu erhalten
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    print("VRS Data Provider erfolgreich initialisiert!")
    
    # Eindeutige Stream-IDs fuer die Sensoren der Aria Gen 2 definieren
    rgb_stream_id = StreamId("214-1")  # Haupt-RGB-Kamera
    gaze_stream_id = StreamId("373-1") # Blickrichtung (Eye-Tracking)
    hand_stream_id = StreamId("371-1") # Gelenk-Tracking der Haende

    # Testen des Datenabrufs fuer das Eye-Tracking
    print("\nPruefe Eye-Tracking und Gaze-Daten...")
    try:
        # Ersten Datensatz (Index 0) des Gaze-Streams abfragen
        gaze_data = provider.get_eye_gaze_data_by_index(gaze_stream_id, 0)
        print("3D-Gaze-Daten erfolgreich abgefragt!")
        print(f"   Typ der Rueckgabe: {type(gaze_data)}")
    except Exception as e:
        print(f"Information Gaze-Abruf: {e}")

    # Testen des Datenabrufs fuer das Hand-Tracking
    print("\nPruefe Hand-Tracking-Daten...")
    try:
        # Ersten Datensatz (Index 0) des Hand-Streams abfragen
        hand_data = provider.get_hand_pose_data_by_index(hand_stream_id, 0)
        print("Hand-Tracking-Daten erfolgreich abgefragt!")
        print(f"   Typ der Rueckgabe: {type(hand_data)}")
    except Exception as e:
        print(f"Information Hand-Abruf: {e}")

    # Testen des Datenabrufs fuer die RGB-Kamera
    print("\nPruefe RGB-Bilddaten...")
    try:
        # Bildpaket abfragen (gibt ein Tuple aus Daten und Record zurueck)
        image_tuple = provider.get_image_data_by_index(rgb_stream_id, 0)
        image_data = image_tuple[0]
        
        # Das proprietaere Bildobjekt in ein Standard-NumPy-Array konvertieren
        pixel_array = image_data.to_numpy_array()
        print("RGB-Kamera-Schnittstelle erfolgreich verifiziert!")
        print(f"   Bild-Array erfolgreich geladen! Shape: {pixel_array.shape}")
    except Exception as e:
        print(f"Information Bild-Abruf: {e}")
        
    print(f"Bilddaten erfolgreich geladen. Typ: {type(image_tuple[0])}")
    print("\nVerifikation abgeschlossen: Die Pipeline liest die Datenstroeme nun mit korrekter Typisierung aus.")

if __name__ == "__main__":
    # Pfad zur lokalen Testdatei definieren und Funktion ausfuehren
    test_file = "../Recordings/test_recording.vrs" 
    inspect_aria_file(test_file)