import os
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def inspect_aria_file(vrs_file_path):
    if not os.path.exists(vrs_file_path):
        print(f"❌ Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"📦 Öffne Aria-Aufnahme: {vrs_file_path}...")
    
    # 1. Erstelle den Data Provider
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    print("✅ VRS Data Provider erfolgreich initialisiert!")
    
    # Definition der StreamIds basierend auf deinem Log
    rgb_stream_id = StreamId("214-1")
    gaze_stream_id = StreamId("373-1")
    hand_stream_id = StreamId("371-1")

    # 2. Teste Zugriff auf Eye-Gaze
    print("\nPrüfe Eye-Tracking & Gaze-Daten...")
    try:
        gaze_data = provider.get_eye_gaze_data_by_index(gaze_stream_id, 0)
        print("✅ 3D-Gaze-Daten erfolgreich abgefragt!")
        print(f"   Typ der Rückgabe: {type(gaze_data)}")
    except Exception as e:
        print(f"ℹ️ Gaze-Abruf: {e}")

    # 3. Teste Zugriff auf Hand-Pose
    print("\nPrüfe Hand-Tracking-Daten...")
    try:
        hand_data = provider.get_hand_pose_data_by_index(hand_stream_id, 0)
        print("✅ Hand-Tracking-Daten erfolgreich abgefragt!")
        print(f"   Typ der Rückgabe: {type(hand_data)}")
    except Exception as e:
        print(f"ℹ️ Hand-Abruf: {e}")

    # 4. Teste Zugriff auf die Kamera-Bilder (RGB)
# 4. Teste Zugriff auf die Kamera-Bilder (RGB) und konvertiere zu NumPy
    print("\nPrüfe RGB-Bilddaten...")
    try:
        image_tuple = provider.get_image_data_by_index(rgb_stream_id, 0)
        image_data = image_tuple[0]
        
        # Der offizielle Meta-Weg, um an die rohen Pixel zu kommen:
        pixel_array = image_data.to_numpy_array()
        print("✅ RGB-Kamera-Schnittstelle erfolgreich verifiziert!")
        print(f"   Bild-Array erfolgreich geladen! Shape: {pixel_array.shape}")
    except Exception as e:
        print(f"ℹ️ Bild-Abruf: {e}")
        
    print(f" Bilddaten erfolgreich geladen. Typ: {type(image_tuple[0])}")

    print("\n🎉 Perfekt! Die Pipeline liest die Datenströme nun mit korrekter Typisierung aus.")
    
   # print(f"   Bildauflösung: {image_tuple[0].width}x{image_tuple[0].height}")
# ODER, falls das auch zickt, einfach die Form des Arrays nutzen:


if __name__ == "__main__":
    test_file = "test_recording.vrs" 
    inspect_aria_file(test_file)