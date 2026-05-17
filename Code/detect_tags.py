import os
import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def run_apriltag_detection(vrs_file_path):
    if not os.path.exists(vrs_file_path):
        print(f"❌ Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"📦 Öffne Aria-Aufnahme: {vrs_file_path}...")
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    rgb_stream_id = StreamId("214-1")
    
    # 1. Echte Kamera-Kalibrierung aus der VRS-Datei laden
    print("📐 Lade Aria Kamera-Kalibrierung...")
    src_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    
    focal_lengths = src_calib.get_focal_lengths()      # [fx, fy]
    principal_point = src_calib.get_principal_point()  # [cx, cy]
    
    # Intrinsische Kamera-Matrix K aufbauen
    K = np.array([
        [focal_lengths[0], 0,                principal_point[0]],
        [0,                focal_lengths[1], principal_point[1]],
        [0,                0,                1]
    ], dtype=np.float64)
    
    # Da Aria-Bilder im SDK rektifiziert bereitgestellt werden, setzen wir die Distorsion auf 0
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 2. Modernen OpenCV ArUco/AprilTag Detektor initialisieren (Ab v4.7.0)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, detector_parameters)
    
    # 3. Den ersten Frame (Index 0) laden
    print("📸 Lade RGB-Frame aus der Aufnahme...")
    image_tuple = provider.get_image_data_by_index(rgb_stream_id, 0)
    rgb_image = image_tuple[0].to_numpy_array()
    
    # OpenCV braucht für die Erkennung Graustufen
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # 4. Physische Größe deines ausgedruckten AprilTags definieren
    TAG_SIZE_METERS = 0.10  # 10 Zentimeter

    print("🔍 Suche nach AprilTags im Bild...")
    corners, ids, rejected = detector.detectMarkers(gray_image)

    if ids is None:
        print("\n📊 Gefundene Tags im Frame: 0")
        print("=" * 60)
    else:
        print(f"\n📊 Gefundene Tags im Frame: {len(ids)}")
        print("=" * 60)
        
        # Ideale 3D-Eckpunkte des Tags im eigenen Koordinatensystem definieren
        # Zentriert um den Ursprung auf der Z=0 Ebene
        half_size = TAG_SIZE_METERS / 2.0
        obj_pts = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for i in range(len(ids)):
            tag_id = int(ids[i][0])
            
            # Filter auf deine IDs (0 = Basis, 1-4 = Tisch, 5 = Extra)
            if tag_id in [0, 1, 2, 3, 4, 5]:
                print(f"🎯 [TAG {tag_id}] erfolgreich erkannt!")
                
                # 2D-Eckpunkte aus dem Bild holen
                img_pts = corners[i][0].astype(np.float32)
                
                # Pixel-Zentrum berechnen
                center_x = np.mean(img_pts[:, 0])
                center_y = np.mean(img_pts[:, 1])
                print(f"   -> Pixel-Zentrum im Bild: [{center_x:.1f}, {center_y:.1f}]")
                
                # Pose robust über SolvePnP schätzen
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if success:
                    print(f"   -> 3D-Position relativ zur Aria-Brille:")
                    print(f"      Links(-) / Rechts(+) (X): {tvec[0][0]:.3f} Meter")
                    print(f"      Oben(-)  / Unten(+)  (Y): {tvec[1][0]:.3f} Meter")
                    print(f"      Abstand (Tiefe)      (Z): {tvec[2][0]:.3f} Meter")
                print("-" * 60)

    print("\n🎉 Detektion abgeschlossen!")

if __name__ == "__main__":
    run_apriltag_detection("../Recordings/Test2.vrs")