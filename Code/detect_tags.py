import os
import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def run_apriltag_detection(vrs_file_path):
    # Prüfen, ob die Datei existiert
    if not os.path.exists(vrs_file_path):
        print(f"Fehler: Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"Öffne Aria-Aufnahme: {vrs_file_path}...")
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    rgb_stream_id = StreamId("214-1")
    
    # 1. Echte Kamera-Kalibrierung aus den VRS-Metadaten laden
    print("Lade Aria Kamera-Kalibrierung...")
    src_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    
    focal_lengths = src_calib.get_focal_lengths()      # Brennweite [fx, fy]
    principal_point = src_calib.get_principal_point()  # Optischer Mittelpunkt [cx, cy]
    
    # Intrinsische Kamera-Matrix K aufbauen
    K = np.array([
        [focal_lengths[0], 0,                principal_point[0]],
        [0,                focal_lengths[1], principal_point[1]],
        [0,                0,                1]
    ], dtype=np.float64)
    
    # Da Aria-Bilder im SDK rektifiziert bereitgestellt werden, ist die Distorsion 0
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 2. OpenCV ArUco/AprilTag Detektor initialisieren (Verwendet Familie 36h11)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, detector_parameters)
    
    # 3. Den ersten Frame (Index 0) laden
    print("Lade RGB-Frame aus der Aufnahme...")
    image_tuple = provider.get_image_data_by_index(rgb_stream_id, 0)
    rgb_image = image_tuple[0].to_numpy_array()
    
    # Farbkonvertierung fuer OpenCV-Operationen
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # 4. Physische Groesse des ausgedruckten AprilTags definieren
    TAG_SIZE_METERS = 0.10  # Seitenlaenge von 10 Zentimetern

    print("Suche nach AprilTags im Bild...")
    corners, ids, rejected = detector.detectMarkers(gray_image)

    if ids is None:
        print("\nGefundene Tags im Frame: 0")
        print("=" * 60)
    else:
        print(f"\nGefundene Tags im Frame: {len(ids)}")
        print("=" * 60)
        
        # Ideale 3D-Eckpunkte des Tags im eigenen Koordinatensystem definieren
        half_size = TAG_SIZE_METERS / 2.0
        obj_pts = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for i in range(len(ids)):
            tag_id = int(ids[i][0])
            
            # Filter auf die relevanten Versuchs-IDs loeschen
            if tag_id in [0, 1, 2, 3, 4, 5]:
                print(f"Tag {tag_id} erfolgreich erkannt!")
                
                # 2D-Eckpunkte aus dem Bild extrahieren
                img_pts = corners[i][0].astype(np.float32)
                
                # Pixel-Zentrum im Bild berechnen
                center_x = np.mean(img_pts[:, 0])
                center_y = np.mean(img_pts[:, 1])
                print(f"   -> Pixel-Zentrum im Bild: [{center_x:.1f}, {center_y:.1f}]")
                
                # 3D-Pose (Rotation und Translation) relativ zur Brille ueber SolvePnP berechnen
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if success:
                    print(f"   -> 3D-Position relativ zur Aria-Brille:")
                    print(f"      Links(-) / Rechts(+) (X): {tvec[0][0]:.3f} Meter")
                    print(f"      Oben(-)  / Unten(+)  (Y): {tvec[1][0]:.3f} Meter")
                    print(f"      Abstand (Tiefe)      (Z): {tvec[2][0]:.3f} Meter")

                    # Achsen-Endpunkte im 3D-Raum definieren (Laenge entspricht halber Tag-Groesse)
                    axis_length = TAG_SIZE_METERS / 2.0
                    axis_points = np.array([
                        [0, 0, 0],             # Ursprung im Zentrum
                        [axis_length, 0, 0],   # X-Achse (Rechts)
                        [0, axis_length, 0],   # Y-Achse (Unten)
                        [0, 0, -axis_length]   # Z-Achse (Aus dem Tag heraus)
                    ], dtype=np.float32)

                    # 3D-Achsenpunkte zur visuellen Kontrolle zurueck auf das 2D-Bild projizieren
                    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
                    img_points = img_points.reshape(-1, 2).astype(int)

                    # Die drei Koordinatenachsen zeichnen
                    p_center = tuple(img_points[0])
                    p_x = tuple(img_points[1])
                    p_y = tuple(img_points[2])
                    p_z = tuple(img_points[3])

                    cv2.line(bgr_image, p_center, p_x, (0, 0, 255), 5)  # X-Achse in Rot
                    cv2.line(bgr_image, p_center, p_y, (0, 255, 0), 5)  # Y-Achse in Gruen
                    cv2.line(bgr_image, p_center, p_z, (255, 0, 0), 5)  # Z-Achse in Blau
                print("-" * 60)

        # Kontrollbild abspeichern
        output_filename = "kalibrierung_kontrolle2.jpg"
        cv2.imwrite(output_filename, bgr_image)
        print(f"\nKontrollbild {output_filename} erfolgreich gespeichert!")

    print("\nDetektion abgeschlossen!")

if __name__ == "__main__":
    # Funktion mit dem Pfad zur lokalen Aufnahme ausfuehren
    run_apriltag_detection("../Recordings/Test2.vrs")