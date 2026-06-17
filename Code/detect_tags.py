import os
import cv2
import csv
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

def extract_and_visualize_aruco(vrs_file_path, output_video_path, output_csv_path):
    if not os.path.exists(vrs_file_path):
        print(f"Fehler: Datei nicht gefunden: {vrs_file_path}")
        return

    print(f"Öffne Aria-Aufnahme: {os.path.basename(vrs_file_path)}...")
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    rgb_stream_id = StreamId("214-1")
    
    num_frames = provider.get_num_data(rgb_stream_id)
    if num_frames == 0:
        print("Fehler: Keine RGB-Frames gefunden.")
        return

    # 1. Kamera-Kalibrierung laden
    src_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    focal_lengths = src_calib.get_focal_lengths()
    principal_point = src_calib.get_principal_point()
    
    K = np.array([
        [focal_lengths[0], 0,                principal_point[0]],
        [0,                focal_lengths[1], principal_point[1]],
        [0,                0,                1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 2. Detektoren initialisieren (Der Mix)
    dict_april = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_april = cv2.aruco.ArucoDetector(dict_april, cv2.aruco.DetectorParameters())

    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_aruco = cv2.aruco.ArucoDetector(dict_aruco, cv2.aruco.DetectorParameters())

    # 3. Video-Writer einrichten
    first_image = provider.get_image_data_by_index(rgb_stream_id, 0)[0].to_numpy_array()
    h, w = first_image.shape[:2]
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Hier speichern wir alle rohen Daten für die KI!
    extracted_data = []

    print(f"\nStarte Frame-by-Frame Extraktion ({num_frames} Frames)...")
    print("-" * 60)

    # Hilfsfunktion, die zeichnet UND die Daten speichert
    def process_markers(corners, ids, current_image, is_apriltag, frame_idx, timestamp_ns):
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(current_image, corners, ids)
            for j in range(len(ids)):
                tag_id = int(ids[j][0])
                
                # Größe bestimmen
                if is_apriltag:
                    size_m = 0.10 if tag_id == 0 else 0.08
                else:
                    size_m = 0.05
                
                half_size = size_m / 2.0
                obj_pts = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)

                img_pts = corners[j][0].astype(np.float32)
                
                # 3D-Pose berechnen
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if success:
                    # ---> DIE WICHTIGSTE ZEILE FÜR DEIN TRAINING <---
                    # Wir speichern Frame-Nummer, Zeitstempel, Marker-ID und die X,Y,Z Koordinaten (in Metern)
                    extracted_data.append([
                        frame_idx, 
                        timestamp_ns, 
                        tag_id, 
                        round(tvec[0][0], 5), # X (Links/Rechts)
                        round(tvec[1][0], 5), # Y (Oben/Unten)
                        round(tvec[2][0], 5)  # Z (Tiefe/Abstand)
                    ])

                    # Visualisierung (Achsen einzeichnen)
                    axis_length = size_m / 2.0
                    axis_points = np.array([
                        [0, 0, 0],             
                        [axis_length, 0, 0],   
                        [0, axis_length, 0],   
                        [0, 0, -axis_length]   
                    ], dtype=np.float32)

                    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
                    img_points = img_points.reshape(-1, 2).astype(int)

                    p_center = tuple(img_points[0])
                    cv2.line(current_image, p_center, tuple(img_points[1]), (0, 0, 255), 3)  
                    cv2.line(current_image, p_center, tuple(img_points[2]), (0, 255, 0), 3)  
                    cv2.line(current_image, p_center, tuple(img_points[3]), (255, 0, 0), 3)

    # 4. Schleife durch alle Frames
    for i in range(num_frames):
        if i % 50 == 0 or i == num_frames - 1:
            print(f"Verarbeite Frame {i}/{num_frames} ({(i/num_frames)*100:.1f}%)")

        image_tuple = provider.get_image_data_by_index(rgb_stream_id, i)
        
        # Den extrem wichtigen Hardware-Zeitstempel auslesen!
        timestamp_ns = image_tuple[1].capture_timestamp_ns 
        
        rgb_image = image_tuple[0].to_numpy_array()
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # AprilTags suchen
        corners_april, ids_april, _ = detector_april.detectMarkers(gray_image)
        process_markers(corners_april, ids_april, bgr_image, True, i, timestamp_ns)

        # ArUco suchen
        corners_aruco, ids_aruco, _ = detector_aruco.detectMarkers(gray_image)
        process_markers(corners_aruco, ids_aruco, bgr_image, False, i, timestamp_ns)

        # Frame ins Video schreiben
        out_video.write(bgr_image)

    out_video.release()
    
    # 5. DIE DATEN ALS CSV SPEICHERN
    print("\nSpeichere extrahierte Posen in CSV...")
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header-Zeile schreiben
        writer.writerow(['frame_index', 'timestamp_ns', 'marker_id', 'pos_x_m', 'pos_y_m', 'pos_z_m'])
        # Alle gesammelten Daten schreiben
        writer.writerows(extracted_data)

    print("-" * 60)
    print(f"✅ Video generiert: {output_video_path}")
    print(f"✅ Daten extrahiert: {output_csv_path} ({len(extracted_data)} Einträge)")

if __name__ == "__main__":
    # Pfade anpassen
    input_vrs = "../Data_collection/Data_vrs/Edu_5_20260604_170944.vrs"
    output_mp4 = "../Data_collection/aruco_Edu_5.mp4"
    output_csv = "../Data_collection/aruco_poses_Edu_5.csv"
    
    extract_and_visualize_aruco(input_vrs, output_mp4, output_csv)