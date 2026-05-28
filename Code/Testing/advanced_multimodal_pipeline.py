import os
import json
import numpy as np
from datetime import timedelta
import cv2  # Für solvePnP-Konvertierungen falls nötig

# Meta Project Aria SDK
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core import mps  # Wichtig für SLAM-Trajektorien

class MultimodalExtractionPipeline:
    def __init__(self, vrs_path, trajectory_csv_path, target_object_positions=None):
        self.vrs_path = vrs_path
        self.trajectory_csv_path = trajectory_csv_path
        # Fest definierte 3D-Weltkoordinaten der Objekte/Zonen im Raum
        self.target_object_positions = target_object_positions or {
            "target_object_1": np.array([1.2, 0.4, -0.1])  # Beispiel-Mittelpunkt [X,Y,Z] relativ zur Roboterbasis
        }
        
        # 1. SLAM/VIO-Trajektorie aus MPS (Machine Perception Services) laden
        print("Lade geschlossene SLAM-Trajektorie (1kHz IMU-Rate)...")
        self.trajectory = mps.read_closed_loop_trajectory(self.trajectory_csv_path)
        
        # 2. VRS Data Provider initialisieren
        self.provider = data_provider.create_vrs_data_provider(self.vrs_path)
        
        # 3. Statische Transformationsmatrix T_Robot_World (T_R_W) initialisieren
        self.T_R_W = np.eye(4)
        
    def set_static_calibration(self, rvec_aruco, tvec_aruco, T_cam_device, T_robot_aruco):
        """
        Berechnet einmalig die statische Transformation vom SLAM-Weltkoordinatensystem (W) 
        zum Roboter-Basissystem (R) basierend auf dem Kalibrierungszeitpunkt t0.
        """
        # Aruco-Pose aus solvePnP in 4x4 Matrix konvertieren (T_Camera_Aruco)
        R_c_m, _ = cv2.Rodrigues(rvec_aruco)
        T_c_m = np.eye(4)
        T_c_m[:3, :3] = R_c_m
        T_c_m[:3, 3] = tvec_aruco.flatten()
        
        # Beispielhafter Kalibrierungs-Timestamp t0 (erster Frame)
        t0_ns = self.provider.get_first_time_ns(StreamId("371-1"), TimeDomain.DEVICE_TIME)
        pose_t0 = self.get_closest_slam_pose(t0_ns)
        
        # Matrix-Konvertierung aus Sophus SE3 (Projekt-Aria Format) -> 4x4 NumPy
        T_W_D_t0 = pose_t0.T_world_device.to_matrix() 
        
        # Komplette Transformationskette zum Roboter-Nullpunkt in der SLAM-Welt
        T_W_R = T_W_D_t0 @ T_cam_device @ T_c_m @ T_robot_aruco
        
        # Inverse: Transformiert Punkte VON der SLAM-Welt IN das Roboter-System
        self.T_R_W = np.linalg.inv(T_W_R)
        print("Statische Welt-zu-Roboter-Kalibrierung erfolgreich berechnet.")

    def get_closest_slam_pose(self, timestamp_ns):
        """Sucht hocheffizient die zeitlich am nächsten liegende SLAM-Pose."""
        # Sucht den passenden Eintrag aus der geladenen MPS-Liste
        closest_pose = min(
            self.trajectory, 
            key=lambda x: abs(x.tracking_timestamp.count() - timestamp_ns)
        )
        return closest_pose

    def transform_point_to_robot(self, pt_device, T_W_D):
        """Transformiert einen 3D-Punkt: Device -> SLAM-Welt -> Roboterbasis"""
        # Homogene Koordinaten erstellen [X, Y, Z, 1]
        pt_h = np.append(pt_device, 1.0)
        # Schritt 1: Brille zu SLAM-Welt
        pt_w = T_W_D @ pt_h
        # Schritt 2: SLAM-Welt zu Roboterbasis
        pt_r = self.T_R_W @ pt_w
        return pt_r[:3]

    def process_and_export(self, output_json_path):
        hand_stream_id = StreamId("371-1")
        eye_stream_id = StreamId("373-1")
        num_hand_samples = self.provider.get_num_data(hand_stream_id)
        
        extracted_records = []
        
        # Variablen zur Berechnung von Ableitungen (Geschwindigkeit/Beschleunigung)
        last_positions = {"left": None, "right": None}
        last_velocities = {"left": None, "right": None}
        last_time_sec = None

        print(f"Starte Transformation und Feature-Engineering für {num_hand_samples} Frames...")

        for i in range(num_hand_samples):
            hand_data = self.provider.get_hand_pose_data_by_index(hand_stream_id, i)
            if hand_data is None:
                continue
                
            timestamp_ns = (hand_data.tracking_timestamp // timedelta(microseconds=1)) * 1000
            timestamp_sec = timestamp_ns / 1e9
            
            # 1. Aktuelle 6DoF Brillen-Pose aus SLAM-Daten holen
            slam_pose = self.get_closest_slam_pose(timestamp_ns)
            T_W_D = slam_pose.T_world_device.to_matrix()
            
            # Zeitdifferenz für Ableitungen berechnen
            dt = (timestamp_sec - last_time_sec) if last_time_sec is not None else 0.0

            hand_frame_info = {"left_hand": None, "right_hand": None}
            
            # 2. Hände verarbeiten & transformieren
            for label in ("left", "right"):
                hand_obj = getattr(hand_data, f"{label}_hand")
                
                if hand_obj is not None and hand_obj.landmark_positions_device is not None:
                    # Rohdaten aus Device-Koordinaten extrahieren
                    raw_joints = [np.array([float(pt[0]), float(pt[1]), float(pt[2])]) for pt in hand_obj.landmark_positions_device]
                    raw_wrist = np.array([float(pos) for pos in hand_obj.get_wrist_position_device()])
                    
                    # --- KOORDINATENTRANSFORMATION ---
                    # Transformiere alle Gelenke in das Roboter-Basissystem
                    transformed_joints = [self.transform_point_to_robot(pt, T_W_D).tolist() for pt in raw_joints]
                    transformed_wrist = self.transform_point_to_robot(raw_wrist, T_W_D)
                    
                    # --- FEATURE ENGINEERING ---
                    # Feature A: Hand-Kompaktheit (Mittlere Distanz aller Gelenke zum Handgelenk)
                    compactness = float(np.mean([np.linalg.norm(np.array(j) - transformed_wrist) for j in transformed_joints]))
                    
                    # Feature B: Kinematik (Geschwindigkeit & Beschleunigung im Raum)
                    velocity = [0.0, 0.0, 0.0]
                    acceleration = [0.0, 0.0, 0.0]
                    
                    if dt > 0 and last_positions[label] is not None:
                        velocity = ((transformed_wrist - last_positions[label]) / dt).tolist()
                        if last_velocities[label] is not None:
                            acceleration = ((np.array(velocity) - np.array(last_velocities[label])) / dt).tolist()
                    
                    # Zustand für nächsten Frame puffern
                    last_positions[label] = transformed_wrist
                    last_velocities[label] = velocity
                    
                    hand_frame_info[f"{label}_hand"] = {
                        "confidence": float(hand_obj.confidence),
                        "wrist_position_robot": transformed_wrist.tolist(),
                        "joints_robot": transformed_joints,
                        "feature_compactness": compactness,
                        "feature_velocity": velocity,
                        "feature_acceleration": acceleration
                    }

            # 3. Eye-Gaze fusionieren & Feature berechnen
            eye_idx = self.provider.get_index_by_time_ns(eye_stream_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            eye_data = self.provider.get_eye_gaze_data_by_index(eye_stream_id, eye_idx)
            
            gaze_info = None
            if eye_data is not None and eye_data.combined_gaze_valid:
                # Blickvektor im CPF (Central Pupil Frame)
                yaw, pitch = float(eye_data.yaw), float(eye_data.pitch)
                
                # Konvertierung von Winkeln (Yaw/Pitch) in einen 3D-Richtungsvektor im Device-Frame
                v_device = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
                # Rotationsmatrix der Brille in die Welt extrahieren (3x3)
                R_W_D = T_W_D[:3, :3]
                v_world = R_W_D @ v_device  # Blickrichtung global im Raum
                
                # --- FEATURE ENGINEERING ---
                # Feature C: Gaze-Object Angular Error (Winkel zum Zielobjekt)
                gaze_errors = {}
                brillen_pos_world = T_W_D[:3, 3]
                
                for obj_name, obj_pos_robot in self.target_object_positions.items():
                    # Objekt von Roboter zurück in Welt transformieren für den Vergleich
                    T_W_R = np.linalg.inv(self.T_R_W)
                    obj_pos_world = (T_W_R @ np.append(obj_pos_robot, 1.0))[:3]
                    
                    # Vektor von Brille zum Objekt
                    v_to_obj = obj_pos_world - brillen_pos_world
                    v_to_obj_norm = v_to_obj / np.linalg.norm(v_to_obj)
                    
                    # Winkeldifferenz über Skalarprodukt
                    dot_product = np.clip(np.dot(v_world, v_to_obj_norm), -1.0, 1.0)
                    gaze_errors[f"angle_to_{obj_name}"] = float(np.degrees(np.arccos(dot_product)))

                gaze_info = {
                    "yaw": yaw, "pitch": pitch, "depth": float(eye_data.depth),
                    "feature_gaze_object_angles": gaze_errors
                }

            # Datensatz schreiben
            extracted_records.append({
                "timestamp_ns": timestamp_ns,
                "timestamp_vrs_seconds": round(timestamp_sec, 3),
                "eye_gaze": gaze_info,
                "hands": hand_frame_info
            })
            last_time_sec = timestamp_sec

        # Speichern als JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_records, f, indent=2)
        print(f"Erfolgreich extrahiert und berechnet. Datei gespeichert unter: {output_json_path}")


# --- WHISPER METADATA LOOK-BACK ADJUSTMENT ---
def adjust_whisper_metadata_with_lookback(input_metadata_json, output_metadata_json, lookback_seconds=1.0):
    """
    Lädt das von Whisper generierte virtuelle Schnittbuch und verschiebt 
    den Startzeitpunkt von Intent 2 (Object Fetching) nach vorne.
    """
    with open(input_metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Durchlaufe alle aufgezeichneten Sequenzen/Trials
    for sequence in metadata.get("sequences", []):
        if sequence.get("intention_class") == 2:  # 2 = Object Fetching
            # Startzeitpunkt um 1 Sekunde nach vorne verschieben (Sakkaden-Auffangfenster)
            original_start = sequence["start_timestamp_vrs_seconds"]
            adjusted_start = max(0.0, original_start - lookback_seconds)
            sequence["start_timestamp_vrs_seconds"] = round(adjusted_start, 3)
            sequence["whisper_latency_corrected"] = True
            
    with open(output_metadata_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Whisper-Metadaten korrigiert (Look-Back von {lookback_seconds}s angewandt).")


if __name__ == "__main__":
    # Beispielhafter Aufruf der Pipeline
    pipeline = MultimodalExtractionPipeline(
        vrs_path="../Recordings/test_recording.vrs",
        trajectory_csv_path="../Recordings/mps_output/closed_loop_trajectory.csv"
    )
    
    # Hier fütterst du die ermittelten Matrizen deiner Aruco-Kalibrierung ein
    # pipeline.set_static_calibration(rvec_aruco, tvec_aruco, T_cam_device, T_robot_aruco)
    
    # Ausführen
    pipeline.process_and_export("transformed_features_output.json")
    
    # Nachgelagert: Whisper-Korrektur anwenden
    if os.path.exists("metadata.json"):
        adjust_whisper_metadata_with_lookback("metadata.json", "metadata_final_dataloader.json")