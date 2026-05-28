import os
import json
import numpy as np
from datetime import timedelta
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

def extract_tracking_data(vrs_file_path, output_json_path):
    # Check if the file exists
    if not os.path.exists(vrs_file_path):
        print(f"Error: VRS file not found: {vrs_file_path}")
        return

    print(f"Opening Aria recording: {vrs_file_path}...")
    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    
    # Define sensor streams
    eye_stream_id = StreamId("373-1")   # Eye-Tracking
    hand_stream_id = StreamId("371-1")  # Hand-Tracking

    print("Loading eye tracking data...")
    num_eye_samples = provider.get_num_data(eye_stream_id)
    print(f"   -> Found {num_eye_samples} gaze direction entries.")

    print("Loading hand tracking data...")
    num_hand_samples = provider.get_num_data(hand_stream_id)
    print(f"   -> Found {num_hand_samples} hand skeleton entries.")

    extracted_records = []

    print("Synchronizing and extracting multimodal sequences...")
    # Hand-Tracking serves as the temporal baseline for the loop
    for i in range(num_hand_samples):
        hand_data = provider.get_hand_pose_data_by_index(hand_stream_id, i)
        
        if hand_data is None:
            continue
            
        # Convert timedelta to nanoseconds
        timestamp_ns = (hand_data.tracking_timestamp // timedelta(microseconds=1)) * 1000
        timestamp_sec = timestamp_ns / 1e9

        hand_frame_info = {"left_hand": None, "right_hand": None}
        
        # Process left and right hands separately
        for label in ("left", "right"):
            hand_obj = getattr(hand_data, f"{label}_hand")
            
            if hand_obj is not None and hand_obj.landmark_positions_device is not None:
                landmarks = hand_obj.landmark_positions_device
                # Convert 21 joint positions to a standard Python list
                joints_3d = [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in landmarks]
                
                hand_frame_info[f"{label}_hand"] = {
                    "confidence": float(hand_obj.confidence),
                    "wrist_position": [float(pos) for pos in hand_obj.get_wrist_position_device()] if hand_obj.get_wrist_position_device() is not None else None,
                    "palm_position": [float(pos) for pos in hand_obj.get_palm_position_device()] if hand_obj.get_palm_position_device() is not None else None,
                    "joints": joints_3d
                }

        # Determine the temporally closest eye-tracking index
        eye_idx = provider.get_index_by_time_ns(
            eye_stream_id, 
            timestamp_ns, 
            TimeDomain.DEVICE_TIME, 
            TimeQueryOptions.CLOSEST
        )
        
        eye_data = provider.get_eye_gaze_data_by_index(eye_stream_id, eye_idx)
        
        gaze_info = None
        if eye_data is not None and eye_data.combined_gaze_valid:
            gaze_info = {
                "yaw": float(eye_data.yaw),
                "pitch": float(eye_data.pitch),
                "depth": float(eye_data.depth)
            }

        # Merge datasets for this specific time step
        frame_entry = {
            "timestamp_ns": timestamp_ns,
            "timestamp_vrs_seconds": round(timestamp_sec, 3),
            "eye_gaze": gaze_info,
            "hands": hand_frame_info
        }
        
        extracted_records.append(frame_entry)

    # Save results as a structured JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_records, f, indent=2)
        
    print("finished")

if __name__ == "__main__":
    extract_tracking_data(
        vrs_file_path="../Recordings/test_recording.vrs",
        output_json_path="tracking_features_output.json"
    )