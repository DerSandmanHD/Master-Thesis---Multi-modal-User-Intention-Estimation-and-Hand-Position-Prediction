# Overview of Recorded Data, Derived Signals, and Processing

## Objective

This Master's thesis uses multimodal recordings captured with Project Aria glasses. Each recording is initially stored as a VRS file. This file contains multiple timestamped sensor and perception signals that are subsequently converted into a common training representation.

Three processing levels must be distinguished:

1. **Recorded data:** VRS recordings containing RGB, audio, and timestamped perception signals.
2. **Derived primary data:** gaze estimates, MPS hand tracking, MPS SLAM, and marker poses computed from RGB.
3. **Training data:** synchronized, transformed, and normalized features, together with labels and future targets.

Gaze, hand tracking, SLAM, and marker poses are therefore not equivalent to unprocessed sensor pixels. They are already estimated or geometrically derived signals.

## Data Sources

### Eye Gaze

The gaze data describe the direction in which the participant is looking. Available information includes gaze angles, gaze direction, and an estimated gaze depth.

The gaze data are partly three-dimensional:

- gaze direction as a 3D vector,
- an optionally estimated 3D gaze point,
- horizontal and vertical gaze angles.

The gaze direction is later compared with object positions to determine which objects lie close to the current gaze direction.

The 3D gaze direction can remain useful even when no valid metric depth is available. In that case, an internally constructed 3D point must not be interpreted as an actually measured fixation point. For each candidate object, the pipeline primarily computes the angle between the gaze direction and the direction toward the marker center, as well as the distance from the gaze origin to the marker center.

### Hand Tracking

Hand tracking is generated through the Project Aria Machine Perception Services. Three-dimensional hand data are available for both hands.

These data include:

- 3D positions of the hand landmarks,
- 3D position and orientation of the wrist,
- tracking confidence and validity information.

The current model mainly uses the 6-DoF wrist pose, meaning the 3D position and orientation of the left and right wrists.

### SLAM and Head Motion

The SLAM data describe the movement and orientation of the glasses in space.

They contain:

- 3D position of the glasses,
- 3D orientation,
- linear and rotational velocity,
- trajectory quality values.

The absolute SLAM pose is mainly required to transform hand, gaze, and object positions into a common spatial coordinate frame. The features currently passed directly to the model are linear velocity, angular velocity, and the SLAM quality score. Absolute world position and orientation are not used directly as model inputs.

### RGB Images

The RGB camera provides 2D egocentric images.

The sensor baseline does not receive RGB pixels or image embeddings directly.
RGB is used for:

- visual inspection of recordings,
- manual annotation,
- detection of AprilTag and ArUco markers.

### AprilTag and ArUco Markers

The markers are initially detected as 2D structures in the RGB image. Because their physical size is known and the camera is calibrated, a 3D pose relative to the camera can be estimated.

This provides:

- 3D positions of the objects,
- 3D orientations of the markers,
- a reference marker for the robot-marker coordinate frame.

The object markers therefore provide a geometric representation of the scene.

AprilTag 0 serves as the spatial anchor. ArUco IDs 6 to 14 identify the candidate objects. The robot frame used in the current pipeline corresponds to the coordinate frame of AprilTag 0. A fixed transformation from this marker to the actual physical robot base has not yet been incorporated.

### Audio

The audio signal is not used as a model input.

It is used exclusively to detect the spoken phase commands and thereby define the temporal boundaries of the classes:

- `START` to `SECOND`: `continue`
- `SECOND` to `DONE`: `fetch`
- `DONE` to `THIRD`: transition phase
- from `THIRD` onward: `handover`

Automatically detected timestamps are manually corrected when necessary.

In addition, the target object ID and the actual receiving hand are manually annotated for each sequence. The target object ID is not passed directly to the current model as an input feature. Instead, the model receives the geometric information of all candidate objects. The receiving-hand annotation determines which future hand pose is used as the regression target.

## Data Processing

The individual data sources have different sampling rates and timestamps. They are therefore aligned to a common timeline using device time.

The processing can be summarized as follows:

1. Gaze, audio, and RGB data are read from the VRS file.
2. Hand tracking and SLAM are loaded from the MPS outputs.
3. Markers are detected in the RGB images and localized in 3D.
4. The modalities are merged based on their timestamps. For each modality, the temporally nearest sample is selected.
5. Hand, gaze, and object positions are transformed into a common robot-marker coordinate frame using SLAM, camera calibration, and AprilTag 0.
6. Missing or invalid measurements are not treated as true zero values, but are represented using validity masks.
7. For each time step, the actually observed pose of the annotated receiving hand approximately one second later is assigned as the regression target. It is not extrapolated from the current pose.

The default maximum time differences are:

| Modality | Maximum difference from the gaze timestamp |
|---|---:|
| Hand tracking | 12 ms |
| SLAM | 5 ms |
| AprilTag/ArUco markers | 20 ms |

If no source sample lies within the respective tolerance, the corresponding values remain missing. Time offsets are also stored so that the actual synchronization error can be inspected.

## Shortened Examples from the Data

The following examples are taken from the locally available sequence `Jona_6_20260616_182111`. For readability, only relevant columns are shown and some decimal places have been shortened. The gaze rows come directly from `gaze_Jona_6_20260616_182111.csv`. Because the separate hand, SLAM, and marker CSV files are not available locally, the examples use the corresponding source fields preserved unchanged in the local master export and map them back to their original column names. The examples therefore show real values from the recording, but not the full width of the original files.

### Example 1: Gaze CSV

Shortened valid row:

```csv
timestamp_ns,gaze_valid,gaze_yaw_rad,gaze_pitch_rad,gaze_depth_m,gaze_direction_cpf_x,gaze_direction_cpf_y,gaze_direction_cpf_z
2184178902000,1,-0.115458,-0.353685,0.550493,-0.108157,-0.344326,0.932600
```

Interpretation:

- `timestamp_ns` is the recording timestamp in the device-time domain.
- `gaze_valid=1` indicates a valid combined gaze estimate.
- `yaw` and `pitch` are the horizontal and vertical gaze angles in radians.
- `gaze_depth_m=0.550493` corresponds to an estimated gaze depth of approximately 55 cm.
- the three direction values form a normalized 3D direction vector in the Central Pupil Frame.

An invalid row can look like this:

```csv
timestamp_ns,gaze_valid,gaze_yaw_rad,gaze_pitch_rad,gaze_depth_m,gaze_direction_cpf_x,gaze_direction_cpf_y,gaze_direction_cpf_z
2184345569000,0,,,,,,
```

The empty fields are not true zero values. `gaze_valid=0` indicates that the spatial values in this row must not be used.

### Example 2: Hand-Tracking CSV

The actual MPS file contains 21 landmarks for each hand, each with x, y, and z coordinates. The following excerpt shows only landmark 0, the left wrist pose, and both tracking confidences:

```csv
tracking_timestamp_us,left_tracking_confidence,right_tracking_confidence,tx_left_landmark_0_device,ty_left_landmark_0_device,tz_left_landmark_0_device,tx_left_device_wrist,ty_left_device_wrist,tz_left_device_wrist,qx_left_device_wrist,qy_left_device_wrist,qz_left_device_wrist,qw_left_device_wrist
2187449882,0.999886,0.999861,0.124915,0.102885,0.362759,-0.002391,0.142352,0.350480,0.901125,-0.391568,0.164831,-0.086486
```

Interpretation:

- the hand timestamp is given in microseconds and is multiplied by 1000 before merging,
- both hands have very high tracking confidence in this sample,
- landmark 0 corresponds to the left thumb tip,
- the wrist translation describes the wrist position in meters in the device frame,
- `qx,qy,qz,qw` describe the orientation as a quaternion.

A sample without valid hand tracking can look like this:

```csv
tracking_timestamp_us,left_tracking_confidence,right_tracking_confidence,tx_left_landmark_0_device,ty_left_landmark_0_device,tz_left_landmark_0_device,tx_left_device_wrist,ty_left_device_wrist,tz_left_device_wrist,qx_left_device_wrist,qy_left_device_wrist,qz_left_device_wrist,qw_left_device_wrist
2184184303,-1.0,-1.0,,,,,,,,,,
```

MPS uses `-1.0` to indicate missing tracking. The pipeline defines a hand as valid when its confidence is greater than 0 and sets the spatial fields of an invalid hand to missing.

### Example 3: Closed-Loop SLAM CSV

```csv
graph_uid,tracking_timestamp_us,tx_world_device,ty_world_device,tz_world_device,qx_world_device,qy_world_device,qz_world_device,qw_world_device,device_linear_velocity_x_device,device_linear_velocity_y_device,device_linear_velocity_z_device,angular_velocity_x_device,angular_velocity_y_device,angular_velocity_z_device,quality_score
9490ef9f-9ab6-2cbe-0447-6ee1b76634a4,2185378594,-0.002986,0.007549,-0.001776,-0.830973,-0.202605,0.090885,0.510074,0.006503,0.009065,-0.023220,0.060076,-0.096881,-0.181835,1.0
```

Interpretation:

- `graph_uid` identifies the sequence-specific SLAM world frame,
- `t*_world_device` is the position of the glasses in the world frame in meters,
- `q*_world_device` is their orientation,
- linear velocity is given in m/s,
- angular velocity is given in rad/s,
- `quality_score=1.0` indicates high SLAM quality in this row.

The SLAM world frame is consistent within one sequence, but it is not automatically identical across different recordings. Model-relevant positions are therefore additionally transformed into the AprilTag-0 frame.

### Example 4: Marker CSV

A shortened AprilTag-0 row:

```csv
sequence_id,timestamp_ns,marker_family,marker_id,marker_role,marker_size_m,tx_camera_m,ty_camera_m,tz_camera_m,qx_camera_marker,qy_camera_marker,qz_camera_marker,qw_camera_marker,reprojection_error_px,marker_area_px2
Jona_6_20260616_182111,2184184303815,apriltag_36h11,0,robot,0.10,0.576793,0.269472,0.727162,0.036775,-0.772419,0.633354,0.029642,0.88039,8205.0
```

An object-marker row from the same RGB timestamp:

```csv
sequence_id,timestamp_ns,marker_family,marker_id,marker_role,marker_size_m,tx_camera_m,ty_camera_m,tz_camera_m,qx_camera_marker,qy_camera_marker,qz_camera_marker,qw_camera_marker,reprojection_error_px,marker_area_px2
Jona_6_20260616_182111,2184184303815,aruco_4x4_50,6,object,0.05,0.025512,0.262624,0.722630,0.759888,0.120692,-0.116409,0.628055,0.66805,1998.0
```

Interpretation:

- AprilTag 0 has a physical side length of 10 cm and defines the robot-marker frame.
- Object marker 6 has a side length of 5 cm.
- `t*_camera_m` describes the marker position relative to the RGB camera.
- the quaternion describes the marker orientation relative to the camera.
- `reprojection_error_px` measures how well the estimated 3D pose explains the detected 2D corners; lower values are generally better.
- `marker_area_px2` is the visible marker area in the image. Larger marker areas often provide more stable poses.

The current code stores reprojection error and marker area for quality analysis, but it does not yet reject poses using a fixed threshold.

### Example 5: Merged Master Row

After synchronization, a strongly shortened excerpt looks like this:

```csv
sequence_id,timestamp_ns,intent_label,gaze_valid,hand_left_valid,hand_right_valid,slam_time_offset_ms,apriltag_0_valid,aruco_6_valid,aruco_6_gaze_angle_rad,aruco_6_gaze_distance_m
Jona_6_20260616_182111,2187445569000,continue,1,1,1,0.025,1,1,0.251063,0.791002
```

This row shows that gaze, both hands, SLAM, the robot anchor, and object marker 6 are usable at the common timestamp. The SLAM match lies only 0.025 ms away from the gaze timestamp. The gaze ray is approximately 0.251 rad, or 14.4 degrees, away from the direction toward the marker center. The object center is approximately 0.791 m away from the gaze origin.

### Example 6: Future Target at `t + 1 s`

```csv
timestamp_ns,intent_label,left_wrist_robot_x_m,left_wrist_robot_y_m,left_wrist_robot_z_m,future_1s_left_wrist_robot_x_m,future_1s_left_wrist_robot_y_m,future_1s_left_wrist_robot_z_m,future_1s_left_wrist_valid,future_1s_time_error_ms
2209912233000,handover,0.523034,0.600817,-0.034252,0.533835,0.596539,-0.030100,1,0.0
```

The first three position values are the current left-wrist position in the AprilTag-0 frame. The `future_1s_*` values come from the actually observed hand pose one second later. `future_1s_time_error_ms=0.0` shows that, in this example, a sample existed exactly at the desired future timestamp. Only later does the model attempt to predict this measured target from the observation window.

## Resulting Model Representation

At the end of the pipeline, each recording is represented as a synchronized time series containing:

- 3D gaze information,
- 3D wrist positions and orientations,
- motion data of the glasses,
- 3D object positions,
- visibility and quality values,
- intention labels,
- future hand poses as regression targets.

The sensor baseline data are therefore predominantly **3D geometric data**.
The original RGB images are additionally used for marker extraction and manual review;
separate visual ablations use time-aligned CLIP embeddings.

The model inputs consist of temporal windows of these synchronized features. Missing measurements are represented using additional observation masks, allowing the model to distinguish between a missing value and a true numerical value.

The sensor baseline feature profile mainly uses gaze angles and robot-relative gaze direction, robot-relative wrist poses, hand confidence values, SLAM velocities, SLAM quality, robot-relative object positions, gaze-object angles, gaze-object distances, and visibility flags. Full hand landmarks, RGB pixels, absolute SLAM world poses, and marker reprojection errors are not direct sensor-baseline inputs. Separate visual ablations append time-aligned CLIP embeddings.

## Compact Overview

| Modality | Original representation | Use |
|---|---|---|
| Gaze | 3D direction, angles, optional depth | gaze direction and object relation |
| Hand tracking | 3D landmarks and 6-DoF wrist pose | hand motion and future hand pose |
| SLAM | 3D position, orientation, and motion | absolute pose for transformations; velocity and quality as model features |
| RGB | 2D images | marker extraction and review |
| Markers | 2D detection, converted to 3D pose | object and robot-marker position |
| Audio | 1D time signal | phase segmentation, not a model input |

## Short Summary for a Supervisor Meeting

The recordings consist of gaze, hand tracking, SLAM, RGB, and audio. Gaze, hands, head motion, and markers are processed predominantly as 3D data. The RGB recordings are initially 2D, but they are mainly used for marker and object localization. Because the marker size is known and the camera is calibrated, 3D object poses can be estimated from the 2D detections. The absolute SLAM pose is mainly used for coordinate transformations, while velocity, angular velocity, and SLAM quality are used as direct motion features.

Audio is used only for temporal segmentation of the interaction phases and is not a model input. The target object and receiving hand are additionally annotated manually. All modalities are synchronized using their timestamps and transformed into a common AprilTag-based robot-marker frame. This frame currently corresponds to AprilTag 0 rather than directly to the physical robot base. The model then receives temporal windows containing gaze, wrist, motion, and object features, together with validity masks. The targets are the current intention and the actually observed future position and orientation of the receiving hand at approximately `t + 1 s`.
