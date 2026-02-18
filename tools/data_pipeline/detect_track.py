# run_batch_processing.py
#
# Description:
# This script scans a specified directory structure for video files, processes each
# video to perform object detection (YOLOv8), tracking (BotSort), and depth
# estimation (Depth Anything V2).
#
# Instead of creating an output video, it extracts the following data for each
# tracked object: frame number, track ID, class, bounding box, confidence,
# estimated depth, and an 'is_important' flag based on a depth threshold.
#
# All data from all videos within a single subfolder (e.g., 'Front') is
# aggregated and saved into a single JSON file for that subfolder (e.g., 'Front_detections.json').
#
# Instructions:
# 1. Place this script in the directory containing your '01 - main folder'.
# 2. Ensure the directory structure is as specified:
#    ./run_batch_processing.py
#    ./01 - main folder/
#        ├── Aria/Pupil/
#        │   ├── video1.mp4
#        │   └── ...
#        ├── Front/
#        ├── Rear/
#        └── Helmet/
# 3. Update the 'root_dir' and 'yolo_model_file' variables if needed.
# 4. Run the script: python run_batch_processing.py

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
import time
from collections import defaultdict
import os
import json
import gc # Import garbage collector
import re # Import regular expression module for natural sorting

def calculate_overlap(box1, box2):
    """Calculates the overlap of box1 relative to box2 (intersection_area / area_of_box1)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    if area1 == 0: return 0
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2) # Corrected min for y2
    inter_width, inter_height = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height
    return intersection_area / area1

def custom_clip_sort_key(filename):
    """
    Key for sorting filenames that contain a 'clip_XXX' sequence numerically.
    Extracts the number following 'clip_' and uses it for sorting.
    If 'clip_' pattern is not found, it falls back to regular string sorting.
    """
    match = re.search(r'clip_(\d+)', filename, re.IGNORECASE)
    if match:
        # Return a tuple: (0, integer_clip_number, original_filename)
        # The 0 ensures files with clip numbers sort before those without (if any)
        return (0, int(match.group(1)), filename)
    else:
        # Fallback for files that don't match the pattern
        # Return a tuple: (1, original_filename) to sort them after matched files
        return (1, filename)

def process_video_and_extract_data(video_path, yolo_model, depth_pipe, device):
    """
    Processes a video to extract tracking, and depth data, returning it as a list of dicts.
    """
    print(f"\n--- Processing video: {video_path} ---")
    detections_data = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at '{video_path}'")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Initialize Tracking and State Management ---
        track_history = defaultdict(lambda: {"is_important": False})
        frame_count = 0
        video_name = os.path.basename(video_path)

        while cap.isOpened():
            ret, frame_cv = cap.read()
            if not ret: break

            loop_start_time = time.time()
            image = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
            
            try:
                # The YOLO model and tracker are already loaded for the current subfolder
                track_results = yolo_model.track(image, persist=True, tracker="botsort.yaml", verbose=False)
            except Exception as e:
                print(f"Warning: BotSort tracker failed on frame {frame_count} in {video_name}. Error: {e}. Skipping frame.")
                frame_count += 1
                continue
            
            # --- Depth Estimation (once per frame) ---
            depth_result = depth_pipe(image)
            predicted_depth = depth_result['predicted_depth']
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(0).unsqueeze(0),
                size=(frame_height, frame_width), mode="bicubic", align_corners=False
            ).squeeze()
            depth_map = prediction.cpu().numpy()

            # --- Process Tracked Objects and Store Data ---
            if track_results and track_results[0].boxes.id is not None:
                boxes = track_results[0].boxes.xyxy.cpu().numpy()
                track_ids = track_results[0].boxes.id.int().cpu().tolist()
                class_ids = track_results[0].boxes.cls.int().cpu().tolist()
                confs = track_results[0].boxes.conf.cpu().tolist()
                
                # Rider-merging logic
                persons_data = [(i, boxes[i]) for i, cid in enumerate(class_ids) if yolo_model.names[cid] == 'person']
                motorcycles_data = [(i, boxes[i]) for i, cid in enumerate(class_ids) if yolo_model.names[cid] == 'motorcycle']
                indices_to_remove = set()
                for p_idx, p_box in persons_data:
                    for m_idx, m_box in motorcycles_data:
                        if calculate_overlap(p_box, m_box) >= 0.2:
                            indices_to_remove.add(p_idx)
                            break
                
                for i in range(len(track_ids)):
                    if i in indices_to_remove: continue

                    track_id = track_ids[i]
                    box = boxes[i]
                    conf = confs[i]
                    class_name = yolo_model.names[class_ids[i]]
                    x1, y1, x2, y2 = map(int, box)
                    
                    object_depth_region = depth_map[y1:y2, x1:x2]
                    average_depth = np.median(object_depth_region) if object_depth_region.size > 0 else 0

                    # Update state if object comes closer than 20m
                    if average_depth <= 20:
                        track_history[track_id]['is_important'] = True
                    
                    # Store data only for objects within 30 meters
                    if average_depth <= 30:
                        detection_log = {
                            "video_name": video_name,
                            "frame": frame_count,
                            "track_id": track_id,
                            "class_name": class_name,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "confidence": float(f"{conf:.4f}"),
                            "estimated_depth_m": float(f"{average_depth:.2f}"),
                            "is_important": track_history[track_id]['is_important']
                        }
                        detections_data.append(detection_log)
            
            frame_count += 1
            processing_time = time.time() - loop_start_time
            print(f"  Processed frame {frame_count}/{total_frames} (Time: {processing_time:.2f}s)", end='\r')
        
        print(f"\nFinished processing {video_name}. Found {len(detections_data)} relevant detections.")
        cap.release()

    except Exception as e:
        print(f"An unexpected error occurred while processing {video_path}: {e}")
    
    return detections_data

if __name__ == "__main__":
    # --- Configuration ---
    root_dir = ""
    subfolder_names = ["Aria", "Front", "Rear", "Helmet", "Pupil"]
    yolo_model_file = 'yolo11x.pt' # Can be a custom model like 'yolo11x.pt'

    # --- Setup Device (Once) ---
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Iterate Through Directories and Process Videos ---
    for subfolder in subfolder_names:
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            print(f"\nWarning: Subfolder not found, skipping: {subfolder_path}")
            continue

        print(f"\n======================================================")
        print(f"Scanning subfolder: {subfolder_path}")
        print(f"======================================================")

        # --- Load Models for the Current Subfolder ---
        print(f"Loading YOLO model for {subfolder}: {yolo_model_file}...")
        try:
            yolo_model = YOLO(yolo_model_file)
            print("YOLO model loaded.")
        except Exception as e:
            print(f"Fatal Error: Could not load YOLO model for {subfolder}. Error: {e}")
            # Decide if you want to skip the subfolder or exit entirely
            continue # Skip to next subfolder

        print(f"Loading Depth Anything V2 Metric model for {subfolder}...")
        depth_pipeline_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
        try:
            depth_pipe = pipeline(task="depth-estimation", model=depth_pipeline_id, device=device)
            print("Depth Anything pipeline loaded.")
        except Exception as e:
            print(f"Fatal Error: Could not load Depth Anything pipeline for {subfolder}. Error: {e}")
            # Clean up YOLO model before skipping
            del yolo_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue # Skip to next subfolder


        # Find all video files in the current subfolder
        video_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"No video files found in {subfolder_path}.")
            # Clean up models even if no videos are found
            del yolo_model
            del depth_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # --- Sort video files using the custom key for 'clip_XXX' ---
        video_files.sort(key=custom_clip_sort_key)
        print(f"Found {len(video_files)} video files. Processing in sorted order.")

        all_subfolder_detections = []
        for video_filename in video_files:
            video_path = os.path.join(subfolder_path, video_filename)
            # Process one video and get its data
            video_data = process_video_and_extract_data(video_path, yolo_model, depth_pipe, device)
            # Add the data from this video to the subfolder's master list
            if video_data:
                all_subfolder_detections.extend(video_data)
        
        # --- 3. Save Aggregated Data to JSON ---
        if not all_subfolder_detections:
            print(f"\nNo detections were recorded for the entire subfolder: {subfolder}")
        else:
            # Create a clean filename for the JSON output (e.g., "Aria/Pupil" -> "Aria_Pupil")
            sanitized_subfolder_name = subfolder.replace('/', '_').replace('\\', '_')
            output_json_path = os.path.join(root_dir, f"{sanitized_subfolder_name}_detections.json")
            
            print(f"\nAggregating results for subfolder '{subfolder}'...")
            try:
                with open(output_json_path, 'w') as f:
                    json.dump(all_subfolder_detections, f, indent=4)
                print(f"✅ Success! All detections for '{subfolder}' saved to: {output_json_path}")
            except Exception as e:
                print(f"❌ Error: Could not write JSON file to {output_json_path}. Error: {e}")

        # --- Unload Models and Clear GPU Memory for the Current Subfolder ---
        print(f"Unloading models and clearing GPU memory for subfolder: {subfolder}...")
        del yolo_model
        del depth_pipe
        # Explicitly clear CUDA cache if running on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() # Trigger Python's garbage collector
        print(f"Models unloaded and GPU memory cleared for {subfolder}.")

    print("\n\nBatch processing complete.")