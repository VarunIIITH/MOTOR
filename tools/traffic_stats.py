import os
import cv2
import json
import argparse # Import argparse
from ultralytics import YOLO

# YOLOv11x and BotSort placeholder classes are no longer needed
# as Ultralytics' YOLO object will handle both detection and tracking.

def load_yolo_model(model_path='yolo11x.pt'):
    """
    Load the YOLO model from Ultralytics.
    
    Args:
        model_path (str): Path to the YOLO model weights file (e.g., 'yolov8x.pt').
                          The model will be downloaded automatically if not present.
    
    Returns:
        ultralytics.YOLO: Initialized YOLO model instance.
    """
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
    return model

def process_video(video_path, yolo_model):
    """
    Process a single video: detect objects with YOLO, track with Bot-Sort, 
    and save tracks in the same directory.
    
    Args:
        video_path (str): Path to the input video.
        yolo_model (ultralytics.YOLO): Initialized YOLO model instance.
    """
    output_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f'{video_filename}_tracks.json')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    interested_classes_names = [
        'car', 'bus', 'motorcycle', 'person', 
        'truck', 'traffic light',           
        'bicycle', 'dog', 'cat', 'horse'
    ]
    
    model_class_mapping = yolo_model.model.names 
    interested_class_indices = []
    valid_model_class_names = []

    if isinstance(model_class_mapping, dict):
        name_to_id_map = {name: k for k, name in model_class_mapping.items()}
        for name in interested_classes_names:
            if name in name_to_id_map:
                interested_class_indices.append(name_to_id_map[name])
                valid_model_class_names.append(name)
            else:
                print(f"Warning: Class '{name}' not found in model vocabulary. Available classes: {list(model_class_mapping.values())}")
    else:
        print("Warning: Could not determine class indices precisely from model_class_mapping. Ensure model.model.names is a dictionary.")
        valid_model_class_names = interested_classes_names

    if not interested_class_indices and isinstance(model_class_mapping, dict):
        print("Warning: None of the specified interested classes were found in the model's vocabulary.")
    elif interested_class_indices:
        print(f"Tracking the following classes (indices): {valid_model_class_names} ({interested_class_indices})")

    all_tracks = {}
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            results = yolo_model.track(
                source=frame, 
                persist=True, 
                tracker="botsort.yaml", 
                classes=interested_class_indices if interested_class_indices else None,
                verbose=False,
                conf = 0.4 
            )
        except:
            print("Error at frame: ", frame_id)
            continue
        
        current_frame_tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            # Ensure data is on CPU and converted to NumPy arrays
            track_ids_np = results[0].boxes.id.cpu().numpy() 
            detected_boxes_xywh_np = results[0].boxes.xywh.cpu().numpy()
            detected_class_ids_np = results[0].boxes.cls.cpu().numpy()

            for i in range(len(track_ids_np)):
                # --- FIX: Convert NumPy types to Python native types ---
                obj_track_id = int(track_ids_np[i])
                class_id = int(detected_class_ids_np[i])
                # --- End of FIX ---
                
                if class_id not in model_class_mapping:
                    # This check might be redundant if yolo_model.track already filters by valid classes from model_class_mapping
                    # print(f"Warning: Detected class_id {class_id} not in model_class_mapping. Skipping.")
                    continue
                class_name = model_class_mapping[class_id] 

                if class_name in valid_model_class_names:
                    x_center, y_center, w, h = detected_boxes_xywh_np[i]
                    
                    # --- FIX: Ensure all bbox components are Python floats ---
                    x_min = float(x_center - w / 2)
                    y_min = float(y_center - h / 2)
                    bbox_w = float(w)
                    bbox_h = float(h)
                    bbox = [x_min, y_min, bbox_w, bbox_h]
                    # --- End of FIX ---
                    
                    current_frame_tracks.append({
                        'id': obj_track_id, # Now a Python int
                        'class': class_name, # Already a string
                        'bbox': bbox         # Now a list of Python floats
                    })
        
        for track_in_frame in current_frame_tracks: # Renamed 'track' to 'track_in_frame' to avoid confusion
            track_id_val = track_in_frame['id'] # This is now a Python int
            cls = track_in_frame['class']
            bbox_val = track_in_frame['bbox'] 
            
            if track_id_val not in all_tracks:
                all_tracks[track_id_val] = {'class': cls, 'frames': []}
            all_tracks[track_id_val]['class'] = cls 
            # frame_id is already a Python int
            all_tracks[track_id_val]['frames'].append({'frame': frame_id, 'bbox': bbox_val})
            
        if frame_id > 0 and frame_id % 100 == 0:
            print(f"Processed frame {frame_id} for {os.path.basename(video_path)}")

        frame_id += 1
    
    cap.release()
    
    # The keys (tid) and values in all_tracks should now be Python native types
    tracks_list = [
        {'track_id': tid, 'class': data['class'], 'frames': data['frames']}
        for tid, data in all_tracks.items()
    ]
    
    try:
        with open(output_path, 'w') as f:
            json.dump(tracks_list, f, indent=4)
        print(f"Saved tracks to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}") # This error should now be gone


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process videos for object detection and tracking.")
    parser.add_argument(
        "main_dir", 
        type=str, 
        help="The main directory containing video files to process."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo11x.pt",
        help="Path to the YOLO model weights file (e.g., 'yolov8x.pt', 'yolov9c.pt'). Defaults to 'yolov8x.pt'."
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default="Front_combined_output_synced.mp4",
        help="The specific video filename to look for in subdirectories. Defaults to 'Front_combined_output_synced.mp4'."
    )

    args = parser.parse_args()
    
    main_processing_dir = args.main_dir
    model_weights_path = args.model_path
    target_video_filename = args.video_filename

    if not os.path.isdir(main_processing_dir):
        print(f"Error: The specified main directory does not exist: {main_processing_dir}")
        return

    video_paths = []
    for root, dirs, files in os.walk(main_processing_dir):
        for file in files:
            if file == target_video_filename: 
                video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        print(f"No videos named '{target_video_filename}' found in '{main_processing_dir}' or its subdirectories.")
        return
    
    print(f"Found {len(video_paths)} video(s) to process.")
    
    try:
        yolo_model = load_yolo_model(model_path=model_weights_path) 
    except Exception as e:
        print(f"Failed to load YOLO model from {model_weights_path}: {e}")
        print("Please ensure you have 'ultralytics' installed and the model path is correct or downloadable.")
        return
        
    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")
        process_video(video_path, yolo_model)

if __name__ == '__main__':
    # Before running:
    # 1. Make sure you have the ultralytics package installed: pip install ultralytics
    # 2. Ensure you have a stable internet connection the first time you run,
    #    so the YOLO model weights (e.g., 'yolov8x.pt') can be downloaded if not already present.
    #
    # To run the script:
    # python your_script_name.py /path/to/your/video_directory
    # Example:
    # python your_script_name.py 
    #
    # To specify a different model or video filename:
    # python your_script_name.py /path/to/videos --model_path yolov8n.pt --video_filename my_video.mp4
    main()