import os
import json
import numpy as np
import cv2
import torch
import argparse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# === Command-line arguments ===
parser = argparse.ArgumentParser(
    description='Extract CLIP features per video clip'
)
parser.add_argument(
    'clips_dir',
    help='Directory containing video clips'
)
parser.add_argument(
    'metadata_file',
    help='Path to metadata JSON file mapping clips to labels'
)
parser.add_argument(
    '--sample_fps',
    type=int,
    default=10,
    help='Sampling rate (frames per second)'
)
parser.add_argument(
    '--model_name',
    default='openai/clip-vit-base-patch32',
    help='Hugging Face CLIP model identifier'
)
args = parser.parse_args()

# Configuration from args
CLIPS_DIR     = args.clips_dir
METADATA_FILE = args.metadata_file
SAMPLE_FPS    = args.sample_fps
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME    = args.model_name

# Derive FEATURES_DIR as a sibling of 'clips' and 'gt'
# Find common base path between clips_dir and metadata_dir
metadata_dir = os.path.dirname(METADATA_FILE)
base_dir = os.path.commonpath([CLIPS_DIR, metadata_dir])
FEATURES_DIR = os.path.join(base_dir, 'clip_features')

# Create output directory
os.makedirs(FEATURES_DIR, exist_ok=True)

# Initialize CLIP model and processor
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)
clip_to_label = {item['clip_filename']: item['event'] for item in metadata}

# Process each clip
for clip_file, label in clip_to_label.items():
    clip_path = os.path.join(CLIPS_DIR, clip_file)
    if not os.path.isfile(clip_path):
        print(f"Warning: clip not found: {clip_path}")
        continue

    cap = cv2.VideoCapture(clip_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or SAMPLE_FPS
    frame_interval = max(1, int(round(video_fps / SAMPLE_FPS)))

    frame_feats = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = processor(images=pil_img, return_tensors='pt')
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            feat = outputs.cpu().numpy()  # (1, hidden_dim)
            frame_feats.append(feat)
        frame_idx += 1
    cap.release()

    if not frame_feats:
        print(f"No frames sampled for clip: {clip_file}")
        continue

    # Aggregate features: mean pooling
    features = np.vstack(frame_feats)
    video_feat = features.mean(axis=0)

    # Save per-clip features and label
    base_name = os.path.splitext(clip_file)[0]
    feature_path = os.path.join(FEATURES_DIR, f"{base_name}_feat.npz")
    np.savez_compressed(
        feature_path,
        feature=video_feat,
        label=label
    )
    print(f"Saved features for {clip_file} to {feature_path}")
