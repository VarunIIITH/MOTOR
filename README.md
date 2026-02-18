# MOTOR — Multimodal twO-wheeler Traffic behaviOuR Dataset

## Overview

MOTOR is a multimodal dataset capturing rider behavior in chaotic, lane-free traffic with:
- **RGB egocentric video** (helmet-mounted front camera)
- **Gaze data** (Pupil Invisible / Aria eye trackers)
- **Telemetry** (speed + lean angle from GoPro GPS/IMU)
- **11 maneuver classes** with legal/illegal annotations

This repository contains the model training pipeline and data processing tools.

## Setup

```bash
conda create -n motor python=3.10
conda activate motor
pip install -r requirements.txt
```

## Training

All experiments are controlled via YAML config files:

```bash
# Swin3D-T with RGB + Gaze + Telemetry fusion
python train.py --config configs/swin3dt_gaze_tele.yaml

# ResNet3D-18 base (RGB only)
python train.py --config configs/r3d18_base.yaml

# MViTv2-S for legality classification
python train.py --config configs/legality/mvitv2_base_legal.yaml

# Cross-attention fusion (R3D-18 backbone)
python train.py --config configs/r3d18_cross_attn.yaml
```

### Available Backbones
| Key | Architecture |
|-----|-------------|
| `swin3d_t` | Swin Transformer 3D-Tiny |
| `r3d_18` | ResNet3D-18 |
| `mvit_v2_s` | MViTv2-Small |
| `s3d` | S3D |

### Fusion Strategies
| Key | Modalities | Description |
|-----|-----------|-------------|
| `single` | RGB | Single-stream backbone |
| `two_stream` | RGB + Gaze | Late fusion (concatenate features) |
| `rgb_tele` | RGB + Telemetry | Late fusion with telemetry head |
| `three_stream` | RGB + Gaze + Tele | Three-stream late fusion |
| `cross_attn` | RGB + Gaze | Cross-attention (R3D-18 only) |

## Config Format

```yaml
model:
  backbone: swin3d_t
  fusion: three_stream
  modalities: [rgb, gaze, telemetry]
  head_hidden: [512, 128]
  dropout: 0.4
  tele_len: 256
  tele_dim: 32

training:
  epochs: 50
  batch_size: 8
  lr_head: 1e-4
  lr_tail: 3e-5
  tail_blocks_k: 1
  focal_gamma: 2.0

data:
  manifest_csv: data/train_val_manifest.csv
  task: behavior          # behavior | legality
  frames_per_clip: 16
  img_size: 224

output:
  plots_dir: outputs/plots
  checkpoints_dir: outputs/checkpoints
```

## Data Pipeline

The `tools/data_pipeline/` directory contains scripts for raw recording processing:

| Script | Description |
|--------|-------------|
| `combine_gopro.py` | Merge split GoPro files + interpolate telemetry |
| `aria_vrs_to_mp4.py` | Convert Aria VRS to MP4 |
| `undistort_aria.py` | Undistort Aria fisheye video |
| `undistort_pupil.py` | Undistort Pupil Invisible video |
| `time_sync.py` | QR-based multi-video synchronization (3 or 4 streams) |
| `metadata_process.py` | Sync metadata across time-aligned views |
| `video_split.py` | Split synced videos into 1-minute clips |
| `detect_track.py` | YOLOv8 detection + BotSort tracking + depth estimation |

The `tools/` directory contains scripts for dataset preparation:

| Script | Description |
|--------|-------------|
| `preprocess_data.py` | Extract annotated behavior clips from video |
| `preprocess_data_gaze.py` | Extract annotated gaze clips |
| `create_gaze.py` | Generate gaze patch/overlay videos |
| `gaze_transfer.py` | Homography-based gaze transfer across views |
| `feature_extraction_clip.py` | Extract CLIP features from clips |
| `mlp_classifier.py` | Train MLP on pre-extracted features |
| `traffic_stats.py` | Traffic detection statistics |
| `visualise_data.py` | Dataset distribution visualization |

## Project Structure

```
ICRA_MOTOR/
├── train.py                    # Unified training entry point
├── configs/                    # YAML experiment configs
│   ├── swin3dt_base.yaml
│   ├── r3d18_gaze_tele.yaml
│   ├── r3d18_cross_attn.yaml
│   ├── legality/               # Legality task configs
│   └── ...
├── models/
│   ├── backbones.py            # S3D, R3D18, MViTv2, Swin3D encoders
│   ├── heads.py                # MLP classification + telemetry heads
│   └── fusion.py               # All fusion architectures + factory
├── datasets/
│   ├── transforms.py           # Per-backbone transforms
│   ├── clip_dataset.py         # Single-stream (RGB only)
│   ├── rgb_gaze_dataset.py     # Two-stream (RGB + Gaze)
│   ├── rgb_tele_dataset.py     # RGB + Telemetry
│   └── rgb_gaze_tele_dataset.py # Three-stream
├── utils/
│   ├── losses.py               # Focal cross-entropy
│   ├── metrics.py              # F1, confusion matrix, top-k
│   ├── plotting.py             # Training curves, confusion matrices
│   ├── training.py             # Epoch runner, unfreezing, optimizer
│   └── telemetry.py            # Telemetry vector parsing
├── tools/                      # Data processing tools
│   └── data_pipeline/          # Raw recording processing
└── scripts/
    └── train.sh                # SLURM batch script template
```

## Manifest CSV Format

The training manifest is a CSV file with one row per video clip. Which columns are required depends on the fusion type and task.

### Required Columns (all configs)

| Column | Type | Description |
|--------|------|-------------|
| `path_video` | `str` | Absolute or relative path to the RGB video clip |
| `label` | `int` | Behavior class label (0–10, see table below) |
| `split` | `str` | `train` or `val` |

### Optional Columns (depending on config)

| Column | Needed for | Type | Description |
|--------|-----------|------|-------------|
| `path_gaze` | `two_stream`, `three_stream`, `cross_attn` | `str` | Path to the corresponding gaze video clip |
| `label_legality` | legality task configs | `int` | Legality class label (0 = legal, 1 = illegal) |
| `telemetry_vec_64` | `rgb_tele`, `three_stream` | `str` (JSON list) | Telemetry feature vector as a JSON list of 64 floats |
| `speed_vec_32` | fallback for telemetry | `str` (JSON list) | Speed-only vector (32 floats), concatenated with `gyro_vec_32` |
| `gyro_vec_32` | fallback for telemetry | `str` (JSON list) | Gyro/lean-angle vector (32 floats), concatenated with `speed_vec_32` |

> **Note:** For telemetry, the code first looks for `telemetry_vec_64`. If absent, it falls back to concatenating `speed_vec_32 + gyro_vec_32`. Vectors are zero-padded or truncated to `tele_len` (default 256).

### Behavior Class Labels

| Label | Class |
|-------|-------|
| 0 | Overtake |
| 1 | Lane change |
| 2 | Going straight |
| 3 | Left turn |
| 4 | Obstruction avoidance |
| 5 | Weaving through traffic |
| 6 | Distracted driving |
| 7 | Stop |
| 8 | Right turn |
| 9 | Violation |
| 10 | U-turn |

### Example CSV

```csv
path_video,path_gaze,label,label_legality,split,telemetry_vec_64
data/clips/01_clip_001.mp4,data/gaze/01_clip_001.mp4,3,0,train,"[0.12, 0.34, ..., 0.56]"
data/clips/01_clip_002.mp4,data/gaze/01_clip_002.mp4,0,1,train,"[0.78, 0.91, ..., 0.23]"
data/clips/02_clip_001.mp4,data/gaze/02_clip_001.mp4,2,0,val,"[0.45, 0.67, ..., 0.89]"
```

