# MOTOR — A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding

Code release accompanying the ICRA 2026 paper *MOTOR: A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding*.

- Project page: https://varuniiith.github.io/MOTOR-Dataset/
- Dataset: https://huggingface.co/datasets/varunpaturkar/MOTOR

## Abstract

Two-wheelers account for a disproportionately high share of road fatalities in the Global South. Research on two-wheeler rider behavior, however, lags far behind four-wheelers, where multimodal datasets have driven major advances in Advanced Driver Assistance Systems (ADAS). To address this gap, we present the MOtorized TwO-wheeler Rider (MOTOR) dataset, the first large-scale, multi-view, multimodal resource dedicated to two-wheelers in dense, unstructured traffic. MOTOR comprises 1,629 annotated sequences (25+ hours of video data) collected from 16 riders and integrates synchronized front, rear, and helmet videos, rider eye-gaze from wearable trackers, on-road audio, and telemetry (GPS, accelerometer, gyroscope). Rich annotations capture traffic context, rider state, 12 riding maneuvers spanning conventional and unconventional behaviors, and legality labels (Legal, Illegal, Unspecified). We benchmark rider behavior recognition and maneuver legality classification using state-of-the-art video action recognition backbones (CNN and Transformer-based), extended with multimodal fusion, and find that combining RGB, gaze, and telemetry consistently yields the best performance. MOTOR thus provides a unique foundation for advancing safety-critical understanding of two-wheeler riding.

## Setup

```bash
conda create -n motor python=3.10 -y
conda activate motor
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/swin3dt_gaze_tele.yaml
```

The shipped config trains a three-stream fusion model (RGB + gaze + telemetry) on a Swin3D-T backbone — the configuration that performed best across our benchmarks. Edit the YAML or `train.py` to adapt to other backbones, modalities, or the legality task.