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

## Data preparation

Download the dataset from [Hugging Face](https://huggingface.co/datasets/varunpaturkar/MOTOR) and build the training manifest:

```bash
python tools/make_manifest.py --data_root /path/to/MOTOR --task behavior --out data/train_val_manifest.csv
```

The manifest uses the dataset's rider-disjoint train/val split (`Split` column of `annotations.csv`). Use `--task legality` for the legality task, and `--gaze_dir` to point to gaze-centred clips for the gaze stream.

**Telemetry input.** As in the paper, the telemetry stream uses two signals from each clip's telemetry CSV: speed (`speed_2d_mps`) and lean, taken from the front-camera gyroscope axis the motorcycle leans around (`gyro_z`, roll rate in rad/s). Each is averaged into 64 equal-time bins and the two are concatenated into a 128-dimensional vector (`utils/telemetry.py`), which a three-layer MLP projects to the telemetry embedding.

## Training

```bash
python train.py --config configs/swin3dt_gaze_tele.yaml
```

The shipped config trains a three-stream fusion model (RGB + gaze + telemetry) on a Swin3D-T backbone — the configuration that performed best across our benchmarks. Edit the YAML or `train.py` to adapt to other backbones, modalities, or the legality task.

## Citation

```bibtex
@INPROCEEDINGS{11696157,
  author={Paturkar, Varun A. and Gangisetty, Shankar and Jawahar, C.V.},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  title={MOTOR: A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding},
  year={2026},
  volume={},
  number={},
  pages={5406-5413},
  doi={10.1109/ICRA57385.2026.11696157}}
```
