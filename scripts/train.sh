#!/bin/bash

#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=motor_train
#SBATCH --output=outputs/logs/%x_%j.txt

conda activate motor

CONFIG=${1:-configs/swin3dt_base.yaml}

echo "Config: $CONFIG"
echo "Start: $(date)"

python train.py --config "$CONFIG"

echo "Done: $(date)"
