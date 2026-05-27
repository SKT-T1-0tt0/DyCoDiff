#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

python scripts/train_content.py \
  --num_workers 0 \
  --gpus 1 \
  --batch_size 1 \
  --data_path datasets/post_URMP/ \
  --save_dir saved_ckpts/your_directory_path \
  --resolution 64 \
  --sequence_length 16 \
  --text_stft_cond \
  --diffusion_steps 4000 \
  --noise_schedule cosine \
  --lr 5e-5 \
  --num_channels 64 \
  --num_res_blocks 2 \
  --class_cond False \
  --log_interval 50 \
  --save_interval 10000 \
  --image_size 64 \
  --learn_sigma True \
  --in_channels 3
