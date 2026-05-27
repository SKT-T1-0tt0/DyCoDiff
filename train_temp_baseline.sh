#!/usr/bin/env bash
# Baseline 训练：train_temp.py，不启用 MCFL（不加 --use_mcfl）
# 需要先有 content 模型：model_path 指向 train_content 产出的 ckpt（如 ema_0.9999_xxxxx.pt）

set -euo pipefail
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 请把 model_path 改成你的 content 模型路径（train_content.py 训练得到的 ckpt）
MODEL_PATH="${MODEL_PATH:-saved_ckpts/your_content_model.pt}"

python scripts/train_temp.py \
  --num_workers 8 \
  --batch_size 1 \
  --data_path datasets/post_URMP/ \
  --model_path "$MODEL_PATH" \
  --save_dir saved_ckpts/your_directory_path \
  --resolution 64 \
  --sequence_length 16 \
  --text_stft_cond \
  --audio_emb_model beats \
  --diffusion_steps 4000 \
  --noise_schedule cosine \
  --num_channels 64 \
  --num_res_blocks 2 \
  --class_cond False \
  --image_size 64 \
  --learn_sigma True \
  --in_channels 3 \
  --lr 5e-5 \
  --log_interval 50 \
  --save_interval 5000 \
  --gpus 1
