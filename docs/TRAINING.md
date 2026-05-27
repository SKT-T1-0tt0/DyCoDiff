# Training Guide

This guide records the training commands used for DyCoDiff. Run all commands from the repository root.

## Environment

Use the training environment exported in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate dycodiff
```

## Stage 1: Content Model

If released TIA content checkpoints are unavailable, train the content model first:

```bash
python -m scripts.train_content \
  --num_workers 8 \
  --gpus 1 \
  --batch_size 1 \
  --data_path datasets/post_audioset_drums \
  --save_dir saved_ckpts/audioset_content \
  --resolution 64 \
  --sequence_length 16 \
  --text_stft_cond \
  --diffusion_steps 4000 \
  --noise_schedule cosine \
  --lr 5e-5 \
  --num_channels 64 \
  --num_res_blocks 2 \
  --class_cond False \
  --image_size 64 \
  --learn_sigma True \
  --in_channels 3
```

## Stage 2: DyCoDiff Temporal Training

The paper configuration is provided as a script:

```bash
bash scripts/train_paper_collab_dynamic_cosine_v7_unified_adaptive.sh
```

By default, the script trains on AudioSet-Drums-VAT. Override paths through environment variables:

```bash
DATA_PATH=datasets/post_landscape \
MODEL_PATH=saved_ckpts/Landscape-VAT_tia.pt \
SAVE_DIR=saved_ckpts/paper_collab_dynamic_cosine_v7_landscape \
bash scripts/train_paper_collab_dynamic_cosine_v7_unified_adaptive.sh
```

## Full Paper Command

```bash
python -m scripts.train_temp \
  --num_workers 8 \
  --gpus 1 \
  --batch_size 1 \
  --data_path datasets/post_audioset_drums \
  --model_path saved_ckpts/AudioSet-Drums-VAT_tia.pt \
  --save_dir saved_ckpts/paper_collab_dynamic_cosine_v7_audioset_drums \
  --resolution 64 \
  --image_size 64 \
  --sequence_length 16 \
  --text_stft_cond \
  --audio_emb_model beats \
  --diffusion_steps 4000 \
  --noise_schedule cosine \
  --num_channels 64 \
  --num_res_blocks 2 \
  --class_cond False \
  --learn_sigma True \
  --in_channels 3 \
  --lr 5e-5 \
  --log_interval 50 \
  --save_interval 10000 \
  --lr_anneal_steps 20000 \
  --audio_response compand \
  --audio_random_gain True \
  --audio_gain_low 0.5 \
  --audio_gain_high 2.0 \
  --modality_dropout_prob 0.2 \
  --paper_collab_mode True \
  --paper_collab_no_mcfl False \
  --paper_collab_hybrid_legacy True \
  --use_mcfl True \
  --mcfl_embed_dim 768 \
  --mcfl_pooling_mode mean \
  --mcfl_gate_lambda 0.1 \
  --mcfl_conservative False \
  --use_baseline_imitation True \
  --mcfl_gate_use_zscore True \
  --mcfl_gate_norm_mu 8.4 \
  --mcfl_gate_norm_sigma 0.5 \
  --mcfl_gate_z_low -1.5 \
  --mcfl_gate_z_high 1.5 \
  --mcfl_gate_lambda_max 0.2 \
  --mcfl_gate_use_av_conf False \
  --mcfl_collab_weight 0.001 \
  --learned_gate_enable False \
  --omni_encoder_ckpt saved_ckpts/omni_encoder.pt \
  --collab_metric dynamic_cosine \
  --collab_dyn_stride 1 \
  --collab_dyn_gate_k 2.0 \
  --collab_dyn_beta 2.0 \
  --collab_dyn_rank_mode hybrid \
  --collab_dyn_rank_low_q 0.35 \
  --collab_dyn_rank_high_q 0.9 \
  --collab_dyn_rank_sharpness 6.0 \
  --collab_dyn_hist_size 256 \
  --collab_av_score_mix_av_ratio 0.7 \
  --collab_av_score_mix_av_conf 0.3 \
  --collab_av_conf_beta 0.5 \
  --collab_av_conf_sim_low 0.0 \
  --collab_av_conf_sim_high 0.3 \
  --collab_audio_ema_alpha 0.9 \
  --collab_vt_weight 0.0010 \
  --collab_va_weight 0.00005 \
  --collab_vi_weight 0.0005 \
  --collab_start_step 10000 \
  --collab_warmup_steps 5000 \
  --collab_imit_scale 0.0 \
  --collab_obi_scale 0.0 \
  --collab_obi_start_step 0 \
  --collab_obi_warmup_steps 10000 \
  --collab_obi_layer_name temporal_last \
  --collab_obi_min_weight 0.05 \
  --collab_obi_min_scale 0.2
```

## Ablation Settings

- **Baseline**: disable MCFL and collaborative losses.
- **Only-MCFL**: enable `--use_mcfl True` but disable the full dynamic collaborative loss configuration.
- **DyCoDiff**: use the paper script above with MCFL, dynamic cosine collaboration, and online baseline attention imitation.
