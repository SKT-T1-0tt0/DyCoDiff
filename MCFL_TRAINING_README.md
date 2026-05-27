# MCFL 训练模式与 Online Baseline 模仿

本文档说明 motion 训练中 **MCFL（Multi-modal Condition Fusion Layer）** 的开关与模式，以及后续 **Online baseline 模仿** 的接入方式。

---

## 1. 训练模式概览

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Baseline** | `--use_mcfl False`，不做多模态融合 | 纯 baseline 复现 / 对比 |
| **MCFL 保守版** | `--use_mcfl True`，`--mcfl_conservative True`（默认） | 当前推荐：Alpha + 8k 冻结 + lambda_temp curriculum |
| **MCFL 非保守版** | `--use_mcfl True`，`--mcfl_conservative False` | 准备接 Online baseline 模仿时使用 |
| **Online baseline 模仿** | `--use_baseline_imitation True`（占位，待实现） | 用 baseline 输出模仿平衡 FVD/FID/FFC 与 AV_ALIGN |

---

## 2. 开关说明

### 2.1 MCFL 总开关

- **`--use_mcfl`**（默认 `False`）
  - `False`：Baseline，条件只用 concat，无 MCFL。
  - `True`：启用 MCFL，c_at 使用 gated residual 融合（text+image+audio）。

### 2.2 保守版开关（保存当前 MCFL 行为）

- **`--mcfl_conservative`**（默认 `True`）
  - **`True`（保守版）**：
    - Alpha curriculum：0–4k 从 0.2 线性到 0.2，4k–8k 到 0.7，8k–10k 为 0.5，10k+ 固定 0.2。
    - 8k 步后冻结 MCFL 参数。
    - 若 `--lambda_temp > 0`：启用 lambda_temp 的 0→0.02 curriculum（0–10k 步）。
  - **`False`（非保守版）**：
    - 不做 alpha 调制，不冻结 MCFL，不做 lambda_temp curriculum。
    - 用于后续接 **Online baseline 模仿**（放开 MCFL，用模仿约束 FVD/FID/FFC）。

### 2.3 Online baseline 模仿（占位）

- **`--use_baseline_imitation`**（默认 `False`）
  - 当前为占位参数，逻辑尚未实现。
  - 计划：同 batch 下用冻结 baseline（相同 x_t、c_ti，原始 c_at）的预测做 target，对 MCFL 模型加 MSE 模仿 loss，λ 可配。
  - 实现时在 `diffusion/tacm_train_temp_util.py` 的 `forward_backward` 中接好 baseline 前向与 loss 即可。

### 2.4 其他 MCFL 相关参数

- **`--mcfl_gate_lambda`**（默认 `0.2`）：Gated residual 门控，越小 MCFL 影响越弱（如保守实验可用 `0.1`）。
- **`--mcfl_pooling_mode`**：`mean` 或 `attention`。
- **`--lambda_temp`**：时序平滑权；仅在 **保守版** 下会按 curriculum 使用，非保守版下不启用 curriculum。

---

## 3. 推荐命令示例

### 3.1 Baseline（无 MCFL）

```bash
python -m scripts.train_temp \
  --num_workers 8 --gpus 1 --batch_size 1 \
  --data_path datasets/post_URMP/ \
  --model_path saved_ckpts/URMP-VAT_tia.pt \
  --save_dir saved_ckpts/temp_baseline \
  --resolution 64 --sequence_length 16 --text_stft_cond --audio_emb_model beats \
  --diffusion_steps 4000 --noise_schedule cosine \
  --num_channels 64 --num_res_blocks 2 --class_cond False --image_size 64 \
  --learn_sigma True --in_channels 3 --lr 5e-5 --log_interval 50 --save_interval 10000
```

### 3.2 MCFL 保守版（当前默认行为，已“保存”为开关）

```bash
python -m scripts.train_temp \
  --num_workers 8 --gpus 1 --batch_size 1 \
  --data_path datasets/post_URMP/ \
  --model_path saved_ckpts/URMP-VAT_tia.pt \
  --save_dir saved_ckpts/mcfl_curriculum_20k_conservative \
  --resolution 64 --sequence_length 16 --text_stft_cond --audio_emb_model beats \
  --diffusion_steps 4000 --noise_schedule cosine \
  --num_channels 64 --num_res_blocks 2 --class_cond False --image_size 64 \
  --learn_sigma True --in_channels 3 --lr 5e-5 --log_interval 50 --save_interval 10000 \
  --use_mcfl True --mcfl_embed_dim 768 --mcfl_pooling_mode mean --mcfl_gate_lambda 0.1 \
  --lr_anneal_steps 20000 --lambda_temp 0.01
  # mcfl_conservative 默认为 True，无需显式写
```

### 3.3 MCFL 非保守版（为 Online baseline 模仿预留）

```bash
python -m scripts.train_temp \
  ... \
  --use_mcfl True \
  --mcfl_conservative False \
  --mcfl_gate_lambda 0.3
  # 不加 lambda_temp 或设为 0；use_baseline_imitation 实现后加 --use_baseline_imitation True
```

---

## 4. 代码位置（便于接 Online baseline 模仿）

- **训练循环与 MCFL 条件**：`diffusion/tacm_train_temp_util.py`
  - `run_loop`：alpha 与 MCFL 冻结 受 `mcfl_conservative` 控制。
  - `forward_backward`：diffusion loss 计算；lambda_temp 受 `mcfl_conservative` 控制；可在此处加 baseline 前向与 `loss_imitate`。
- **条件构建（c_ti / c_at）**：`diffusion/condition_builder.py`
- **入口与参数**：`scripts/train_temp.py`（`mcfl_conservative`、`use_baseline_imitation` 已接入 TrainLoop）

---

## 5. 小结

- **当前“保守版”MCFL** 已通过 **`--mcfl_conservative True`**（默认）完整保留，无需改代码即可复现。
- 要做 **Online baseline 模仿** 时：使用 **`--mcfl_conservative False`** 放开 MCFL，再在 `forward_backward` 中实现 `use_baseline_imitation` 的 loss 并挂上 `--use_baseline_imitation True` 即可。
