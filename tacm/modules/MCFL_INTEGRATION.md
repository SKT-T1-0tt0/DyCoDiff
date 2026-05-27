# MCFL 集成指南

## 模块说明

MCFL (Multi-modal Condition Fusion Layer) 是一个多模态条件融合模块，用于融合文本、图像和音频条件。

## 模块位置

- **文件**: `tacm/modules/mcfl.py`
- **类名**: `MCFL`
- **已注册**: 在 `tacm/modules/__init__.py` 中已导入

## 使用方法

### 基本用法

```python
from tacm import MCFL
import torch

# 初始化模块
mcfl = MCFL(
    embed_dim=768,  # 必须与输入维度 D 一致
    num_heads=8,
    dropout=0.1
)

# 输入条件（必须是 [B, D] 格式）
c_text = torch.randn(2, 768)   # [B, D] - 文本条件
c_image = torch.randn(2, 768) # [B, D] - 图像条件
c_audio = torch.randn(2, 768) # [B, D] - 音频条件

# 融合
c_fused = mcfl(c_text, c_image, c_audio)  # [B, D]
```

## 集成到训练流程

### 插入位置

MCFL 应该插入在 encoder 输出之后、diffusion 模型之前。

**当前代码结构** (`diffusion/tacm_train_temp_util.py` 第 214-260 行):

```python
# ----get text----
c_t = sample['text'].squeeze(1).to(dist_util.dev())  # [B, N, D]

# ----get image----
image_cat = ...  # [B, 1, D]

# ----get audio----
c_temp = ...  # [B, T, D]

# 当前方式：直接 concat
c_ti = th.concat((c_t, image_cat), dim=1)  # [B, N+1, D]
c_at = th.concat((c_temp, c_t), 1)        # [B, T+N, D]

# 传递给 diffusion
self.run_step(batch, cond, c_ti, c_at)
```

### 集成方案

由于当前 condition 是序列格式 `[B, N, D]`，而 MCFL 需要 `[B, D]`，需要先进行池化操作：

```python
# 在 TrainLoop.__init__ 中添加
from tacm import MCFL

class TrainLoop():
    def __init__(self, ..., use_mcfl=False, mcfl_embed_dim=768):
        # ... 现有代码 ...
        
        # 初始化 MCFL（可选）
        self.use_mcfl = use_mcfl
        if self.use_mcfl:
            self.mcfl = MCFL(
                embed_dim=mcfl_embed_dim,
                num_heads=8,
                dropout=0.1
            ).to(dist_util.dev())
        else:
            self.mcfl = None

    def run_loop(self):
        # ... 现有代码 ...
        
        # ----get text----
        c_t = sample['text'].squeeze(1).to(dist_util.dev())  # [B, N, D]
        
        # ----get image----
        image_cat = ...  # [B, 1, D]
        
        # ----get audio----
        c_temp = ...  # [B, T, D]
        
        if self.use_mcfl and self.mcfl is not None:
            # 从序列中提取单个 embedding（平均池化）
            c_text_single = c_t.mean(dim=1)  # [B, D] - 文本序列的平均
            c_image_single = image_cat.squeeze(1)  # [B, D] - 图像（已经是单个）
            c_audio_single = c_temp.mean(dim=1)  # [B, D] - 音频序列的平均
            
            # 使用 MCFL 融合
            c_fused = self.mcfl(c_text_single, c_image_single, c_audio_single)  # [B, D]
            
            # 将融合后的条件扩展回序列格式（用于兼容现有接口）
            # 方式1: 重复融合结果作为新的 text token
            c_t_fused = c_fused.unsqueeze(1).repeat(1, c_t.shape[1], 1)  # [B, N, D]
            c_ti = th.concat((c_t_fused, image_cat), dim=1)  # [B, N+1, D]
            
            # 方式2: 将融合结果与 audio 组合
            c_fused_audio = c_fused.unsqueeze(1).repeat(1, c_temp.shape[1], 1)  # [B, T, D]
            c_at = th.concat((c_temp, c_fused_audio), dim=1)  # [B, T+T, D]
        else:
            # 原有逻辑（完全不变）
            c_ti = th.concat((c_t, image_cat), dim=1)
            c_at = th.concat((c_temp, c_t), 1)
        
        c_at = c_at.to(dist_util.dev())
        self.run_step(batch, cond, c_ti, c_at)
```

### 注意事项

1. **维度匹配**: 确保 `mcfl_embed_dim` 与 condition 的维度 `D` 一致（通常是 768）

2. **序列处理**: 
   - Text 和 Audio 是序列，需要池化（平均或取第一个 token）
   - Image 已经是单个 embedding，直接使用

3. **向后兼容**: 
   - 当 `use_mcfl=False` 时，代码行为与当前版本完全一致
   - 所有修改都是可选的

4. **训练参数**: 
   - MCFL 的参数会通过 optimizer 自动更新
   - 建议使用较小的学习率（如 1e-5）初始化 MCFL

## 集成到推理流程

在 `scripts/sample_motion_optim.py` 中也可以类似集成：

```python
# 在 main() 函数中
if args.use_mcfl:
    from tacm import MCFL
    mcfl = MCFL(embed_dim=768, num_heads=8, dropout=0.1).to(dist_util.dev())
    
    # 在 condition 构建后
    c_text_single = c_t.mean(dim=1)  # [B, D]
    c_image_single = image_cat.squeeze(1)  # [B, D]
    c_audio_single = c_temp.mean(dim=1)  # [B, D]
    
    c_fused = mcfl(c_text_single, c_image_single, c_audio_single)
    # ... 后续处理 ...
```

## 参数说明

- `embed_dim`: 嵌入维度，必须与输入条件维度一致（默认 768）
- `num_heads`: 注意力头数（默认 8）
- `dropout`: Dropout 率（默认 0.1）

## 模块内部逻辑

1. **Step 1: Joint Self-Attention**
   - 将 text、image、audio 三个条件 stack 成 `[B, 3, D]`
   - 执行 multi-head self-attention
   - 使用 residual + layer norm

2. **Step 2: Text-Centric Cross-Attention**
   - 将 text token 作为 query
   - 将 image + audio tokens 作为 key/value
   - 执行 cross-attention
   - 使用 residual + layer norm

3. **Step 3: 输出**
   - 返回融合后的 text token `[B, D]`

## 测试示例

```python
import torch
from tacm import MCFL

# 创建模块
mcfl = MCFL(embed_dim=768, num_heads=8)

# 创建测试数据
B, D = 4, 768
c_text = torch.randn(B, D)
c_image = torch.randn(B, D)
c_audio = torch.randn(B, D)

# 前向传播
c_fused = mcfl(c_text, c_image, c_audio)

# 验证输出形状
assert c_fused.shape == (B, D), f"Expected ({B}, {D}), got {c_fused.shape}"
print(f"Input shape: text={c_text.shape}, image={c_image.shape}, audio={c_audio.shape}")
print(f"Output shape: {c_fused.shape}")
```
