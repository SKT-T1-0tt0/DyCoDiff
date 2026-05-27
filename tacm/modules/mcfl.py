# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
Multi-modal Condition Fusion Layer (MCFL).

MCFL 通过两阶段注意力机制融合 text / image / audio 条件：
  - AttnPool: 可学习注意力池化（可选）
  - MultiHeadSelfAttention: 多模态对称交互
  - MultiHeadCrossAttention: 文本中心的跨模态融合
  - MCFL: 两阶段注意力融合主模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """可学习注意力池化：自动聚焦重要 token（如文本中的动作词、音频中的强节奏段）。

    Args:
        dim: 嵌入维度 D。

    Input:
        x: [B, N, D] token 序列。

    Output:
        pooled: [B, D] 注意力加权池化向量。
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        """attn = softmax(x @ q / sqrt(D)); out = sum(attn * x)."""
        attn = torch.softmax((x @ self.query) / (x.shape[-1] ** 0.5), dim=1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力。MCFL 中用于三模态 token 的对称交互。"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x [B, N, D] -> out [B, N, D]."""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MultiHeadCrossAttention(nn.Module):
    """多头跨注意力。MCFL 中用于 text 作为 query、image+audio 作为 key/value。"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """query [B, Nq, D], key_value [B, Nkv, D] -> out [B, Nq, D]."""
        B, Nq, D = query.shape
        B_kv, Nkv, D_kv = key_value.shape
        assert B == B_kv and D == D_kv, "Batch size and embed_dim must match"
        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(key_value).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.proj(out)


class MCFL(nn.Module):
    """Multi-modal Condition Fusion Layer：两阶段注意力融合 text / image / audio。

    1. Joint Self-Attention：三模态对称交互
    2. Text-Centric Cross-Attention：text 为 query，从 image+audio 抽取信息
    3. Optional Collaborative Loss：对 self-attention 后的 image/audio token 做一致性约束
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        c_text: torch.Tensor = None,
        c_image: torch.Tensor = None,
        c_audio: torch.Tensor = None,
        return_collab_loss: bool = False,
    ):
        """Self-Attn(stack) -> Cross-Attn(text q, image+audio kv) -> c_fused [B, D].

        return_collab_loss=True 时，返回 (c_fused, collab_loss)，其中
        collab_loss = 1 - cos(z_img_post, z_aud_post)，促进 image/audio token 对齐。
        """
        assert c_text is not None or c_image is not None or c_audio is not None

        if c_text is not None:
            B, D = c_text.shape[0], c_text.shape[1]
            device = c_text.device
            dtype = c_text.dtype
        elif c_image is not None:
            B, D = c_image.shape[0], c_image.shape[1]
            device = c_image.device
            dtype = c_image.dtype
        else:
            B, D = c_audio.shape[0], c_audio.shape[1]
            device = c_audio.device
            dtype = c_audio.dtype

        assert D == self.embed_dim

        c_text = c_text if c_text is not None else torch.zeros(B, D, device=device, dtype=dtype)
        c_image = c_image if c_image is not None else torch.zeros(B, D, device=device, dtype=dtype)
        c_audio = c_audio if c_audio is not None else torch.zeros(B, D, device=device, dtype=dtype)
        assert c_text.shape == (B, D) and c_image.shape == (B, D) and c_audio.shape == (B, D)

        # Step 1: Joint Self-Attention
        tokens = torch.stack([c_text, c_image, c_audio], dim=1)  # [B, 3, D]
        tokens = self.self_attn_norm(tokens + self.self_attn(tokens))

        collab_loss = None
        if return_collab_loss:
            c_img_post = tokens[:, 1, :]  # [B, D]
            c_aud_post = tokens[:, 2, :]  # [B, D]
            collab_loss = (1.0 - F.cosine_similarity(c_img_post, c_aud_post, dim=-1)).mean()

        # Step 2: Text-Centric Cross-Attention
        t = tokens[:, 0:1, :]
        ia = tokens[:, 1:, :]
        t = self.cross_attn_norm(t + self.cross_attn(t, ia))

        c_fused = t.squeeze(1)  # [B, D]

        if return_collab_loss:
            return c_fused, collab_loss
        return c_fused
