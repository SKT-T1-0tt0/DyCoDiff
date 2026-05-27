import torch
import numpy as np
import os
from collections import defaultdict

BASELINE = "saved_ckpts/temp_baseline"
MCFL = "saved_ckpts/temp_mcfl"
STEP = "010000"  # 10000 步
USE_EMA = True   # 强烈建议 True（评估）

def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点文件不存在: {path}")
    return torch.load(path, map_location="cpu")

def find_available_steps(directory, use_ema=True):
    """查找可用的检查点步数"""
    if use_ema:
        pattern = "ema_0.9999_"
    else:
        pattern = "model"
    
    steps = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if pattern in f and f.endswith(".pt"):
                if use_ema:
                    # ema_0.9999_000000.pt -> 000000
                    step = f.split("_")[-1].replace(".pt", "")
                else:
                    # model000000.pt -> 000000
                    step = f.replace("model", "").replace(".pt", "")
                steps.append(step)
    return sorted(steps)

def stats(state_dict):
    vals = []
    for v in state_dict.values():
        if torch.is_floating_point(v):
            vals.append(v.float().view(-1))
    if not vals:
        return None
    x = torch.cat(vals)
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "abs_mean": x.abs().mean().item(),
        "abs_max": x.abs().max().item(),
    }

def diff(a, b):
    diffs = []
    for k in a:
        if k in b and torch.is_floating_point(a[k]):
            diffs.append((a[k] - b[k]).abs().mean())
    return torch.stack(diffs).mean().item()

# -------- 检查可用步数 --------
baseline_steps = find_available_steps(BASELINE, USE_EMA)
mcfl_steps = find_available_steps(MCFL, USE_EMA)

print(f"Baseline 可用步数: {baseline_steps}")
print(f"MCFL 可用步数: {mcfl_steps}")

# 严格要求 50000 步，不存在则报错
if STEP not in baseline_steps:
    raise FileNotFoundError(
        f"❌ 错误: Baseline 没有 {STEP} 步的检查点\n"
        f"   可用步数: {baseline_steps}\n"
        f"   请确保存在 {STEP} 步的检查点"
    )

if STEP not in mcfl_steps:
    raise FileNotFoundError(
        f"❌ 错误: MCFL 没有 {STEP} 步的检查点\n"
        f"   可用步数: {mcfl_steps}\n"
        f"   请确保存在 {STEP} 步的检查点"
    )

print(f"\n✓ 找到 {STEP} 步的检查点")

# -------- load checkpoints --------
if USE_EMA:
    # EMA 文件名格式: ema_0.9999_000000.pt
    name_0 = "ema_0.9999_000000.pt"
    name_step = f"ema_0.9999_{STEP}.pt"
else:
    # model 文件名格式: model000000.pt
    name_0 = "model000000.pt"
    name_step = f"model{STEP}.pt"

print(f"\n加载检查点...")
print(f"  Baseline 0: {BASELINE}/{name_0}")
print(f"  Baseline {STEP}: {BASELINE}/{name_step}")
print(f"  MCFL 0: {MCFL}/{name_0}")
print(f"  MCFL {STEP}: {MCFL}/{name_step}")

b_0 = load(f"{BASELINE}/{name_0}")
b_5 = load(f"{BASELINE}/{name_step}")

m_0 = load(f"{MCFL}/{name_0}")
m_5 = load(f"{MCFL}/{name_step}")

# -------- stats --------
print(f"\n=== Parameter Statistics @{STEP} ===")
print("Baseline:", stats(b_5))
print("MCFL    :", stats(m_5))

print(f"\n=== Update Magnitude (0 -> {STEP}) ===")
print("Baseline Δ:", diff(b_0, b_5))
print("MCFL     Δ:", diff(m_0, m_5))

print(f"\n=== Baseline vs MCFL Difference @{STEP} ===")
print("EMA diff :", diff(b_5, m_5))
