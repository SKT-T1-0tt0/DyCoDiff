#!/usr/bin/env python3
"""
读取多个 audio_beats_stats 输出的 JSON，汇总 BEATs pooled_norm，
输出统一标定：绝对区间 [low, high] 与 z-score 默认 (mu, sigma, z_low, z_high)。

用法:
  python scripts/unify_gate_calibration.py stats_drums.json stats_urmp.json stats_landscape.json
"""

import argparse
import json
import sys


def load_pooled(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    beats = d.get("beats")
    if not beats or "pooled_norm" not in beats:
        return None
    return beats["pooled_norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="Paths to stats JSON (with beats.pooled_norm)")
    ap.add_argument("--margin", type=float, default=0.2, help="Extra margin on unified [low, high]")
    args = ap.parse_args()

    all_p5, all_p95, all_means, all_stds, all_mins, all_maxs = [], [], [], [], [], []
    for path in args.json_files:
        p = load_pooled(path)
        if p is None:
            print(f"  skip (no beats.pooled_norm): {path}", file=sys.stderr)
            continue
        all_p5.append(p["p5"])
        all_p95.append(p["p95"])
        all_means.append(p["mean"])
        all_stds.append(p["std"])
        all_mins.append(p["min"])
        all_maxs.append(p["max"])
        print(f"  {path}: mean={p['mean']:.4f} std={p['std']:.4f} p5={p['p5']:.4f} p95={p['p95']:.4f}")

    if not all_p5:
        print("No valid stats. Exit.")
        return

    low = min(all_p5) - args.margin
    high = max(all_p95) + args.margin
    mu = sum(all_means) / len(all_means)
    sigma = (sum(all_stds) / len(all_stds)) * 1.2  # 略放大以覆盖跨集方差

    print("\n--- 统一标定（同时覆盖所有数据集）---")
    print(f"  绝对区间: mcfl_gate_norm_low={low:.2f}, mcfl_gate_norm_high={high:.2f}")
    print(f"  Z-score 默认: mcfl_gate_norm_mu={mu:.2f}, mcfl_gate_norm_sigma={sigma:.2f}")
    print(f"  建议 z 范围: mcfl_gate_z_low=-1.5, mcfl_gate_z_high=1.5")
    print("\n  训练/推理可直接使用上述参数，无需 per-dataset 手调。")


if __name__ == "__main__":
    main()
