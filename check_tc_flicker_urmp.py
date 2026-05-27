#!/usr/bin/env python3
"""检查 post_URMP 的 10 条视频在 baseline(3_tacm_) 与 mcfl(6_tacm_) 上的逐条 TC_FLICKER。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tc_flicker import tc_flicker_revised

BASELINE_DIR = "results/3_tacm_/fake1_30fps"
MCFL_DIR = "results/6_tacm_/fake1_30fps"

def main():
    names = sorted([f for f in os.listdir(BASELINE_DIR) if f.endswith(".mp4")])
    # 只取两边都有的
    mcfl_names = set(f for f in os.listdir(MCFL_DIR) if f.endswith(".mp4"))
    names = [n for n in names if n in mcfl_names]
    print(f"共 {len(names)} 个视频（两边都有）: {names}\n")

    baseline_vals = []
    mcfl_vals = []
    for n in names:
        b_path = os.path.join(BASELINE_DIR, n)
        m_path = os.path.join(MCFL_DIR, n)
        b_val = tc_flicker_revised(b_path, smooth_k=3)
        m_val = tc_flicker_revised(m_path, smooth_k=3)
        baseline_vals.append(b_val)
        mcfl_vals.append(m_val)
        print(f"  {n}:  baseline={b_val:.6f}   mcfl={m_val:.6f}   Δ={m_val - b_val:+.6f}")

    baseline_vals = np.array(baseline_vals)
    mcfl_vals = np.array(mcfl_vals)
    print()
    print("Baseline (3_tacm_):  mean = %.6f   std = %.6f   median = %.6f" % (baseline_vals.mean(), baseline_vals.std(), np.median(baseline_vals)))
    print("MCFL (6_tacm_):      mean = %.6f   std = %.6f   median = %.6f" % (mcfl_vals.mean(), mcfl_vals.std(), np.median(mcfl_vals)))
    print()
    # 找出 MCFL 里特别大的
    for i, n in enumerate(names):
        if mcfl_vals[i] > 50 or mcfl_vals[i] > 2 * baseline_vals[i]:
            print("  [异常] %s  MCFL=%.2f  Baseline=%.2f" % (n, mcfl_vals[i], baseline_vals[i]))

if __name__ == "__main__":
    main()
