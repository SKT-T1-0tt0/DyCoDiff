#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_URMP 三路对比评估：3=Baseline, 6=MCFL, 12=MCFL 2.0

用法:
  python3 eval_urmp_3_6_12.py --metrics fvd fid ffc clip av_align tc_flicker --output evaluation_report_urmp_3_6_12.txt --no_cache
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_all import EvaluationRunner

# 3=Baseline, 6=MCFL, 12=MCFL 2.0
REAL_DIR = "results/3_tacm_/real"
BASELINE_DIR = "results/3_tacm_/fake1_30fps"   # run 3: baseline
MCFL_DIR = "results/6_tacm_/fake1_30fps"       # run 6: MCFL
MCFL2_DIR = "results/12_tacm_/fake1_30fps"     # run 12: MCFL 2.0


def write_report(all_results, output_file, metrics):
    """写入三路对比报告（Baseline | MCFL | MCFL 2.0）"""
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "config": {
            "real": REAL_DIR,
            "baseline_3": BASELINE_DIR,
            "mcfl_6": MCFL_DIR,
            "mcfl2_12": MCFL2_DIR,
        },
        "results": all_results,
    }
    json_file = output_file.replace(".txt", ".json") if output_file.endswith(".txt") else output_file + ".json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("post_URMP 三路对比：3=Baseline, 6=MCFL, 12=MCFL 2.0\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"  real:     {REAL_DIR}\n")
        f.write(f"  baseline (3):  {BASELINE_DIR}\n")
        f.write(f"  mcfl (6):      {MCFL_DIR}\n")
        f.write(f"  mcfl2 (12):    {MCFL2_DIR}\n")
        f.write("=" * 80 + "\n\n")

        cols = ["Metric", "Baseline(3)", "MCFL(6)", "MCFL2(12)", "Δ6", "Δ12"]
        table_data = []
        for metric_name in metrics:
            metric_name = metric_name.lower()
            if metric_name not in all_results:
                continue
            mr = all_results[metric_name]
            if "error" in mr:
                table_data.append({
                    "Metric": metric_name.upper(),
                    "Baseline(3)": "-",
                    "MCFL(6)": "-",
                    "MCFL2(12)": "-",
                    "Δ6": "-",
                    "Δ12": "-",
                })
                continue

            def mean_val(k):
                r = mr.get(k, {})
                if metric_name == "tc_flicker" and "flicker" in r:
                    return r["flicker"].get("mean")
                return r.get("mean")

            if metric_name == "tc_flicker":
                for sub in ["flicker", "tc"]:
                    for agg in ["mean", "median"]:
                        lbl = "median" if agg == "median" else "mean"
                        rb = mr.get("baseline", {}).get(sub, {})
                        rm = mr.get("mcfl", {}).get(sub, {})
                        rm2 = mr.get("mcfl2", {}).get(sub, {})
                        vb = rb.get("value_median" if agg == "median" else "value", rb.get("value", "-"))
                        vm = rm.get("value_median" if agg == "median" else "value", rm.get("value", "-"))
                        vm2 = rm2.get("value_median" if agg == "median" else "value", rm2.get("value", "-"))
                        mb = rb.get(agg)
                        mm = rm.get(agg)
                        mm2 = rm2.get(agg)
                        d6 = f"{mm - mb:+.4f}" if mb is not None and mm is not None else "-"
                        d12 = f"{mm2 - mb:+.4f}" if mb is not None and mm2 is not None else "-"
                        table_data.append({
                            "Metric": f"TC_FLICKER ({sub},{lbl})",
                            "Baseline(3)": vb,
                            "MCFL(6)": vm,
                            "MCFL2(12)": vm2,
                            "Δ6": d6,
                            "Δ12": d12,
                        })
            else:
                b = mr.get("baseline", {}).get("value", "N/A")
                m6 = mr.get("mcfl", {}).get("value", "N/A")
                m12 = mr.get("mcfl2", {}).get("value", "N/A")
                bm = mean_val("baseline")
                m6m = mean_val("mcfl")
                m12m = mean_val("mcfl2")
                d6 = f"{m6m - bm:+.4f}" if bm is not None and m6m is not None else "-"
                d12 = f"{m12m - bm:+.4f}" if bm is not None and m12m is not None else "-"
                display = "FVD-32" if metric_name == "fvd_32" else metric_name.upper()
                table_data.append({
                    "Metric": display,
                    "Baseline(3)": b,
                    "MCFL(6)": m6,
                    "MCFL2(12)": m12,
                    "Δ6": d6,
                    "Δ12": d12,
                })

        if table_data:
            w = {c: max(len(str(r.get(c, "-"))) for r in table_data) + 2 for c in cols}
            header = " | ".join(f"{c:<{w[c]}}" for c in cols)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in table_data:
                f.write(" | ".join(f"{str(row.get(c, '-')):<{w[c]}}" for c in cols) + "\n")

    print(f"\n💾 报告已保存: {output_file}")
    print(f"💾 JSON: {json_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="post_URMP 三路对比：3=Baseline, 6=MCFL, 12=MCFL 2.0")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["fvd", "fid", "ffc", "clip", "av_align", "tc_flicker"],
                        choices=["fvd", "fvd_32", "fid", "ffc", "clip", "av_align", "tc_flicker"])
    parser.add_argument("--output", type=str, default="evaluation_report_urmp_3_6_12.txt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--bootstrap_k", type=int, default=10)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt")
    args = parser.parse_args()

    if not os.path.exists(REAL_DIR):
        print(f"❌ 真实目录不存在: {REAL_DIR}")
        return
    if not os.path.exists(BASELINE_DIR):
        print(f"❌ Baseline(3) 目录不存在: {BASELINE_DIR}")
        return
    if not os.path.exists(MCFL_DIR):
        print(f"❌ MCFL(6) 目录不存在: {MCFL_DIR}")
        return
    if not os.path.exists(MCFL2_DIR):
        print(f"❌ MCFL 2.0(12) 目录不存在: {MCFL2_DIR}")
        return

    metrics = [m.lower() for m in args.metrics]
    print("\n" + "=" * 80)
    print("📂 post_URMP 三路对比：3=Baseline, 6=MCFL, 12=MCFL 2.0")
    print("=" * 80)

    runner = EvaluationRunner(
        real_dir=REAL_DIR,
        baseline_dir=BASELINE_DIR,
        mcfl_dir=MCFL_DIR,
        mcfl2_dir=MCFL2_DIR,
        prompt_file=args.prompt_file,
        device=args.device,
        use_bootstrap=not args.no_bootstrap,
        bootstrap_k=args.bootstrap_k,
        use_cache=not args.no_cache,
        output_file=None,
    )
    runner.run_all(metrics)
    runner.print_report()
    write_report(runner.results, args.output, metrics)


if __name__ == "__main__":
    main()
