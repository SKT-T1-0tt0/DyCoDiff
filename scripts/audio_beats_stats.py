#!/usr/bin/env python3
"""
post_audioset_drums（或任意 TAV 数据集）的原始音频统计 + BEATs embedding 范数分布。
用于调 mcfl_gate_norm_low/high、归一化/compand 参数。

用法:
  python scripts/audio_beats_stats.py --data_path datasets/post_audioset_drums
  python scripts/audio_beats_stats.py --data_path datasets/post_audioset_drums --load_vid_len 90 --max_files 50 --with_beats
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import librosa

# 项目根目录
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

def find_wavs(data_path, split=None):
    """与 TAVDataset 一致：train/test 下找 wav，或从 stft_paths 推导。"""
    wavs = []
    for sub in ["train", "test"]:
        d = osp.join(data_path, sub)
        if not osp.isdir(d):
            continue
        # 可能布局: train/wav/*.wav 或 train/**/*.wav
        found = glob.glob(osp.join(d, "**", "*.wav"), recursive=True)
        if not found:
            # 从 mp4 推导: train/mp4/*.mp4 -> train/wav/*.wav
            mp4s = glob.glob(osp.join(d, "**", "*.mp4"), recursive=True)
            for p in mp4s:
                w = p.replace("/mp4/", "/wav/").replace("\\mp4\\", "\\wav\\").replace(".mp4", ".wav")
                if osp.isfile(w):
                    found.append(w)
        wavs.extend(found)
    if not wavs:
        # 无 train/test 时直接 data_path 下
        wavs = glob.glob(osp.join(data_path, "**", "*.wav"), recursive=True)
    return sorted(set(wavs))


def audio_stats(audio):
    """单段波形: peak, RMS, 是否近似已归一 (peak 接近 1)."""
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {"peak": peak, "rms": rms, "approx_normalized": 0.95 <= peak <= 1.05 if peak > 0 else False}


def collect_audio_stats(wav_paths, sr, load_vid_len, max_files=None):
    """收集原始音频统计（峰值、RMS、采样率、是否已归一）。"""
    paths = wav_paths[:max_files] if max_files else wav_paths
    peaks, rms_list, lengths_sec = [], [], []
    per_file = []
    for i, path in enumerate(paths):
        try:
            y, sr_actual = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        duration = len(y) / sr_actual
        st = audio_stats(y)
        peaks.append(st["peak"])
        rms_list.append(st["rms"])
        lengths_sec.append(duration)
        per_file.append({"path": path, "peak": st["peak"], "rms": st["rms"], "duration_sec": duration, "sr": sr_actual})
    peaks = np.array(peaks)
    rms_list = np.array(rms_list)
    return {
        "n_files": len(peaks),
        "sr": sr,
        "peak": {"mean": float(np.mean(peaks)), "std": float(np.std(peaks)), "min": float(np.min(peaks)), "max": float(np.max(peaks)),
                 "p5": float(np.percentile(peaks, 5)), "p50": float(np.percentile(peaks, 50)), "p95": float(np.percentile(peaks, 95))},
        "rms": {"mean": float(np.mean(rms_list)), "std": float(np.std(rms_list)), "min": float(np.min(rms_list)), "max": float(np.max(rms_list)),
                "p5": float(np.percentile(rms_list, 5)), "p50": float(np.percentile(rms_list, 50)), "p95": float(np.percentile(rms_list, 95))},
        "duration_sec": {"mean": float(np.mean(lengths_sec)), "min": float(np.min(lengths_sec)), "max": float(np.max(lengths_sec))},
        "approx_normalized_ratio": float(np.mean([1 if 0.95 <= p["peak"] <= 1.05 else 0 for p in per_file])) if per_file else 0.0,
        "per_file_sample": per_file[:5],
    }


def collect_beats_norms(wav_paths, sr, load_vid_len, sequence_length, max_files, max_clips_per_file, beats_ckpt, device="cpu",
                       gate_norm_low=7.2, gate_norm_high=10.0, gate_use_zscore=False, gate_norm_mu=8.4, gate_norm_sigma=0.5):
    """用 BEATs 跑与训练一致的 16 段 clip，统计 embedding 范数（每帧、pooled）及 conf 分布（护栏2）。"""
    import torch as th
    from einops import rearrange
    from beats.BEATs import BEATs, BEATsConfig

    checkpoint = th.load(beats_ckpt, map_location=device)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # 与 data.py 一致：整段重采样后 reshape (load_vid_len, -1)，每行是一段

    per_frame_norms = []   # 每个 (clip, frame) 的 768-d 向量范数
    pooled_norms = []      # 每个 clip 的 mean-pooled [768] 的范数

    paths = wav_paths[:max_files] if max_files else wav_paths
    rng = np.random.default_rng(42)

    for path in paths:
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            continue
        # reshape to (load_vid_len, -1)：与 TAVDataset 一致，每段长度 = len(y)//load_vid_len
        n_seg = load_vid_len
        segment_samples = len(y) // n_seg
        if segment_samples < 100:
            continue
        total_want = n_seg * segment_samples
        y = y[:total_want].reshape(n_seg, -1)

        for _ in range(max_clips_per_file):
            if n_seg < sequence_length:
                break
            start = rng.integers(0, n_seg - sequence_length + 1)
            end = start + sequence_length
            chunk = y[start:end]  # (16, segment_samples)
            # (16, segment_samples) -> (16*1, segment_samples) for BEATs
            audio = th.from_numpy(chunk.astype(np.float32)).unsqueeze(0)  # (1, 16, segment_samples)
            audio = rearrange(audio, "b f g -> (b f) g")  # (16, segment_samples)

            with th.no_grad():
                out = model.extract_features(audio.to(device), padding_mask=None)[0]  # (16, 8, 768)
            # 每帧: mean over 8 tokens -> (16, 768), norm -> (16,)
            frame_emb = out.mean(dim=1)  # (16, 768)
            frame_norms = frame_emb.norm(dim=-1).cpu().numpy()
            per_frame_norms.extend(frame_norms.tolist())
            # pooled: mean over (16, 8, 768) -> (768,), norm
            pooled = out.mean(dim=(0, 1)).cpu().numpy()
            pooled_norms.append(float(np.linalg.norm(pooled)))

    per_frame_norms = np.array(per_frame_norms)
    pooled_norms = np.array(pooled_norms)

    # 护栏2：conf 分布（与 condition_builder 一致的三角形或 z-score），便于 eval 时对比
    def _conf_from_norm(n, low, high, use_zscore=False, mu=8.4, sigma=0.5, z_low=-1.5, z_high=1.5):
        if use_zscore:
            z = (n - mu) / (sigma + 1e-6)
            return np.clip((z - z_low) / (z_high - z_low + 1e-6), 0.0, 1.0)
        mid = (low + high) * 0.5
        span = (high - low) + 1e-8
        return np.clip(1.0 - 2.0 * np.abs(n - mid) / span, 0.0, 1.0)

    confs = _conf_from_norm(pooled_norms, gate_norm_low, gate_norm_high, use_zscore=gate_use_zscore, mu=gate_norm_mu, sigma=gate_norm_sigma)

    return {
        "n_clips": len(pooled_norms),
        "n_frame_embeddings": len(per_frame_norms),
        "per_frame_norm": {
            "mean": float(np.mean(per_frame_norms)), "std": float(np.std(per_frame_norms)),
            "min": float(np.min(per_frame_norms)), "max": float(np.max(per_frame_norms)),
            "p5": float(np.percentile(per_frame_norms, 5)), "p50": float(np.percentile(per_frame_norms, 50)),
            "p95": float(np.percentile(per_frame_norms, 95)),
            "p1": float(np.percentile(per_frame_norms, 1)), "p99": float(np.percentile(per_frame_norms, 99)),
        },
        "pooled_norm": {
            "mean": float(np.mean(pooled_norms)), "std": float(np.std(pooled_norms)),
            "min": float(np.min(pooled_norms)), "max": float(np.max(pooled_norms)),
            "p5": float(np.percentile(pooled_norms, 5)), "p50": float(np.percentile(pooled_norms, 50)),
            "p95": float(np.percentile(pooled_norms, 95)),
            "p1": float(np.percentile(pooled_norms, 1)), "p99": float(np.percentile(pooled_norms, 99)),
        },
        "conf": {
            "mean": float(np.mean(confs)), "std": float(np.std(confs)),
            "min": float(np.min(confs)), "max": float(np.max(confs)),
            "p1": float(np.percentile(confs, 1)), "p99": float(np.percentile(confs, 99)),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Audio + BEATs embedding stats for TAV dataset")
    ap.add_argument("--data_path", type=str, default="datasets/post_audioset_drums", help="Path to dataset (train/test or wav root)")
    ap.add_argument("--load_vid_len", type=int, default=30, help="Segment count per file (30 or 90, matches training)")
    ap.add_argument("--sr", type=int, default=None, help="Sample rate (default: 48000 if load_vid_len<=30 else 96000)")
    ap.add_argument("--sequence_length", type=int, default=16)
    ap.add_argument("--max_files", type=int, default=100, help="Max wav files to scan")
    ap.add_argument("--with_beats", action="store_true", help="Run BEATs and report embedding norms")
    ap.add_argument("--beats_ckpt", type=str, default="saved_ckpts/BEATs_iter3_plus_AS20K.pt")
    ap.add_argument("--max_clips_per_file", type=int, default=2, help="Random clips per file for BEATs stats")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path")
    ap.add_argument("--gate_norm_low", type=float, default=7.2, help="Gate 区间下界（用于 conf 统计）")
    ap.add_argument("--gate_norm_high", type=float, default=10.0, help="Gate 区间上界")
    ap.add_argument("--gate_use_zscore", action="store_true", help="Conf 用 z-score 计算")
    ap.add_argument("--gate_norm_mu", type=float, default=8.4)
    ap.add_argument("--gate_norm_sigma", type=float, default=0.5)
    args = ap.parse_args()

    if args.sr is None:
        args.sr = 96000 if args.load_vid_len == 90 else 48000

    wav_paths = find_wavs(args.data_path)
    print(f"data_path={args.data_path}, load_vid_len={args.load_vid_len}, sr={args.sr}")
    print(f"Found {len(wav_paths)} wav files (using up to {args.max_files})")

    if not wav_paths:
        print("No wav files found. Exit.")
        return

    # ----- 原始音频统计 -----
    audio_report = collect_audio_stats(wav_paths, args.sr, args.load_vid_len, max_files=args.max_files)
    print("\n" + "=" * 60)
    print("原始音频统计 (Raw audio stats)")
    print("=" * 60)
    print(f"  采样率 (sr):           {audio_report['sr']} Hz")
    print(f"  文件数:                {audio_report['n_files']}")
    print("  Peak (峰值):")
    for k, v in audio_report["peak"].items():
        print(f"    {k}: {v:.6f}")
    print("  RMS:")
    for k, v in audio_report["rms"].items():
        print(f"    {k}: {v:.6f}")
    print("  时长 (秒):")
    for k, v in audio_report["duration_sec"].items():
        print(f"    {k}: {v:.4f}")
    print(f"  近似已归一比例 (peak∈[0.95,1.05]): {audio_report.get('approx_normalized_ratio', 0):.2%}")
    print("  per_file_sample (前几条):")
    for p in audio_report.get("per_file_sample", [])[:3]:
        print(f"    {osp.basename(p['path'])}: peak={p['peak']:.4f} rms={p['rms']:.4f} dur={p['duration_sec']:.2f}s")

    # ----- BEATs embedding 范数 -----
    beats_report = None
    if args.with_beats:
        print("\n" + "=" * 60)
        print("BEATs embedding 范数分布 (after mean-pool over 8 tokens)")
        print("=" * 60)
        beats_report = collect_beats_norms(
            wav_paths,
            sr=args.sr,
            load_vid_len=args.load_vid_len,
            sequence_length=args.sequence_length,
            max_files=min(args.max_files, 50),
            max_clips_per_file=args.max_clips_per_file,
            beats_ckpt=args.beats_ckpt,
            gate_norm_low=args.gate_norm_low,
            gate_norm_high=args.gate_norm_high,
            gate_use_zscore=args.gate_use_zscore,
            gate_norm_mu=args.gate_norm_mu,
            gate_norm_sigma=args.gate_norm_sigma,
        )
        print(f"  clips 数: {beats_report['n_clips']}, 帧嵌入数: {beats_report['n_frame_embeddings']}")
        print("  Per-frame norm (每帧 768-d 向量范数):")
        for k, v in beats_report["per_frame_norm"].items():
            print(f"    {k}: {v:.4f}")
        print("  Pooled norm (整 clip mean-pool 后 768-d 范数，用于 MCFL gate 置信度):")
        for k, v in beats_report["pooled_norm"].items():
            print(f"    {k}: {v:.4f}")
        print("  Conf (置信度，护栏2 写入 eval JSON 便于排查):")
        for k, v in beats_report["conf"].items():
            print(f"    {k}: {v:.4f}")
        p = beats_report["pooled_norm"]
        low_sug = max(0.5, p["p5"] - 0.5)
        high_sug = p["p95"] + 0.5
        beats_report["suggested_gate_norm_range"] = {"mcfl_gate_norm_low": low_sug, "mcfl_gate_norm_high": high_sug}
        print(f"\n  → 建议（本数据集）: mcfl_gate_norm_low={low_sug:.2f}, mcfl_gate_norm_high={high_sug:.2f}")
        print("     (使 pooled_norm 落在 [low, high] 内，置信度三角形中心约在 (low+high)/2)")

    out = {"audio": audio_report}
    if beats_report:
        out["beats"] = beats_report
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWritten {args.out}")


if __name__ == "__main__":
    main()
