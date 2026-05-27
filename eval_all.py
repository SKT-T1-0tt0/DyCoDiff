#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版视频评估脚本
统一管理所有评估指标：FVD, FID, FFC, CLIP, AV-Align, TC-Flicker

使用方法:
    python eval_all.py --real_dir results/1_tacm_/real \
                       --baseline_dir results/0_tacm_/fake1_30fps \
                       --mcfl_dir results/1_tacm_/fake1_30fps \
                       --metrics fvd fid ffc clip av_align tc_flicker \
                       --output report.txt

或者只运行特定指标:
    python eval_all.py --metrics fvd fid --real_dir results/1_tacm_/real
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# 全局导入常用库
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，创建一个简单的替代
    def tqdm(iterable, desc=""):
        if desc:
            print(desc)
        return iterable

try:
    import numpy as np
except ImportError:
    print("⚠️  NumPy 未安装，某些功能可能不可用")
    np = None

try:
    import cv2
except ImportError:
    print("⚠️  OpenCV 未安装，某些功能可能不可用")
    cv2 = None

import re  # 用于正则表达式匹配

# 导入各个评估模块
try:
    from eval_fvd import (
        load_i3d_model, extract_features as extract_fvd_features,
        compute_fvd, bootstrap_fvd as bootstrap_fvd_func
    )
    FVD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  FVD 模块导入失败: {e}")
    FVD_AVAILABLE = False

try:
    from eval_fid_video import (
        load_fid_inception_model, extract_features as extract_inception_features,
        compute_fid, bootstrap_fid as bootstrap_fid_func
    )
    FID_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  FID 模块导入失败: {e}")
    FID_AVAILABLE = False

try:
    from eval_ffc import (
        load_raft_model, compute_ffc as compute_ffc_func,
        bootstrap_ffc as bootstrap_ffc_func
    )
    FFC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  FFC 模块导入失败: {e}")
    FFC_AVAILABLE = False

try:
    import clip
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import re
    CLIP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CLIP 模块导入失败: {e}")
    CLIP_AVAILABLE = False

# 全局导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，创建一个简单的替代
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable

try:
    from scipy.io import wavfile
    AV_ALIGN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AV-Align 模块导入失败: {e}")
    AV_ALIGN_AVAILABLE = False

TC_FLICKER_AVAILABLE = True  # 只依赖 cv2, numpy

try:
    from eval_tc_flicker import tc_flicker_revised
except ImportError:
    tc_flicker_revised = None
    TC_FLICKER_AVAILABLE = False


class EvaluationRunner:
    """统一的评估运行器"""
    
    def __init__(
        self,
        real_dir: str,
        baseline_dir: Optional[str] = None,
        mcfl_dir: Optional[str] = None,
        mcfl2_dir: Optional[str] = None,
        prompt_file: str = "prompts.txt",
        device: str = "cuda",
        use_bootstrap: bool = True,
        bootstrap_k: int = 10,
        use_cache: bool = True,
        output_file: Optional[str] = None,
    ):
        """
        Args:
            real_dir: 真实视频目录
            baseline_dir: Baseline 生成视频目录（可选）
            mcfl_dir: MCFL 生成视频目录（可选）
            mcfl2_dir: MCFL 2.0 生成视频目录（可选，三路对比时使用）
            prompt_file: 提示词文件路径（用于 CLIP 评估）
            device: 计算设备
            use_bootstrap: 是否使用 bootstrap 采样
            bootstrap_k: Bootstrap 采样次数
            use_cache: 是否使用特征缓存
            output_file: 输出报告文件路径（可选）
        """
        self.real_dir = real_dir
        self.baseline_dir = baseline_dir
        self.mcfl_dir = mcfl_dir
        self.mcfl2_dir = mcfl2_dir
        self.prompt_file = prompt_file
        self.device = device
        self.use_bootstrap = use_bootstrap
        self.bootstrap_k = bootstrap_k
        self.use_cache = use_cache
        self.output_file = output_file
        
        self.results = {}
        
    def run_fvd(self, num_frames=16) -> Dict:
        """运行 FVD 评估 (num_frames: 16=FVD, 32=FVD-32)"""
        if not FVD_AVAILABLE:
            return {"error": "FVD 模块不可用"}
        
        label = f"FVD-{num_frames}" if num_frames != 16 else "FVD"
        print("\n" + "="*60)
        print(f"📊 {label} (Fréchet Video Distance) 评估")
        print("="*60)
        
        try:
            import torch
            import numpy as np
            from tqdm import tqdm
            
            # 设置全局变量（eval_fvd.py 使用全局变量）
            import eval_fvd as fvd_module
            fvd_module.DEVICE = self.device
            fvd_module.USE_CACHE = self.use_cache
            
            # 加载 I3D 模型（设置全局变量）
            print("🔄 加载 I3D 模型...")
            fvd_module.i3d = load_i3d_model(self.device)
            
            # 提取特征
            print(f"📹 提取真实视频特征: {self.real_dir} (num_frames={num_frames})")
            feats_real = extract_fvd_features(
                self.real_dir, use_cache=self.use_cache, num_frames=num_frames
            )
            
            results = {}
            
            if self.baseline_dir:
                print(f"📹 提取 Baseline 视频特征: {self.baseline_dir}")
                feats_baseline = extract_fvd_features(
                    self.baseline_dir, use_cache=self.use_cache, num_frames=num_frames
                )
                
                # 对齐数量
                min_len = min(len(feats_real), len(feats_baseline))
                feats_real_aligned = feats_real[:min_len]
                feats_baseline_aligned = feats_baseline[:min_len]
                
                if self.use_bootstrap:
                    mean_fvd, std_fvd = bootstrap_fvd_func(
                        feats_real_aligned, feats_baseline_aligned, k=self.bootstrap_k
                    )
                    results["baseline"] = {
                        "mean": float(mean_fvd),
                        "std": float(std_fvd),
                        "value": f"{mean_fvd:.2f} ± {std_fvd:.2f}"
                    }
                else:
                    fvd_val = compute_fvd(feats_real_aligned, feats_baseline_aligned)
                    results["baseline"] = {
                        "mean": float(fvd_val),
                        "std": 0.0,
                        "value": f"{fvd_val:.2f}"
                    }
                print(f"  ✓ Baseline {label}: {results['baseline']['value']}")
            
            if self.mcfl_dir:
                print(f"📹 提取 MCFL 视频特征: {self.mcfl_dir}")
                feats_mcfl = extract_fvd_features(
                    self.mcfl_dir, use_cache=self.use_cache, num_frames=num_frames
                )
                
                # 对齐数量
                min_len = min(len(feats_real), len(feats_mcfl))
                feats_real_aligned = feats_real[:min_len]
                feats_mcfl_aligned = feats_mcfl[:min_len]
                
                if self.use_bootstrap:
                    mean_fvd, std_fvd = bootstrap_fvd_func(
                        feats_real_aligned, feats_mcfl_aligned, k=self.bootstrap_k
                    )
                    results["mcfl"] = {
                        "mean": float(mean_fvd),
                        "std": float(std_fvd),
                        "value": f"{mean_fvd:.2f} ± {std_fvd:.2f}"
                    }
                else:
                    fvd_val = compute_fvd(feats_real_aligned, feats_mcfl_aligned)
                    results["mcfl"] = {
                        "mean": float(fvd_val),
                        "std": 0.0,
                        "value": f"{fvd_val:.2f}"
                    }
                print(f"  ✓ MCFL {label}: {results['mcfl']['value']}")
            
            if self.mcfl2_dir:
                print(f"📹 提取 MCFL 2.0 视频特征: {self.mcfl2_dir}")
                feats_mcfl2 = extract_fvd_features(
                    self.mcfl2_dir, use_cache=self.use_cache, num_frames=num_frames
                )
                min_len = min(len(feats_real), len(feats_mcfl2))
                feats_real_aligned = feats_real[:min_len]
                feats_mcfl2_aligned = feats_mcfl2[:min_len]
                if self.use_bootstrap:
                    mean_fvd, std_fvd = bootstrap_fvd_func(
                        feats_real_aligned, feats_mcfl2_aligned, k=self.bootstrap_k
                    )
                    results["mcfl2"] = {
                        "mean": float(mean_fvd),
                        "std": float(std_fvd),
                        "value": f"{mean_fvd:.2f} ± {std_fvd:.2f}"
                    }
                else:
                    fvd_val = compute_fvd(feats_real_aligned, feats_mcfl2_aligned)
                    results["mcfl2"] = {
                        "mean": float(fvd_val),
                        "std": 0.0,
                        "value": f"{fvd_val:.2f}"
                    }
                print(f"  ✓ MCFL 2.0 {label}: {results['mcfl2']['value']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ FVD 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_fvd_32(self) -> Dict:
        """运行 FVD-32 评估（32 帧，更稳定的估计）"""
        return self.run_fvd(num_frames=32)
    
    def run_fid(self) -> Dict:
        """运行 FID 评估"""
        if not FID_AVAILABLE:
            return {"error": "FID 模块不可用"}
        
        print("\n" + "="*60)
        print("📊 FID (Fréchet Inception Distance) 评估")
        print("="*60)
        
        try:
            # 设置全局变量（eval_fid_video.py 使用全局变量）
            import eval_fid_video as fid_module
            fid_module.DEVICE = self.device
            fid_module.USE_CACHE = self.use_cache
            
            # 加载 FID Inception 模型（设置全局变量）
            print("🔄 加载 FID Inception 模型...")
            fid_module.inception_model = load_fid_inception_model(self.device)
            
            # 提取特征
            print(f"📹 提取真实视频特征: {self.real_dir}")
            feats_real = extract_inception_features(
                self.real_dir, use_cache=self.use_cache
            )
            
            results = {}
            
            if self.baseline_dir:
                print(f"📹 提取 Baseline 视频特征: {self.baseline_dir}")
                feats_baseline = extract_inception_features(
                    self.baseline_dir, use_cache=self.use_cache
                )
                
                min_len = min(len(feats_real), len(feats_baseline))
                feats_real_aligned = feats_real[:min_len]
                feats_baseline_aligned = feats_baseline[:min_len]
                
                if self.use_bootstrap:
                    mean_fid, std_fid = bootstrap_fid_func(
                        feats_real_aligned, feats_baseline_aligned, k=self.bootstrap_k
                    )
                    results["baseline"] = {
                        "mean": float(mean_fid),
                        "std": float(std_fid),
                        "value": f"{mean_fid:.2f} ± {std_fid:.2f}"
                    }
                else:
                    fid_val = compute_fid(feats_real_aligned, feats_baseline_aligned)
                    results["baseline"] = {
                        "mean": float(fid_val),
                        "std": 0.0,
                        "value": f"{fid_val:.2f}"
                    }
                print(f"  ✓ Baseline FID: {results['baseline']['value']}")
            
            if self.mcfl_dir:
                print(f"📹 提取 MCFL 视频特征: {self.mcfl_dir}")
                feats_mcfl = extract_inception_features(
                    self.mcfl_dir, use_cache=self.use_cache
                )
                
                min_len = min(len(feats_real), len(feats_mcfl))
                feats_real_aligned = feats_real[:min_len]
                feats_mcfl_aligned = feats_mcfl[:min_len]
                
                if self.use_bootstrap:
                    mean_fid, std_fid = bootstrap_fid_func(
                        feats_real_aligned, feats_mcfl_aligned, k=self.bootstrap_k
                    )
                    results["mcfl"] = {
                        "mean": float(mean_fid),
                        "std": float(std_fid),
                        "value": f"{mean_fid:.2f} ± {std_fid:.2f}"
                    }
                else:
                    fid_val = compute_fid(feats_real_aligned, feats_mcfl_aligned)
                    results["mcfl"] = {
                        "mean": float(fid_val),
                        "std": 0.0,
                        "value": f"{fid_val:.2f}"
                    }
                print(f"  ✓ MCFL FID: {results['mcfl']['value']}")
            
            if self.mcfl2_dir:
                print(f"📹 提取 MCFL 2.0 视频特征: {self.mcfl2_dir}")
                feats_mcfl2 = extract_inception_features(
                    self.mcfl2_dir, use_cache=self.use_cache
                )
                min_len = min(len(feats_real), len(feats_mcfl2))
                feats_real_aligned = feats_real[:min_len]
                feats_mcfl2_aligned = feats_mcfl2[:min_len]
                if self.use_bootstrap:
                    mean_fid, std_fid = bootstrap_fid_func(
                        feats_real_aligned, feats_mcfl2_aligned, k=self.bootstrap_k
                    )
                    results["mcfl2"] = {
                        "mean": float(mean_fid),
                        "std": float(std_fid),
                        "value": f"{mean_fid:.2f} ± {std_fid:.2f}"
                    }
                else:
                    fid_val = compute_fid(feats_real_aligned, feats_mcfl2_aligned)
                    results["mcfl2"] = {
                        "mean": float(fid_val),
                        "std": 0.0,
                        "value": f"{fid_val:.2f}"
                    }
                print(f"  ✓ MCFL 2.0 FID: {results['mcfl2']['value']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ FID 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_ffc(self) -> Dict:
        """运行 FFC (Frame-to-Frame Consistency) 评估"""
        if not FFC_AVAILABLE:
            return {"error": "FFC 模块不可用"}
        
        print("\n" + "="*60)
        print("📊 FFC (Frame-to-Frame Consistency) 评估")
        print("="*60)
        
        try:
            # 设置全局变量（eval_ffc.py 使用全局变量）
            import eval_ffc as ffc_module
            ffc_module.DEVICE = self.device
            ffc_module.USE_CACHE = self.use_cache
            
            # 加载 RAFT 模型（设置全局变量）
            print("🔄 加载 RAFT 模型...")
            ffc_module.raft_model = load_raft_model(self.device)
            
            results = {}
            
            if self.baseline_dir:
                print(f"📹 评估 Baseline 视频: {self.baseline_dir}")
                mean_ffc, scores = compute_ffc_func(self.baseline_dir, use_cache=self.use_cache)
                
                if self.use_bootstrap:
                    mean_ffc_bs, std_ffc = bootstrap_ffc_func(scores, k=self.bootstrap_k)
                    results["baseline"] = {
                        "mean": float(mean_ffc_bs),
                        "std": float(std_ffc),
                        "value": f"{mean_ffc_bs:.4f} ± {std_ffc:.4f}"
                    }
                else:
                    results["baseline"] = {
                        "mean": float(mean_ffc),
                        "std": 0.0,
                        "value": f"{mean_ffc:.4f}"
                    }
                print(f"  ✓ Baseline FFC: {results['baseline']['value']}")
            
            if self.mcfl_dir:
                print(f"📹 评估 MCFL 视频: {self.mcfl_dir}")
                mean_ffc, scores = compute_ffc_func(self.mcfl_dir, use_cache=self.use_cache)
                
                if self.use_bootstrap:
                    mean_ffc_bs, std_ffc = bootstrap_ffc_func(scores, k=self.bootstrap_k)
                    results["mcfl"] = {
                        "mean": float(mean_ffc_bs),
                        "std": float(std_ffc),
                        "value": f"{mean_ffc_bs:.4f} ± {std_ffc:.4f}"
                    }
                else:
                    results["mcfl"] = {
                        "mean": float(mean_ffc),
                        "std": 0.0,
                        "value": f"{mean_ffc:.4f}"
                    }
                print(f"  ✓ MCFL FFC: {results['mcfl']['value']}")
            
            if self.mcfl2_dir:
                print(f"📹 评估 MCFL 2.0 视频: {self.mcfl2_dir}")
                mean_ffc, scores = compute_ffc_func(self.mcfl2_dir, use_cache=self.use_cache)
                if self.use_bootstrap:
                    mean_ffc_bs, std_ffc = bootstrap_ffc_func(scores, k=self.bootstrap_k)
                    results["mcfl2"] = {
                        "mean": float(mean_ffc_bs),
                        "std": float(std_ffc),
                        "value": f"{mean_ffc_bs:.4f} ± {std_ffc:.4f}"
                    }
                else:
                    results["mcfl2"] = {
                        "mean": float(mean_ffc),
                        "std": 0.0,
                        "value": f"{mean_ffc:.4f}"
                    }
                print(f"  ✓ MCFL 2.0 FFC: {results['mcfl2']['value']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ FFC 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_clip(self) -> Dict:
        """运行 CLIP Text-Video Similarity 评估"""
        if not CLIP_AVAILABLE:
            return {"error": "CLIP 模块不可用"}
        
        print("\n" + "="*60)
        print("📊 CLIP Text-Video Similarity 评估")
        print("="*60)
        
        try:
            # 加载提示词
            if not os.path.exists(self.prompt_file):
                return {"error": f"提示词文件不存在: {self.prompt_file}"}
            
            prompts = open(self.prompt_file).read().strip().splitlines()
            
            # 加载 CLIP 模型
            print("🔄 加载 CLIP 模型...")
            device = self.device if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.eval()
            
            def sorted_videos(video_dir):
                videos = [v for v in os.listdir(video_dir) if v.endswith(".mp4")]
                videos.sort(key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
                return videos
            
            def sample_frames(video_path, num_frames=8):
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                idxs = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
                
                frames = []
                for i in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                cap.release()
                return frames
            
            def clip_video_text_score(video_dir, prompts):
                videos = sorted_videos(video_dir)
                # 对齐视频和提示词数量（取最小值）
                min_count = min(len(videos), len(prompts))
                videos = videos[:min_count]
                prompts = prompts[:min_count]
                
                if len(videos) == 0:
                    raise ValueError(f"No videos found in {video_dir}")
                
                print(f"  使用 {len(videos)} 个视频和 {len(prompts)} 个提示词进行评估")
                
                # Encode all text once
                text_tokens = clip.tokenize(prompts).to(device)
                with torch.no_grad():
                    text_feats = model.encode_text(text_tokens)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                
                scores = []
                
                for idx, video_name in enumerate(tqdm(videos, desc=f"Evaluating {video_dir}")):
                    video_path = os.path.join(video_dir, video_name)
                    frames = sample_frames(video_path, num_frames=8)
                    assert len(frames) > 0, f"No frames read from {video_path}"
                    
                    img_feats = []
                    for frame in frames:
                        img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feat = model.encode_image(img)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                            img_feats.append(feat)
                    
                    # average over frames
                    video_feat = torch.mean(torch.cat(img_feats, dim=0), dim=0, keepdim=True)
                    
                    score = (video_feat @ text_feats[idx].unsqueeze(1)).item()
                    scores.append(score)
                
                return float(np.mean(scores)), float(np.std(scores))
            
            results = {}
            
            if self.baseline_dir:
                print(f"📹 评估 Baseline 视频: {self.baseline_dir}")
                mean_score, std_score = clip_video_text_score(self.baseline_dir, prompts)
                results["baseline"] = {
                    "mean": float(mean_score),
                    "std": float(std_score),
                    "value": f"{mean_score:.4f} ± {std_score:.4f}"
                }
                print(f"  ✓ Baseline CLIP: {results['baseline']['value']}")
            
            if self.mcfl_dir:
                print(f"📹 评估 MCFL 视频: {self.mcfl_dir}")
                mean_score, std_score = clip_video_text_score(self.mcfl_dir, prompts)
                results["mcfl"] = {
                    "mean": float(mean_score),
                    "std": float(std_score),
                    "value": f"{mean_score:.4f} ± {std_score:.4f}"
                }
                print(f"  ✓ MCFL CLIP: {results['mcfl']['value']}")

            if self.mcfl2_dir:
                print(f"📹 评估 MCFL 2.0 视频: {self.mcfl2_dir}")
                mean_score, std_score = clip_video_text_score(self.mcfl2_dir, prompts)
                results["mcfl2"] = {
                    "mean": float(mean_score),
                    "std": float(std_score),
                    "value": f"{mean_score:.4f} ± {std_score:.4f}"
                }
                print(f"  ✓ MCFL 2.0 CLIP: {results['mcfl2']['value']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ CLIP 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_av_align(self) -> Dict:
        """运行 Audio-Video Alignment 评估"""
        if not AV_ALIGN_AVAILABLE:
            return {"error": "AV-Align 模块不可用"}
        
        print("\n" + "="*60)
        print("📊 Audio-Video Alignment 评估")
        print("="*60)
        
        try:
            def motion_energy(video_path, smooth_k=3):
                """运动能量序列，可选 3-frame MA 平滑以压 AV_ALIGN std"""
                cap = cv2.VideoCapture(video_path)
                prev = None
                energies = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev is not None:
                        energies.append(np.abs(gray.astype(np.float32) - prev.astype(np.float32)).mean())
                    prev = gray
                
                cap.release()
                energies = np.array(energies)
                
                if smooth_k > 1 and len(energies) >= smooth_k:
                    kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
                    energies = np.convolve(energies, kernel, mode="same")
                
                return energies
            
            def audio_energy(wav_path, target_len):
                sr, audio = wavfile.read(wav_path)
                audio = audio.astype(np.float32)
                frame_len = len(audio) // target_len
                return np.array([
                    np.mean(np.abs(audio[i*frame_len:(i+1)*frame_len]))
                    for i in range(target_len)
                ])
            
            def max_corr(a, b, max_lag=5):
                a = (a - a.mean()) / (a.std() + 1e-6)
                b = (b - b.mean()) / (b.std() + 1e-6)
                corrs = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        corrs.append(np.corrcoef(a[:lag], b[-lag:])[0, 1])
                    elif lag > 0:
                        corrs.append(np.corrcoef(a[lag:], b[:-lag])[0, 1])
                    else:
                        corrs.append(np.corrcoef(a, b)[0, 1])
                return np.nanmax(corrs)
            
            results = {}
            
            if self.baseline_dir:
                print(f"📹 评估 Baseline 视频: {self.baseline_dir}")
                audio_dir = self.baseline_dir.replace("/fake1_30fps", "/audio").replace("/fake1_6fps", "/audio")
                
                scores = []
                videos = sorted([v for v in os.listdir(self.baseline_dir) if v.endswith(".mp4")])
                
                for video_name in tqdm(videos[:50], desc="AV-Align Baseline"):
                    video_path = os.path.join(self.baseline_dir, video_name)
                    # 尝试匹配音频文件名
                    video_id = re.findall(r"\d+", video_name)
                    if video_id:
                        audio_name = f"groundtruth_{video_id[0]}.wav"
                    else:
                        audio_name = video_name.replace(".mp4", ".wav")
                    audio_path = os.path.join(audio_dir, audio_name)
                    
                    if os.path.exists(audio_path):
                        m = motion_energy(video_path)
                        ae = audio_energy(audio_path, len(m))
                        scores.append(max_corr(m, ae, max_lag=5))
                
                if scores:
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    results["baseline"] = {
                        "mean": mean_score,
                        "std": std_score,
                        "value": f"{mean_score:.4f} ± {std_score:.4f}"
                    }
                    print(f"  ✓ Baseline AV-Align: {results['baseline']['value']}")
            
            if self.mcfl_dir:
                print(f"📹 评估 MCFL 视频: {self.mcfl_dir}")
                audio_dir = self.mcfl_dir.replace("/fake1_30fps", "/audio").replace("/fake1_6fps", "/audio")
                
                scores = []
                videos = sorted([v for v in os.listdir(self.mcfl_dir) if v.endswith(".mp4")])
                
                for video_name in tqdm(videos[:50], desc="AV-Align MCFL"):
                    video_path = os.path.join(self.mcfl_dir, video_name)
                    video_id = re.findall(r"\d+", video_name)
                    if video_id:
                        audio_name = f"groundtruth_{video_id[0]}.wav"
                    else:
                        audio_name = video_name.replace(".mp4", ".wav")
                    audio_path = os.path.join(audio_dir, audio_name)
                    
                    if os.path.exists(audio_path):
                        m = motion_energy(video_path)
                        ae = audio_energy(audio_path, len(m))
                        scores.append(max_corr(m, ae, max_lag=5))
                
                if scores:
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    results["mcfl"] = {
                        "mean": mean_score,
                        "std": std_score,
                        "value": f"{mean_score:.4f} ± {std_score:.4f}"
                    }
                    print(f"  ✓ MCFL AV-Align: {results['mcfl']['value']}")

            if self.mcfl2_dir:
                print(f"📹 评估 MCFL 2.0 视频: {self.mcfl2_dir}")
                audio_dir = self.mcfl2_dir.replace("/fake1_30fps", "/audio").replace("/fake1_6fps", "/audio")
                scores = []
                videos = sorted([v for v in os.listdir(self.mcfl2_dir) if v.endswith(".mp4")])
                for video_name in tqdm(videos[:50], desc="AV-Align MCFL 2.0"):
                    video_path = os.path.join(self.mcfl2_dir, video_name)
                    video_id = re.findall(r"\d+", video_name)
                    if video_id:
                        audio_name = f"groundtruth_{video_id[0]}.wav"
                    else:
                        audio_name = video_name.replace(".mp4", ".wav")
                    audio_path = os.path.join(audio_dir, audio_name)
                    if os.path.exists(audio_path):
                        m = motion_energy(video_path)
                        ae = audio_energy(audio_path, len(m))
                        scores.append(max_corr(m, ae, max_lag=5))
                if scores:
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    results["mcfl2"] = {
                        "mean": mean_score,
                        "std": std_score,
                        "value": f"{mean_score:.4f} ± {std_score:.4f}"
                    }
                    print(f"  ✓ MCFL 2.0 AV-Align: {results['mcfl2']['value']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ AV-Align 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_tc_flicker(self) -> Dict:
        """运行 Temporal Consistency (Flicker) 评估 - 使用 TC_FLICKER-R 修订版"""
        if not TC_FLICKER_AVAILABLE or tc_flicker_revised is None:
            return {"error": "TC-Flicker 模块不可用"}
        
        print("\n" + "="*60)
        print("📊 Temporal Consistency (Flicker) 评估 [TC_FLICKER-R]")
        print("="*60)
        
        try:
            results = {}
            
            if self.baseline_dir:
                print(f"📹 评估 Baseline 视频: {self.baseline_dir}")
                videos = sorted([v for v in os.listdir(self.baseline_dir) if v.endswith(".mp4")])
                vals = []
                for video_name in tqdm(videos[:50], desc="TC-Flicker Baseline"):
                    video_path = os.path.join(self.baseline_dir, video_name)
                    vals.append(tc_flicker_revised(video_path, smooth_k=3))
                vals = np.array(vals)
                n_b = len(vals)
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals)) if n_b > 1 else 0.0
                median_val = float(np.median(vals))
                results["baseline"] = {
                    "flicker": {
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "value": f"{mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{median_val:.6f}",
                    },
                    "tc": {
                        "mean": -mean_val,
                        "std": std_val,
                        "median": -median_val,
                        "value": f"{-mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{-median_val:.6f}",
                    },
                    "n_videos": n_b,
                }
                print(f"  ✓ Baseline Flicker: mean={results['baseline']['flicker']['value']}  median={results['baseline']['flicker']['value_median']} (N={n_b})")
                print(f"  ✓ Baseline TC:      mean={results['baseline']['tc']['value']}  median={results['baseline']['tc']['value_median']}")
            
            if self.mcfl_dir:
                print(f"📹 评估 MCFL 视频: {self.mcfl_dir}")
                videos = sorted([v for v in os.listdir(self.mcfl_dir) if v.endswith(".mp4")])
                vals = []
                for video_name in tqdm(videos[:50], desc="TC-Flicker MCFL"):
                    video_path = os.path.join(self.mcfl_dir, video_name)
                    vals.append(tc_flicker_revised(video_path, smooth_k=3))
                vals = np.array(vals)
                n_m = len(vals)
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals)) if n_m > 1 else 0.0
                median_val = float(np.median(vals))
                results["mcfl"] = {
                    "flicker": {
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "value": f"{mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{median_val:.6f}",
                    },
                    "tc": {
                        "mean": -mean_val,
                        "std": std_val,
                        "median": -median_val,
                        "value": f"{-mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{-median_val:.6f}",
                    },
                    "n_videos": n_m,
                }
                print(f"  ✓ MCFL Flicker: mean={results['mcfl']['flicker']['value']}  median={results['mcfl']['flicker']['value_median']} (N={n_m})")
                print(f"  ✓ MCFL TC:      mean={results['mcfl']['tc']['value']}  median={results['mcfl']['tc']['value_median']}")

            if self.mcfl2_dir:
                print(f"📹 评估 MCFL 2.0 视频: {self.mcfl2_dir}")
                videos = sorted([v for v in os.listdir(self.mcfl2_dir) if v.endswith(".mp4")])
                vals = []
                for video_name in tqdm(videos[:50], desc="TC-Flicker MCFL 2.0"):
                    video_path = os.path.join(self.mcfl2_dir, video_name)
                    vals.append(tc_flicker_revised(video_path, smooth_k=3))
                vals = np.array(vals)
                n_m2 = len(vals)
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals)) if n_m2 > 1 else 0.0
                median_val = float(np.median(vals))
                results["mcfl2"] = {
                    "flicker": {
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "value": f"{mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{median_val:.6f}",
                    },
                    "tc": {
                        "mean": -mean_val,
                        "std": std_val,
                        "median": -median_val,
                        "value": f"{-mean_val:.6f} ± {std_val:.6f}",
                        "value_median": f"{-median_val:.6f}",
                    },
                    "n_videos": n_m2,
                }
                print(f"  ✓ MCFL 2.0 Flicker: mean={results['mcfl2']['flicker']['value']}  median={results['mcfl2']['flicker']['value_median']} (N={n_m2})")
                print(f"  ✓ MCFL 2.0 TC:      mean={results['mcfl2']['tc']['value']}  median={results['mcfl2']['tc']['value_median']}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ TC-Flicker 评估失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_all(self, metrics: List[str]) -> Dict:
        """运行所有指定的评估指标"""
        print("\n" + "="*80)
        print("🚀 开始批量评估")
        print("="*80)
        print(f"真实视频目录: {self.real_dir}")
        if self.baseline_dir:
            print(f"Baseline 目录: {self.baseline_dir}")
        if self.mcfl_dir:
            print(f"MCFL 目录: {self.mcfl_dir}")
        if self.mcfl2_dir:
            print(f"MCFL 2.0 目录: {self.mcfl2_dir}")
        print(f"评估指标: {', '.join(metrics)}")
        print("="*80)
        
        all_results = {}
        
        metric_map = {
            "fvd": self.run_fvd,
            "fvd_32": self.run_fvd_32,
            "fid": self.run_fid,
            "ffc": self.run_ffc,
            "clip": self.run_clip,
            "av_align": self.run_av_align,
            "tc_flicker": self.run_tc_flicker,
        }
        
        for metric in metrics:
            if metric.lower() not in metric_map:
                print(f"⚠️  未知的评估指标: {metric}")
                continue
            
            try:
                result = metric_map[metric.lower()]()
                all_results[metric.lower()] = result
            except Exception as e:
                print(f"❌ {metric} 评估失败: {e}")
                all_results[metric.lower()] = {"error": str(e)}
        
        self.results = all_results
        return all_results
    
    def print_report(self):
        """打印评估报告"""
        print("\n" + "="*80)
        print("📋 评估报告")
        print("="*80)
        
        # 构建表格
        table_data = []
        
        for metric_name, metric_results in self.results.items():
            if "error" in metric_results:
                continue
            
            has_mcfl2 = "mcfl2" in metric_results
            if metric_name == "tc_flicker":
                for sub_metric in ["flicker", "tc"]:
                    for agg in ["mean", "median"]:
                        label = "median" if agg == "median" else "mean"
                        row = {
                            "Metric": f"{metric_name.upper()} ({sub_metric}, {label})",
                            "Baseline": "-",
                            "MCFL": "-",
                            "Δ": "-"
                        }
                        if has_mcfl2:
                            row["MCFL2"] = "-"
                        b = metric_results.get("baseline", {}).get(sub_metric, {})
                        m = metric_results.get("mcfl", {}).get(sub_metric, {})
                        m2 = metric_results.get("mcfl2", {}).get(sub_metric, {}) if has_mcfl2 else {}
                        if b:
                            row["Baseline"] = b.get("value_median" if agg == "median" else "value", b.get("value", "-"))
                        if m:
                            row["MCFL"] = m.get("value_median" if agg == "median" else "value", m.get("value", "-"))
                        if has_mcfl2 and m2:
                            row["MCFL2"] = m2.get("value_median" if agg == "median" else "value", m2.get("value", "-"))
                        if b and m and agg in b and agg in m:
                            row["Δ"] = f"{m[agg] - b[agg]:+.4f}"
                        table_data.append(row)
            else:
                display_name = "FVD-32" if metric_name == "fvd_32" else metric_name.upper()
                row = {
                    "Metric": display_name,
                    "Baseline": "-",
                    "MCFL": "-",
                    "Δ": "-"
                }
                if has_mcfl2:
                    row["MCFL2"] = "-"
                if "baseline" in metric_results:
                    row["Baseline"] = metric_results["baseline"]["value"]
                if "mcfl" in metric_results:
                    row["MCFL"] = metric_results["mcfl"]["value"]
                if has_mcfl2 and "mcfl2" in metric_results:
                    row["MCFL2"] = metric_results["mcfl2"]["value"]
                if "baseline" in metric_results and "mcfl" in metric_results:
                    row["Δ"] = f"{metric_results['mcfl']['mean'] - metric_results['baseline']['mean']:+.4f}"
                table_data.append(row)
        
        # 打印表格
        if table_data:
            cols = ["Metric", "Baseline", "MCFL"]
            if any("MCFL2" in row for row in table_data):
                cols.append("MCFL2")
            cols.append("Δ")
            col_widths = {c: max(len(str(row.get(c, "-"))) for row in table_data) + 2 for c in cols}
            header = " | ".join(f"{c:<{col_widths[c]}}" for c in cols)
            print(header)
            print("-" * len(header))
            for row in table_data:
                print(" | ".join(f"{str(row.get(c, '-')):<{col_widths[c]}}" for c in cols))
        
        print("="*80)
        
        # 保存到文件
        if self.output_file:
            self.save_report()
    
    def save_report(self):
        """保存评估报告到文件"""
        if not self.output_file:
            return
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "real_dir": self.real_dir,
                "baseline_dir": self.baseline_dir,
                "mcfl_dir": self.mcfl_dir,
                "mcfl2_dir": self.mcfl2_dir,
                "use_bootstrap": self.use_bootstrap,
                "bootstrap_k": self.bootstrap_k,
            },
            "results": self.results,
        }
        
        # 保存 JSON
        json_file = self.output_file.replace(".txt", ".json") if self.output_file.endswith(".txt") else self.output_file + ".json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("视频评估报告\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {report['timestamp']}\n")
            f.write(f"真实视频目录: {self.real_dir}\n")
            if self.baseline_dir:
                f.write(f"Baseline 目录: {self.baseline_dir}\n")
            if self.mcfl_dir:
                f.write(f"MCFL 目录: {self.mcfl_dir}\n")
            if self.mcfl2_dir:
                f.write(f"MCFL 2.0 目录: {self.mcfl2_dir}\n")
            f.write("="*80 + "\n\n")
            
            for metric_name, metric_results in self.results.items():
                f.write(f"\n{metric_name.upper()}:\n")
                if "error" in metric_results:
                    f.write(f"  错误: {metric_results['error']}\n")
                else:
                    if "baseline" in metric_results:
                        b = metric_results["baseline"]
                        if metric_name == "tc_flicker" and "flicker" in b:
                            f.write(f"  Baseline: flicker mean={b['flicker'].get('value', 'N/A')}  median={b['flicker'].get('value_median', 'N/A')}")
                            if b.get("n_videos") is not None:
                                f.write(f"  (N={b['n_videos']})")
                            f.write("\n")
                            f.write(f"            tc      mean={b['tc'].get('value', 'N/A')}  median={b['tc'].get('value_median', 'N/A')}\n")
                        else:
                            f.write(f"  Baseline: {b.get('value', 'N/A')}\n")
                    if "mcfl" in metric_results:
                        m = metric_results["mcfl"]
                        if metric_name == "tc_flicker" and "flicker" in m:
                            f.write(f"  MCFL:     flicker mean={m['flicker'].get('value', 'N/A')}  median={m['flicker'].get('value_median', 'N/A')}")
                            if m.get("n_videos") is not None:
                                f.write(f"  (N={m['n_videos']})")
                            f.write("\n")
                            f.write(f"            tc      mean={m['tc'].get('value', 'N/A')}  median={m['tc'].get('value_median', 'N/A')}\n")
                        else:
                            f.write(f"  MCFL: {m.get('value', 'N/A')}\n")
                    if "mcfl2" in metric_results:
                        m2 = metric_results["mcfl2"]
                        if metric_name == "tc_flicker" and "flicker" in m2:
                            f.write(f"  MCFL 2.0: flicker mean={m2['flicker'].get('value', 'N/A')}  median={m2['flicker'].get('value_median', 'N/A')}")
                            if m2.get("n_videos") is not None:
                                f.write(f"  (N={m2['n_videos']})")
                            f.write("\n")
                            f.write(f"            tc      mean={m2['tc'].get('value', 'N/A')}  median={m2['tc'].get('value_median', 'N/A')}\n")
                        else:
                            f.write(f"  MCFL 2.0: {m2.get('value', 'N/A')}\n")
            
            # 小样本说明：TC_FLICKER 视频数较少时，mean 易受单条异常值影响
            n_note = None
            if "tc_flicker" in self.results and "error" not in self.results.get("tc_flicker", {}):
                for key in ("baseline", "mcfl", "mcfl2"):
                    r = self.results["tc_flicker"].get(key, {})
                    if r.get("n_videos") is not None and r["n_videos"] <= 20:
                        n_note = r["n_videos"]
                        break
            if n_note is not None:
                f.write("\n")
                f.write("[小样本说明] 当前 TC_FLICKER 评估视频数 N=%d ≤ 20。此时 mean 易受单条异常值（如某条闪烁严重）影响而拉高，不能据此断定 baseline 比 MCFL 更稳定；请优先参考 median。若 median 上 MCFL 与 baseline 接近或更优，说明多数视频时间一致性相当或更好；若仅 mean 差而 median 接近，多为个别失败样本导致。\n" % n_note)
        
        print(f"\n💾 报告已保存到: {self.output_file}")
        print(f"💾 JSON 数据已保存到: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="整合版视频评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有评估指标
  python eval_all.py --real_dir results/1_tacm_/real \\
                     --baseline_dir results/0_tacm_/fake1_30fps \\
                     --mcfl_dir results/1_tacm_/fake1_30fps \\
                     --metrics fvd fid ffc clip av_align tc_flicker

  # 只运行 FVD 和 FID
  python eval_all.py --real_dir results/1_tacm_/real \\
                     --baseline_dir results/0_tacm_/fake1_30fps \\
                     --metrics fvd fid

  # 保存报告到文件
  python eval_all.py --real_dir results/1_tacm_/real \\
                     --baseline_dir results/0_tacm_/fake1_30fps \\
                     --mcfl_dir results/1_tacm_/fake1_30fps \\
                     --metrics fvd fid \\
                     --output report.txt
        """
    )
    
    parser.add_argument("--real_dir", type=str, required=True,
                        help="真实视频目录（groundtruth）")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Baseline 生成视频目录（可选）")
    parser.add_argument("--mcfl_dir", type=str, default=None,
                        help="MCFL 生成视频目录（可选）")
    parser.add_argument("--mcfl2_dir", type=str, default=None,
                        help="MCFL 2.0 生成视频目录（可选，三路对比时使用）")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt",
                        help="提示词文件路径（用于 CLIP 评估）")
    parser.add_argument("--metrics", type=str, nargs="+",
                        choices=["fvd", "fvd_32", "fid", "ffc", "clip", "av_align", "tc_flicker"],
                        default=["fvd", "fid", "ffc", "clip", "av_align", "tc_flicker"],
                        help="要运行的评估指标")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cuda/cpu)")
    parser.add_argument("--no_bootstrap", action="store_true",
                        help="禁用 bootstrap 采样")
    parser.add_argument("--bootstrap_k", type=int, default=10,
                        help="Bootstrap 采样次数")
    parser.add_argument("--no_cache", action="store_true",
                        help="禁用特征缓存")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告文件路径（可选）")
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.real_dir):
        print(f"❌ 真实视频目录不存在: {args.real_dir}")
        return
    
    if args.baseline_dir and not os.path.exists(args.baseline_dir):
        print(f"⚠️  Baseline 目录不存在: {args.baseline_dir}")
        args.baseline_dir = None
    
    if args.mcfl_dir and not os.path.exists(args.mcfl_dir):
        print(f"⚠️  MCFL 目录不存在: {args.mcfl_dir}")
        args.mcfl_dir = None
    if args.mcfl2_dir and not os.path.exists(args.mcfl2_dir):
        print(f"⚠️  MCFL 2.0 目录不存在: {args.mcfl2_dir}")
        args.mcfl2_dir = None
    
    if not args.baseline_dir and not args.mcfl_dir and not args.mcfl2_dir:
        print("❌ 至少需要提供一个生成视频目录（--baseline_dir 或 --mcfl_dir）")
        return
    
    # 创建评估运行器
    runner = EvaluationRunner(
        real_dir=args.real_dir,
        baseline_dir=args.baseline_dir,
        mcfl_dir=args.mcfl_dir,
        mcfl2_dir=args.mcfl2_dir,
        prompt_file=args.prompt_file,
        device=args.device,
        use_bootstrap=not args.no_bootstrap,
        bootstrap_k=args.bootstrap_k,
        use_cache=not args.no_cache,
        output_file=args.output,
    )
    
    # 运行评估
    runner.run_all(args.metrics)
    
    # 打印报告
    runner.print_report()


if __name__ == "__main__":
    main()
