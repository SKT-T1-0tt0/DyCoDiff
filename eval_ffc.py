import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
import random
import hashlib

# 固定随机种子以确保可复现
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ======================
# Config
# ======================
REAL_DIR     = "results/1_tacm_/real"          # 真实视频（groundtruth）
BASELINE_DIR = "results/0_tacm_/fake1_30fps"  # Baseline 生成视频
MCFL_DIR     = "results/1_tacm_/fake1_30fps"  # MCFL 生成视频

DEVICE = "cuda"

# Bootstrap 配置
BOOTSTRAP_K = 10         # Bootstrap 采样次数
USE_BOOTSTRAP = True     # 是否使用 bootstrap

# 特征缓存配置
CACHE_DIR = "ffc_cache"  # 特征缓存目录
USE_CACHE = True         # 是否使用缓存

# ======================
# Load RAFT Model
# ======================
def load_raft_model(device):
    """Load RAFT optical flow model.
    
    RAFT should be in the RAFT/ directory.
    """
    try:
        import sys
        import argparse
        
        # Add RAFT core to path
        raft_core_path = os.path.join(os.path.dirname(__file__), 'RAFT', 'core')
        if os.path.exists(raft_core_path):
            sys.path.insert(0, raft_core_path)
        else:
            raise ImportError("RAFT/core directory not found")
        
        from raft import RAFT
        
        print("加载 RAFT 模型...")
        
        # Create args object (RAFT requires args)
        # RAFT checks 'dropout' in args and 'alternate_corr' in args
        # So we need to use a dict-like object or implement __contains__
        class Args:
            def __init__(self):
                self.small = False
                self.mixed_precision = False
                self.alternate_corr = False
                self.dropout = 0
            
            def __contains__(self, key):
                # Support 'key in args' checks
                return hasattr(self, key)
        
        args = Args()
        raft_model = RAFT(args)
        
        # Try to load weights
        weight_paths = [
            "RAFT/models/raft-things.pth",
            "RAFT/checkpoints/raft-things.pth",
            "saved_ckpts/raft-things.pth",
            os.path.join(os.path.dirname(__file__), "RAFT", "models", "raft-things.pth"),
        ]
        
        loaded = False
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                print(f"  加载权重: {weight_path}")
                state_dict = torch.load(weight_path, map_location=device)
                
                # 处理 DataParallel 保存的权重（去掉 "module." 前缀）
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                raft_model.load_state_dict(state_dict)
                raft_model.eval()
                raft_model.to(device)
                loaded = True
                break
        
        if not loaded:
            print("  ⚠️  未找到 RAFT 权重文件，使用随机初始化（结果可能不准确）")
            print("  请下载权重文件到以下位置之一：")
            for path in weight_paths:
                print(f"    - {path}")
            print("  或运行: cd RAFT && bash download_models.sh")
        
        raft_model = raft_model.to(device)
        raft_model.eval()
        print("✓ 成功加载 RAFT 模型")
        return raft_model
    except ImportError as e:
        print(f"⚠️  RAFT 导入失败: {e}")
        print("  尝试使用 OpenCV 光流作为替代...")
        return None
    except Exception as e:
        print(f"加载 RAFT 失败: {e}")
        print("⚠️  将使用 OpenCV 光流作为替代（精度较低）")
        return None

print("正在加载光流模型...")
raft_model = load_raft_model(DEVICE)

# ======================
# Utils
# ======================
def warp(x, flow):
    """Warp image x using optical flow.
    
    Args:
        x: [B, C, H, W] input image
        flow: [B, 2, H, W] optical flow
    
    Returns:
        warped: [B, C, H, W] warped image
    """
    B, C, H, W = x.shape
    
    # Create coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
    # Add flow to grid
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    grid = grid + flow
    
    # Normalize to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    
    # Sample
    return torch.nn.functional.grid_sample(x, grid, align_corners=True, mode='bilinear', padding_mode='border')


def compute_flow_opencv(img1, img2):
    """Compute optical flow using OpenCV (fallback when RAFT is not available).
    
    Args:
        img1: [B, C, H, W] first frame
        img2: [B, C, H, W] second frame
    
    Returns:
        flow: [B, 2, H, W] optical flow
    """
    import cv2
    
    # Convert to numpy
    img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
    img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
    
    # Convert to grayscale
    gray1 = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Compute flow using Farneback method
    flow_np = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Convert to tensor [B, 2, H, W]
    flow = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).float().to(img1.device)
    return flow


def get_cache_path(video_dir):
    """Get cache file path for a video directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    normalized_dir = os.path.normpath(video_dir).rstrip("/")
    dir_name = os.path.basename(normalized_dir)
    path_hash = hashlib.md5(normalized_dir.encode()).hexdigest()[:8]
    cache_path = os.path.join(CACHE_DIR, f"{dir_name}_{path_hash}_ffc.npy")
    return cache_path


@torch.no_grad()
def compute_ffc(video_dir, use_cache=None):
    """Compute Frame-to-Frame Consistency (FFC) for videos.
    
    FFC measures temporal consistency by:
    1. Computing optical flow between consecutive frames
    2. Warping first frame using the flow
    3. Computing error between warped frame and second frame
    
    Lower FFC = better temporal consistency
    
    Args:
        video_dir: directory containing video files
        use_cache: whether to use cache (default: USE_CACHE)
    
    Returns:
        mean_ffc: mean FFC score (lower is better)
    """
    if use_cache is None:
        use_cache = USE_CACHE
    
    cache_path = get_cache_path(video_dir)
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        print(f"  📦 Loading FFC scores from cache: {cache_path}")
        scores = np.load(cache_path)
        mean_ffc = np.mean(scores)
        print(f"  ✓ Loaded {len(scores)} FFC scores from cache (mean: {mean_ffc:.4f})")
        return mean_ffc, scores
    
    # Compute FFC
    scores = []
    videos = sorted([v for v in os.listdir(video_dir) if v.endswith(".mp4")])
    if len(videos) == 0:
        raise ValueError(f"No .mp4 files found in {video_dir}")
    
    print(f"  🔄 Computing FFC for {len(videos)} videos...")
    for v in tqdm(videos, desc=f"Computing FFC {video_dir}"):
        reader = imageio.get_reader(os.path.join(video_dir, v), "ffmpeg")
        frames = []
        for frame in reader:
            # Convert to tensor [C, H, W], range [0, 1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE) / 255.0
            frames.append(frame_tensor)
        reader.close()
        
        if len(frames) < 2:
            continue
        
        # Process consecutive frame pairs
        for t in range(len(frames) - 1):
            img1 = frames[t].unsqueeze(0)  # [1, C, H, W]
            img2 = frames[t + 1].unsqueeze(0)  # [1, C, H, W]
            
            # Compute optical flow
            if raft_model is not None:
                # Use RAFT
                try:
                    from utils.utils import InputPadder
                    
                    # RAFT expects images in range [0, 255]
                    img1_raft = img1 * 255.0
                    img2_raft = img2 * 255.0
                    
                    # Pad images to be divisible by 8
                    padder = InputPadder(img1_raft.shape)
                    img1_padded, img2_padded = padder.pad(img1_raft, img2_raft)
                    
                    # Compute flow
                    flow_list = raft_model(img1_padded, img2_padded, iters=20, test_mode=True)
                    flow = flow_list[-1]  # Take the last flow prediction
                    
                    # Unpad flow
                    flow = padder.unpad(flow)
                except Exception as e:
                    print(f"  ⚠️  RAFT 计算失败: {e}，使用 OpenCV 替代")
                    flow = compute_flow_opencv(img1, img2)
            else:
                # Fallback to OpenCV
                flow = compute_flow_opencv(img1, img2)
            
            # Warp first frame using flow
            img1_warped = warp(img1, flow)
            
            # Compute error (L1 distance)
            err = (img1_warped - img2).abs().mean()
            scores.append(err.item())
    
    scores = np.array(scores)
    mean_ffc = np.mean(scores)
    
    # Save to cache
    if use_cache:
        print(f"  💾 Saving FFC scores to cache: {cache_path}")
        np.save(cache_path, scores)
        print(f"  ✓ Cached {len(scores)} FFC scores")
    
    return mean_ffc, scores


def bootstrap_ffc(scores, k=10):
    """Compute FFC with bootstrap sampling for robust statistics.
    
    Args:
        scores: array of FFC scores
        k: number of bootstrap samples
    
    Returns:
        mean_ffc: mean FFC across bootstrap samples
        std_ffc: standard deviation of FFC
    """
    vals = []
    n = len(scores)
    
    for i in range(k):
        # Bootstrap sampling with replacement
        idx = np.random.choice(n, n, replace=True)
        mean_val = np.mean(scores[idx])
        vals.append(mean_val)
    
    return np.mean(vals), np.std(vals)


# ======================
# Main
# ======================
if __name__ == "__main__":
    print("=" * 50)
    print("FFC Evaluation: Frame-to-Frame Consistency")
    print("=" * 50)
    print("(Lower is Better)")
    
    if raft_model is None:
        print("\n⚠️  警告: 使用 OpenCV 光流（精度较低）")
        print("  建议安装 RAFT 以获得更准确的结果:")
        print("    pip install git+https://github.com/princeton-vl/RAFT.git")
        print("    并下载权重文件: raft-things.pth")
    
    # 计算 Baseline FFC
    print("\n[1/3] Computing baseline FFC...")
    ffc_baseline_mean, scores_baseline = compute_ffc(BASELINE_DIR)
    print(f"  ✓ Baseline FFC: {ffc_baseline_mean:.4f} (from {len(scores_baseline)} frame pairs)")
    
    # 计算 MCFL FFC
    print("\n[2/3] Computing MCFL FFC...")
    ffc_mcfl_mean, scores_mcfl = compute_ffc(MCFL_DIR)
    print(f"  ✓ MCFL FFC: {ffc_mcfl_mean:.4f} (from {len(scores_mcfl)} frame pairs)")
    
    # 计算 Real FFC (optional, for reference)
    print("\n[3/3] Computing real video FFC (reference)...")
    try:
        ffc_real_mean, scores_real = compute_ffc(REAL_DIR)
        print(f"  ✓ Real FFC: {ffc_real_mean:.4f} (from {len(scores_real)} frame pairs)")
    except Exception as e:
        print(f"  ⚠️  跳过真实视频: {e}")
        ffc_real_mean = None
    
    # 计算 FFC
    print("\n" + "=" * 50)
    print("Computing FFC scores...")
    print("=" * 50)
    
    if USE_BOOTSTRAP:
        print(f"\n使用 Bootstrap 采样 (k={BOOTSTRAP_K}) 计算 FFD...")
        
        # Bootstrap for Baseline
        print(f"\n[Baseline] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        ffc_baseline_mean_bs, ffc_baseline_std = bootstrap_ffc(scores_baseline, k=BOOTSTRAP_K)
        
        # Bootstrap for MCFL
        print(f"[MCFL] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        ffc_mcfl_mean_bs, ffc_mcfl_std = bootstrap_ffc(scores_mcfl, k=BOOTSTRAP_K)
        
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FFC Results (Lower is Better) - Bootstrap Statistics")
        print("=" * 50)
        print(f"{'Method':<15} {'FFC ↓ (mean ± std)':<25}")
        print("-" * 40)
        print(f"{'Baseline':<15} {ffc_baseline_mean_bs:.4f} ± {ffc_baseline_std:.4f}")
        print(f"{'MCFL':<15} {ffc_mcfl_mean_bs:.4f} ± {ffc_mcfl_std:.4f}")
        if ffc_real_mean is not None:
            print(f"{'Real (ref)':<15} {ffc_real_mean:.4f}")
        print("=" * 50)
        
        # 判断哪个方法更好（基于均值）
        if ffc_mcfl_mean_bs < ffc_baseline_mean_bs:
            improvement = ((ffc_baseline_mean_bs - ffc_mcfl_mean_bs) / ffc_baseline_mean_bs) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif ffc_baseline_mean_bs < ffc_mcfl_mean_bs:
            improvement = ((ffc_mcfl_mean_bs - ffc_baseline_mean_bs) / ffc_mcfl_mean_bs) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
    else:
        # Single FFC computation (no bootstrap)
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FFC Results (Lower is Better)")
        print("=" * 50)
        print(f"{'Method':<15} {'FFC ↓':<15}")
        print("-" * 30)
        print(f"{'Baseline':<15} {ffc_baseline_mean:<15.4f}")
        print(f"{'MCFL':<15} {ffc_mcfl_mean:<15.4f}")
        if ffc_real_mean is not None:
            print(f"{'Real (ref)':<15} {ffc_real_mean:<15.4f}")
        print("=" * 50)
        
        # 判断哪个方法更好
        if ffc_mcfl_mean < ffc_baseline_mean:
            improvement = ((ffc_baseline_mean - ffc_mcfl_mean) / ffc_baseline_mean) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif ffc_baseline_mean < ffc_mcfl_mean:
            improvement = ((ffc_mcfl_mean - ffc_baseline_mean) / ffc_mcfl_mean) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
