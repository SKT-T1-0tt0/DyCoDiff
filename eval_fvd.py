import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
from scipy import linalg
import random

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

NUM_FRAMES = 16          # FVD 通常用 16
RESIZE = 224             # I3D 需要至少 224x224 输入
DEVICE = "cuda"

# Bootstrap 配置
BOOTSTRAP_K = 10         # Bootstrap 采样次数
USE_BOOTSTRAP = True     # 是否使用 bootstrap

# 特征缓存配置
CACHE_DIR = "fvd_cache"  # 特征缓存目录
USE_CACHE = True         # 是否使用缓存

# ======================
# Load I3D
# ======================
def load_i3d_model(device):
    """Load I3D model using alternative methods"""
    # Method 1: Try using local directory first (fastest, no download)
    try:
        import sys
        local_i3d_path = "/root/.cache/torch/hub/piergiaj_pytorch-i3d_05783d1"
        if os.path.exists(local_i3d_path):
            print(f"尝试从本地目录加载 I3D 模型: {local_i3d_path}")
            sys.path.insert(0, local_i3d_path)
            from pytorch_i3d import InceptionI3d
            i3d = InceptionI3d(400, in_channels=3).to(device)
            # Load pretrained weights from local file
            weight_path = os.path.join(local_i3d_path, "models", "rgb_imagenet.pt")
            if os.path.exists(weight_path):
                print(f"  加载预训练权重: {weight_path}")
                state_dict = torch.load(weight_path, map_location=device)
                i3d.load_state_dict(state_dict)
                print("  ✓ 成功加载预训练权重")
            else:
                print(f"  ⚠️  权重文件不存在: {weight_path}")
                print("  将使用随机初始化的权重（可能影响 FVD 计算）")
            i3d.eval()
            # 不要修改 logits 层，使用 extract_features + 手动池化
            print("✓ 成功从本地目录加载 I3D 模型")
            return i3d
    except Exception as e:
        print(f"从本地目录加载失败: {e}")
        if 'local_i3d_path' in locals():
            sys.path.remove(local_i3d_path)
    
    # Method 2: Try using torch.hub (will download if needed)
    try:
        print("尝试从 torch.hub 加载 I3D 模型...")
        i3d = torch.hub.load(
            "piergiaj/pytorch-i3d",
            "i3d",
            pretrained=True,
            trust_repo=True
        ).to(device)
        i3d.eval()
        # 不要修改 logits 层，使用 extract_features + 手动池化
        print("✓ 成功从 torch.hub 加载 I3D 模型")
        return i3d
    except Exception as e:
        print(f"从 torch.hub 加载失败: {e}")
    
    # Method 2: Try using pytorchvideo
    try:
        print("尝试从 pytorchvideo 加载 I3D 模型...")
        from pytorchvideo.models.hub import i3d_r50
        model = i3d_r50(pretrained=True)
        model = model.to(device)
        model.eval()
        # Remove classification head
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            model.blocks[-1] = torch.nn.Identity()
        print("成功加载 I3D 模型 (pytorchvideo)")
        return model
    except ImportError:
        print("pytorchvideo 未安装，跳过...")
    except Exception as e:
        print(f"从 pytorchvideo 加载失败: {e}")
    
    # Method 3: Try using the original deepmind repo with trust_repo
    try:
        print("尝试从 deepmind/kinetics-i3d-pytorch 加载 I3D 模型...")
        i3d = torch.hub.load(
            "deepmind/kinetics-i3d-pytorch",
            "i3d",
            pretrained=True,
            trust_repo=True
        ).to(device)
        i3d.eval()
        # 不要修改 logits 层，使用 extract_features + 手动池化
        print("成功加载 I3D 模型 (deepmind/kinetics-i3d-pytorch)")
        return i3d
    except Exception as e:
        print(f"从 deepmind/kinetics-i3d-pytorch 加载失败: {e}")
    
    raise ImportError(
        "无法从任何源加载 I3D 模型。\n"
        "请尝试以下方法之一：\n"
        "1. 安装 pytorchvideo: pip install pytorchvideo\n"
        "2. 确保网络连接正常，以便从 torch.hub 下载模型\n"
        "3. 手动下载 I3D 模型权重"
    )

print("正在加载 I3D 模型...")
i3d = load_i3d_model(DEVICE)

# ======================
# Utils
# ======================
def load_video(path, num_frames=16):
    reader = imageio.get_reader(path, "ffmpeg")
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()

    idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    frames = [frames[i] for i in idxs]

    frames = np.stack(frames)  # [T, H, W, 3]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = torch.nn.functional.interpolate(
        frames, size=(RESIZE, RESIZE), mode="bilinear"
    )
    return frames  # [T, 3, H, W], range [0, 1]


def get_cache_path(video_dir, num_frames=16):
    """Get cache file path for a video directory.
    
    Uses full path hash to ensure uniqueness even if basenames are the same.
    num_frames in filename to separate FVD-16 vs FVD-32 caches.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Normalize the path
    normalized_dir = os.path.normpath(video_dir).rstrip("/")
    
    # Get basename for readability
    dir_name = os.path.basename(normalized_dir)
    
    # Use full path hash to ensure uniqueness (important for BASELINE vs MCFL)
    import hashlib
    path_hash = hashlib.md5(normalized_dir.encode()).hexdigest()[:8]
    
    # Cache file: {basename}_{hash}_f{num_frames}_feats.npy
    cache_path = os.path.join(CACHE_DIR, f"{dir_name}_{path_hash}_f{num_frames}_feats.npy")
    return cache_path


@torch.no_grad()
def extract_features(video_dir, use_cache=None, num_frames=16):
    """Extract I3D features from videos.
    
    Args:
        video_dir: directory containing video files
        use_cache: whether to use cache (default: USE_CACHE)
        num_frames: number of frames to sample (16 for FVD, 32 for FVD-32)
    
    Returns:
        feats: [N, 1024] array where N is number of videos
    """
    if use_cache is None:
        use_cache = USE_CACHE
    
    cache_path = get_cache_path(video_dir, num_frames=num_frames)
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        print(f"  📦 Loading features from cache: {cache_path}")
        feats = np.load(cache_path)
        print(f"  ✓ Loaded {len(feats)} features from cache")
        return feats
    
    # Extract features
    feats = []
    videos = sorted([v for v in os.listdir(video_dir) if v.endswith(".mp4")])
    if len(videos) == 0:
        raise ValueError(f"No .mp4 files found in {video_dir}")
    
    print(f"  🔄 Extracting features from {len(videos)} videos (num_frames={num_frames})...")
    for v in tqdm(videos, desc=f"Extracting {video_dir}"):
        video = load_video(os.path.join(video_dir, v), num_frames=num_frames)  # [T, 3, H, W], range [0, 1]
        video = video.unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]
        video = video.permute(0, 2, 1, 3, 4)   # [1, 3, T, H, W]
        
        # I3D 标准归一化: 映射到 [-1, 1] (非常重要!)
        video = video * 2.0 - 1.0

        # 使用 extract_features + 手动时空池化（标准做法）
        # extract_features 返回 [1, 1024, T, H, W]，不经过 logits 层
        feat = i3d.extract_features(video)  # [1, 1024, T, H, W]
        # 手动进行时空池化: 平均池化 T, H, W 维度
        feat = feat.mean(dim=[2, 3, 4])     # [1, 1024]
        feat = feat.squeeze(0).cpu().numpy()  # [1024]
        
        feats.append(feat)
    
    feats = np.stack(feats)  # [N, 1024]
    
    # Save to cache
    if use_cache:
        print(f"  💾 Saving features to cache: {cache_path}")
        np.save(cache_path, feats)
        print(f"  ✓ Cached {len(feats)} features")
    
    return feats


def compute_fvd(feats1, feats2):
    """Compute Frechet Video Distance (FVD).
    
    Args:
        feats1: [N1, 1024] real video features
        feats2: [N2, 1024] generated video features
    
    Returns:
        fvd: scalar FVD value
    """
    # Check feature dimensions
    assert feats1.shape[1] == feats2.shape[1], f"Feature dim mismatch: {feats1.shape[1]} vs {feats2.shape[1]}"
    
    mu1, sigma1 = feats1.mean(0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(0), np.cov(feats2, rowvar=False)

    # Numerical stability: add small epsilon to diagonal
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    # Check for NaN or invalid values
    if np.isnan(fvd) or np.isinf(fvd):
        raise ValueError(f"Invalid FVD value: {fvd}. Check feature statistics.")
    
    return fvd


def bootstrap_fvd(real, fake, k=10):
    """Compute FVD with bootstrap sampling for robust statistics.
    
    Args:
        real: [N, 1024] real video features
        fake: [N, 1024] generated video features
        k: number of bootstrap samples
    
    Returns:
        mean_fvd: mean FVD across bootstrap samples
        std_fvd: standard deviation of FVD
    """
    assert len(real) == len(fake), f"Real ({len(real)}) and fake ({len(fake)}) must have same length for bootstrap"
    
    vals = []
    n = len(real)
    
    for i in range(k):
        # Bootstrap sampling with replacement
        idx = np.random.choice(n, n, replace=True)
        fvd_val = compute_fvd(real[idx], fake[idx])
        vals.append(fvd_val)
    
    return np.mean(vals), np.std(vals)


# ======================
# Main
# ======================
if __name__ == "__main__":
    print("=" * 50)
    print("FVD Evaluation: Generated vs Real")
    print("=" * 50)
    
    # 提取真实视频特征
    print("\n[1/3] Extracting real video features...")
    feats_real = extract_features(REAL_DIR)
    print(f"  ✓ Extracted features from {len(feats_real)} real videos (shape: {feats_real.shape})")
    
    # 提取 Baseline 生成视频特征
    print("\n[2/3] Extracting baseline (generated) features...")
    feats_base = extract_features(BASELINE_DIR)
    print(f"  ✓ Extracted features from {len(feats_base)} baseline videos (shape: {feats_base.shape})")
    
    # 提取 MCFL 生成视频特征
    print("\n[3/3] Extracting MCFL (generated) features...")
    feats_mcfl = extract_features(MCFL_DIR)
    print(f"  ✓ Extracted features from {len(feats_mcfl)} MCFL videos (shape: {feats_mcfl.shape})")
    
    # 检查视频数量一致性
    print("\n" + "=" * 50)
    print("Checking video count consistency...")
    print("=" * 50)
    if len(feats_real) != len(feats_base):
        print(f"⚠️  警告: Real ({len(feats_real)}) 和 Baseline ({len(feats_base)}) 数量不一致")
    if len(feats_real) != len(feats_mcfl):
        print(f"⚠️  警告: Real ({len(feats_real)}) 和 MCFL ({len(feats_mcfl)}) 数量不一致")
    if len(feats_base) != len(feats_mcfl):
        print(f"⚠️  警告: Baseline ({len(feats_base)}) 和 MCFL ({len(feats_mcfl)}) 数量不一致")
    
    # 对齐数量（取最小值）
    min_count = min(len(feats_real), len(feats_base), len(feats_mcfl))
    if min_count < len(feats_real) or min_count < len(feats_base) or min_count < len(feats_mcfl):
        print(f"\n📊 对齐到最小数量: {min_count} 个视频")
        feats_real = feats_real[:min_count]
        feats_base = feats_base[:min_count]
        feats_mcfl = feats_mcfl[:min_count]
    
    # 计算 FVD
    print("\n" + "=" * 50)
    print("Computing FVD scores...")
    print("=" * 50)
    
    if USE_BOOTSTRAP:
        print(f"\n使用 Bootstrap 采样 (k={BOOTSTRAP_K}) 计算 FVD...")
        
        # Bootstrap for Baseline
        print(f"\n[Baseline] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        fvd_baseline_mean, fvd_baseline_std = bootstrap_fvd(feats_real, feats_base, k=BOOTSTRAP_K)
        
        # Bootstrap for MCFL
        print(f"[MCFL] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        fvd_mcfl_mean, fvd_mcfl_std = bootstrap_fvd(feats_real, feats_mcfl, k=BOOTSTRAP_K)
        
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FVD Results (Lower is Better) - Bootstrap Statistics")
        print("=" * 50)
        print(f"{'Method':<15} {'FVD ↓ (mean ± std)':<25}")
        print("-" * 40)
        print(f"{'Baseline':<15} {fvd_baseline_mean:.2f} ± {fvd_baseline_std:.2f}")
        print(f"{'MCFL':<15} {fvd_mcfl_mean:.2f} ± {fvd_mcfl_std:.2f}")
        print("=" * 50)
        
        # 判断哪个方法更好（基于均值）
        if fvd_mcfl_mean < fvd_baseline_mean:
            improvement = ((fvd_baseline_mean - fvd_mcfl_mean) / fvd_baseline_mean) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif fvd_baseline_mean < fvd_mcfl_mean:
            improvement = ((fvd_mcfl_mean - fvd_baseline_mean) / fvd_mcfl_mean) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
    else:
        # Single FVD computation (no bootstrap)
        fvd_baseline = compute_fvd(feats_real, feats_base)
        fvd_mcfl = compute_fvd(feats_real, feats_mcfl)
        
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FVD Results (Lower is Better)")
        print("=" * 50)
        print(f"{'Method':<15} {'FVD ↓':<15}")
        print("-" * 30)
        print(f"{'Baseline':<15} {fvd_baseline:<15.2f}")
        print(f"{'MCFL':<15} {fvd_mcfl:<15.2f}")
        print("=" * 50)
        
        # 判断哪个方法更好
        if fvd_mcfl < fvd_baseline:
            improvement = ((fvd_baseline - fvd_mcfl) / fvd_baseline) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif fvd_baseline < fvd_mcfl:
            improvement = ((fvd_mcfl - fvd_baseline) / fvd_mcfl) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
