import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
from scipy import linalg
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

NUM_FRAMES = 16          # FID 通常用 16 帧
RESIZE = 299             # Inception 需要 299x299 输入
DEVICE = "cuda"

# Bootstrap 配置
BOOTSTRAP_K = 10         # Bootstrap 采样次数
USE_BOOTSTRAP = True     # 是否使用 bootstrap

# 特征缓存配置
CACHE_DIR = "fid_cache"  # 特征缓存目录
USE_CACHE = True         # 是否使用缓存

# ======================
# Load FID Inception Model
# ======================
def load_fid_inception_model(device):
    """Load FID Inception model (not standard InceptionV3).
    
    FID requires a specific Inception model with different weights
    than standard torchvision InceptionV3.
    """
    try:
        # Try to use the project's FID Inception implementation
        import sys
        sys.path.insert(0, "calculation/fvd_fid")
        from inception import fid_inception_v3
        
        print("加载 FID Inception 模型...")
        model = fid_inception_v3()
        model = model.to(device)
        model.eval()
        print("✓ 成功加载 FID Inception 模型")
        return model
    except Exception as e:
        print(f"从项目代码加载失败: {e}")
        print("尝试使用标准 InceptionV3（结果可能不可比）...")
        
        # Fallback to standard InceptionV3 (not recommended for FID)
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.aux_logits = False
        model = model.to(device)
        model.eval()
        print("⚠️  使用标准 InceptionV3（FID 结果可能不准确）")
        return model

print("正在加载 FID Inception 模型...")
inception_model = load_fid_inception_model(DEVICE)

# ======================
# Utils
# ======================
def load_video(path, num_frames=16):
    """Load video frames.
    
    Returns:
        frames: [T, 3, H, W] tensor, range [0, 1]
    """
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


def get_cache_path(video_dir):
    """Get cache file path for a video directory.
    
    Uses full path hash to ensure uniqueness even if basenames are the same.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Normalize the path
    normalized_dir = os.path.normpath(video_dir).rstrip("/")
    
    # Get basename for readability
    dir_name = os.path.basename(normalized_dir)
    
    # Use full path hash to ensure uniqueness
    path_hash = hashlib.md5(normalized_dir.encode()).hexdigest()[:8]
    
    # Cache file: {basename}_{hash}_feats.npy
    cache_path = os.path.join(CACHE_DIR, f"{dir_name}_{path_hash}_feats.npy")
    return cache_path


@torch.no_grad()
def extract_features(video_dir, use_cache=None):
    """Extract Inception features from videos.
    
    For FID, we extract features from each frame independently,
    then stack them. The final feature shape is [N*NUM_FRAMES, 2048].
    
    Args:
        video_dir: directory containing video files
        use_cache: whether to use cache (default: USE_CACHE)
    
    Returns:
        feats: [N*NUM_FRAMES, 2048] array where N is number of videos
    """
    if use_cache is None:
        use_cache = USE_CACHE
    
    cache_path = get_cache_path(video_dir)
    
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
    
    print(f"  🔄 Extracting features from {len(videos)} videos...")
    for v in tqdm(videos, desc=f"Extracting {video_dir}"):
        frames = load_video(os.path.join(video_dir, v))  # [T, 3, H, W], range [0, 1]
        
        # Process each frame
        for frame in frames:
            # Inception expects [0, 1] range, normalize to [-1, 1]
            frame = frame.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
            frame = frame * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Extract features
            feat = inception_model(frame)  # [1, 2048]
            feat = feat.squeeze(0).cpu().numpy()  # [2048]
            feats.append(feat)
    
    feats = np.stack(feats)  # [N*NUM_FRAMES, 2048]
    
    # Save to cache
    if use_cache:
        print(f"  💾 Saving features to cache: {cache_path}")
        np.save(cache_path, feats)
        print(f"  ✓ Cached {len(feats)} features")
    
    return feats


def compute_fid(feats1, feats2):
    """Compute Frechet Inception Distance (FID).
    
    Args:
        feats1: [N1, 2048] real video features
        feats2: [N2, 2048] generated video features
    
    Returns:
        fid: scalar FID value
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

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    # Check for NaN or invalid values
    if np.isnan(fid) or np.isinf(fid):
        raise ValueError(f"Invalid FID value: {fid}. Check feature statistics.")
    
    return fid


def bootstrap_fid(real, fake, k=10):
    """Compute FID with bootstrap sampling for robust statistics.
    
    Args:
        real: [N, 2048] real video features
        fake: [N, 2048] generated video features
        k: number of bootstrap samples
    
    Returns:
        mean_fid: mean FID across bootstrap samples
        std_fid: standard deviation of FID
    """
    assert len(real) == len(fake), f"Real ({len(real)}) and fake ({len(fake)}) must have same length for bootstrap"
    
    vals = []
    n = len(real)
    
    for i in range(k):
        # Bootstrap sampling with replacement
        idx = np.random.choice(n, n, replace=True)
        fid_val = compute_fid(real[idx], fake[idx])
        vals.append(fid_val)
    
    return np.mean(vals), np.std(vals)


# ======================
# Main
# ======================
if __name__ == "__main__":
    print("=" * 50)
    print("FID Evaluation: Generated vs Real")
    print("=" * 50)
    
    # 提取真实视频特征
    print("\n[1/3] Extracting real video features...")
    feats_real = extract_features(REAL_DIR)
    print(f"  ✓ Extracted features from {len(feats_real)} frames (shape: {feats_real.shape})")
    print(f"  → Approximately {len(feats_real) // NUM_FRAMES} videos")
    
    # 提取 Baseline 生成视频特征
    print("\n[2/3] Extracting baseline (generated) features...")
    feats_base = extract_features(BASELINE_DIR)
    print(f"  ✓ Extracted features from {len(feats_base)} frames (shape: {feats_base.shape})")
    print(f"  → Approximately {len(feats_base) // NUM_FRAMES} videos")
    
    # 提取 MCFL 生成视频特征
    print("\n[3/3] Extracting MCFL (generated) features...")
    feats_mcfl = extract_features(MCFL_DIR)
    print(f"  ✓ Extracted features from {len(feats_mcfl)} frames (shape: {feats_mcfl.shape})")
    print(f"  → Approximately {len(feats_mcfl) // NUM_FRAMES} videos")
    
    # 检查特征数量一致性
    print("\n" + "=" * 50)
    print("Checking feature count consistency...")
    print("=" * 50)
    if len(feats_real) != len(feats_base):
        print(f"⚠️  警告: Real ({len(feats_real)}) 和 Baseline ({len(feats_base)}) 特征数量不一致")
    if len(feats_real) != len(feats_mcfl):
        print(f"⚠️  警告: Real ({len(feats_real)}) 和 MCFL ({len(feats_mcfl)}) 特征数量不一致")
    if len(feats_base) != len(feats_mcfl):
        print(f"⚠️  警告: Baseline ({len(feats_base)}) 和 MCFL ({len(feats_mcfl)}) 特征数量不一致")
    
    # 对齐数量（取最小值）
    min_count = min(len(feats_real), len(feats_base), len(feats_mcfl))
    if min_count < len(feats_real) or min_count < len(feats_base) or min_count < len(feats_mcfl):
        print(f"\n📊 对齐到最小数量: {min_count} 个特征")
        feats_real = feats_real[:min_count]
        feats_base = feats_base[:min_count]
        feats_mcfl = feats_mcfl[:min_count]
    
    # 计算 FID
    print("\n" + "=" * 50)
    print("Computing FID scores...")
    print("=" * 50)
    
    if USE_BOOTSTRAP:
        print(f"\n使用 Bootstrap 采样 (k={BOOTSTRAP_K}) 计算 FID...")
        
        # Bootstrap for Baseline
        print(f"\n[Baseline] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        fid_baseline_mean, fid_baseline_std = bootstrap_fid(feats_real, feats_base, k=BOOTSTRAP_K)
        
        # Bootstrap for MCFL
        print(f"[MCFL] 进行 {BOOTSTRAP_K} 次 bootstrap 采样...")
        fid_mcfl_mean, fid_mcfl_std = bootstrap_fid(feats_real, feats_mcfl, k=BOOTSTRAP_K)
        
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FID Results (Lower is Better) - Bootstrap Statistics")
        print("=" * 50)
        print(f"{'Method':<15} {'FID ↓ (mean ± std)':<25}")
        print("-" * 40)
        print(f"{'Baseline':<15} {fid_baseline_mean:.2f} ± {fid_baseline_std:.2f}")
        print(f"{'MCFL':<15} {fid_mcfl_mean:.2f} ± {fid_mcfl_std:.2f}")
        print("=" * 50)
        
        # 判断哪个方法更好（基于均值）
        if fid_mcfl_mean < fid_baseline_mean:
            improvement = ((fid_baseline_mean - fid_mcfl_mean) / fid_baseline_mean) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif fid_baseline_mean < fid_mcfl_mean:
            improvement = ((fid_mcfl_mean - fid_baseline_mean) / fid_mcfl_mean) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
    else:
        # Single FID computation (no bootstrap)
        fid_baseline = compute_fid(feats_real, feats_base)
        fid_mcfl = compute_fid(feats_real, feats_mcfl)
        
        # 输出结果表格
        print("\n" + "=" * 50)
        print("FID Results (Lower is Better)")
        print("=" * 50)
        print(f"{'Method':<15} {'FID ↓':<15}")
        print("-" * 30)
        print(f"{'Baseline':<15} {fid_baseline:<15.2f}")
        print(f"{'MCFL':<15} {fid_mcfl:<15.2f}")
        print("=" * 50)
        
        # 判断哪个方法更好
        if fid_mcfl < fid_baseline:
            improvement = ((fid_baseline - fid_mcfl) / fid_baseline) * 100
            print(f"\n✓ MCFL 优于 Baseline (提升 {improvement:.1f}%)")
        elif fid_baseline < fid_mcfl:
            improvement = ((fid_mcfl - fid_baseline) / fid_mcfl) * 100
            print(f"\n✓ Baseline 优于 MCFL (提升 {improvement:.1f}%)")
        else:
            print(f"\n- Baseline 和 MCFL 性能相同")
