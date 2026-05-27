import os
import re
import cv2
import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image

# =====================
# Config (只改这里即可)
# =====================
BASELINE_DIR = "results/0_tacm_/fake1_30fps"
MCFL_DIR     = "results/1_tacm_/fake1_30fps"
PROMPT_FILE  = "prompts.txt"

NUM_FRAMES = 8        # 每个视频抽多少帧
CLIP_MODEL = "ViT-B/32"

# =====================
# Utils
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)
model.eval()


def sorted_videos(video_dir):
    """Sort video_0.mp4, video_1.mp4, ..."""
    videos = [v for v in os.listdir(video_dir) if v.endswith(".mp4")]
    videos.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
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


@torch.no_grad()
def clip_video_text_score(video_dir, prompts):
    videos = sorted_videos(video_dir)
    assert len(videos) == len(prompts), \
        f"Video count {len(videos)} != prompt count {len(prompts)}"

    # Encode all text once
    text_tokens = clip.tokenize(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    scores = []

    for idx, video_name in enumerate(tqdm(videos, desc=f"Evaluating {video_dir}")):
        video_path = os.path.join(video_dir, video_name)
        frames = sample_frames(video_path, NUM_FRAMES)
        assert len(frames) > 0, f"No frames read from {video_path}"

        img_feats = []
        for frame in frames:
            img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_feats.append(feat)

        # average over frames
        video_feat = torch.mean(torch.cat(img_feats, dim=0), dim=0, keepdim=True)

        score = (video_feat @ text_feats[idx].unsqueeze(1)).item()
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))


# =====================
# Main
# =====================
if __name__ == "__main__":
    prompts = open(PROMPT_FILE).read().strip().splitlines()

    base_mean, base_std = clip_video_text_score(BASELINE_DIR, prompts)
    mcfl_mean, mcfl_std = clip_video_text_score(MCFL_DIR, prompts)

    print("\n========== CLIP Text–Video Similarity ==========")
    print(f"Baseline : {base_mean:.4f} ± {base_std:.4f}")
    print(f"MCFL     : {mcfl_mean:.4f} ± {mcfl_std:.4f}")
    print(f"Δ (MCFL - Baseline): {mcfl_mean - base_mean:+.4f}")
    print("===============================================")
