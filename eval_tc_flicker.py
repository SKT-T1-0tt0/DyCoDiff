import os
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_DIR = os.getenv("VIDEO_DIR", "results/0_tacm_/fake1_30fps")

def tc_flicker_revised(video_path, smooth_k=3):
    """
    Revised TC_FLICKER:
    - frame difference
    - remove low-frequency motion via moving average
    - robust median-based energy
    """
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(np.abs(gray - prev).mean())
        prev = gray

    cap.release()
    diffs = np.array(diffs, dtype=np.float32)

    if len(diffs) < smooth_k + 1:
        return 0.0

    # 1️⃣ remove low-frequency motion (temporal smoothing)
    kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
    smooth = np.convolve(diffs, kernel, mode="same")
    high_freq = diffs - smooth

    # 2️⃣ robust flicker energy
    return np.median(high_freq ** 2)


if __name__ == "__main__":
    vals = []
    for i in tqdm(range(50)):
        vals.append(
            tc_flicker_revised(
                os.path.join(VIDEO_DIR, f"video_{i}.mp4"),
                smooth_k=3
            )
        )

    vals = np.array(vals)
    print(f"TC_FLICKER-R (median ↓): {np.median(vals):.6f}")
    print(f"TC_FLICKER-R (mean ↓):   {np.mean(vals):.6f}")
