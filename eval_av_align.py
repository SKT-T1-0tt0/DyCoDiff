import os
import cv2
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

VIDEO_DIR = os.getenv("VIDEO_DIR", "results/0_tacm_/fake1_30fps")
# Auto infer AUDIO_DIR from VIDEO_DIR (e.g., results/0_tacm_/fake1_30fps -> results/0_tacm_/audio)
AUDIO_DIR = VIDEO_DIR.replace("/fake1_30fps", "/audio").replace("/fake1_6fps", "/audio")
MAX_LAG = 5  # frames

def motion_energy(video_path):
    cap = cv2.VideoCapture(video_path)
    prev = None
    energy = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = np.abs(gray.astype(np.float32) - prev.astype(np.float32))
            energy.append(diff.mean())
        prev = gray

    cap.release()
    return np.array(energy)

def audio_energy(wav_path, target_len):
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    frame_len = len(audio) // target_len
    return np.array([
        np.mean(np.abs(audio[i*frame_len:(i+1)*frame_len]))
        for i in range(target_len)
    ])

def max_corr(a, b, max_lag):
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

scores = []

for i in tqdm(range(50)):
    v = os.path.join(VIDEO_DIR, f"video_{i}.mp4")
    a = os.path.join(AUDIO_DIR, f"groundtruth_{i}.wav")

    m = motion_energy(v)
    ae = audio_energy(a, len(m))
    scores.append(max_corr(m, ae, MAX_LAG))

print(f"AV-align: {np.mean(scores):.4f} ± {np.std(scores):.4f}")