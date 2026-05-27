"""
Write OmniEncoder state_dict for paper L_vt/L_va training.

Default shapes match TrainLoop: video_dim = in_channels * image_size**2,
text_dim/audio_dim = 768, shared_dim = omni_shared_dim.

This is an initialized projection head (not a pretrained omni model).
Replace the file with your own --omni_encoder_ckpt when available.
"""
import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusion.omni_encoder import OmniEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="saved_ckpts/omni_encoder.pt")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--audio_dim", type=int, default=768)
    p.add_argument("--shared_dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    video_dim = args.in_channels * args.image_size * args.image_size
    enc = OmniEncoder(
        video_dim=video_dim,
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        shared_dim=args.shared_dim,
    )
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(enc.state_dict(), out)
    print(f"Wrote {out} (video_dim={video_dim}, shared_dim={args.shared_dim})")


if __name__ == "__main__":
    main()
