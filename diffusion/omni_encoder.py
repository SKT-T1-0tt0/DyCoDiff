import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniEncoder(nn.Module):
    """
    Shared-space encoder for video/text/audio/image collaborative losses.

    Original API:
      - encode_video(...) -> [B, D]
      - encode_text(...)  -> [B, D]
      - encode_audio(...) -> [B, D]
      - encode_image(...) -> [B, D]

    Added API for dynamic alignment:
      - encode_video_sequence(...) -> [B, T, D]
      - encode_audio_sequence(..., flattened_bt=...) -> [B, T, D]

    For encode_audio_sequence, pass flattened_bt=True when x is [B*T, T_audio, D_in]
    (first dim flattened with video frames). Pooling uses 0.25*mean + 0.25*max + 0.5*mid
    per frame. Pass False (default) for [B, T_audio, D_in] (mean-pool then repeat to T).
    """

    def __init__(self, video_dim, text_dim, audio_dim, shared_dim=512, image_dim=None):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.image_proj = nn.Linear(image_dim or text_dim, shared_dim)

    def _reduce_bt_to_b(self, x, sequence_length):
        if sequence_length is None or sequence_length <= 1:
            return x
        if x.shape[0] % sequence_length != 0:
            return x
        b = x.shape[0] // sequence_length
        return x.view(b, sequence_length, -1).mean(dim=1)

    def _reshape_bt_to_b_t(self, x, sequence_length):
        assert sequence_length is not None and sequence_length > 1, (
            "sequence_length must be > 1"
        )
        assert x.shape[0] % sequence_length == 0, (
            f"x.shape[0]={x.shape[0]} not divisible by sequence_length={sequence_length}"
        )
        b = x.shape[0] // sequence_length
        return x.view(b, sequence_length, -1)

    def encode_video(self, x, sequence_length=None):
        # x: [B, C, H, W] or [B*T, C, H, W]
        x = x.reshape(x.shape[0], -1)
        x = self._reduce_bt_to_b(x, sequence_length)
        return F.normalize(self.video_proj(x), dim=-1, eps=1e-6)

    def encode_text(self, x, sequence_length=None):
        # x: [B, N, D] or [B, D]
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self._reduce_bt_to_b(x, sequence_length)
        return F.normalize(self.text_proj(x), dim=-1, eps=1e-6)

    def encode_audio(self, x, sequence_length=None):
        # x: [B, T_seq, D] or [B*T, T_seq, D] or [B, D]
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self._reduce_bt_to_b(x, sequence_length)
        return F.normalize(self.audio_proj(x), dim=-1, eps=1e-6)

    def encode_image(self, x, sequence_length=None):
        # x: [B, 1, D] or [B, D]
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self._reduce_bt_to_b(x, sequence_length)
        return F.normalize(self.image_proj(x), dim=-1, eps=1e-6)

    def encode_video_sequence(self, x, sequence_length):
        """
        x: [B*T, C, H, W]
        return: [B, T, D]
        """
        x = x.reshape(x.shape[0], -1)  # [B*T, F]
        x = self.video_proj(x)  # [B*T, D]
        x = F.normalize(x, dim=-1, eps=1e-6)
        x = self._reshape_bt_to_b_t(x, sequence_length)
        return x

    def encode_audio_sequence(self, x, sequence_length, flattened_bt=False):
        """
        Supported input:
          - [B, D]
          - [B, T_audio, D]  when flattened_bt=False
          - [B*T, T_audio, D] when flattened_bt=True

        Return:
          - [B, T, D_shared]
        """
        assert sequence_length is not None and sequence_length > 1, (
            "encode_audio_sequence requires sequence_length > 1"
        )

        if x.dim() == 2:
            # [B, D] -> repeat along time
            x = self.audio_proj(x)  # [B, D_shared]
            x = F.normalize(x, dim=-1, eps=1e-6)
            x = x.unsqueeze(1).repeat(1, sequence_length, 1)  # [B, T, D_shared]
            return x

        elif x.dim() == 3:
            if flattened_bt:
                bt, t_audio, d_in = x.shape
                assert bt % sequence_length == 0, (
                    f"bt={bt} not divisible by sequence_length={sequence_length}"
                )
                b = bt // sequence_length

                x_mean = x.mean(dim=1)            # [B*T, D_in]
                x_max = x.amax(dim=1)             # [B*T, D_in]
                x_mid = x[:, t_audio // 2, :]     # [B*T, D_in]

                x = 0.25 * x_mean + 0.25 * x_max + 0.5 * x_mid
                x = self.audio_proj(x)            # [B*T, D_shared]
                x = F.normalize(x, dim=-1, eps=1e-6)
                x = x.view(b, sequence_length, -1)  # [B, T, D_shared]
                return x
            else:
                # [B, T_audio, D] -> mean pool over T_audio
                x = x.mean(dim=1)
                x = self.audio_proj(x)
                x = F.normalize(x, dim=-1, eps=1e-6)
                x = x.unsqueeze(1).repeat(1, sequence_length, 1)
                return x
        else:
            raise ValueError(f"Unsupported audio shape: {x.shape}")
