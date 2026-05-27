"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import os
import re
from collections import deque

import numpy as np
import torch as th
import torch.nn.functional as F

from .losses import normal_kl, discretized_gaussian_log_likelihood
from .tacm_nn import mean_flat


class _ObiAttnCacheList(list):
    """
    attn_cache collector for OBI: needs a running index on the same object that
    receives .append() from CrossAttention (plain list cannot hold _obi_next_idx).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._obi_next_idx = 0


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        collab_dyn_rank_mode: str = "hybrid",
        collab_dyn_rank_low_q: float = 0.2,
        collab_dyn_rank_high_q: float = 0.8,
        collab_dyn_rank_sharpness: float = 10.0,
        collab_dyn_hist_size: int = 256,
        collab_dyn_score_mix_av_ratio: float = 0.7,
        collab_dyn_score_mix_audio: float = 0.3,
        collab_av_score_mix_av_ratio: float = 0.5,
        collab_av_score_mix_av_conf: float = 0.5,
        collab_av_conf_beta: float = 0.5,
        collab_av_conf_sim_low: float = 0.0,
        collab_av_conf_sim_high: float = 0.3,
        collab_audio_ema_alpha: float = 0.9,
        collab_obi_scale: float = 0.0,
        collab_obi_start_step: int = 0,
        collab_obi_warmup_steps: int = 10000,
        # OBI 取层：temporal_last | temporal_first | temporal_index_<k> | temporal_all_mean
        collab_obi_layer_name: str = "temporal_last",
        collab_obi_min_weight: float = 0.05,
        collab_obi_min_scale: float = 0.1,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        self.collab_dyn_rank_mode = str(collab_dyn_rank_mode)
        self.collab_dyn_rank_low_q = float(collab_dyn_rank_low_q)
        self.collab_dyn_rank_high_q = float(collab_dyn_rank_high_q)
        self.collab_dyn_rank_sharpness = float(collab_dyn_rank_sharpness)
        self.collab_dyn_hist_size = int(collab_dyn_hist_size)
        self.collab_dyn_score_mix_av_ratio = float(collab_dyn_score_mix_av_ratio)
        self.collab_dyn_score_mix_audio = float(collab_dyn_score_mix_audio)
        self.collab_av_score_mix_av_ratio = float(collab_av_score_mix_av_ratio)
        self.collab_av_score_mix_av_conf = float(collab_av_score_mix_av_conf)
        self.collab_av_conf_beta = float(collab_av_conf_beta)
        self.collab_av_conf_sim_low = float(collab_av_conf_sim_low)
        self.collab_av_conf_sim_high = float(collab_av_conf_sim_high)
        self.collab_audio_ema_alpha = float(collab_audio_ema_alpha)
        self.collab_obi_scale = float(collab_obi_scale)
        self.collab_obi_start_step = int(collab_obi_start_step)
        self.collab_obi_warmup_steps = int(collab_obi_warmup_steps)
        self.collab_obi_layer_name = str(collab_obi_layer_name)
        self.collab_obi_min_weight = float(collab_obi_min_weight)
        self.collab_obi_min_scale = float(collab_obi_min_scale)

        # Running history for batch_size=1 friendly reliability.
        self._dyn_score_history = deque(maxlen=max(8, self.collab_dyn_hist_size))

    def _safe_mean(self, x):
        if x.numel() == 0:
            return th.tensor(0.0, device=x.device if hasattr(x, "device") else "cpu")
        return x.mean()

    def _linear_warmup_scale(self, step, start_step, warmup_steps):
        if warmup_steps <= 0:
            return 1.0 if step >= start_step else 0.0
        if step < start_step:
            return 0.0
        return min(1.0, float(step - start_step) / float(warmup_steps))

    def _compute_static_collab_loss(self, x, y, metric="cosine"):
        metric = str(metric).strip().lower()
        if metric == "cosine":
            return 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
        if metric == "mse":
            return ((x - y) ** 2).mean()
        if metric == "mae":
            return (x - y).abs().mean()
        raise ValueError(
            f"Unsupported static collab metric: {metric}. "
            "Expected one of {'cosine', 'mse', 'mae'}."
        )

    def _history_rank_reliability(
        self,
        score,
        low_q=0.2,
        high_q=0.8,
        sharpness=10.0,
        mode="hybrid",
    ):
        """
        History-based rank/quantile reliability for batch_size=1 training.

        Args:
            score: [B] tensor, current per-sample score
            low_q: lower quantile anchor from history
            high_q: upper quantile anchor from history
            sharpness: sigmoid sharpness when mode in {"sigmoid", "hybrid"}
            mode: one of {"linear", "sigmoid", "hybrid"}

        Returns:
            reliability: [B] tensor in [0, 1]
            aux: dict with q_low, q_high, rank_norm
        """
        assert score.dim() == 1, f"score must be [B], got {score.shape}"

        device = score.device
        dtype = score.dtype

        hist = list(self._dyn_score_history)
        cur_vals = score.detach().flatten().tolist()

        # Bootstrap phase: no meaningful history yet.
        if len(hist) < 8:
            reliability = th.full_like(score, 0.5)
            for s in cur_vals:
                self._dyn_score_history.append(float(s))
            return reliability, {
                "q_low": score.mean(),
                "q_high": score.mean(),
                "rank_norm": th.full_like(score, 0.5),
            }

        hist_tensor = th.tensor(hist, device=device, dtype=dtype)

        q = th.tensor([low_q, high_q], device=device, dtype=dtype)
        q_low, q_high = th.quantile(hist_tensor, q)

        denom = (q_high - q_low).clamp(min=1e-6)
        score_norm = ((score - q_low) / denom).clamp(0.0, 1.0)

        # Percentile rank against history, in [0, 1]
        rank_norm = (hist_tensor.unsqueeze(0) <= score.unsqueeze(1)).to(dtype).mean(dim=1)

        if mode == "linear":
            reliability = score_norm
        elif mode == "sigmoid":
            reliability = th.sigmoid((score_norm - 0.5) * sharpness)
        elif mode == "hybrid":
            mixed = 0.5 * score_norm + 0.5 * rank_norm
            reliability = th.sigmoid((mixed - 0.5) * sharpness)
        else:
            raise ValueError(f"Unsupported history rank mode: {mode}")

        for s in cur_vals:
            self._dyn_score_history.append(float(s))

        return reliability, {
            "q_low": q_low,
            "q_high": q_high,
            "rank_norm": rank_norm,
        }

    def _ema_sequence(self, x, alpha=0.9):
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        assert x.dim() == 3, f"x must be [B,T,D], got {x.shape}"
        alpha = float(alpha)

        if x.shape[1] <= 1 or alpha <= 0.0:
            return x

        out = th.zeros_like(x)
        out[:, 0, :] = x[:, 0, :]
        for i in range(1, x.shape[1]):
            out[:, i, :] = alpha * out[:, i - 1, :] + (1.0 - alpha) * x[:, i, :]
        return out

    def _compute_av_confidence(
        self,
        z_v_seq,
        z_a_seq,
        sim_low=0.0,
        sim_high=0.3,
        beta=0.5,
        eps=1e-6,
    ):
        """
        Audio-visual agreement confidence from pooled sequence embeddings.

        Args:
            z_v_seq: [B, T, D]
            z_a_seq: [B, T, D]

        Returns:
            av_conf: [B] in [0, 1]
            av_sim: [B] raw cosine sim
        """
        assert z_v_seq.dim() == 3 and z_a_seq.dim() == 3
        assert z_v_seq.shape == z_a_seq.shape, (
            f"z_v_seq and z_a_seq shape mismatch: {z_v_seq.shape} vs {z_a_seq.shape}"
        )

        v_pool = F.normalize(z_v_seq.mean(dim=1), dim=-1, eps=eps)
        a_pool = F.normalize(z_a_seq.mean(dim=1), dim=-1, eps=eps)

        av_sim = F.cosine_similarity(v_pool, a_pool, dim=-1)

        denom = max(float(sim_high - sim_low), 1e-6)
        av_conf = ((av_sim - float(sim_low)) / denom).clamp(0.0, 1.0)
        av_conf = ((1.0 - float(beta)) + float(beta) * av_conf).clamp(0.0, 1.0)

        return av_conf, av_sim

    def _compute_dynamic_reliability_and_loss(
        self,
        z_v_seq,
        z_a_seq,
        device,
        dtype,
        dyn_stride=1,
        tau_sample=0.03,
        temp_sample=0.01,
        gate_k=2.0,
        dyn_loss_beta=3.0,
        eps=1e-6,
    ):
        """
        Dynamic alignment with:
          - audio delta detach
          - z_a_seq EMA smoothing before da_raw
          - frame-level valid + gate
          - AV agreement + av_ratio sample score + rank-hist reliability
          - robust log1p dynamic loss
        """
        assert z_v_seq.dim() == 3 and z_a_seq.dim() == 3
        assert z_v_seq.shape[:2] == z_a_seq.shape[:2], (
            f"Dynamic collab seq mismatch: z_v_seq={z_v_seq.shape}, z_a_seq={z_a_seq.shape}"
        )

        b, t, _ = z_v_seq.shape
        dyn_stride = max(1, int(dyn_stride))

        if t <= dyn_stride:
            zero = th.zeros((), device=device, dtype=dtype)
            return {
                "loss_va": zero,
                "dyn_cos_mean": zero,
                "dyn_video_mag": zero,
                "dyn_audio_mag": zero,
                "dyn_valid_mean": zero,
                "dyn_gate_mean": zero,
                "dyn_weight_mean": zero,
                "dyn_sample_av_ratio": zero,
                "dyn_sample_on_ratio": zero,
                "dyn_sample_video_strength": zero,
                "dyn_sample_audio_strength": zero,
                "dyn_sample_score_mean": zero,
                "dyn_rank_q_low": zero,
                "dyn_rank_q_high": zero,
                "dyn_rank_mean": zero,
                "dyn_av_conf_mean": zero,
                "dyn_av_sim_mean": zero,
                "dyn_stride": th.as_tensor(float(dyn_stride), device=device, dtype=dtype),
                "sample_reliability": th.zeros((b,), device=device, dtype=dtype),
                "frame_weight": th.zeros((b, 0), device=device, dtype=dtype),
                "z_a_seq_smooth": z_a_seq,
            }

        audio_ema_alpha = float(getattr(self, "collab_audio_ema_alpha", 0.9))
        z_a_seq_smooth = self._ema_sequence(z_a_seq, alpha=audio_ema_alpha)

        dv_raw = z_v_seq[:, dyn_stride:, :] - z_v_seq[:, :-dyn_stride, :]
        da_raw = (
            z_a_seq_smooth[:, dyn_stride:, :] - z_a_seq_smooth[:, :-dyn_stride, :]
        ).detach()

        mag_v = dv_raw.norm(dim=-1)
        mag_a = da_raw.norm(dim=-1)

        mag_a_det = mag_a.detach()
        mag_a_mean = mag_a_det.mean(dim=1, keepdim=True)
        mag_a_std = mag_a_det.std(dim=1, keepdim=True, unbiased=False) + eps
        thr = mag_a_mean * 0.8

        valid = (mag_a_det > thr).to(dtype=dtype)
        gate = th.sigmoid(((mag_a_det - thr) / mag_a_std) * gate_k)
        frame_weight = gate * valid

        av_beta = float(getattr(self, "collab_av_conf_beta", 0.5))
        av_sim_low = float(getattr(self, "collab_av_conf_sim_low", 0.0))
        av_sim_high = float(getattr(self, "collab_av_conf_sim_high", 0.3))

        av_conf, av_sim = self._compute_av_confidence(
            z_v_seq=z_v_seq,
            z_a_seq=z_a_seq_smooth,
            sim_low=av_sim_low,
            sim_high=av_sim_high,
            beta=av_beta,
            eps=eps,
        )

        sample_av_ratio = valid.mean(dim=1)
        sample_video_strength = mag_v.mean(dim=1)
        sample_audio_strength = mag_a_det.mean(dim=1)

        mix_ratio = float(getattr(self, "collab_av_score_mix_av_ratio", 0.5))
        mix_conf = float(getattr(self, "collab_av_score_mix_av_conf", 0.5))
        mix_sum = max(mix_ratio + mix_conf, 1e-6)
        mix_ratio /= mix_sum
        mix_conf /= mix_sum

        # Single sample_score for rank-hist (AV mix only; no audio_strength_norm).
        sample_score = mix_ratio * sample_av_ratio + mix_conf * av_conf

        rank_mode = str(getattr(self, "collab_dyn_rank_mode", "hybrid")).lower()
        rank_low_q = float(getattr(self, "collab_dyn_rank_low_q", 0.2))
        rank_high_q = float(getattr(self, "collab_dyn_rank_high_q", 0.8))
        rank_sharpness = float(getattr(self, "collab_dyn_rank_sharpness", 10.0))

        sample_reliability, rank_aux = self._history_rank_reliability(
            score=sample_score.detach(),
            low_q=rank_low_q,
            high_q=rank_high_q,
            sharpness=rank_sharpness,
            mode=rank_mode,
        )

        full_weight = frame_weight * (0.5 + 0.5 * sample_reliability.unsqueeze(-1))

        dv = F.normalize(dv_raw, dim=-1, eps=eps)
        da = F.normalize(da_raw, dim=-1, eps=eps)

        dyn_cos = F.cosine_similarity(dv, da, dim=-1)
        loss_dyn_raw = 1.0 - dyn_cos

        if dyn_loss_beta > 0:
            denom = math.log1p(float(dyn_loss_beta))
            loss_dyn_robust = th.log1p(float(dyn_loss_beta) * loss_dyn_raw) / denom
        else:
            loss_dyn_robust = loss_dyn_raw

        weight_sum = full_weight.sum()
        if weight_sum.item() > 0:
            loss_va = (full_weight * loss_dyn_robust).sum() / (weight_sum + eps)
        else:
            loss_va = th.zeros((), device=device, dtype=dtype)

        return {
            "loss_va": loss_va,
            "dyn_cos_mean": dyn_cos.mean(),
            "dyn_video_mag": mag_v.mean(),
            "dyn_audio_mag": mag_a_det.mean(),
            "dyn_valid_mean": valid.mean(),
            "dyn_gate_mean": gate.mean(),
            "dyn_weight_mean": full_weight.mean(),
            "dyn_sample_av_ratio": sample_av_ratio.mean(),
            "dyn_sample_on_ratio": sample_reliability.mean(),
            "dyn_sample_video_strength": sample_video_strength.mean(),
            "dyn_sample_audio_strength": sample_audio_strength.mean(),
            "dyn_sample_score_mean": sample_score.mean(),
            "dyn_rank_q_low": rank_aux["q_low"],
            "dyn_rank_q_high": rank_aux["q_high"],
            "dyn_rank_mean": rank_aux["rank_norm"].mean(),
            "dyn_av_conf_mean": av_conf.mean(),
            "dyn_av_sim_mean": av_sim.mean(),
            "dyn_stride": th.as_tensor(float(dyn_stride), device=device, dtype=dtype),
            "sample_reliability": sample_reliability,
            "frame_weight": full_weight,
            "z_a_seq_smooth": z_a_seq_smooth,
        }

    def baseline_pred_fn(self, baseline_model, x_t, t, model_kwargs):
        if baseline_model is None:
            raise ValueError("baseline_model is None")
        if model_kwargs is None:
            model_kwargs = {}
        if "c" not in model_kwargs:
            raise KeyError("model_kwargs must contain 'c' for baseline teacher forward")
        if "c_temp" not in model_kwargs:
            raise KeyError("model_kwargs must contain 'c_temp' for baseline teacher forward")

        c = model_kwargs["c"]
        c_temp = model_kwargs["c_temp"]
        extra_model_kwargs = {
            k: v for k, v in model_kwargs.items() if k not in ["c", "c_temp"]
        }

        model_output = baseline_model(
            x_t,
            self._scale_timesteps(t),
            c,
            c_temp,
            **extra_model_kwargs,
        )

        denoise_pred = model_output
        if self.model_var_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:
            bsz, ch = x_t.shape[:2]
            assert model_output.shape == (bsz, ch * 2, *x_t.shape[2:]), (
                f"Teacher model_output shape mismatch under learned variance: "
                f"got {model_output.shape}, expected {(bsz, ch * 2, *x_t.shape[2:])}"
            )
            model_output, _ = th.split(model_output, ch, dim=1)
            denoise_pred = model_output

        if self.model_mean_type == ModelMeanType.EPSILON:
            x0_pred_teacher = self._predict_xstart_from_eps(
                x_t=x_t,
                t=t,
                eps=denoise_pred,
            )
        elif self.model_mean_type == ModelMeanType.START_X:
            x0_pred_teacher = denoise_pred
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            x0_pred_teacher = self._predict_xstart_from_xprev(
                x_t=x_t,
                t=t,
                xprev=denoise_pred,
            )
        else:
            raise NotImplementedError(
                f"Unsupported model_mean_type for teacher: {self.model_mean_type}"
            )

        return x0_pred_teacher

    def _compute_online_baseline_imitation_loss(
        self,
        x0_pred_student,
        x_t,
        t,
        model_kwargs,
        sample_reliability,
        baseline_model=None,
        baseline_pred_fn=None,
        eps=1e-6,
    ):
        device = x0_pred_student.device
        dtype = x0_pred_student.dtype

        if baseline_model is None or baseline_pred_fn is None:
            zero = th.zeros((), device=device, dtype=dtype)
            return zero, None

        with th.no_grad():
            x0_pred_teacher = baseline_pred_fn(
                baseline_model=baseline_model,
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
            )

        if x0_pred_teacher.shape != x0_pred_student.shape:
            raise ValueError(
                f"Teacher/student x0 shape mismatch: "
                f"teacher={x0_pred_teacher.shape}, student={x0_pred_student.shape}"
            )

        flat = (x0_pred_student - x0_pred_teacher) ** 2
        per_item = flat.reshape(flat.shape[0], -1).mean(dim=1)

        if sample_reliability is not None:
            if per_item.shape[0] != sample_reliability.shape[0]:
                seq_len = int(getattr(self, "sequence_length", 1))
                if seq_len > 1 and per_item.shape[0] == sample_reliability.shape[0] * seq_len:
                    sample_reliability = sample_reliability.repeat_interleave(seq_len)
                else:
                    raise ValueError(
                        f"Cannot align reliability with imitation loss: "
                        f"per_item={per_item.shape}, reliability={sample_reliability.shape}, "
                        f"seq_len={seq_len}"
                    )

            imit_weight = 1.0 - sample_reliability.detach()
            loss_imit = (imit_weight * per_item).sum() / (imit_weight.sum() + eps)
        else:
            loss_imit = per_item.mean()

        return loss_imit, per_item.mean()

    @staticmethod
    def _unwrap_ddp(m):
        return m.module if hasattr(m, "module") else m

    @staticmethod
    def _collect_temporal_cross_attn_tensors(cache):
        """All attn2_temporal_cross tensors in forward append order."""
        if cache is None or not isinstance(cache, (list, tuple)):
            return []
        out = []
        for item in cache:
            if (
                isinstance(item, dict)
                and item.get("source") == "attn2_temporal_cross"
                and th.is_tensor(item.get("attn"))
            ):
                out.append(item["attn"])
        return out

    @staticmethod
    def _select_temporal_cross_attn_from_list(cache, layer_name):
        """
        Pick one temporal-cross-attn tensor from a list cache.

        Supported layer_name: temporal_last, temporal_first, temporal_index_<k>.
        temporal_all_mean is handled in _compute_attention_obi_loss (per-layer scalar mean).
        """
        items = GaussianDiffusion._collect_temporal_cross_attn_tensors(cache)
        if not items:
            return None
        ln = (layer_name or "temporal_last").strip().lower()
        if ln in ("", "temporal_last"):
            return items[-1]
        if ln == "temporal_first":
            return items[0]
        m = re.fullmatch(r"temporal_index_(\d+)", ln)
        if m:
            k = int(m.group(1))
            if k < 0 or k >= len(items):
                raise ValueError(
                    f"collab_obi_layer_name={layer_name!r} out of range: "
                    f"have {len(items)} attn2_temporal_cross tensors (indices 0..{len(items) - 1})"
                )
            return items[k]
        raise ValueError(
            f"Unknown collab_obi_layer_name={layer_name!r}. "
            f"Use temporal_last, temporal_first, temporal_index_<k>, or temporal_all_mean."
        )

    @staticmethod
    def _tensor_from_attn_cache_payload(cache, layer_name="temporal_last"):
        """Resolve attn tensor from a cache list/dict/tensor (OBI / temporal cross)."""
        if cache is None:
            return None

        if isinstance(cache, dict):
            if "attn" in cache and th.is_tensor(cache["attn"]):
                return cache["attn"]
            v = cache.get(layer_name, None)
            if th.is_tensor(v):
                return v
            return None

        if isinstance(cache, (list, tuple)):
            if len(cache) == 0:
                return None
            ln = (layer_name or "temporal_last").strip().lower()
            if ln == "temporal_all_mean":
                return None
            selected = GaussianDiffusion._select_temporal_cross_attn_from_list(
                cache, layer_name
            )
            if selected is not None:
                return selected
            for item in reversed(cache):
                if isinstance(item, dict) and th.is_tensor(item.get("attn")):
                    return item["attn"]
            for item in reversed(cache):
                if th.is_tensor(item):
                    return item
            return None

        if th.is_tensor(cache):
            return cache

        return None

    def _extract_temporal_attn_from_model(self, model, layer_name="temporal_last"):
        """
        Extract temporal attention from either the DDP wrapper or inner module.
        Prefer structured cache entries with source == 'attn2_temporal_cross'.
        """
        inner = self._unwrap_ddp(model)
        carriers = [inner]
        if inner is not model:
            carriers.append(model)

        for carrier in carriers:
            for attr in ("_temp_attn_cache", "_attn_cache"):
                out = self._tensor_from_attn_cache_payload(
                    getattr(carrier, attr, None), layer_name=layer_name
                )
                if out is not None:
                    return out

        return None

    def _obi_delta_mse_one_pair(
        self,
        attn_student,
        attn_teacher,
        sample_reliability,
        eps,
        device,
        dtype,
    ):
        """
        Scalar OBI on one (student, teacher) attn pair [B*H, F, M].
        Returns (loss_scalar, per_head_unweighted_mean) or (None, None) if F < 2.
        """
        if attn_student is None or attn_teacher is None:
            return None, None
        if attn_student.shape != attn_teacher.shape:
            raise ValueError(
                f"OBI attention shape mismatch: student={attn_student.shape}, teacher={attn_teacher.shape}"
            )
        if attn_student.dim() != 3:
            raise ValueError(
                f"OBI expects attn [B*H, F, M], got {attn_student.shape}"
            )
        if attn_student.shape[1] < 2:
            return None, None

        delta_student = attn_student[:, 1:, :] - attn_student[:, :-1, :]
        delta_teacher = (attn_teacher[:, 1:, :] - attn_teacher[:, :-1, :]).detach()
        per_head_loss = ((delta_student - delta_teacher) ** 2).mean(dim=(1, 2))

        if sample_reliability is not None:
            b = sample_reliability.shape[0]
            bh = per_head_loss.shape[0]
            if bh % b != 0:
                raise ValueError(
                    f"Cannot align OBI loss with reliability: per_head={bh}, reliability={b}"
                )
            num_heads = bh // b
            min_obi_weight = float(getattr(self, "collab_obi_min_weight", 0.05))
            obi_weight = (1.0 - sample_reliability.detach()).clamp(min=min_obi_weight)
            obi_weight = obi_weight.repeat_interleave(num_heads)
            loss_obi = (obi_weight * per_head_loss).sum() / (obi_weight.sum() + eps)
        else:
            loss_obi = per_head_loss.mean()

        return loss_obi, per_head_loss.mean()

    def _compute_attention_obi_loss(
        self,
        student_model,
        teacher_model,
        x_t,
        t,
        c,
        c_temp,
        model_kwargs,
        sample_reliability,
        obi_layer_name="temporal_last",
        eps=1e-6,
    ):
        """
        OBI: weighted || Δattn_student - Δattn_teacher ||^2
        DDP-safe: pass explicit attn_cache=list into UNet kwargs so CrossAttention
        appends to a caller-owned list (see tacm_unet_temp_dual.UNetModel.forward).

        collab_obi_layer_name:
          - temporal_last (default): last attn2_temporal_cross in forward order
          - temporal_first: first such tensor
          - temporal_index_<k>: k-th (0-based) such tensor
          - temporal_all_mean: mean of per-layer scalar OBI (handles varying [B*H,F,M] shapes)
        """
        device = x_t.device
        dtype = x_t.dtype

        if teacher_model is None:
            zero = th.zeros((), device=device, dtype=dtype)
            return zero, None

        extra_model_kwargs = {
            k: v
            for k, v in (model_kwargs or {}).items()
            if k not in ("c", "c_temp", "return_attn", "attn_cache")
        }

        tea_inner = self._unwrap_ddp(teacher_model)

        teacher_cache = _ObiAttnCacheList()
        student_cache = _ObiAttnCacheList()

        # Pass explicit list into UNet kwargs so appends use this object regardless of DDP / self.
        with th.no_grad():
            _ = tea_inner(
                x_t,
                self._scale_timesteps(t),
                c,
                c_temp,
                return_attn=True,
                attn_cache=teacher_cache,
                **extra_model_kwargs,
            )

        _ = student_model(
            x_t,
            self._scale_timesteps(t),
            c,
            c_temp,
            return_attn=True,
            attn_cache=student_cache,
            **extra_model_kwargs,
        )

        if os.environ.get("OBI_DEBUG_CACHE", "").lower() in ("1", "true", "yes"):
            # Under DDP, model._temp_attn_cache may not reflect the explicit list; trust list len.
            print("teacher _temp_attn_cache len =", len(teacher_cache))
            print("teacher _attn_cache len =", len(teacher_cache))
            print("student _temp_attn_cache len =", len(student_cache))
            print("student _attn_cache len =", len(student_cache))
            n_t = len(self._collect_temporal_cross_attn_tensors(teacher_cache))
            n_s = len(self._collect_temporal_cross_attn_tensors(student_cache))
            print("temporal_cross_attn tensors teacher/student =", n_t, n_s)

        ln = str(obi_layer_name).strip().lower()
        if ln == "temporal_all_mean":
            ts = self._collect_temporal_cross_attn_tensors(teacher_cache)
            ss = self._collect_temporal_cross_attn_tensors(student_cache)
            if len(ts) != len(ss):
                raise ValueError(
                    f"OBI temporal_all_mean: teacher has {len(ts)} attn2_temporal_cross tensors, "
                    f"student has {len(ss)} (forward order must match)."
                )
            if not ts:
                zero = th.zeros((), device=device, dtype=dtype)
                return zero, None
            layer_losses = []
            head_means = []
            for at, a_s in zip(ts, ss):
                lo, hm = self._obi_delta_mse_one_pair(
                    a_s, at, sample_reliability, eps, device, dtype
                )
                if lo is None:
                    continue
                layer_losses.append(lo)
                if hm is not None:
                    head_means.append(hm)
            if not layer_losses:
                zero = th.zeros((), device=device, dtype=dtype)
                return zero, None
            loss_obi = th.stack(layer_losses).mean()
            obi_mean = th.stack(head_means).mean() if head_means else None
            return loss_obi, obi_mean

        attn_teacher = self._tensor_from_attn_cache_payload(
            teacher_cache, layer_name=obi_layer_name
        )
        attn_student = self._tensor_from_attn_cache_payload(
            student_cache, layer_name=obi_layer_name
        )

        if attn_student is None or attn_teacher is None:
            zero = th.zeros((), device=device, dtype=dtype)
            return zero, None

        loss_obi, obi_mean = self._obi_delta_mse_one_pair(
            attn_student,
            attn_teacher,
            sample_reliability,
            eps,
            device,
            dtype,
        )
        if loss_obi is None:
            zero = th.zeros((), device=device, dtype=dtype)
            return zero, None
        return loss_obi, obi_mean

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start) #(16,3,64,64)
        # first frame pipeline
        #zeros = th.zeros_like(x_start)
        #noise[0] = zeros[0]

        assert noise.shape == x_start.shape
        
        x_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        
        #x_t[0] = x_start[0]
        
        return x_t
            
        #return (
        #    _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        #    + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        #)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, c, c_temp, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param c: can be text condition.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), c, c_temp, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, c, c_temp, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), c, c_temp, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self, model, x, t, c, c_temp, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            c,
            c_temp,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, c, c_temp, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        c,
        c_temp,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            c=c,
            c_temp=c_temp,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
        ):
            final = sample
        #return final["pred_xstart"]
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        c,
        c_temp,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        postprocess_fn=None,
        randomize_class=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            frames = max(1, int(init_image.shape[0]))
            assert shape[0] % frames == 0, (
                f"Sample shape[0]={shape[0]} not divisible by init_image frames={frames}"
            )
            batch_size = int(shape[0] // frames)
            init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
            img = self.q_sample(
                x_start=init_image_batch,
                t=th.tensor(indices[0], dtype=th.long, device=device),
                noise=img,
            )

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and "y" in model_kwargs:
                model_kwargs["y"] = th.randint(
                    low=0,
                    high=model.num_classes,
                    size=model_kwargs["y"].shape,
                    device=model_kwargs["y"].device,
                )
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    c,
                    c_temp,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                if postprocess_fn is not None:
                    out = postprocess_fn(out, t)

                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        postprocess_fn=None,
        randomize_class=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            batch_size = shape[0]
            init_image_batch = th.tile(init_image, dims=(batch_size, 1, 1, 1))
            img = self.q_sample(init_image_batch, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and "y" in model_kwargs:
                model_kwargs["y"] = th.randint(
                    low=0,
                    high=model.num_classes,
                    size=model_kwargs["y"].shape,
                    device=model_kwargs["y"].device,
                )
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                if postprocess_fn is not None:
                    out = postprocess_fn(out, t)

                yield out
                img = out["sample"]

    def _vb_terms_bpd(self, model, x_start, x_t, t, c, c_temp, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, c, c_temp, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, c, c_temp, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                c=c,
                c_temp=c_temp,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), c, c_temp, **model_kwargs)
            denoise_pred = model_output

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                denoise_pred = model_output
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    c=c,
                    c_temp=c_temp,
                    clip_denoised=True,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
            terms["loss_diff_base"] = terms["loss"]

            # Paper-style collaborative losses:
            #   L_vt = Colla(C(x0_pred), C(text))
            #   L_va = dynamic temporal alignment for audio-video changes
            #   L_vi = Colla(C(x0_pred), C(image))
            collab_scale = float(getattr(self, "collab_scale", 0.0))
            collab_vt_scale = float(getattr(self, "collab_vt_scale", getattr(self, "collab_vt_weight", collab_scale)))
            collab_va_scale = float(getattr(self, "collab_va_scale", getattr(self, "collab_va_weight", collab_scale)))
            collab_vi_scale = float(getattr(self, "collab_vi_scale", getattr(self, "collab_vi_weight", collab_scale)))
            collab_imit_scale = float(getattr(self, "collab_imit_scale", 0.0))
            collab_obi_scale = float(getattr(self, "collab_obi_scale", 0.0))
            omni_encoder = getattr(self, "omni_encoder", None)

            if omni_encoder is not None and (
                collab_vt_scale > 0.0
                or collab_va_scale > 0.0
                or collab_vi_scale > 0.0
                or collab_imit_scale > 0.0
                or collab_obi_scale > 0.0
            ):
                if self.model_mean_type == ModelMeanType.EPSILON:
                    x0_pred = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=denoise_pred)
                elif self.model_mean_type == ModelMeanType.START_X:
                    x0_pred = denoise_pred
                elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                    x0_pred = self._predict_xstart_from_xprev(x_t=x_t, t=t, xprev=denoise_pred)
                else:
                    raise NotImplementedError(f"Unsupported model_mean_type: {self.model_mean_type}")

                seq_len = int(getattr(self, "sequence_length", 1))
                collab_metric = str(getattr(self, "collab_metric", "cosine")).lower()

                # c is c_ti = [text tokens, image token]; split for explicit text/image losses.
                c_text = c[:, :-1] if (c.dim() == 3 and c.shape[1] > 1) else c
                c_image = c[:, -1:] if c.dim() == 3 else c

                # Content-level embeddings
                z_v = omni_encoder.encode_video(x0_pred, sequence_length=seq_len)
                z_t = omni_encoder.encode_text(c_text, sequence_length=seq_len)
                z_a = omni_encoder.encode_audio(c_temp, sequence_length=seq_len)
                z_i = omni_encoder.encode_image(c_image, sequence_length=seq_len)

                assert z_v.shape[0] == z_t.shape[0] == z_a.shape[0] == z_i.shape[0], (
                    f"Collaborative embedding batch mismatch: "
                    f"z_v={z_v.shape}, z_t={z_t.shape}, z_a={z_a.shape}, z_i={z_i.shape}"
                )

                if z_v.shape[0] > 0:
                    step_now = int(getattr(self, "step", getattr(self, "train_step", 0)))
                    collab_start_step = int(getattr(self, "collab_start_step", 0))
                    collab_warmup_steps = int(getattr(self, "collab_warmup_steps", 1))
                    va_warmup_steps = int(getattr(self, "va_warmup_steps", max(1, collab_warmup_steps // 2)))

                    collab_dyn_stride = int(getattr(self, "collab_dyn_stride", 1))
                    collab_dyn_sample_tau = float(getattr(self, "collab_dyn_sample_tau", 0.03))
                    collab_dyn_sample_temp = float(getattr(self, "collab_dyn_sample_temp", 0.01))
                    collab_dyn_gate_k = float(getattr(self, "collab_dyn_gate_k", 2.0))
                    collab_dyn_beta = float(getattr(self, "collab_dyn_beta", 3.0))

                    collab_obi_start_step = int(
                        getattr(self, "collab_obi_start_step", collab_start_step)
                    )
                    collab_obi_warmup_steps = int(
                        getattr(self, "collab_obi_warmup_steps", collab_warmup_steps)
                    )
                    collab_obi_layer_name = str(
                        getattr(self, "collab_obi_layer_name", "temporal_last")
                    )

                    vt_scale_warm = self._linear_warmup_scale(
                        step=step_now,
                        start_step=collab_start_step,
                        warmup_steps=collab_warmup_steps,
                    )
                    va_scale_warm = self._linear_warmup_scale(
                        step=step_now,
                        start_step=collab_start_step,
                        warmup_steps=va_warmup_steps,
                    )
                    obi_scale_warm = self._linear_warmup_scale(
                        step=step_now,
                        start_step=collab_obi_start_step,
                        warmup_steps=collab_obi_warmup_steps,
                    )

                    static_collab_metric = (
                        "cosine" if collab_metric == "dynamic_cosine" else collab_metric
                    )
                    loss_vt = self._compute_static_collab_loss(
                        z_v, z_t, metric=static_collab_metric
                    )
                    loss_vi = self._compute_static_collab_loss(
                        z_v, z_i, metric=static_collab_metric
                    )
                    loss_va = th.zeros_like(loss_vt)
                    loss_imit = th.zeros_like(loss_vt)
                    loss_obi = th.zeros_like(loss_vt)
                    sample_reliability = None

                    if collab_metric in {"cosine", "mse", "mae"}:
                        loss_va = self._compute_static_collab_loss(
                            z_v, z_a, metric=collab_metric
                        )
                    elif collab_metric == "dynamic_cosine":
                        if seq_len <= 1:
                            loss_va = th.zeros_like(loss_vt)
                            sample_reliability = th.zeros(
                                (z_v.shape[0],),
                                device=z_v.device,
                                dtype=z_v.dtype,
                            )
                        else:
                            z_v_seq = omni_encoder.encode_video_sequence(
                                x0_pred, sequence_length=seq_len
                            )
                            _audio_flat_bt = (
                                c_temp.dim() == 3 and c_temp.shape[0] == x0_pred.shape[0]
                            )
                            z_a_seq = omni_encoder.encode_audio_sequence(
                                c_temp,
                                sequence_length=seq_len,
                                flattened_bt=_audio_flat_bt,
                            )

                            dyn_stats = self._compute_dynamic_reliability_and_loss(
                                z_v_seq=z_v_seq,
                                z_a_seq=z_a_seq,
                                device=terms["loss"].device,
                                dtype=terms["loss"].dtype,
                                dyn_stride=collab_dyn_stride,
                                tau_sample=collab_dyn_sample_tau,
                                temp_sample=collab_dyn_sample_temp,
                                gate_k=collab_dyn_gate_k,
                                dyn_loss_beta=collab_dyn_beta,
                            )
                            loss_va = dyn_stats["loss_va"]
                            sample_reliability = dyn_stats["sample_reliability"]

                            terms["dyn_stride"] = dyn_stats["dyn_stride"]
                            terms["dyn_video_mag"] = dyn_stats["dyn_video_mag"]
                            terms["dyn_audio_mag"] = dyn_stats["dyn_audio_mag"]
                            terms["dyn_cos_mean"] = dyn_stats["dyn_cos_mean"]
                            terms["dyn_gate_mean"] = dyn_stats["dyn_gate_mean"]
                            terms["dyn_valid_mean"] = dyn_stats["dyn_valid_mean"]
                            terms["dyn_weight_mean"] = dyn_stats["dyn_weight_mean"]
                            terms["dyn_sample_av_ratio"] = dyn_stats["dyn_sample_av_ratio"]
                            terms["dyn_sample_on_ratio"] = dyn_stats["dyn_sample_on_ratio"]
                            terms["dyn_sample_video_strength"] = dyn_stats["dyn_sample_video_strength"]
                            terms["dyn_sample_audio_strength"] = dyn_stats["dyn_sample_audio_strength"]
                            terms["dyn_sample_score_mean"] = dyn_stats["dyn_sample_score_mean"]
                            terms["dyn_rank_q_low"] = dyn_stats["dyn_rank_q_low"]
                            terms["dyn_rank_q_high"] = dyn_stats["dyn_rank_q_high"]
                            terms["dyn_rank_mean"] = dyn_stats["dyn_rank_mean"]
                            terms["dyn_av_conf_mean"] = dyn_stats["dyn_av_conf_mean"]
                            terms["dyn_av_sim_mean"] = dyn_stats["dyn_av_sim_mean"]

                            try:
                                sr = sample_reliability.detach()
                                q = th.quantile(sr, th.tensor([0.25, 0.5, 0.75], device=sr.device))
                                terms["dyn_sample_on_ratio_q0"] = sr[sr <= q[0]].mean() if (sr <= q[0]).any() else sr.mean()
                                terms["dyn_sample_on_ratio_q1"] = sr[(sr > q[0]) & (sr <= q[1])].mean() if ((sr > q[0]) & (sr <= q[1])).any() else sr.mean()
                                terms["dyn_sample_on_ratio_q2"] = sr[(sr > q[1]) & (sr <= q[2])].mean() if ((sr > q[1]) & (sr <= q[2])).any() else sr.mean()
                                terms["dyn_sample_on_ratio_q3"] = sr[sr > q[2]].mean() if (sr > q[2]).any() else sr.mean()
                            except Exception:
                                pass
                    else:
                        raise ValueError(
                            f"Unsupported collab_metric: {collab_metric}. "
                            "Expected one of {'cosine', 'mse', 'mae', 'dynamic_cosine'}."
                        )

                    if sample_reliability is None:
                        sample_reliability = th.zeros(
                            (z_v.shape[0],),
                            device=z_v.device,
                            dtype=z_v.dtype,
                        )

                    baseline_model = getattr(self, "baseline_model", None)

                    if collab_obi_scale > 0.0 and baseline_model is not None:
                        loss_obi, obi_mean = self._compute_attention_obi_loss(
                            student_model=model,
                            teacher_model=baseline_model,
                            x_t=x_t,
                            t=t,
                            c=c,
                            c_temp=c_temp,
                            model_kwargs=model_kwargs,
                            sample_reliability=sample_reliability,
                            obi_layer_name=collab_obi_layer_name,
                        )
                        if obi_mean is not None:
                            terms["collab_obi_mean"] = obi_mean.detach()
                    else:
                        loss_obi = th.zeros_like(loss_vt)

                    loss_collab_vt_weighted = collab_vt_scale * vt_scale_warm * loss_vt
                    loss_collab_va_weighted = collab_va_scale * va_scale_warm * loss_va
                    loss_collab_vi_weighted = collab_vi_scale * vt_scale_warm * loss_vi
                    loss_collab_imit_weighted = th.zeros_like(loss_collab_vt_weighted)
                    min_obi_scale = float(getattr(self, "collab_obi_min_scale", 0.1))
                    obi_scale_eff = float(collab_obi_scale) * max(
                        float(obi_scale_warm), min_obi_scale
                    )
                    loss_collab_obi_weighted = obi_scale_eff * loss_obi

                    terms["collab_vt"] = loss_vt.detach()
                    terms["collab_va"] = loss_va.detach()
                    terms["collab_vi"] = loss_vi.detach()
                    terms["collab_imit"] = loss_imit.detach()
                    terms["collab_obi"] = loss_obi.detach()
                    terms["collab_scale"] = th.as_tensor(collab_scale, device=terms["loss"].device, dtype=terms["loss"].dtype)
                    terms["collab_vt_scale"] = th.as_tensor(collab_vt_scale * vt_scale_warm, device=terms["loss"].device, dtype=terms["loss"].dtype)
                    terms["collab_va_scale"] = th.as_tensor(collab_va_scale * va_scale_warm, device=terms["loss"].device, dtype=terms["loss"].dtype)
                    terms["collab_vi_scale"] = th.as_tensor(collab_vi_scale * vt_scale_warm, device=terms["loss"].device, dtype=terms["loss"].dtype)
                    # x0 baseline imitation removed from loss; keep log at 0 to avoid confusion.
                    terms["collab_imit_scale"] = th.as_tensor(
                        0.0, device=terms["loss"].device, dtype=terms["loss"].dtype
                    )
                    terms["collab_obi_scale"] = th.as_tensor(
                        obi_scale_eff, device=terms["loss"].device, dtype=terms["loss"].dtype
                    )
                    terms["debug_obi_raw"] = loss_obi.detach()
                    terms["debug_obi_scale_eff"] = th.as_tensor(
                        obi_scale_eff, device=terms["loss"].device, dtype=terms["loss"].dtype
                    )
                    terms["debug_obi_final"] = loss_collab_obi_weighted.detach()
                    terms["debug_reliability_mean"] = sample_reliability.mean().detach()
                    terms["loss_collab_vt_weighted"] = loss_collab_vt_weighted.detach()
                    terms["loss_collab_va_weighted"] = loss_collab_va_weighted.detach()
                    terms["loss_collab_vi_weighted"] = loss_collab_vi_weighted.detach()
                    terms["loss_collab_imit_weighted"] = loss_collab_imit_weighted.detach()
                    terms["loss_collab_obi_weighted"] = loss_collab_obi_weighted.detach()

                    terms["loss"] = (
                        terms["loss"]
                        + loss_collab_vt_weighted
                        + loss_collab_va_weighted
                        + loss_collab_vi_weighted
                        + loss_collab_obi_weighted
                    )
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
   # res = th.from_numpy(arr.astype(np.float32)).to(device=timesteps.device)[timesteps].float()
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
