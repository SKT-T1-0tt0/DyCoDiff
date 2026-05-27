import copy
import functools
import os
import warnings
from collections import deque

import transformers.image_transforms
from einops import rearrange, repeat

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from tacm.download import load_vqgan
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
from .tacm_nn import update_ema
from .condition_builder import build_conditions
from .omni_encoder import OmniEncoder
from .attention import TemporalTransformer
from tacm import AudioCLIP
from tacm.modules.learned_gate import LearnedGateRefiner
import wav2clip
from beats.BEATs import BEATs, BEATsConfig

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def _split_train_checkpoint(raw):
    """
    Supports:
      - legacy: flat UNet state_dict only
      - new: dict with keys 'model', optional 'mcfl', 'learned_gate_refiner', attn pools
    """
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw
    return {"model": raw}


def _get_last_temporal_attn(model):
    """Get attn from collector (last temporal cross-attn block, attn2 only)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_attn_cache") and m._attn_cache is not None and len(m._attn_cache) > 0:
        for item in reversed(m._attn_cache):
            if isinstance(item, dict) and th.is_tensor(item.get("attn")):
                return item["attn"]
            if th.is_tensor(item):
                return item
    return None


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class TrainLoop():
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_dir,
        vqgan_ckpt,
        sequence_length,
        audio_emb_model,
        use_mcfl=False,
        mcfl_embed_dim=768,
        mcfl_pooling_mode="mean",  # "mean" or "attention"
        mcfl_gate_lambda=0.1,  # MCFL v2-A gate parameter (0.1 降低 TC_FLICKER，原 0.2)
        lambda_temp=0.0,  # Temporal smooth regularization weight (default 0.0 = disabled, set > 0 to enable, e.g., 0.01)
        mcfl_conservative=False,  # True: alpha 课程 + 8k 冻结 + lambda_temp 课程。False: 完整 MCFL（与论文 L_vt/L_va 协同项正交，默认关保守策略）
        use_baseline_imitation=False,  # If True: add online baseline output imitation loss (placeholder; implement when needed).
        mcfl_norm_modality=True,  # 跨分布鲁棒：送入 MCFL 前对 image/audio 做 L2 归一化
        audio_response="tanh",  # 送入 BEATs 前强度响应（只做一次）: "none" | "tanh" | "compand"
        audio_random_gain=False,  # 默认关闭随机增益；需要增强鲁棒性时再显式开启
        audio_gain_range=(0.25, 4.0),  # (low, high)，可改为 (0.25, 8)
        audio_random_response_strength=False,  # 默认关闭随机响应强度；需要时再显式开启
        modality_dropout_prob=0.0,  # 默认关闭 modality dropout
        mcfl_gate_adaptive=True,  # gate 随音频范数置信度自适应，异常时→0
        mcfl_gate_norm_low=7.2,  # 统一标定：覆盖 drums/URMP/landscape p5–p95
        mcfl_gate_norm_high=10.0,
        mcfl_gate_time_smooth=True,
        mcfl_gate_ema=0.9,  # 更强时间平滑，减轻 flicker
        mcfl_gate_use_zscore=False,
        mcfl_gate_norm_mu=8.4,
        mcfl_gate_norm_sigma=0.5,
        mcfl_gate_z_low=-1.5,
        mcfl_gate_z_high=1.5,
        mcfl_gate_lambda_max=0.2,  # 护栏1：gate 硬上限
        mcfl_gate_norm_clip_clamp=True,  # 护栏3：per-frame norm 按 clip p5–p95 限幅
        # 新增：audio-visual agreement gate 因子
        mcfl_gate_use_av_conf: bool = False,
        mcfl_gate_av_sim_low: float = 0.0,
        mcfl_gate_av_sim_high: float = 0.3,
        mcfl_gate_av_beta: float = 0.5,
        mcfl_collab_weight: float = 0.0,  # 协同损失权重，0=关闭；建议 0.01/0.05/0.10，从 0.05 起步
        learned_gate_enable: bool = False,
        learned_gate_hidden_dim: int = 16,
        learned_gate_dropout: float = 0.0,
        learned_gate_detach_input: bool = True,
        learned_gate_reg_weight: float = 0.0,
        collab_vt_weight: float = 0.10,
        collab_va_weight: float = 0.01,
        collab_vi_weight: float = 0.0,
        omni_shared_dim: int = 512,
        image_size: int = 64,
        in_channels: int = 3,
        collab_metric: str = "cosine",
        omni_encoder_ckpt: str = "",
        collab_start_step: int = 3000,
        collab_warmup_steps: int = 10000,
        collab_max_scale: float = 1.0,
        collab_max_ratio: float = 0.10,
        collab_margin_vt: float = 0.20,
        collab_margin_va: float = 0.10,
        collab_dyn_stride: int = 1,
        collab_dyn_sample_tau: float = 0.03,
        collab_dyn_sample_temp: float = 0.01,
        collab_dyn_gate_k: float = 2.0,
        collab_dyn_beta: float = 3.0,
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
        collab_obi_layer_name: str = "temporal_last",
        collab_obi_min_weight: float = 0.05,
        collab_obi_min_scale: float = 0.1,
        collab_imit_scale: float = 0.0,
        collab_imit_start_step: int = 0,
        collab_imit_warmup_steps: int = 10000,
        baseline_model_path: str = "",
        baseline_model=None,
        baseline_pred_fn=None,
        collab_smooth_weight: float = 0.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.save_dir = save_dir
        self.vqgan_ckpt = vqgan_ckpt
        self.sequence_length = sequence_length
        self.audio_emb_model = audio_emb_model
        self.use_mcfl = use_mcfl
        self.mcfl_pooling_mode = mcfl_pooling_mode
        self.mcfl_gate_lambda = mcfl_gate_lambda  # MCFL v2-A gate parameter
        self.lambda_temp = lambda_temp  # Temporal smooth regularization weight
        self.mcfl_conservative = mcfl_conservative  # Conservative curriculum (alpha, freeze, lambda_temp) vs full MCFL
        self.use_baseline_imitation = use_baseline_imitation  # Online baseline imitation (placeholder)
        self.mcfl_norm_modality = mcfl_norm_modality
        self.audio_response = audio_response
        self.audio_random_gain = audio_random_gain
        self.audio_gain_range = tuple(audio_gain_range) if hasattr(audio_gain_range, '__len__') else (0.25, 4.0)
        self.audio_random_response_strength = audio_random_response_strength
        self.modality_dropout_prob = modality_dropout_prob
        self.mcfl_gate_adaptive = mcfl_gate_adaptive
        self.mcfl_gate_norm_low = mcfl_gate_norm_low
        self.mcfl_gate_norm_high = mcfl_gate_norm_high
        self.mcfl_gate_time_smooth = mcfl_gate_time_smooth
        self.mcfl_gate_ema = mcfl_gate_ema
        self.mcfl_gate_use_zscore = mcfl_gate_use_zscore
        self.mcfl_gate_norm_mu = mcfl_gate_norm_mu
        self.mcfl_gate_norm_sigma = mcfl_gate_norm_sigma
        self.mcfl_gate_z_low = mcfl_gate_z_low
        self.mcfl_gate_z_high = mcfl_gate_z_high
        self.mcfl_gate_lambda_max = mcfl_gate_lambda_max
        self.mcfl_gate_norm_clip_clamp = mcfl_gate_norm_clip_clamp
        self.mcfl_gate_use_av_conf = mcfl_gate_use_av_conf
        self.mcfl_gate_av_sim_low = mcfl_gate_av_sim_low
        self.mcfl_gate_av_sim_high = mcfl_gate_av_sim_high
        self.mcfl_gate_av_beta = mcfl_gate_av_beta
        self.mcfl_collab_weight = mcfl_collab_weight
        self.learned_gate_enable = learned_gate_enable
        self.learned_gate_hidden_dim = learned_gate_hidden_dim
        self.learned_gate_dropout = learned_gate_dropout
        self.learned_gate_detach_input = learned_gate_detach_input
        self.learned_gate_reg_weight = learned_gate_reg_weight
        self.collab_vt_weight = collab_vt_weight
        self.collab_va_weight = collab_va_weight
        self.collab_vi_weight = collab_vi_weight
        self.omni_shared_dim = omni_shared_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.collab_metric = collab_metric
        self.omni_encoder_ckpt = omni_encoder_ckpt
        self.collab_start_step = collab_start_step
        self.collab_warmup_steps = collab_warmup_steps
        self.collab_max_scale = collab_max_scale
        self.collab_max_ratio = collab_max_ratio
        self.collab_margin_vt = collab_margin_vt
        self.collab_margin_va = collab_margin_va
        self.collab_dyn_stride = int(collab_dyn_stride)
        self.collab_dyn_sample_tau = float(collab_dyn_sample_tau)
        self.collab_dyn_sample_temp = float(collab_dyn_sample_temp)
        self.collab_dyn_gate_k = float(collab_dyn_gate_k)
        self.collab_dyn_beta = float(collab_dyn_beta)
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
        self.collab_imit_scale = float(collab_imit_scale)
        self.collab_imit_start_step = int(collab_imit_start_step)
        self.collab_imit_warmup_steps = int(collab_imit_warmup_steps)
        self.baseline_model_path = baseline_model_path
        self.baseline_model = baseline_model
        self.baseline_pred_fn = baseline_pred_fn
        self.collab_smooth_weight = float(collab_smooth_weight)

        _seq_len_tt = int(self.sequence_length)
        for _mod in self.model.modules():
            if isinstance(_mod, TemporalTransformer):
                _mod.sequence_length = _seq_len_tt

        # Initialize MCFL if enabled
        if self.use_mcfl:
            from tacm import MCFL, AttnPool
            self.mcfl = MCFL(
                embed_dim=mcfl_embed_dim,
                num_heads=8,
                dropout=0.1  # Changed from 0.0 to 0.1 to prevent overfitting
            ).to(dist_util.dev())
            
            # Initialize attention pooling modules if using attention pooling
            if self.mcfl_pooling_mode == "attention":
                self.attn_pool_text = AttnPool(dim=mcfl_embed_dim).to(dist_util.dev())
                self.attn_pool_audio = AttnPool(dim=mcfl_embed_dim).to(dist_util.dev())
            else:
                self.attn_pool_text = None
                self.attn_pool_audio = None
        else:
            self.mcfl = None
            self.attn_pool_text = None
            self.attn_pool_audio = None

        if self.learned_gate_enable:
            self.learned_gate_refiner = LearnedGateRefiner(
                in_dim=4,
                hidden_dim=self.learned_gate_hidden_dim,
                dropout=self.learned_gate_dropout,
            ).to(dist_util.dev())
        else:
            self.learned_gate_refiner = None

        self._mcfl_frozen = False  # Flag for curriculum: freeze MCFL in Stage 3

        # Optional baseline teacher for online imitation.
        if self.baseline_model is None and self.baseline_model_path:
            if dist.get_rank() == 0:
                logger.log(f"loading baseline teacher from checkpoint: {self.baseline_model_path}")
            teacher = copy.deepcopy(self.model).to(dist_util.dev())
            teacher_raw = dist_util.load_state_dict(
                self.baseline_model_path, map_location=dist_util.dev()
            )
            teacher_bundle = _split_train_checkpoint(teacher_raw)
            teacher.load_state_dict(teacher_bundle["model"], strict=False)
            teacher.eval()
            teacher.train = disabled_train
            for p in teacher.parameters():
                p.requires_grad = False
            self.baseline_model = teacher
        elif self.baseline_model is not None:
            self.baseline_model = self.baseline_model.to(dist_util.dev())
            self.baseline_model.eval()
            self.baseline_model.train = disabled_train
            for p in self.baseline_model.parameters():
                p.requires_grad = False

        # Paper-style collaborative loss encoder C:
        # L_vt = Colla(C(v_hat), C(t)), L_va = Colla(C(v_hat), C(a)).
        if (
            (self.collab_vt_weight > 0.0)
            or (self.collab_va_weight > 0.0)
            or (self.collab_vi_weight > 0.0)
            or (self.collab_imit_scale > 0.0)
            or (self.collab_obi_scale > 0.0)
        ):
            self.omni_encoder = OmniEncoder(
                video_dim=self.in_channels * self.image_size * self.image_size,
                text_dim=768,
                audio_dim=768,
                shared_dim=self.omni_shared_dim,
                image_dim=768,
            ).to(dist_util.dev())
            if self.omni_encoder_ckpt:
                state = th.load(self.omni_encoder_ckpt, map_location=dist_util.dev())
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.omni_encoder.load_state_dict(state, strict=False)
            else:
                warnings.warn(
                    "OmniEncoder is randomly initialized (placeholder). "
                    "For high-fidelity reproduction, load a pretrained omni encoder via --omni_encoder_ckpt.",
                    UserWarning,
                )
            for p in self.omni_encoder.parameters():
                p.requires_grad = False
            self.omni_encoder.eval()
            self.diffusion.omni_encoder = self.omni_encoder
            self.diffusion.collab_vt_weight = float(self.collab_vt_weight)
            self.diffusion.collab_va_weight = float(self.collab_va_weight)
            self.diffusion.collab_vi_weight = float(self.collab_vi_weight)
            self.diffusion.sequence_length = int(self.sequence_length)
            self.diffusion.collab_metric = self.collab_metric
            self.diffusion.collab_start_step = int(self.collab_start_step)
            self.diffusion.collab_warmup_steps = int(self.collab_warmup_steps)
            self.diffusion.collab_max_scale = float(self.collab_max_scale)
            self.diffusion.collab_max_ratio = float(self.collab_max_ratio)
            self.diffusion.collab_margin_vt = float(self.collab_margin_vt)
            self.diffusion.collab_margin_va = float(self.collab_margin_va)
            self.diffusion.collab_dyn_stride = int(self.collab_dyn_stride)
            self.diffusion.collab_dyn_sample_tau = float(self.collab_dyn_sample_tau)
            self.diffusion.collab_dyn_sample_temp = float(self.collab_dyn_sample_temp)
            self.diffusion.collab_dyn_gate_k = float(self.collab_dyn_gate_k)
            self.diffusion.collab_dyn_beta = float(self.collab_dyn_beta)
            self.diffusion.collab_dyn_rank_mode = self.collab_dyn_rank_mode
            self.diffusion.collab_dyn_rank_low_q = float(self.collab_dyn_rank_low_q)
            self.diffusion.collab_dyn_rank_high_q = float(self.collab_dyn_rank_high_q)
            self.diffusion.collab_dyn_rank_sharpness = float(self.collab_dyn_rank_sharpness)
            self.diffusion.collab_dyn_hist_size = int(self.collab_dyn_hist_size)
            self.diffusion.collab_dyn_score_mix_av_ratio = float(
                self.collab_dyn_score_mix_av_ratio
            )
            self.diffusion.collab_dyn_score_mix_audio = float(
                self.collab_dyn_score_mix_audio
            )
            self.diffusion.collab_av_score_mix_av_ratio = float(
                self.collab_av_score_mix_av_ratio
            )
            self.diffusion.collab_av_score_mix_av_conf = float(
                self.collab_av_score_mix_av_conf
            )
            self.diffusion.collab_av_conf_beta = float(self.collab_av_conf_beta)
            self.diffusion.collab_av_conf_sim_low = float(self.collab_av_conf_sim_low)
            self.diffusion.collab_av_conf_sim_high = float(self.collab_av_conf_sim_high)
            self.diffusion.collab_audio_ema_alpha = float(self.collab_audio_ema_alpha)
            self.diffusion.collab_obi_scale = float(self.collab_obi_scale)
            self.diffusion.collab_obi_start_step = int(self.collab_obi_start_step)
            self.diffusion.collab_obi_warmup_steps = int(self.collab_obi_warmup_steps)
            self.diffusion.collab_obi_layer_name = str(self.collab_obi_layer_name)
            self.diffusion.collab_obi_min_weight = float(self.collab_obi_min_weight)
            self.diffusion.collab_obi_min_scale = float(self.collab_obi_min_scale)

            _hist_sz = max(8, int(self.collab_dyn_hist_size))
            _dh = getattr(self.diffusion, "_dyn_score_history", None)
            if _dh is None or getattr(_dh, "maxlen", None) != _hist_sz:
                self.diffusion._dyn_score_history = deque(_dh or (), maxlen=_hist_sz)

            self.diffusion.collab_imit_scale = float(self.collab_imit_scale)
            self.diffusion.collab_imit_start_step = int(self.collab_imit_start_step)
            self.diffusion.collab_imit_warmup_steps = int(self.collab_imit_warmup_steps)
            self.diffusion.baseline_model = self.baseline_model
            if self.baseline_pred_fn is not None:
                self.diffusion.baseline_pred_fn = self.baseline_pred_fn
            self.diffusion.collab_scale = float(
                max(self.collab_vt_weight, self.collab_va_weight, self.collab_vi_weight)
            )
            self.diffusion.collab_vt_scale = float(self.collab_vt_weight)
            self.diffusion.collab_va_scale = float(self.collab_va_weight)
            self.diffusion.collab_vi_scale = float(self.collab_vi_weight)
            self.diffusion.collab_smooth_weight = float(self.collab_smooth_weight)
            self.diffusion.train_step = 0
        else:
            self.omni_encoder = None

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        #for pn, p in self.mp_trainer.model.named_parameters():
        #    if 'temporal_conv' in pn:
        #        continue
        #    elif '2.transformer_blocks' in pn:
        #        continue
        #    else:
        #        p.requires_grad = False
               
        #params = filter(lambda p : p.requires_grad, self.mp_trainer.model.parameters())   
        #self.opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        opt_params = list(self.mp_trainer.master_params)
        if self.learned_gate_refiner is not None:
            opt_params += [p for p in self.learned_gate_refiner.parameters() if p.requires_grad]
        self.opt = AdamW(opt_params, lr=self.lr, weight_decay=self.weight_decay)
             
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            
        # self.audioclip = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
        # self.audioclip = self.audioclip.to(dist_util.dev())
        
        # self.wav2clip_model = wav2clip.get_model()
        # self.wav2clip_model = self.wav2clip_model.to(dist_util.dev())
        # for p in self.wav2clip_model.parameters():
        #     p.requires_grad = False
            
        checkpoint = th.load('saved_ckpts/BEATs_iter3_plus_AS20K.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.BEATs_model = BEATs(cfg)
        self.BEATs_model = self.BEATs_model.to('cpu')  # 固定在 CPU 上运行
        self.BEATs_model.load_state_dict(checkpoint['model'])
        self.BEATs_model.eval()
        
        self.processor = CLIPProcessor.from_pretrained("tacm/modules/cache/clip-vit-large-patch14",return_unused_kwargs=False)
        self.clipmodel = CLIPModel.from_pretrained("tacm/modules/cache/clip-vit-large-patch14").to(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                bundle = _split_train_checkpoint(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
                self.model.load_state_dict(bundle["model"])
                if self.mcfl is not None and "mcfl" in bundle:
                    self.mcfl.load_state_dict(bundle["mcfl"], strict=True)
                if self.learned_gate_refiner is not None and "learned_gate_refiner" in bundle:
                    self.learned_gate_refiner.load_state_dict(
                        bundle["learned_gate_refiner"], strict=True
                    )
                if self.attn_pool_text is not None and "attn_pool_text" in bundle:
                    self.attn_pool_text.load_state_dict(bundle["attn_pool_text"], strict=True)
                if self.attn_pool_audio is not None and "attn_pool_audio" in bundle:
                    self.attn_pool_audio.load_state_dict(bundle["attn_pool_audio"], strict=True)
        self.model.to(dist_util.dev())
        dist_util.sync_params(self.model.parameters())
        if self.mcfl is not None:
            dist_util.sync_params(self.mcfl.parameters())
        if self.learned_gate_refiner is not None:
            dist_util.sync_params(self.learned_gate_refiner.parameters())
        if self.attn_pool_text is not None:
            dist_util.sync_params(self.attn_pool_text.parameters())
            dist_util.sync_params(self.attn_pool_audio.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                raw = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                bundle = _split_train_checkpoint(raw)
                ema_params = self.mp_trainer.state_dict_to_master_params(bundle["model"])

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            for i, sample in enumerate(self.data):
                batch, cond = sample['video'], {}
                # ----get text----
                c_t = sample['text'].squeeze(1).to(dist_util.dev())
                
                # ----get image----
                image = batch[:,:,0]+0.5
                image_cat=None
                for j in range(image.shape[0]):
                    image_j = transformers.image_transforms.to_pil_image(image[j])
                    image_input = self.processor(images=image_j, return_tensors="pt", padding=True).to(dist_util.dev())
                    with th.no_grad():
                        image_features = self.clipmodel.get_image_features(image_input.pixel_values)

                    if image_cat is None:
                        image_cat = image_features.unsqueeze(0)
                    else:
                        image_cat = th.concat((image_cat, image_features.unsqueeze(0)), dim=0) #torch.Size([1, 1, 768])

                batch = rearrange(batch, "b c t h w -> (b t) c h w")
                c_ti = th.concat((c_t,image_cat), dim=1)
                #c_i = image_cat
                
                # ----get audio----
                if self.audio_emb_model == 'STFT':      
                    stft = sample['stft'] #torch.Size([1, 1, 16, 64, 16])
                else:
                    audio = sample['audio'].to(dist_util.dev()) #torch.Size([1, 16, 1600])
                
                # if self.audio_emb_model == 'audioclip':
                #     ((audio_embed, _, _), _), _ = self.audioclip(audio=audio.squeeze())
                #     c_temp = audio_embed.unsqueeze(0) #(1,16,1024)
                # elif self.audio_emb_model == 'wav2clip':
                #     audio_embed = th.from_numpy(wav2clip.embed_audio(audio.cpu().numpy().squeeze(), self.wav2clip_model)) #(16,512)
                #     c_temp = audio_embed.unsqueeze(1) #(1,16,512) #(16,1,512)
                if self.audio_emb_model == 'STFT':
                    c_temp = stft.squeeze(1)              
                elif self.audio_emb_model == 'beats':
                    audio = rearrange(audio, "b f g -> (b f) g")
                    # 将音频移到 CPU 上进行处理，因为 BEATs 模型在 CPU 上
                    c_temp = self.BEATs_model.extract_features(audio.cpu(), padding_mask=None)[0] #torch.Size([16, 8, 768])
                    # 处理完成后移回 GPU
                    c_temp = c_temp.to(dist_util.dev())

                # 🔧 修改 1：对 c_temp (BEATs audio tokens) 做跨帧 EMA 平滑
                # BEATs 每帧独立编码 → 条件帧间跳变 → flicker
                # EMA 低通滤波 → 平滑帧间过渡
                T = self.sequence_length  # 16
                BT, M, D = c_temp.shape
                B = BT // T
                c_temp_reshaped = c_temp.view(B, T, M, D)  # [B, T, 8, 768]
                
                alpha = 0.9  # EMA 系数，越大越平滑
                c_smoothed = c_temp_reshaped.clone()
                for t in range(1, T):
                    c_smoothed[:, t] = alpha * c_smoothed[:, t-1] + (1 - alpha) * c_temp_reshaped[:, t]
                
                c_temp = c_smoothed.view(BT, M, D)  # [B*T, 8, 768]

                # 生成条件：调用 build_conditions（内部调用 mcfl）
                # mcfl_collab_weight > 0 时开启协同损失，需 return_collab_loss=True 并塞入 cond
                use_collab = self.use_mcfl and getattr(self, "mcfl_collab_weight", 0.0) > 0
                use_gate_reg = self.use_mcfl and getattr(self, "learned_gate_reg_weight", 0.0) > 0
                _build_kw = dict(
                    c_t=c_t,
                    image_cat=image_cat,
                    c_temp=c_temp,
                    mcfl=self.mcfl,
                    use_mcfl=self.use_mcfl,
                    pooling_mode=self.mcfl_pooling_mode,
                    attn_pool_text=self.attn_pool_text,
                    attn_pool_audio=self.attn_pool_audio,
                    mcfl_gate_lambda=getattr(self, 'mcfl_gate_lambda', 0.2),
                    mcfl_norm_modality=getattr(self, 'mcfl_norm_modality', True),
                    mcfl_gate_adaptive=getattr(self, 'mcfl_gate_adaptive', True),
                    mcfl_gate_norm_low=getattr(self, 'mcfl_gate_norm_low', 7.2),
                    mcfl_gate_norm_high=getattr(self, 'mcfl_gate_norm_high', 10.0),
                    mcfl_gate_time_smooth=getattr(self, 'mcfl_gate_time_smooth', True),
                    mcfl_gate_ema=getattr(self, 'mcfl_gate_ema', 0.9),
                    mcfl_gate_use_zscore=getattr(self, 'mcfl_gate_use_zscore', False),
                    mcfl_gate_norm_mu=getattr(self, 'mcfl_gate_norm_mu', 8.4),
                    mcfl_gate_norm_sigma=getattr(self, 'mcfl_gate_norm_sigma', 0.5),
                    mcfl_gate_z_low=getattr(self, 'mcfl_gate_z_low', -1.5),
                    mcfl_gate_z_high=getattr(self, 'mcfl_gate_z_high', 1.5),
                    mcfl_gate_lambda_max=getattr(self, 'mcfl_gate_lambda_max', 0.2),
                    mcfl_gate_norm_clip_clamp=getattr(self, 'mcfl_gate_norm_clip_clamp', True),
                    mcfl_gate_use_av_conf=getattr(self, 'mcfl_gate_use_av_conf', False),
                    mcfl_gate_av_sim_low=getattr(self, 'mcfl_gate_av_sim_low', 0.0),
                    mcfl_gate_av_sim_high=getattr(self, 'mcfl_gate_av_sim_high', 0.3),
                    mcfl_gate_av_beta=getattr(self, 'mcfl_gate_av_beta', 0.5),
                    learned_gate_refiner=self.learned_gate_refiner,
                    learned_gate_enable=self.learned_gate_enable,
                    learned_gate_detach_input=self.learned_gate_detach_input,
                )
                if use_collab and use_gate_reg:
                    c_ti, c_at, collab_loss, gate_reg_loss = build_conditions(
                        **_build_kw, return_collab_loss=True, return_gate_reg_loss=True
                    )
                    cond["mcfl_collab_loss"] = collab_loss
                    cond["learned_gate_reg_loss"] = gate_reg_loss
                elif use_collab:
                    c_ti, c_at, collab_loss = build_conditions(
                        **_build_kw, return_collab_loss=True, return_gate_reg_loss=False
                    )
                    cond["mcfl_collab_loss"] = collab_loss
                elif use_gate_reg:
                    c_ti, c_at, gate_reg_loss = build_conditions(
                        **_build_kw, return_collab_loss=False, return_gate_reg_loss=True
                    )
                    cond["learned_gate_reg_loss"] = gate_reg_loss
                else:
                    c_ti, c_at = build_conditions(**_build_kw, return_collab_loss=False, return_gate_reg_loss=False)
                c_at = c_at.to(dist_util.dev())

                s = self.step + self.resume_step  # global step

                # =====================================================
                # MCFL: conservative curriculum only when mcfl_conservative=True
                # mcfl_conservative=True: alpha schedule + MCFL freeze at 8k (saves current MCFL behavior)
                # mcfl_conservative=False: full c_at, no freeze (for use with online baseline imitation)
                # =====================================================
                if self.use_mcfl and getattr(self, "mcfl_conservative", False):
                    # Alpha schedule (final)
                    if s < 4000:
                        alpha = 0.2 * (s / 4000.0)
                    elif s < 8000:
                        alpha = 0.2 + (0.7 - 0.2) * ((s - 4000.0) / 4000.0)
                    elif s < 10000:
                        alpha = 0.5
                    else:
                        # FINAL trade-off (0.2 for better FID/FVD/FFC)
                        alpha = 0.2
                    c_at = alpha * c_at + (1.0 - alpha) * c_at.detach()
                    logger.logkv_mean("alpha_audio", alpha)

                    # Curriculum: freeze MCFL in refinement stage (8k for more visual refinement)
                    if (
                        (not self._mcfl_frozen)
                        and (s >= 8000)
                        and self.mcfl is not None
                    ):
                        for p in self.mcfl.parameters():
                            p.requires_grad = False
                        self._mcfl_frozen = True
                        logger.log("MCFL frozen at step %d for refinement stage." % s)

                # batch = batch.to(dist_util.dev())
                self.diffusion.train_step = int(s)
                self.run_step(batch, cond, c_ti, c_at)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, c, c_temp):
        self.forward_backward(batch, cond, c, c_temp)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, c, c_temp):
        self.mp_trainer.zero_grad()
        self.microbatch = self.batch_size * self.sequence_length
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            # mcfl_collab_loss 为标量，不参与切片
            micro_cond = {
                k: (
                    v.to(dist_util.dev())
                    if k in {"mcfl_collab_loss", "learned_gate_reg_loss"}
                    else v[i : i + self.microbatch].to(dist_util.dev())
                )
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                c,
                c_temp,
                model_kwargs=micro_cond,
            )

            # For baseline imitation: use same noise so x_t matches, and capture attn from MCFL forward
            if getattr(self, "use_baseline_imitation", False) and self.use_mcfl:
                micro_cond["return_attn"] = True
                noise = th.randn_like(micro)
                if last_batch or not self.use_ddp:
                    losses = compute_losses(noise=noise)
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses(noise=noise)
            else:
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # ========== OPTIONAL: Temporal Δ-Attention Imitation (use_baseline_imitation=True) ==========
            # Imitate baseline's temporal attention dynamics to reduce flicker.
            # L_attn = ((Δ_mcfl - Δ_base) ** 2).mean(), Δ = attn[:, 1:, :] - attn[:, :-1, :] (沿 query/time 维 dim 1)
            # ========================================================================
            if getattr(self, "use_baseline_imitation", False) and self.use_mcfl:
                # attn_mcfl was captured from compute_losses forward (return_attn=True)
                attn_mcfl = _get_last_temporal_attn(self.ddp_model)

                # Build c_at_baseline (no MCFL)
                c_t = c[:, :-1]  # [B, N, D]
                image_cat = c[:, -1:]  # [B, 1, D]
                c_temp_raw = c_temp[:, :8, :]  # [B*T, 8, D] - audio tokens
                _, c_at_baseline = build_conditions(
                    c_t=c_t,
                    image_cat=image_cat,
                    c_temp=c_temp_raw,
                    use_mcfl=False,
                    mcfl=None,
                )
                c_at_baseline = c_at_baseline.to(dist_util.dev())

                x_t = self.diffusion.q_sample(micro, t, noise=noise)

                # Baseline forward (no grad)
                with th.no_grad():
                    _ = self.ddp_model(
                        x_t,
                        self.diffusion._scale_timesteps(t),
                        c,
                        c_at_baseline,
                        return_attn=True,
                    )
                attn_base = _get_last_temporal_attn(self.ddp_model)

                if attn_base is not None and attn_mcfl is not None:
                    # Δ must be along query/time dim (dim 1), NOT context (dim 2)
                    # attn: [(b*h*w*heads), F, M]  F=time frames, M=context tokens
                    assert attn_mcfl.shape == attn_base.shape, (
                        f"attn shape mismatch: mcfl {attn_mcfl.shape} vs base {attn_base.shape}"
                    )
                    # F=query/time dim (通常 16), M=context dim
                    if attn_mcfl.shape[1] != 16:
                        logger.log(
                            f"WARN: attn time dim F={attn_mcfl.shape[1]} (expected 16). "
                            "Check sequence_length / token align."
                        )
                    delta_mcfl = attn_mcfl[:, 1:, :] - attn_mcfl[:, :-1, :]   # [*, F-1, M]
                    delta_base = attn_base[:, 1:, :] - attn_base[:, :-1, :]
                    loss_attn = ((delta_mcfl - delta_base.detach()) ** 2).mean()

                    # 幅度约束：限制 Δ 抖得有多狠，不改方向
                    loss_attn_energy = (delta_mcfl ** 2).mean()
                    beta = getattr(self, "attn_energy_beta", 0.005)  # 更保守：保留约束但减少对主损失的牵制

                    s = self.step + self.resume_step
                    # 更激进衰减：FVD 已受益，再强 imitation 会限制 MCFL 自由
                    if s < 3000:
                        lambda_attn = 0.05
                    elif s < 6000:
                        lambda_attn = 0.015
                    else:
                        lambda_attn = 0.002

                    loss = loss + lambda_attn * loss_attn + beta * loss_attn_energy
                    logger.logkv_mean("loss_attn", loss_attn.item())
                    logger.logkv_mean("loss_attn_energy", loss_attn_energy.item())
                    logger.logkv_mean("lambda_attn", lambda_attn)
            # ========================================================================

            # ========== OPTIONAL: Temporal Smooth (only when use_mcfl + mcfl_conservative) ==========
            # mcfl_conservative=True: lambda_temp curriculum (0 -> 0.02 over 0-10k steps).
            # mcfl_conservative=False: skip (for use with online baseline imitation).
            # ========================================================================
            if (
                self.use_mcfl
                and getattr(self, "mcfl_conservative", False)
                and getattr(self, "lambda_temp", 0.0) > 0
            ):
                T = self.sequence_length
                BT, C, H, W = micro.shape
                assert BT % T == 0, f"micro batch {BT} not divisible by sequence_length {T}"
                B = BT // T

                # reshape back to [B, T, C, H, W]
                micro_seq = micro.view(B, T, C, H, W)

                # first-order temporal difference
                diff = micro_seq[:, 1:] - micro_seq[:, :-1]   # [B, T-1, C, H, W]
                loss_temp = (diff * diff).mean()

                # =====================================================
                # Final temporal smooth curriculum (20k steps)
                # =====================================================
                s = self.step + self.resume_step

                if s < 4000:
                    # Stage 1: no temporal constraint (learn motion freely)
                    lambda_temp_now = 0.0

                elif s < 8000:
                    # Stage 2: gently suppress early jitter
                    lambda_temp_now = 0.005 * ((s - 4000.0) / 4000.0)

                elif s < 10000:
                    # Stage 3: stabilize audio-driven motion
                    lambda_temp_now = 0.005 + (0.02 - 0.005) * ((s - 8000.0) / 2000.0)

                else:
                    # Stage 4: FINAL trade-off (do NOT increase further)
                    lambda_temp_now = 0.02

                # add temporal smooth loss
                loss = loss + lambda_temp_now * loss_temp

                # optional logging
                logger.logkv_mean("lambda_temp", lambda_temp_now)
                logger.logkv_mean("loss_temp", loss_temp.item())
            # ========================================================================

            # ========== OPTIONAL: MCFL Collaborative Loss（协同损失最小化）==========
            # L_collab = 1 - cos(z_img_post, z_aud_post)，促进 MCFL 内 image/audio token 对齐
            # 开关：mcfl_collab_weight > 0 时启用
            # ========================================================================
            if self.use_mcfl and getattr(self, "mcfl_collab_weight", 0.0) > 0:
                collab_loss = cond.get("mcfl_collab_loss", None)
                if collab_loss is not None:
                    loss = loss + self.mcfl_collab_weight * collab_loss
                    logger.logkv_mean("loss_collab", collab_loss.item())
            # ========================================================================

            # ========== OPTIONAL: Learned Gate Regularization ==========
            if self.use_mcfl and getattr(self, "learned_gate_reg_weight", 0.0) > 0:
                gate_reg_loss = cond.get("learned_gate_reg_loss", None)
                if gate_reg_loss is not None:
                    loss = loss + self.learned_gate_reg_weight * gate_reg_loss
                    logger.logkv_mean("loss_gate_reg", gate_reg_loss.item())
            # ===========================================================

            # Paper-style collaborative losses (computed inside diffusion.training_losses).
            if "collab_vt" in losses:
                logger.logkv_mean("loss_collab_vt", losses["collab_vt"].mean().item())
            if "collab_va" in losses:
                logger.logkv_mean("loss_collab_va", losses["collab_va"].mean().item())
            if "collab_vi" in losses:
                logger.logkv_mean("loss_collab_vi", losses["collab_vi"].mean().item())
            if "loss_diff_base" in losses:
                logger.logkv_mean("loss_diff_base", losses["loss_diff_base"].mean().item())
            if "loss_collab_vt_weighted" in losses:
                logger.logkv_mean("loss_collab_vt_weighted", losses["loss_collab_vt_weighted"].mean().item())
            if "loss_collab_va_weighted" in losses:
                logger.logkv_mean("loss_collab_va_weighted", losses["loss_collab_va_weighted"].mean().item())
            if "loss_collab_vi_weighted" in losses:
                logger.logkv_mean("loss_collab_vi_weighted", losses["loss_collab_vi_weighted"].mean().item())
            if "collab_vi_scale" in losses:
                logger.logkv_mean("collab_vi_scale", losses["collab_vi_scale"].mean().item())
            if "collab_imit" in losses:
                logger.logkv_mean("loss_collab_imit", losses["collab_imit"].mean().item())
            if "loss_collab_imit_weighted" in losses:
                logger.logkv_mean("loss_collab_imit_weighted", losses["loss_collab_imit_weighted"].mean().item())
            if "collab_imit_scale" in losses:
                logger.logkv_mean("collab_imit_scale", losses["collab_imit_scale"].mean().item())
            if "collab_obi" in losses:
                logger.logkv_mean("loss_collab_obi", losses["collab_obi"].mean().item())
            if "loss_collab_obi_weighted" in losses:
                logger.logkv_mean(
                    "loss_collab_obi_weighted", losses["loss_collab_obi_weighted"].mean().item()
                )
            if "collab_obi_scale" in losses:
                logger.logkv_mean("collab_obi_scale", losses["collab_obi_scale"].mean().item())
            if "collab_obi_mean" in losses:
                logger.logkv_mean("collab_obi_mean", losses["collab_obi_mean"].mean().item())
            if "debug_obi_raw" in losses:
                logger.logkv_mean("debug_obi_raw", losses["debug_obi_raw"].mean().item())
            if "debug_obi_scale_eff" in losses:
                logger.logkv_mean("debug_obi_scale_eff", losses["debug_obi_scale_eff"].mean().item())
            if "debug_obi_final" in losses:
                logger.logkv_mean("debug_obi_final", losses["debug_obi_final"].mean().item())
            if "debug_reliability_mean" in losses:
                logger.logkv_mean("debug_reliability_mean", losses["debug_reliability_mean"].mean().item())
            if "dyn_av_conf_mean" in losses:
                logger.logkv_mean("dyn_av_conf_mean", losses["dyn_av_conf_mean"].mean().item())
            if "dyn_av_sim_mean" in losses:
                logger.logkv_mean("dyn_av_sim_mean", losses["dyn_av_sim_mean"].mean().item())
            if "collab_scale" in losses:
                logger.logkv_mean("collab_scale", losses["collab_scale"].mean().item())
            if "dyn_stride" in losses:
                logger.logkv_mean("dyn_stride", losses["dyn_stride"].mean().item())
            if "dyn_video_mag" in losses:
                logger.logkv_mean("dyn_video_mag", losses["dyn_video_mag"].mean().item())
            if "dyn_audio_mag" in losses:
                logger.logkv_mean("dyn_audio_mag", losses["dyn_audio_mag"].mean().item())
            if "dyn_gate_mean" in losses:
                logger.logkv_mean("dyn_gate_mean", losses["dyn_gate_mean"].mean().item())
            if "dyn_cos_mean" in losses:
                logger.logkv_mean("dyn_cos_mean", losses["dyn_cos_mean"].mean().item())
            if "dyn_mag_gap" in losses:
                logger.logkv_mean("dyn_mag_gap", losses["dyn_mag_gap"].mean().item())
            if "loss_smooth" in losses:
                logger.logkv_mean("loss_smooth", losses["loss_smooth"].mean().item())

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # Bundle MCFL / learned gate / attn pools so sampling can load the same weights.
            if (
                self.mcfl is not None
                or self.learned_gate_refiner is not None
                or self.attn_pool_text is not None
            ):
                payload = {"model": state_dict}
                if self.mcfl is not None:
                    payload["mcfl"] = self.mcfl.state_dict()
                if self.learned_gate_refiner is not None:
                    payload["learned_gate_refiner"] = self.learned_gate_refiner.state_dict()
                if self.attn_pool_text is not None:
                    payload["attn_pool_text"] = self.attn_pool_text.state_dict()
                    payload["attn_pool_audio"] = self.attn_pool_audio.state_dict()
                to_save = payload
            else:
                to_save = state_dict
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                    th.save(to_save, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
