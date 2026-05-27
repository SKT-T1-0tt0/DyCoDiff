"""
Temporal diffusion training (train_temp).
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import-time backend for diffusion/respace_temp; must run before diffusion imports.
if "--use_gan_diffusion" in sys.argv:
    os.environ["TACM_USE_GAN_DIFFUSION"] = "1"
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pytorch_lightning as pl

from diffusion.resample import create_named_schedule_sampler
from diffusion import dist_util, logger
from diffusion.tacm_script_temp_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.tacm_train_temp_util import TrainLoop
from diffusion.dist_util import save_video_grid
from tacm import VideoData
from einops import rearrange, repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _apply_paper_collab_mode(args):
    """
    论文 3.2「协同损失最小化 / 协同均衡」专用实验：
    训练目标仅保留 L_diff + α·L_vt + β·L_va + γ·L_vi（由 OmniEncoder 与 collab_* 权重控制）。
    关闭与论文主式无关的附加项（MCFL 内部 collab、imitation、gate 正则、保守课程等）。
    传 --paper_collab_mode False 可恢复手动控制上述开关。
    若 paper_collab_no_mcfl=True（默认）：强制 --use_mcfl False，条件走 baseline（音频 token + 文本展开），
    不受旧版 MCFL 融合与 gate 等影响；若仍要与 MCFL 融合一起训，请传 --paper_collab_no_mcfl False --use_mcfl True。
    若 paper_collab_hybrid_legacy=True：保留 legacy MCFL / imitation / AV gate / 内部 collab 等辅助项，
    用于“新 collab + 旧框架”混合实验。
    """
    if not getattr(args, "paper_collab_mode", True):
        return
    hybrid_legacy = bool(getattr(args, "paper_collab_hybrid_legacy", False))
    if not hybrid_legacy:
        args.use_baseline_imitation = False
        args.mcfl_collab_weight = 0.0
        args.learned_gate_enable = False
        args.learned_gate_reg_weight = 0.0
        args.lambda_temp = 0.0
        args.mcfl_conservative = False
        args.mcfl_gate_use_av_conf = False
        args.modality_dropout_prob = 0.0
        args.audio_random_gain = False
        args.audio_random_response_strength = False
    if getattr(args, "paper_collab_no_mcfl", True):
        prev_mcfl = bool(getattr(args, "use_mcfl", False))
        args.use_mcfl = False
        if prev_mcfl:
            # 供 main 打日志：命令行曾开 MCFL，已被论文干净实验策略覆盖
            setattr(args, "_paper_collab_overrode_mcfl", True)


def main():
    args = create_argparser().parse_args()
    _apply_paper_collab_mode(args)

    dist_util.setup_dist()
    logger.configure()
    if dist.get_rank() == 0:
        _gan = os.environ.get("TACM_USE_GAN_DIFFUSION", "").lower() in ("1", "true", "yes")
        logger.log(
            "diffusion backend: %s"
            % ("tacm_gaussian_diffusion_gan" if _gan else "tacm_gaussian_diffusion_temp (paper L_vt/L_va when weights>0)")
        )
    if getattr(args, "paper_collab_mode", True) and dist.get_rank() == 0:
        if getattr(args, "paper_collab_hybrid_legacy", False):
            msg = (
                "paper_collab_mode=True + paper_collab_hybrid_legacy=True: "
                "启用新 collab 主线，并保留 legacy MCFL / imitation / AV gate / 内部 collab 等辅助项。"
            )
        else:
            msg = (
                "paper_collab_mode=True: 仅论文协同项（扩散主损失 + L_vt/L_va/L_vi）；"
                "已关闭 imitation / MCFL 内部 collab / learned gate / lambda_temp / "
                "保守课程 / AV gate / 音频随机增广与 modality dropout。"
            )
        if getattr(args, "paper_collab_no_mcfl", True):
            msg += " paper_collab_no_mcfl=True: 已关闭 MCFL 融合（use_mcfl=False），条件为 baseline 拼接。"
        if getattr(args, "_paper_collab_overrode_mcfl", False):
            msg += " （已忽略命令行 --use_mcfl True，避免 MCFL 与 L_vt/L_va 同时拉扯。）"
        logger.log(msg)

    logger.log("creating model and diffusion...")
    new_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # load original model parameters
    original_model_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # get new model parameters dictionary
    new_model_dict = new_model.state_dict()
    # delete keys in original parameters which are not same as new_model
    pretrained_dict = {k: v for k, v in original_model_dict.items() if k in new_model_dict}
    # keep original parameters (spatial layers) fixed
    for v in original_model_dict.values():
        v.requires_grad = False
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)

    #print(new_model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.train_dataloader()

    logger.log("training...")
    TrainLoop(
        model=new_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        vqgan_ckpt=args.vqgan_ckpt,
        sequence_length=args.sequence_length,
        audio_emb_model=args.audio_emb_model,
        use_mcfl=getattr(args, 'use_mcfl', False),
        mcfl_embed_dim=getattr(args, 'mcfl_embed_dim', 768),
        mcfl_pooling_mode=getattr(args, 'mcfl_pooling_mode', 'mean'),  # "mean" or "attention"
        mcfl_gate_lambda=getattr(args, 'mcfl_gate_lambda', 0.1),  # MCFL v2-A gate (0.1 降低 TC_FLICKER)
        lambda_temp=getattr(args, 'lambda_temp', 0.0),  # Temporal smooth regularization weight (default 0.0 = disabled)
        mcfl_conservative=getattr(args, 'mcfl_conservative', False),  # False：与论文协同损失实验一致，不做 alpha/冻结课程；旧行为请传 True
        use_baseline_imitation=getattr(args, 'use_baseline_imitation', False),  # Online baseline imitation (implement when needed)
        mcfl_norm_modality=getattr(args, 'mcfl_norm_modality', True),
        audio_response=getattr(args, 'audio_response', 'tanh'),
        audio_random_gain=getattr(args, 'audio_random_gain', True),
        audio_gain_range=(getattr(args, 'audio_gain_low', 0.25), getattr(args, 'audio_gain_high', 4.0)),
        audio_random_response_strength=getattr(args, 'audio_random_response_strength', True),
        modality_dropout_prob=getattr(args, 'modality_dropout_prob', 0.2),
        mcfl_gate_adaptive=getattr(args, 'mcfl_gate_adaptive', True),
        mcfl_gate_norm_low=getattr(args, 'mcfl_gate_norm_low', 7.2),
        mcfl_gate_norm_high=getattr(args, 'mcfl_gate_norm_high', 10.0),
        mcfl_gate_time_smooth=getattr(args, 'mcfl_gate_time_smooth', True),
        mcfl_gate_ema=getattr(args, 'mcfl_gate_ema', 0.9),
        mcfl_gate_use_zscore=getattr(args, 'mcfl_gate_use_zscore', False),
        mcfl_gate_norm_mu=getattr(args, 'mcfl_gate_norm_mu', 8.4),
        mcfl_gate_norm_sigma=getattr(args, 'mcfl_gate_norm_sigma', 0.5),
        mcfl_gate_z_low=getattr(args, 'mcfl_gate_z_low', -1.5),
        mcfl_gate_z_high=getattr(args, 'mcfl_gate_z_high', 1.5),
        mcfl_gate_lambda_max=getattr(args, 'mcfl_gate_lambda_max', 0.2),
        mcfl_gate_norm_clip_clamp=getattr(args, 'mcfl_gate_norm_clip_clamp', True),
        mcfl_gate_use_av_conf=getattr(args, 'mcfl_gate_use_av_conf', False),
        mcfl_gate_av_sim_low=getattr(args, 'mcfl_gate_av_sim_low', 0.0),
        mcfl_gate_av_sim_high=getattr(args, 'mcfl_gate_av_sim_high', 0.3),
        mcfl_gate_av_beta=getattr(args, 'mcfl_gate_av_beta', 0.5),
        mcfl_collab_weight=getattr(args, 'mcfl_collab_weight', 0.0),  # 协同损失权重，0=关闭
        learned_gate_enable=getattr(args, 'learned_gate_enable', False),
        learned_gate_hidden_dim=getattr(args, 'learned_gate_hidden_dim', 16),
        learned_gate_dropout=getattr(args, 'learned_gate_dropout', 0.0),
        learned_gate_detach_input=getattr(args, 'learned_gate_detach_input', True),
        learned_gate_reg_weight=getattr(args, 'learned_gate_reg_weight', 0.0),
        collab_vt_weight=getattr(args, 'collab_vt_weight', 0.10),
        collab_va_weight=getattr(args, 'collab_va_weight', 0.01),
        collab_vi_weight=getattr(args, 'collab_vi_weight', 0.0),
        omni_shared_dim=getattr(args, 'omni_shared_dim', 512),
        collab_metric=getattr(args, 'collab_metric', 'cosine'),
        omni_encoder_ckpt=getattr(args, 'omni_encoder_ckpt', ''),
        collab_start_step=getattr(args, 'collab_start_step', 3000),
        collab_warmup_steps=getattr(args, 'collab_warmup_steps', 10000),
        collab_max_scale=getattr(args, 'collab_max_scale', 1.0),
        collab_max_ratio=getattr(args, 'collab_max_ratio', 0.10),
        collab_margin_vt=getattr(args, 'collab_margin_vt', 0.20),
        collab_margin_va=getattr(args, 'collab_margin_va', 0.10),
        collab_dyn_stride=getattr(args, 'collab_dyn_stride', 1),
        collab_dyn_sample_tau=getattr(args, 'collab_dyn_sample_tau', 0.03),
        collab_dyn_sample_temp=getattr(args, 'collab_dyn_sample_temp', 0.01),
        collab_dyn_gate_k=getattr(args, 'collab_dyn_gate_k', 2.0),
        collab_dyn_beta=getattr(args, 'collab_dyn_beta', 3.0),
        collab_dyn_rank_mode=getattr(args, 'collab_dyn_rank_mode', 'hybrid'),
        collab_dyn_rank_low_q=getattr(args, 'collab_dyn_rank_low_q', 0.2),
        collab_dyn_rank_high_q=getattr(args, 'collab_dyn_rank_high_q', 0.8),
        collab_dyn_rank_sharpness=getattr(args, 'collab_dyn_rank_sharpness', 10.0),
        collab_dyn_hist_size=getattr(args, 'collab_dyn_hist_size', 256),
        collab_dyn_score_mix_av_ratio=getattr(args, 'collab_dyn_score_mix_av_ratio', 0.7),
        collab_dyn_score_mix_audio=getattr(args, 'collab_dyn_score_mix_audio', 0.3),
        collab_av_score_mix_av_ratio=getattr(args, 'collab_av_score_mix_av_ratio', 0.5),
        collab_av_score_mix_av_conf=getattr(args, 'collab_av_score_mix_av_conf', 0.5),
        collab_av_conf_beta=getattr(args, 'collab_av_conf_beta', 0.5),
        collab_av_conf_sim_low=getattr(args, 'collab_av_conf_sim_low', 0.0),
        collab_av_conf_sim_high=getattr(args, 'collab_av_conf_sim_high', 0.3),
        collab_audio_ema_alpha=getattr(args, 'collab_audio_ema_alpha', 0.9),
        collab_obi_scale=getattr(args, 'collab_obi_scale', 0.0),
        collab_obi_start_step=getattr(args, 'collab_obi_start_step', 0),
        collab_obi_warmup_steps=getattr(args, 'collab_obi_warmup_steps', 10000),
        collab_obi_layer_name=getattr(args, 'collab_obi_layer_name', 'temporal_last'),
        collab_obi_min_weight=getattr(args, 'collab_obi_min_weight', 0.05),
        collab_obi_min_scale=getattr(args, 'collab_obi_min_scale', 0.1),
        collab_imit_scale=getattr(args, 'collab_imit_scale', 0.0),
        collab_imit_start_step=getattr(args, 'collab_imit_start_step', 0),
        collab_imit_warmup_steps=getattr(args, 'collab_imit_warmup_steps', 10000),
        baseline_model_path=getattr(args, 'baseline_model_path', ''),
        collab_smooth_weight=getattr(args, 'collab_smooth_weight', 0.0),
        image_size=args.image_size,
        in_channels=args.in_channels,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # MCFL switches (all off = baseline; pass --use_mcfl to enable MCFL)
        use_mcfl=False,
        mcfl_embed_dim=768,
        mcfl_pooling_mode="mean",
        mcfl_gate_lambda=0.1,
        lambda_temp=0.0,
        mcfl_conservative=False,  # False：全程完整 MCFL，不冻结；True：alpha 课程 + 8k 冻结 MCFL + 可选 lambda_temp 课程
        use_baseline_imitation=False,  # Online baseline output imitation (placeholder; implement when needed)
        mcfl_norm_modality=True,
        audio_response='compand',
        audio_random_gain=True,
        audio_gain_low=0.5,
        audio_gain_high=2.0,
        audio_random_response_strength=False,
        modality_dropout_prob=0.2,
        mcfl_gate_adaptive=True,
        mcfl_gate_norm_low=7.2,
        mcfl_gate_norm_high=10.0,
        mcfl_gate_time_smooth=True,
        mcfl_gate_ema=0.9,
        mcfl_gate_use_zscore=True,
        mcfl_gate_norm_mu=8.4,
        mcfl_gate_norm_sigma=0.5,
        mcfl_gate_z_low=-1.5,
        mcfl_gate_z_high=1.5,
        mcfl_gate_lambda_max=0.2,
        mcfl_gate_norm_clip_clamp=True,
        # 新增：audio-visual agreement gate 因子（默认关闭，显式打开更安全）
        mcfl_gate_use_av_conf=False,
        mcfl_gate_av_sim_low=0.0,
        mcfl_gate_av_sim_high=0.3,
        mcfl_gate_av_beta=0.5,
        # 协同损失：统一跨数据集默认先关闭内部 collab，避免与主扩散/论文协同项同时拉扯
        mcfl_collab_weight=0.0,
        # learned refinement gate: g = g_hand * sigmoid(MLP(x))
        learned_gate_enable=False,
        learned_gate_hidden_dim=16,
        learned_gate_dropout=0.0,
        learned_gate_detach_input=True,
        learned_gate_reg_weight=0.0,
        # Paper-style collaborative losses:
        # L_total = L_diff + alpha*L_vt + beta*L_va + gamma*L_vi (+ optional MCFL internal regularizer)
        collab_vt_weight=0.0002,
        collab_va_weight=0.0,
        collab_vi_weight=0.0,
        omni_shared_dim=512,
        collab_metric="cosine",
        # 与 OmniEncoder 维度一致；可用 scripts/save_omni_encoder_checkpoint.py 生成；可改为你自己的预训练路径
        omni_encoder_ckpt="saved_ckpts/omni_encoder.pt",
        collab_start_step=12000,
        collab_warmup_steps=5000,
        collab_max_scale=1.0,
        collab_max_ratio=0.10,
        collab_margin_vt=0.20,
        collab_margin_va=0.10,
        # 复原清单：须为 1；勿改为 2/4（否则 dynamic_magnitude 等分支会偏离逐帧设定）
        collab_dyn_stride=1,
        collab_dyn_sample_tau=0.03,
        collab_dyn_sample_temp=0.01,
        collab_dyn_gate_k=2.0,
        collab_dyn_beta=3.0,
        collab_dyn_rank_mode="hybrid",
        collab_dyn_rank_low_q=0.2,
        collab_dyn_rank_high_q=0.8,
        collab_dyn_rank_sharpness=10.0,
        collab_dyn_hist_size=256,
        collab_dyn_score_mix_av_ratio=0.7,
        collab_dyn_score_mix_audio=0.3,
        collab_av_score_mix_av_ratio=0.5,
        collab_av_score_mix_av_conf=0.5,
        collab_av_conf_beta=0.5,
        collab_av_conf_sim_low=0.0,
        collab_av_conf_sim_high=0.3,
        collab_audio_ema_alpha=0.9,
        collab_obi_scale=0.0,
        collab_obi_start_step=0,
        collab_obi_warmup_steps=10000,
        collab_obi_layer_name="temporal_last",
        collab_obi_min_weight=0.05,
        collab_obi_min_scale=0.1,
        collab_imit_scale=0.0,
        collab_imit_start_step=0,
        collab_imit_warmup_steps=10000,
        baseline_model_path="",
        collab_smooth_weight=0.0,
        # True：仅论文 3.2 协同损失实验（见 _apply_paper_collab_mode）；False：自行组合其它正则
        paper_collab_mode=True,
        # True（默认）：论文干净实验下不启用 MCFL 融合，避免 legacy 条件分支干扰 L_vt/L_va 消融
        paper_collab_no_mcfl=True,
        # True：在 paper_collab_mode 下保留 legacy MCFL / imitation / gate / 内部 collab 等辅助项
        paper_collab_hybrid_legacy=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--vqgan_ckpt', type=str)
    parser.add_argument(
        "--use_gan_diffusion",
        action="store_true",
        help="Use tacm_gaussian_diffusion_gan (discriminator in training_losses). Disables paper L_vt/L_va; must appear on argv so import sees it.",
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
