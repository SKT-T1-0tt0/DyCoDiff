"""
Generate a large batch of video samples from a model.
"""
import argparse
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import time
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms.functional as F
import torch.nn.functional as Func

from diffusion import dist_util, logger
from diffusion.condition_builder import build_conditions
from diffusion.tacm_script_temp_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.dist_util import save_video_grid
from tacm import VideoData
from tacm.modules.learned_gate import LearnedGateRefiner
from tacm.download import load_vqgan
from einops import rearrange, repeat
import wav2clip
# from tacm import AudioCLIP
from beats.BEATs import BEATs, BEATsConfig
from optimization.video_editor import VideoEditor
# from optimization.video_editor_simple import VideoEditor  # 已改为使用 video_editor.py
from optimization.arguments import get_arguments

import soundfile
from shutil import copyfile

import transformers.image_transforms
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def _split_train_checkpoint(raw):
    """Legacy flat UNet dict vs bundled checkpoint from TrainLoop.save()."""
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw
    return {"model": raw}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def d_clip_loss(x, y, use_cosine=False):
    x = th.nn.functional.normalize(x, dim=-1)
    y = th.nn.functional.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance

                
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    ckpt_raw = dist_util.load_state_dict(args.model_path, map_location="cpu")
    ckpt = _split_train_checkpoint(ckpt_raw)
    model.load_state_dict(ckpt["model"])
    model.to(dist_util.dev())
    model.eval()

    processor = CLIPProcessor.from_pretrained("tacm/modules/cache/clip-vit-large-patch14")
    clipmodel = CLIPModel.from_pretrained("tacm/modules/cache/clip-vit-large-patch14").to(dist_util.dev())
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.test_dataloader()
    
    # load audio
    logger.log("loading audio embedding model...")
    # if args.audio_emb_model == 'audioclip':
    #     audioclip_model = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
    #     audioclip_model = audioclip_model.to(dist_util.dev())
    # if args.audio_emb_model == 'wav2clip':
    #     wav2clip_model = wav2clip.get_model()
    #     wav2clip_model = wav2clip_model.to(dist_util.dev())
    #     for p in wav2clip_model.parameters():
    #         p.requires_grad = False
    if args.audio_emb_model == 'beats':
        checkpoint = th.load('saved_ckpts/BEATs_iter3_plus_AS20K.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model = BEATs_model.to('cpu')  # 固定在 CPU 上运行，避免 cuFFT 错误
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
    
    # Initialize MCFL if enabled
    mcfl = None
    attn_pool_text = None
    attn_pool_audio = None
    mcfl_pooling_mode = getattr(args, 'mcfl_pooling_mode', 'mean')
    
    learned_gate_refiner = None
    if args.use_mcfl:
        from tacm import MCFL, AttnPool
        mcfl = MCFL(
            embed_dim=args.mcfl_embed_dim,
            num_heads=8,
            dropout=0.1  # Changed from 0.0 to 0.1 to prevent overfitting
        ).to(dist_util.dev())
        
        # Initialize attention pooling modules if using attention pooling
        if mcfl_pooling_mode == "attention":
            attn_pool_text = AttnPool(dim=args.mcfl_embed_dim).to(dist_util.dev())
            attn_pool_audio = AttnPool(dim=args.mcfl_embed_dim).to(dist_util.dev())

        if getattr(args, "learned_gate_enable", False):
            learned_gate_refiner = LearnedGateRefiner(
                in_dim=4,
                hidden_dim=getattr(args, "learned_gate_hidden_dim", 16),
                dropout=getattr(args, "learned_gate_dropout", 0.0),
            ).to(dist_util.dev())

        # Load MCFL / learned gate / attn pools from bundled checkpoint (if present)
        if "mcfl" in ckpt:
            mcfl.load_state_dict(ckpt["mcfl"], strict=True)
        if learned_gate_refiner is not None and "learned_gate_refiner" in ckpt:
            learned_gate_refiner.load_state_dict(ckpt["learned_gate_refiner"], strict=True)
        if mcfl_pooling_mode == "attention":
            if "attn_pool_text" in ckpt:
                attn_pool_text.load_state_dict(ckpt["attn_pool_text"], strict=True)
            if "attn_pool_audio" in ckpt:
                attn_pool_audio.load_state_dict(ckpt["attn_pool_audio"], strict=True)

        if dist.get_rank() == 0:
            if "mcfl" not in ckpt and mcfl is not None:
                logger.log(
                    "WARN: checkpoint has no 'mcfl' weights; MCFL is randomly initialized. "
                    "Re-save with latest TrainLoop.save() (bundled ckpt) or re-train."
                )
            if (
                getattr(args, "learned_gate_enable", False)
                and learned_gate_refiner is not None
                and "learned_gate_refiner" not in ckpt
            ):
                logger.log(
                    "WARN: learned_gate_enable=True but checkpoint has no 'learned_gate_refiner'; "
                    "gate MLP is random. Use a checkpoint saved after bundling fix, or disable."
                )

    # sampling
    for i in range(args.num_samples):
        batch = data.dataset.__getitem__(i) #sample_id
        
        os.makedirs('./results/%d_tacm_%s/real/'%(args.run, args.dataset), exist_ok=True)
        save_video_grid(th.clamp(batch['video'].unsqueeze(0), -0.5, 0.5) + 0.5, os.path.join('./results/%d_tacm_%s/real/'%(args.run, args.dataset), 'groundtruth_%d.mp4'%(i)),
                        1, fps=6)
        
        # get text from text_data
        c_t = batch['text'].to(dist_util.dev()) #torch.Size([1, 77, 768])
        raw_text = batch['raw_text'] 
        
        # get image
        image = batch['video'][:,0]+0.5
        image = image.unsqueeze(0)
        image_cat=None
        for j in range(image.shape[0]):
            image_j = transformers.image_transforms.to_pil_image(image[j])
            image_input = processor(images=image_j, return_tensors="pt", padding=True).to(dist_util.dev())
            with th.no_grad():
                image_features = clipmodel.get_image_features(image_input.pixel_values)

                if image_cat is None:
                    image_cat = image_features.unsqueeze(0)
                else:
                    image_cat = th.concat((image_cat, image_features), dim=0) #torch.Size([1, 1, 768])
        c_ti = th.concat((c_t,image_cat), dim=1)
        #c_i = image_cat
        
        
        if args.audio_emb_model == 'STFT':
            stft = batch['stft']
        else:
            audio = batch['audio'].to(dist_util.dev()) 
               
        # if args.audio_emb_model == 'audioclip':
        #     ((audio_embed, _, _), _), _ = audioclip_model(audio=audio)
        #     c_temp = audio_embed.unsqueeze(0) #(1,16,1024)
        # if args.audio_emb_model == 'wav2clip':
        #     audio_embed = th.from_numpy(wav2clip.embed_audio(audio.cpu().numpy(), wav2clip_model)) #(16,512)
        #     c_temp = audio_embed.unsqueeze(1) #(16,1,512)
        if args.audio_emb_model == 'STFT':
            c_temp = stft
        elif args.audio_emb_model == 'beats':
            audio = rearrange(audio.unsqueeze(0), "b f g -> (b f) g")
            # 与训练一致：BEATs 前单次 response（数据侧 audio_norm_mode/soft_clip 当前工程未接入，勿传）
            resp = getattr(args, 'audio_response', 'compand')
            if resp == 'tanh':
                audio = th.tanh(audio)
            elif resp == 'compand':
                mu = 5.0
                audio = th.sign(audio) * th.log1p(mu * th.abs(audio)) / math.log1p(mu)
            # 将音频移到 CPU 上进行处理，因为 BEATs 模型在 CPU 上
            c_temp = BEATs_model.extract_features(audio.cpu(), padding_mask=None)[0] #torch.Size([16, 8, 768])
            # 处理完成后移回 GPU
            c_temp = c_temp.to(dist_util.dev())

        # 与 train_temp 一致：对 BEATs token 做时间维 EMA，减轻条件帧间跳变
        T = getattr(args, "sequence_length", 16)
        if c_temp.dim() == 3 and c_temp.shape[0] % T == 0:
            BT, M, D = c_temp.shape
            B = BT // T
            c_temp_reshaped = c_temp.view(B, T, M, D)
            alpha = 0.9
            c_smoothed = c_temp_reshaped.clone()
            for t in range(1, T):
                c_smoothed[:, t] = alpha * c_smoothed[:, t - 1] + (1 - alpha) * c_temp_reshaped[:, t]
            c_temp = c_smoothed.view(BT, M, D)
        
        # Use common condition builder (shared with training)
        # MCFL v2-A: Temporal-only + Gated Residual
        c_ti, c_at = build_conditions(
            c_t=c_t,
            image_cat=image_cat,
            c_temp=c_temp,
            mcfl=mcfl,
            use_mcfl=args.use_mcfl,
            pooling_mode=mcfl_pooling_mode,
            attn_pool_text=attn_pool_text,
            attn_pool_audio=attn_pool_audio,
            mcfl_gate_lambda=getattr(args, 'mcfl_gate_lambda', 0.2),
            mcfl_norm_modality=getattr(args, 'mcfl_norm_modality', True),
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
            learned_gate_refiner=learned_gate_refiner,
            learned_gate_enable=getattr(args, "learned_gate_enable", False),
            learned_gate_detach_input=getattr(args, "learned_gate_detach_input", True),
        )
        c_at = c_at.to(dist_util.dev())
        #c_temp = c_temp.to(dist_util.dev()) 

        init_video = batch['video'].unsqueeze(0)  # [1, C, T, H, W]
        init_video = rearrange(init_video, "b c t h w -> (b t) c h w")
        # Dataset resolution (default 96) may differ from model image_size (e.g. 64).
        # Diffusion q_sample requires init_image and noise to match exactly.
        ih, iw = init_video.shape[-2], init_video.shape[-1]
        if ih != args.image_size or iw != args.image_size:
            init_video = Func.interpolate(
                init_video,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )
        seq_len = getattr(args, "sequence_length", 16)
        n_frames = init_video.shape[0]
        need_frames = args.batch_size * seq_len
        if n_frames != need_frames:
            if n_frames > need_frames:
                init_video = init_video[:need_frames]
            else:
                last = init_video[-1:].clone()
                init_video = th.cat(
                    [init_video, last.repeat(need_frames - n_frames, 1, 1, 1)], dim=0
                )
        zeros = th.zeros_like(init_video)
        init_video[1:, :, :, :] = zeros[1:, :, :, :]

        logger.log("sampling...")
        t1 = time.time()    
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            logger.log(classes)
            model_kwargs["y"] = classes
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size * seq_len, args.in_channels, args.image_size, args.image_size),
            c_ti,
            c_at,
            clip_denoised=args.clip_denoised,
            cond_fn=None,
            model_kwargs=model_kwargs,
            progress=True,
            skip_timesteps=100,
            init_image=init_video.to(dist_util.dev()),
            #init_image=None,
        )
        
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #sample = sample.permute(0, 2, 3, 4, 1)
        #sample = sample.contiguous()
        #print('samples:', sample.shape) #torch.Size([1, 256, 16, 8, 8])

        if args.in_channels == 3:
            sample = rearrange(sample, '(b f) c h w -> b c f h w', f=seq_len)
            sample_recon = th.clamp(sample, -0.5, 0.5)
            
        logger.log("save to mp4 format...")
        os.makedirs("./results/%d_tacm_%s/fake1_6fps"%(args.run, args.dataset), exist_ok=True)
        save_video_grid(sample_recon+0.5, os.path.join("./results/%d_tacm_%s"%(args.run, args.dataset), "fake1_6fps", f"video_%d.mp4"%(i)),
                        1, fps=6)

        os.makedirs("./results/%d_tacm_%s/fake1_30fps" % (args.run, args.dataset), exist_ok=True)
        save_video_grid(sample_recon + 0.5,
                        os.path.join("./results/%d_tacm_%s" % (args.run, args.dataset), "fake1_30fps", f"video_%d.mp4" % (i)),
                        1, fps=30)

        os.makedirs('./results/%d_tacm_%s/txt/'%(args.run, args.dataset), exist_ok=True)
        copyfile(batch['path'].replace("/mp4/", "/txt/").replace(".mp4", ".txt"), os.path.join('./results/%d_tacm_%s/txt/'%(args.run, args.dataset), 'groundtruth_%d.txt'%(i)))
        
        os.makedirs('./results/%d_tacm_%s/audio/'%(args.run, args.dataset), exist_ok=True)
        soundfile.write(os.path.join('./results/%d_tacm_%s/audio/'%(args.run, args.dataset), 'groundtruth_%d.wav'%(i)), batch['audio'].reshape(-1).numpy(), 96000)


        #video editing - 暂时关闭
        # video = sample_recon.squeeze()
        # video = Func.interpolate(video, size=(128, 128), mode='bilinear',align_corners=False)
        # logger.log("creating video editor...")
        # video_editor = VideoEditor(args)
        # pred_video = video_editor.edit_video_by_prompt(video, audio=None, raw_text=None, text=batch['text'].to(dist_util.dev()))

        # logger.log("save to mp4 format...")
        # os.makedirs("./results/%d_tacm_%s/fake2_6fps"%(args.run, args.dataset), exist_ok=True)
        # save_video_grid(pred_video+0.5, os.path.join("./results/%d_tacm_%s"%(args.run, args.dataset), "fake2_6fps", f"video_%d.mp4"%(i)), 1)
        '''
        true_video = th.clamp(batch['video'], -0.5, 0.5)
        true_video = Func.interpolate(true_video, size=(128, 128), mode='bilinear',align_corners=False)
        pred_true_video = video_editor.edit_video_by_prompt(true_video, audio=None, raw_text=None, text=c_t)
        
        os.makedirs('./results/%d_tacm/real_stage2/'%(args.run), exist_ok=True)
        save_video_grid(pred_true_video, os.path.join('./results/%d_tacm/real_stage2/'%(args.run), 'groundtruth_%d.mp4'%(i)), 1)
        '''

    #dist.barrier()
    logger.log("sampling complete")
    t2 = time.time()
    sampling_time = t2 - t1
    logger.log(f"sampling time: {sampling_time:.2f} seconds.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        #batch_size=8,
        use_ddim=False,
        model_path="",
        vqgan_ckpt="",
        run=0,
        dataset="",
        use_mcfl=False,  # MCFL flag: enable multi-modal condition fusion
        mcfl_embed_dim=768,  # MCFL embedding dimension (must match condition dimension)
        mcfl_pooling_mode="mean",  # Pooling mode: "mean" (average) or "attention" (learned attention weights)
        mcfl_gate_lambda=0.1,  # 与训练默认一致
        mcfl_norm_modality=True,
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
        # 新增：audio-visual agreement gate 因子（默认关闭）
        mcfl_gate_use_av_conf=False,
        mcfl_gate_av_sim_low=0.0,
        mcfl_gate_av_sim_high=0.3,
        mcfl_gate_av_beta=0.5,
        audio_response='compand',  # 与训练稳健默认一致（单次压缩）
        learned_gate_enable=False,
        learned_gate_hidden_dim=16,
        learned_gate_dropout=0.0,
        learned_gate_detach_input=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = VideoData.add_data_specific_args(parser)
    parser = get_arguments(parser)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
