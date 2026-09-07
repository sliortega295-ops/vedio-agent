# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for sparse frame generation (FIS-DiT style)
import argparse
import gc
import logging
import math
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

_DEFAULT_WAN_SOURCE = Path(__file__).resolve().parent
_WAN_SOURCE = Path(os.environ.get("SPARSEMAC_WAN_SOURCE_ROOT", _DEFAULT_WAN_SOURCE))
if str(_WAN_SOURCE) not in sys.path:
    sys.path.insert(0, str(_WAN_SOURCE))

warnings.filterwarnings('ignore')

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video


def interpolate_non_key_frames(key_latents, key_indices, total_frames):
    """
    从键帧 latent 通过线性插值恢复全帧 latent。

    Args:
        key_latents: Tensor [C, K, H, W]，K 个键帧的 latent
        key_indices: List[int]，键帧在全帧中的索引 [0, 2, 4, ..., 20]
        total_frames: int，全帧总数 (21)

    Returns:
        full_latents: Tensor [C, total_frames, H, W]
    """
    C, K, H, W = key_latents.shape
    full_latents = key_latents.new_zeros(C, total_frames, H, W)

    # 放置键帧
    for i, idx in enumerate(key_indices):
        full_latents[:, idx] = key_latents[:, i]

    # 线性插值非键帧
    for i in range(len(key_indices) - 1):
        idx_a = key_indices[i]
        idx_b = key_indices[i + 1]
        for j in range(idx_a + 1, idx_b):
            alpha = (j - idx_a) / (idx_b - idx_a)
            full_latents[:, j] = (1 - alpha) * full_latents[:, idx_a] + alpha * full_latents[:, idx_b]

    return full_latents


def generate_sparse(
    config,
    checkpoint_dir,
    prompt,
    size=(832, 480),
    frame_num=81,
    shift=5.0,
    sampling_steps=50,
    guide_scale=5.0,
    seed=-1,
    save_file=None,
    device_id=0,
    sparse_stride=2,
    key_only=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    save_warmup_latent=None,
):
    """
    使用稀疏帧策略生成视频。

    可选 warmup 阶段先对完整 latent temporal path 去噪若干步，
    之后切到键帧去噪。key_only 模式下最终仍只解码键帧 latent。
    """
    device = torch.device(f"cuda:{device_id}")

    # ============ 加载模型 ============
    logging.info("Loading T5 text encoder...")
    t5_start = time.time()
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
    )
    logging.info(f"T5 loaded in {time.time() - t5_start:.1f}s")

    logging.info("Loading VAE...")
    vae = WanVAE(
        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        device=device,
    )

    logging.info("Loading DiT model...")
    dit_start = time.time()
    model = WanModel.from_pretrained(checkpoint_dir)
    model.eval().requires_grad_(False)
    model.to(device)
    logging.info(f"DiT loaded in {time.time() - dit_start:.1f}s")

    # ============ 预处理 ============
    vae_stride = config.vae_stride
    patch_size = config.patch_size

    F = frame_num  # 81
    total_latent_frames = (F - 1) // vae_stride[0] + 1  # 21
    lat_h = size[1] // vae_stride[1]  # 480 // 8 = 60
    lat_w = size[0] // vae_stride[2]  # 832 // 8 = 104
    z_dim = vae.model.z_dim

    target_shape = (z_dim, total_latent_frames, lat_h, lat_w)

    # 键帧索引 (隐空间帧)：[0, 2, 4, ..., 20]
    if sparse_stride == 9:
        key_indices = [round(i * (total_latent_frames - 1) / 8) for i in range(9)]
    elif sparse_stride == 11:
        key_indices = [0, 1, 2, 3, 9, 10, 11, 17, 18, 19, 20]
    elif sparse_stride == 13:
        key_indices = [round(i * (total_latent_frames - 1) / 12) for i in range(13)]
    elif sparse_stride == 15:
        key_indices = [round(i * (total_latent_frames - 1) / 14) for i in range(15)]
    elif sparse_stride == 4:
        key_indices = [8, 9, 10, 11]
    elif sparse_stride == 7:
        key_indices = [7, 8, 9, 10, 11, 12, 13]
    elif sparse_stride == 91:
        key_indices = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    elif sparse_stride == 92:
        key_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    elif sparse_stride == 12:
        key_indices = [0, 1, 2, 3, 8, 9, 10, 11, 17, 18, 19, 20]
    else:
        key_indices = list(range(0, total_latent_frames, sparse_stride))
    num_key_frames = len(key_indices)
    key_indices_tensor = torch.tensor(key_indices, dtype=torch.long, device=device)

    logging.info(f"Total latent frames: {total_latent_frames}, Key frames: {num_key_frames}")
    logging.info(f"Key frame indices: {key_indices}")

    # seq_len 计算 (基于键帧数)
    seq_len_full = math.ceil(
        (lat_h * lat_w) / (patch_size[1] * patch_size[2]) * total_latent_frames
    )
    seq_len_key = math.ceil(
        (lat_h * lat_w) / (patch_size[1] * patch_size[2]) * num_key_frames
    )

    # ============ 文本编码 ============
    logging.info(f"Encoding prompt: {prompt}")
    # Run text encoding completely on CPU to save VRAM and prevent OOM
    context = [c.to(device) for c in text_encoder([prompt], torch.device('cpu'))]
    context_null = [c.to(device) for c in text_encoder([config.sample_neg_prompt], torch.device('cpu'))]
    torch.cuda.empty_cache()

    # ============ 初始化噪声 ============
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    logging.info(f"Using seed: {seed}")

    noise_full = torch.randn(
        target_shape, dtype=torch.float32, device=device, generator=seed_g
    )

    # ============ 设置采样调度器 ============
    def make_sample_scheduler():
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        return scheduler

    sample_scheduler = make_sample_scheduler()
    timesteps = sample_scheduler.timesteps

    # ============ 去噪循环 ============
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = math.ceil(sampling_steps * warmup_ratio)
    warmup_steps = max(0, min(int(warmup_steps), sampling_steps))
    latents_full = noise_full  # [C, 21, H, W]
    latents_key = None
    warmup_latent_saved = False

    def save_warmup_state(next_step_index):
        nonlocal warmup_latent_saved
        if not save_warmup_latent or warmup_latent_saved:
            return
        metadata = {
            "prompt": prompt,
            "seed": int(seed),
            "sample_steps": int(sampling_steps),
            "warmup_steps": int(warmup_steps),
            "next_step_index": int(next_step_index),
            "latent_shape": list(latents_full.shape),
            "size": f"{size[0]}*{size[1]}",
            "frame_num": int(frame_num),
            "sample_shift": float(shift),
            "guide_scale": float(guide_scale),
        }
        os.makedirs(os.path.dirname(save_warmup_latent) or ".", exist_ok=True)
        torch.save(
            {
                "latent": latents_full.detach().to("cpu"),
                "metadata": metadata,
            },
            save_warmup_latent,
        )
        warmup_latent_saved = True
        logging.info(
            f"Warmup latent saved to {save_warmup_latent} "
            f"(next_step_index={next_step_index}, shape={tuple(latents_full.shape)})"
        )

    logging.info(f"Starting sparse denoising loop ({sampling_steps} steps)...")
    if key_only:
        logging.info("Key-only mode enabled: non-key latent frames will not be interpolated back.")
    if warmup_steps:
        logging.info(f"Warmup enabled: {warmup_steps}/{sampling_steps} full-frame denoising steps before sparse key-frame denoising.")
    denoise_start = time.time()

    with amp.autocast(dtype=config.param_dtype), torch.no_grad():
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sparse Denoising")):
            timestep = torch.stack([t])

            if step_idx < warmup_steps:
                noise_pred_cond = model(
                    [latents_full],
                    t=timestep,
                    context=context,
                    seq_len=seq_len_full,
                )[0]

                noise_pred_uncond = model(
                    [latents_full],
                    t=timestep,
                    context=context_null,
                    seq_len=seq_len_full,
                )[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents_full.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents_full = temp_x0.squeeze(0)
                if step_idx + 1 == warmup_steps:
                    save_warmup_state(step_idx + 1)
                continue

            if step_idx == warmup_steps and warmup_steps:
                # UniPC is a multistep solver and caches previous model outputs.
                # Full-frame warmup outputs have 21 latent frames, while sparse
                # key outputs have 11, so the sparse phase needs a fresh history.
                sample_scheduler = make_sample_scheduler()
                logging.info("Reset scheduler history for sparse key-frame phase after warmup.")

            # 1. warmup 后首次从全帧 latent 提取键帧；非 key_only 模式每步都从插值后的全帧再提取
            if latents_key is None or not key_only:
                latents_key = latents_full[:, key_indices]  # [C, 11, H, W]

            # 2. 送入 DiT 进行噪声预测（只有键帧）
            noise_pred_cond = model(
                [latents_key],
                t=timestep,
                context=context,
                seq_len=seq_len_key,
                temporal_indices=key_indices_tensor,
            )[0]

            noise_pred_uncond = model(
                [latents_key],
                t=timestep,
                context=context_null,
                seq_len=seq_len_key,
                temporal_indices=key_indices_tensor,
            )[0]

            # 3. CFG 引导
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # 4. Scheduler 更新键帧 latent
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents_key.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents_key_updated = temp_x0.squeeze(0)  # [C, 11, H, W]

            # 5. key_only 模式直接进入下一步；否则线性插值恢复非键帧
            if key_only:
                latents_key = latents_key_updated
            else:
                latents_full = interpolate_non_key_frames(
                    latents_key_updated, key_indices, total_latent_frames
                )
        if warmup_steps == sampling_steps:
            save_warmup_state(warmup_steps)
        if latents_key is None:
            latents_key = latents_full[:, key_indices]

    denoise_time = time.time() - denoise_start
    logging.info(f"Sparse denoising completed in {denoise_time:.1f}s")

    # ============ VAE 解码 ============
    logging.info("Decoding with VAE...")
    model.cpu()
    torch.cuda.empty_cache()

    vae_start = time.time()
    latents_to_decode = latents_key if key_only else latents_full
    logging.info(f"Decoding latent shape: {tuple(latents_to_decode.shape)}")
    videos = vae.decode([latents_to_decode])
    vae_time = time.time() - vae_start
    logging.info(f"VAE decoding completed in {vae_time:.1f}s")

    # ============ 保存视频 ============
    video = videos[0]
    if save_file is None:
        prefix = "key_only" if key_only else "sparse"
        save_file = f"outputs/{prefix}_s{sparse_stride}_{seed}.mp4"

    os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
    cache_video(
        tensor=video[None],
        save_file=save_file,
        fps=config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info(f"Video saved to {save_file}")
    logging.info(f"=== Timing Summary ===")
    logging.info(f"  Denoising: {denoise_time:.1f}s")
    logging.info(f"  VAE decode: {vae_time:.1f}s")
    logging.info(f"  Total: {denoise_time + vae_time:.1f}s")

    del model, vae, text_encoder
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse frame video generation")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--sparse_stride", type=int, default=2, help="Frame stride for sparse generation. 2=every other frame")
    parser.add_argument("--key_only", action="store_true", help="Decode only key latents without interpolating non-key latent frames")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Fraction of denoising steps to run on full latent frames before sparse denoising")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Exact number of full-frame warmup denoising steps; overrides warmup_ratio when > 0")
    parser.add_argument("--save_warmup_latent", type=str, default=None, help="Path to save full-frame latent after warmup")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    w, h = args.size.split("*")
    size = (int(w), int(h))

    config = WAN_CONFIGS["t2v-1.3B"]

    generate_sparse(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        prompt=args.prompt,
        size=size,
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        save_file=args.save_file,
        sparse_stride=args.sparse_stride,
        key_only=args.key_only,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        save_warmup_latent=args.save_warmup_latent,
    )
