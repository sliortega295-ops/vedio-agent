"""Source-reviewed LongLive 1.0 adapter; checkpoint/GPU validation is NOT yet run.

Integration references: NVlabs/LongLive/LongLive1.0/pipeline/
interactive_causal_inference.py (Apache-2.0). Only video is supported.
Stateless full-prefix VAE decoding is an intentionally expensive correctness
baseline; saved committed RGB is never replaced by subsequent decodes.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from pathlib import Path

from .core import Candidate, Stage, write_json


class LongLiveDriver:
    name, mock, native_ar, joint_av = "longlive1-video-reference", False, True, False

    def __init__(self, pipeline, *, device="cuda", max_latent_frames=600, fps=16):
        import torch
        self.torch, self.pipe = torch, pipeline
        self.device, self.dtype = torch.device(device), torch.bfloat16
        self.max_frames, self.fps = int(max_latent_frames), float(fps)
        self.block = int(pipeline.num_frame_per_block)
        if self.max_frames < self.block or self.fps != 16:
            raise ValueError("Reference adapter expects positive budget and official 16 fps")
        if pipeline.frame_seq_length != 1560:
            raise ValueError("Only reviewed Wan1.3B latent shape C=16,H=60,W=104 is supported")
        self.history, self.pixel_count, self.current_prompt = None, 0, None
        self.conditional, self.pending_id = None, None

    @classmethod
    def from_config(cls, config: dict):
        import torch
        from omegaconf import OmegaConf
        root = Path(config["upstream_root"]).resolve()
        if not (root / "pipeline/interactive_causal_inference.py").is_file():
            raise FileNotFoundError("upstream_root must be the LongLive1.0 directory, not repository root")
        args = OmegaConf.load(Path(config["upstream_config"]).resolve())
        if getattr(args, "adapter", None):
            raise ValueError("LoRA loading is not implemented by this reference factory")
        checkpoint = Path(config["checkpoint"]).resolve()
        device = config.get("device", "cuda")
        sys.path.insert(0, str(root))
        old_cwd = Path.cwd()
        try:
            os.chdir(root)  # Upstream resolves its model directories relative to cwd.
            module = importlib.import_module("pipeline.interactive_causal_inference")
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError("An unrelated 'pipeline' package is already imported; use a fresh worker")
            pipe = module.InteractiveCausalInferencePipeline(args, device=torch.device(device))
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            key = "generator_ema" if getattr(args, "use_ema", False) else "generator"
            weights = weights.get(key, weights.get("model", weights))
            weights = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in weights.items()}
            pipe.generator.load_state_dict(weights, strict=True)
            pipe = pipe.to(device=device, dtype=torch.bfloat16).eval()
        finally:
            os.chdir(old_cwd)
        return cls(pipe, device=device, max_latent_frames=config.get("max_latent_frames", 600))

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset(self):
        p, t = self.pipe, self.torch
        capacity = self.max_frames if p.local_attn_size == -1 else p.local_attn_size
        p._initialize_kv_cache(1, dtype=self.dtype, device=self.device,
                               kv_cache_size_override=capacity * p.frame_seq_length)
        p._initialize_crossattn_cache(batch_size=1, dtype=self.dtype, device=self.device)
        p.generator.model.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)
        self.history = t.empty((1, 0, 16, 60, 104), dtype=self.dtype, device=self.device)
        self.pixel_count, self.current_prompt, self.conditional, self.pending_id = 0, None, None, None

    def snapshot(self):
        # deepcopy clones tensor state, NOT model weights. Latent history is append-only.
        return (copy.deepcopy(self.pipe.kv_cache1), copy.deepcopy(self.pipe.crossattn_cache),
                self.history, self.pixel_count, self.current_prompt, self.conditional,
                getattr(self.pipe.generator.model, "block_mask", None), self.pending_id)

    def restore(self, state):
        (self.pipe.kv_cache1, self.pipe.crossattn_cache, self.history, self.pixel_count,
         self.current_prompt, self.conditional, mask, self.pending_id) = state
        self.pipe.generator.model.block_mask = mask

    def sample(self, stage: Stage, control: dict, seed: int, out: Path, start_s: float) -> Candidate:
        from PIL import Image
        from torchvision.io import write_video
        t, p = self.torch, self.pipe
        position = self.history.shape[1]
        if position + self.block > self.max_frames:
            raise RuntimeError("Latent budget exhausted; increase max_latent_frames explicitly")
        if control.get("audio_prompt"):
            raise ValueError("LongLive reference driver does not generate audio")
        times = {}
        with t.inference_mode():
            self.synchronize()
            timer = time.perf_counter()
            if control["video_prompt"] != self.current_prompt:
                self.conditional = p.text_encoder(text_prompts=[control["video_prompt"]])
                p._recache_after_switch(self.history, position, self.conditional)
                self.current_prompt = control["video_prompt"]
            self.synchronize()
            times["condition_recache_s"] = time.perf_counter() - timer
            timer = time.perf_counter()
            rng = t.Generator(device=self.device).manual_seed(seed)
            x = t.randn((1, self.block, 16, 60, 104), generator=rng, device=self.device, dtype=self.dtype)
            for index, value in enumerate(p.denoising_step_list):
                timestep = t.full((1, self.block), int(value), device=self.device, dtype=t.int64)
                _, clean = p.generator(noisy_image_or_video=x, conditional_dict=self.conditional,
                                       timestep=timestep, kv_cache=p.kv_cache1,
                                       crossattn_cache=p.crossattn_cache,
                                       current_start=position * p.frame_seq_length)
                if index + 1 < len(p.denoising_step_list):
                    flat = clean.flatten(0, 1)
                    noise = t.randn(flat.shape, generator=rng, device=self.device, dtype=flat.dtype)
                    next_time = t.full((self.block,), int(p.denoising_step_list[index + 1]),
                                       device=self.device, dtype=t.long)
                    x = p.scheduler.add_noise(flat, noise, next_time).unflatten(0, clean.shape[:2])
            p.generator(noisy_image_or_video=clean, conditional_dict=self.conditional,
                        timestep=t.ones_like(timestep) * p.args.context_noise,
                        kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                        current_start=position * p.frame_seq_length)
            self.history = t.cat((self.history, clean.detach()), dim=1)
            self.synchronize()
            times["denoise_and_clean_cache_s"] = time.perf_counter() - timer
            timer = time.perf_counter()
            video = (p.vae.decode_to_pixel(self.history, use_cache=False) * .5 + .5).clamp(0, 1)
            new = video[0, self.pixel_count:]
            if len(new) < 1:
                raise RuntimeError("VAE did not emit new frames")
            rgb = (new.permute(0, 2, 3, 1) * 255).round().to(t.uint8).cpu()
            self.pixel_count = video.shape[1]
            self.synchronize()
            times["full_prefix_decode_s"] = time.perf_counter() - timer
        out.mkdir(parents=True, exist_ok=True)
        video_path = out / "candidate.mp4"
        write_video(str(video_path), rgb, fps=self.fps, video_codec="libx264", options={"crf": "18"})
        indices = sorted(set(round(i * (len(rgb) - 1) / 7) for i in range(8)))
        frames = []
        for index in indices:
            path = out / f"frame-{index:05d}.png"
            Image.fromarray(rgb[index].numpy()).save(path)
            frames.append(str(path))
        self.pending_id = out.name
        return Candidate(out.name, str(video_path), tuple(frames), None,
                         (str(video_path), *frames), start_s, len(rgb), self.fps, driver_costs=times)

    def commit(self, candidate: Candidate):
        if candidate.id != self.pending_id:
            raise RuntimeError("Attempted to commit a stale candidate")
        self.pending_id = None
