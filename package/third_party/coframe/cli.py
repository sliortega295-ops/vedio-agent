from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .config import CoFrameConfig
from .wan.pipeline import coframe_wan_generate

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


@torch.inference_mode()
def _decode_wan_latents(pipe: Any, latents: torch.Tensor) -> list[Any]:
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    latent_mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    )
    inverse_std = 1.0 / torch.tensor(
        pipe.vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents = latents / inverse_std + latent_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="np")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate dense, fixed, Rhyme, and CoFrame frame meshes on Wan2.1-T2V-1.3B."
    )
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--method",
        choices=["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode", "sparse_evolution"],
        default="coframe",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-anchors", type=int, default=9)
    parser.add_argument("--sparse-block-start", type=int, default=3)
    parser.add_argument("--sparse-block-end", type=int, default=27)
    parser.add_argument("--block-group-size", type=int, default=3)
    parser.add_argument("--kv-mode", choices=["anchor_only", "full_kv"], default="anchor_only")
    parser.add_argument("--interpolation-target", choices=["delta", "state"], default="delta")
    parser.add_argument("--defect-target", choices=["delta", "state"], default="delta")
    parser.add_argument("--refresh-signal", choices=["defect", "none", "gap_only", "shuffled"], default="defect")
    parser.add_argument("--shuffle-defect-seed", type=int, default=20260810)
    parser.add_argument("--fis-anchor-stride", type=int, default=0)
    parser.add_argument("--fis-dense-tail-steps", type=int, default=0)
    parser.add_argument("--rhyme-similarity-threshold", type=float, default=0.98)
    parser.add_argument("--rhyme-prior-weight", type=float, default=0.35)
    parser.add_argument("--risk-ema", type=float, default=0.75)
    parser.add_argument("--gap-power", type=float, default=2.0)
    parser.add_argument("--move-penalty", type=float, default=0.02)
    parser.add_argument("--min-refresh-gain", type=float, default=1.0e-4)
    parser.add_argument("--max-swaps-per-refresh", type=int, default=1)
    parser.add_argument("--refresh-every-groups", type=int, default=1)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--oracle-probe-steps", type=_csv_ints, default=())
    parser.add_argument("--oracle-probe-blocks", type=_csv_ints, default=())
    parser.add_argument("--oracle-probe-horizons", type=_csv_ints, default=(1, 3))
    parser.add_argument("--oracle-metric-chunk-size", type=int, default=65_536)
    parser.add_argument("--probe-counterfactual-methods", type=_csv_strings, default=("rhyme", "fis", "fixed"))
    parser.add_argument("--probe-curvature-signals", action="store_true")
    parser.add_argument("--curvature-shuffle-seed", type=int, default=20260811)
    parser.add_argument("--adaptive-k-policy", choices=["none", "step_block", "mean_defect", "max_defect"], default="none")
    parser.add_argument("--adaptive-k-values", type=_csv_ints, default=(6, 9, 12, 21))
    parser.add_argument("--adaptive-k-thresholds", type=_csv_floats, default=())
    parser.add_argument("--adaptive-k-schedule-json", type=Path, default=None)
    parser.add_argument("--no-adaptive-k-carry", action="store_true")
    parser.add_argument("--evolution-dense-tail-steps", type=int, default=1)
    parser.add_argument("--ode-target-average-k", type=float, default=0.0)
    parser.add_argument("--ode-min-anchors", type=int, default=0)
    parser.add_argument("--ode-max-anchors", type=int, default=0)
    parser.add_argument("--ode-budget-values", type=_csv_ints, default=())
    parser.add_argument("--ode-signal-ema", type=float, default=0.9)
    parser.add_argument("--ode-signal-clip", type=float, default=4.0)
    parser.add_argument("--ode-direction-weight", type=float, default=0.5)
    parser.add_argument("--ode-endpoint-weight", type=float, default=0.5)
    parser.add_argument("--ode-difficulty-power", type=float, default=1.0 / 3.0)
    parser.add_argument("--ode-anchor-stride", type=int, default=0)
    parser.add_argument("--ode-interleave-penalty", type=float, default=2.0)
    parser.add_argument("--no-ode-interleave-across-steps", action="store_true")

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--decode", action="store_true", help="Decode and export MP4 after latent validation")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-unsupported-diffusers", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, parser: argparse.ArgumentParser | None = None) -> int:
    args = (parser or build_parser()).parse_args(argv)
    run_name = args.run_name or f"wan21_1.3b_{args.method}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    summary_path = run_dir / "summary.json"
    latent_path = run_dir / "latents.pt"
    trace_path = run_dir / "trace.json"
    video_path = run_dir / "video.mp4"

    adaptive_k_schedule = {}
    if args.adaptive_k_schedule_json is not None:
        adaptive_k_schedule = json.loads(args.adaptive_k_schedule_json.read_text(encoding="utf-8"))
        if not isinstance(adaptive_k_schedule, dict):
            raise ValueError("adaptive-k schedule JSON must contain an object mapping step:group to K")

    config = CoFrameConfig(
        method=args.method,
        warmup_steps=args.warmup_steps,
        num_anchors=args.num_anchors,
        sparse_block_start=args.sparse_block_start,
        sparse_block_end=args.sparse_block_end,
        block_group_size=args.block_group_size,
        kv_mode=args.kv_mode,
        interpolation_target=args.interpolation_target,
        defect_target=args.defect_target,
        refresh_signal=args.refresh_signal,
        shuffle_defect_seed=args.shuffle_defect_seed,
        fis_anchor_stride=args.fis_anchor_stride,
        fis_dense_tail_steps=args.fis_dense_tail_steps,
        rhyme_similarity_threshold=args.rhyme_similarity_threshold,
        rhyme_prior_weight=args.rhyme_prior_weight,
        risk_ema=args.risk_ema,
        interval_gap_power=args.gap_power,
        move_penalty=args.move_penalty,
        min_refresh_gain=args.min_refresh_gain,
        max_swaps_per_refresh=args.max_swaps_per_refresh,
        refresh_every_groups=args.refresh_every_groups,
        sketch_dim=args.sketch_dim,
        oracle_probe_steps=args.oracle_probe_steps,
        oracle_probe_blocks=args.oracle_probe_blocks,
        oracle_probe_horizons=args.oracle_probe_horizons,
        oracle_metric_chunk_size=args.oracle_metric_chunk_size,
        probe_counterfactual_methods=args.probe_counterfactual_methods,
        probe_curvature_signals=args.probe_curvature_signals,
        curvature_shuffle_seed=args.curvature_shuffle_seed,
        adaptive_k_policy=args.adaptive_k_policy or ("max_defect" if args.method == "sparse_evolution" else "none"),
        adaptive_k_values=args.adaptive_k_values,
        adaptive_k_thresholds=args.adaptive_k_thresholds,
        adaptive_k_schedule={str(key): int(value) for key, value in adaptive_k_schedule.items()},
        adaptive_k_carry_across_steps=not args.no_adaptive_k_carry,
        evolution_dense_tail_steps=args.evolution_dense_tail_steps,
        ode_target_average_k=args.ode_target_average_k,
        ode_min_anchors=args.ode_min_anchors,
        ode_max_anchors=args.ode_max_anchors,
        ode_budget_values=args.ode_budget_values,
        ode_signal_ema=args.ode_signal_ema,
        ode_signal_clip=args.ode_signal_clip,
        ode_direction_weight=args.ode_direction_weight,
        ode_endpoint_weight=args.ode_endpoint_weight,
        ode_difficulty_power=args.ode_difficulty_power,
        ode_anchor_stride=args.ode_anchor_stride,
        ode_interleave_penalty=args.ode_interleave_penalty,
        ode_interleave_across_steps=not args.no_ode_interleave_across_steps,
        trace_path=str(trace_path),
        strict_diffusers_version=not args.allow_unsupported_diffusers,
    )

    # Validate user-supplied thresholds before loading models or writing output.
    config.validate(num_blocks=30, num_frames=(args.num_frames - 1) // 4 + 1)
    if args.steps < 1 or args.num_frames < 1 or args.num_frames % 4 != 1:
        raise ValueError("steps must be positive and Wan2.1 num-frames must equal 4n+1")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1-1.3B validation requires a CUDA GPU")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    run_dir.mkdir(parents=True, exist_ok=False)

    local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
    load_start = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    pipe.scheduler = scheduler
    pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    if args.vae_slicing:
        pipe.vae.enable_slicing()
    load_time = time.perf_counter() - load_start

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    result = coframe_wan_generate(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        config=config,
    )
    torch.cuda.synchronize()
    generation_time = time.perf_counter() - total_start

    torch.save(
        {
            "latents": result.frames.detach().cpu(),
            "metadata": result.metadata,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
        },
        latent_path,
    )

    decode_time = None
    if args.decode:
        decode_start = time.perf_counter()
        videos = _decode_wan_latents(pipe, result.frames)
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        export_to_video(videos[0], str(video_path), fps=args.fps)

    summary = {
        "status": "success",
        "run_name": run_name,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "method": args.method,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "load_time_sec": load_time,
        "generation_time_sec": generation_time,
        "denoise_time_sec": result.metadata["denoise_time_sec"],
        "decode_time_sec": decode_time,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1.0e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1.0e9,
        "latent_path": str(latent_path),
        "trace_path": str(trace_path),
        "video_path": str(video_path) if args.decode else None,
        "initial_anchors": result.metadata.get("initial_anchors"),
        "final_anchors": result.metadata.get("final_anchors"),
        "dense_steps": result.metadata.get("dense_steps"),
        "sparse_steps": result.metadata.get("sparse_steps"),
        "config": config.to_dict(),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
