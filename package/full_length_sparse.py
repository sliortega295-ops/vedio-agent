"""Full-length Diffusers Wan candidate backend, independent of native-Wan N9/N6.

Use a separate pipeline/environment from the historical benchmark. Calls on one
pipeline must be serial: scheduler state is mutable even with fresh controllers.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Match the repository's vendored-backend convention; no external coframe install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
from coframe import cli  # noqa: E402
from coframe.config import CoFrameConfig  # noqa: E402
from coframe.wan.pipeline import coframe_wan_generate  # noqa: E402


@dataclass
class SparseCandidate:
    prompt: str
    seed: int
    latents: torch.Tensor
    video: Any | None  # decoded [F,H,W,C] numpy array, NOT sparse anchor frames
    metadata: dict[str, Any]


@torch.inference_mode()
def generate_candidate(
    pipe: Any, *, prompt: str, seed: int, config: CoFrameConfig,
    decode: bool = True, **generation_kwargs: Any,
) -> SparseCandidate:
    """Generate one candidate for an outer verifier/search loop.

    The caller owns proposal, scoring, iteration, and final selection. Pass a
    dense config to regenerate a selected (prompt, seed) pair without sparsity.
    Supply a distinct config.trace_path for each candidate if saving traces.
    """
    if not isinstance(prompt, str):
        raise ValueError("generate_candidate requires one prompt string")
    if generation_kwargs.get("num_videos_per_prompt", 1) != 1:
        raise ValueError("generate_candidate supports one candidate per call")
    frames = generation_kwargs.get("num_frames", 81)
    steps = generation_kwargs.get("num_inference_steps", 50)
    temporal_scale = pipe.vae_scale_factor_temporal
    if steps < 1 or frames < 1 or frames % temporal_scale != 1:
        raise ValueError("generate_candidate requires positive steps and num_frames=4n+1; no implicit rounding")
    latents = generation_kwargs.get("latents")
    if latents is not None and (latents.ndim != 5 or latents.shape[0] != 1):
        raise ValueError("generate_candidate requires one candidate's [1,C,F,H,W] latents")
    if latents is not None and latents.shape[2] != (frames - 1) // temporal_scale + 1:
        raise ValueError("generate_candidate requires the requested full latent frame count")
    if "generator" in generation_kwargs:
        raise ValueError("pass seed, not generator; each candidate owns its RNG")
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    try:
        result = coframe_wan_generate(
            pipe, prompt=prompt, generator=generator, config=config, **generation_kwargs,
        )
        video = cli._decode_wan_latents(pipe, result.frames)[0] if decode else None
        return SparseCandidate(prompt, seed, result.frames, video, result.metadata)
    finally:
        pipe._current_timestep = None


def build_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    parser.description = "SVA full-length residual-budget sparsity (Diffusers Wan2.1-1.3B backend)."
    parser.set_defaults(
        method="sparse_evolution", warmup_steps=0, kv_mode="full_kv",
        adaptive_k_policy=None, decode=True, local_files_only=True,
    )
    parser.add_argument("--no-decode", action="store_false", dest="decode",
                        help="Save full latents only; do not pass these to a video verifier")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only",
                        help="Explicitly allow fetching a Diffusers checkpoint")
    parser.epilog = (
        "Budget thresholds are required for sparse_evolution: supply one fewer "
        "--adaptive-k-thresholds than --adaptive-k-values. Calibrate relative RMS "
        "thresholds on held-out data; no validated thresholds are bundled."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return cli.main(argv, parser=build_parser())


if __name__ == "__main__":
    raise SystemExit(main())
