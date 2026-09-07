from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ..config import CoFrameConfig
from ..controller import AdaptiveMeshController
from ..ode_budget import ODEPathBudgetController
from ..selection import (
    fis_interleaved_select,
    frame_representations_from_clean_latents,
    rhyme_select,
    transition_scores,
    uniform_select,
)
from ..trace import CoFrameTrace
from .sparse_forward import coframe_transformer_forward, require_diffusers_034


@dataclass(slots=True)
class CoFrameGenerationOutput:
    frames: torch.Tensor
    metadata: dict[str, Any]


def _dense_predict(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_kwargs: dict[str, Any] | None,
) -> torch.Tensor:
    transformer_dtype = pipe.transformer.dtype
    return pipe.transformer(
        hidden_states=latents.to(transformer_dtype),
        timestep=timestep.expand(latents.shape[0]),
        encoder_hidden_states=encoder_hidden_states,
        attention_kwargs=attention_kwargs,
        return_dict=False,
    )[0].float()


def _make_controller(
    *,
    config: CoFrameConfig,
    clean_proxy: torch.Tensor,
) -> tuple[AdaptiveMeshController, list[int], torch.Tensor]:
    frame_representations = frame_representations_from_clean_latents(clean_proxy)
    prior = transition_scores(frame_representations).detach().cpu()
    num_frames = int(clean_proxy.shape[2])

    rhyme_anchors = rhyme_select(
        frame_representations,
        config.num_anchors,
        similarity_threshold=config.rhyme_similarity_threshold,
        force_boundaries=config.force_boundaries,
        min_gap=config.min_anchor_gap,
    )
    fixed_anchors = uniform_select(num_frames, config.num_anchors, config.force_boundaries)

    controller_prior = prior
    controller_prior_weight = config.rhyme_prior_weight
    if config.method in {"fixed", "adaptive_k", "coframe_ode", "sparse_evolution"}:
        anchors = fixed_anchors
        controller_prior = torch.zeros_like(prior)
        controller_prior_weight = 0.0
    elif config.method == "fis":
        anchors = fis_interleaved_select(
            num_frames,
            config.num_anchors,
            config.sparse_block_start,
            config.sparse_block_start,
            force_boundaries=config.force_boundaries,
            anchor_stride=config.fis_anchor_stride,
        )
        controller_prior = torch.zeros_like(prior)
        controller_prior_weight = 0.0
    elif config.method in {"rhyme", "coframe"}:
        anchors = rhyme_anchors
        if config.method == "coframe" and config.refresh_signal == "gap_only":
            controller_prior = torch.zeros_like(prior)
            controller_prior_weight = 0.0
    else:
        raise ValueError(f"A sparse controller is not defined for method={config.method}")

    controller = AdaptiveMeshController(
        num_frames=num_frames,
        num_anchors=config.num_anchors,
        initial_anchors=anchors,
        prior_scores=controller_prior,
        force_boundaries=config.force_boundaries,
        min_gap=config.min_anchor_gap,
        risk_ema=config.risk_ema,
        prior_weight=controller_prior_weight,
        risk_floor=(1.0 if config.method == "coframe" and config.refresh_signal == "gap_only" else config.risk_floor),
        gap_power=config.interval_gap_power,
        move_penalty=config.move_penalty,
        min_refresh_gain=config.min_refresh_gain,
        max_swaps_per_refresh=(
            config.max_swaps_per_refresh
            if (config.method == "coframe_ode" or (config.method == "coframe" and config.refresh_signal != "none"))
            else 0
        ),
        defect_clip=config.defect_clip,
    )
    controller.rhyme_reference_anchors = list(rhyme_anchors)
    controller.fixed_reference_anchors = list(fixed_anchors)
    return controller, anchors, prior


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_flow_prediction_scheduler(scheduler: Any) -> None:
    """Fail closed before applying the flow clean-endpoint conversion."""
    scheduler_name = type(scheduler).__name__
    scheduler_config = getattr(scheduler, "config", None)
    prediction_type = getattr(scheduler_config, "prediction_type", None)
    use_flow_sigmas = bool(getattr(scheduler_config, "use_flow_sigmas", False))
    native_flow_scheduler = scheduler_name.startswith("FlowMatch")
    if not native_flow_scheduler and not (prediction_type == "flow_prediction" and use_flow_sigmas):
        raise RuntimeError(
            "coframe_ode requires a flow-prediction scheduler: either a FlowMatch scheduler or "
            "prediction_type='flow_prediction' with use_flow_sigmas=True"
        )


@torch.no_grad()
def coframe_wan_generate(
    pipe: Any,
    *,
    prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    max_sequence_length: int = 512,
    config: CoFrameConfig | None = None,
) -> CoFrameGenerationOutput:
    """Run dense and sparse CoFrame variants under one Wan2.1 sampler contract."""
    config = config or CoFrameConfig()
    require_diffusers_034(config.strict_diffusers_version)
    config.validate(num_blocks=len(pipe.transformer.blocks))
    if config.method == "sparse_evolution":
        if num_inference_steps < 1 or num_frames < 1 or num_frames % pipe.vae_scale_factor_temporal != 1:
            raise ValueError("sparse_evolution requires positive steps and num_frames=4n+1; no implicit rounding")

    pipe.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        ["latents"],
    )

    if num_frames % pipe.vae_scale_factor_temporal != 1:
        num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False
    device = pipe._execution_device

    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        if prompt_embeds is None:
            raise ValueError("Either prompt or prompt_embeds must be provided")
        batch_size = prompt_embeds.shape[0]

    if config.method == "sparse_evolution" and batch_size * num_videos_per_prompt != 1:
        raise ValueError("sparse_evolution supports one candidate per call; candidates need independent controllers")

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if not hasattr(pipe.scheduler, "sigmas"):
        raise RuntimeError("CoFrame requires a scheduler exposing sigmas")
    if config.method == "coframe_ode":
        _require_flow_prediction_scheduler(pipe.scheduler)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)

    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        pipe.transformer.config.in_channels,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )
    latent_frame_count = int(latents.shape[2])
    if config.method == "sparse_evolution" and (
        latents.shape[0] != 1
        or latent_frame_count != (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    ):
        raise ValueError("sparse_evolution requires one candidate with the requested full latent frame count")
    config.validate(num_blocks=len(pipe.transformer.blocks), num_frames=latent_frame_count)

    trace = CoFrameTrace(
        run={
            "method": config.method,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "latent_frames": latent_frame_count,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "config": config.to_dict(),
        }
    )

    controller: AdaptiveMeshController | None = None
    initial_anchors: list[int] | None = None
    prior_scores: torch.Tensor | None = None
    transformer_events: list[dict[str, Any]] = []
    last_sparse_anchors: list[int] | None = None
    ode_budget_controller: ODEPathBudgetController | None = None
    dense_step_count = 0
    sparse_step_count = 0

    # FIS is prompt-agnostic and can start sparsity at the first denoising
    # step.  Rhyme/CoFrame keep their semantic warmup.
    if config.method in {"fis", "sparse_evolution"}:
        controller, initial_anchors, prior_scores = _make_controller(config=config, clean_proxy=latents)
        trace.add("mesh_initialization", step=-1, timestep=None, anchors=initial_anchors, prior_scores=prior_scores)
    if config.method == "sparse_evolution":
        # Unlike semantic-mesh modes, this path has no required warmup.
        # Honor all-dense runs as well; never silently shorten a requested tail.
        effective_warmup = min(config.warmup_steps, num_inference_steps)
    elif config.method in {"dense", "fis"} or num_inference_steps <= 1:
        effective_warmup = 0
    else:
        # Always leave at least one sparse denoising step, including few-step
        # smoke tests. The canonical 50-step setting still uses warmup=5.
        effective_warmup = max(1, min(config.warmup_steps, num_inference_steps - 1))

    if config.method == "coframe_ode":
        controller, initial_anchors, prior_scores = _make_controller(config=config, clean_proxy=latents)
        total_sparse_steps = sum(
            1
            for index in range(num_inference_steps)
            if index >= effective_warmup and config.is_sparse_step(index, num_inference_steps)
        )
        ode_budget_controller = ODEPathBudgetController.from_config(
            config,
            num_frames=latent_frame_count,
            total_sparse_steps=total_sparse_steps,
        )
        trace.add(
            "mesh_initialization",
            step=-1,
            timestep=None,
            anchors=initial_anchors,
            prior_scores=prior_scores,
            placement_policy="coverage_interleaved",
        )
        trace.add("ode_budget_initialization", **ode_budget_controller.state_dict())

    _cuda_sync()
    denoise_start = time.perf_counter()
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for step_index, timestep_scalar in enumerate(timesteps):
            if pipe.interrupt:
                progress_bar.update()
                continue
            pipe._current_timestep = timestep_scalar
            timestep = timestep_scalar.expand(latents.shape[0])

            use_sparse = (
                config.method != "dense"
                and controller is not None
                and step_index >= effective_warmup
                and config.is_sparse_step(step_index, num_inference_steps)
            )
            if not use_sparse:
                noise_cond = _dense_predict(pipe, latents, timestep_scalar, prompt_embeds, attention_kwargs)
                if pipe.do_classifier_free_guidance:
                    if negative_prompt_embeds is None:
                        raise RuntimeError("CFG is enabled but negative_prompt_embeds is missing")
                    noise_uncond = _dense_predict(
                        pipe,
                        latents,
                        timestep_scalar,
                        negative_prompt_embeds,
                        attention_kwargs,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond
                dense_step_count += 1

                if config.method != "dense" and controller is None and step_index + 1 >= effective_warmup:
                    sigma = sigmas[step_index].to(device=latents.device, dtype=torch.float32)
                    clean_proxy = latents.float() - sigma * noise_pred.float()
                    controller, initial_anchors, prior_scores = _make_controller(config=config, clean_proxy=clean_proxy)
                    trace.add(
                        "mesh_initialization",
                        step=step_index,
                        timestep=int(timestep_scalar.item()),
                        anchors=initial_anchors,
                        prior_scores=prior_scores,
                    )
            else:
                assert controller is not None
                noise_cond, cond_metadata = coframe_transformer_forward(
                    pipe.transformer,
                    latents.to(transformer_dtype),
                    timestep,
                    prompt_embeds,
                    config=config,
                    controller=controller,
                    step_index=step_index,
                    replay_block_anchors=None,
                    update_controller=config.method in {"coframe", "adaptive_k", "coframe_ode", "sparse_evolution"},
                    attention_kwargs=attention_kwargs,
                    trace=trace,
                )
                if pipe.do_classifier_free_guidance:
                    if negative_prompt_embeds is None:
                        raise RuntimeError("CFG is enabled but negative_prompt_embeds is missing")
                    noise_uncond, uncond_metadata = coframe_transformer_forward(
                        pipe.transformer,
                        latents.to(transformer_dtype),
                        timestep,
                        negative_prompt_embeds,
                        config=config,
                        controller=controller,
                        step_index=step_index,
                        replay_block_anchors=cond_metadata.block_anchors,
                        update_controller=False,
                        attention_kwargs=attention_kwargs,
                        trace=None,
                    )
                    if uncond_metadata.block_anchors != cond_metadata.block_anchors:
                        raise RuntimeError("Conditional/unconditional sparse schedules diverged")
                    noise_pred = noise_uncond.float() + guidance_scale * (noise_cond.float() - noise_uncond.float())
                else:
                    noise_pred = noise_cond.float()
                sparse_step_count += 1
                transformer_events.append(cond_metadata.to_dict())
                if cond_metadata.block_anchors:
                    final_block = max(cond_metadata.block_anchors)
                    last_sparse_anchors = list(cond_metadata.block_anchors[final_block])

            if ode_budget_controller is not None:
                sigma = sigmas[step_index].to(device=latents.device, dtype=torch.float32)
                signal = ode_budget_controller.observe(
                    step_index=step_index,
                    sample=latents,
                    velocity=noise_pred,
                    sigma=sigma,
                )
                trace.add("ode_path_signal", **signal.to_dict())
                next_step = step_index + 1
                if (
                    next_step < num_inference_steps
                    and next_step >= effective_warmup
                    and config.is_sparse_step(next_step, num_inference_steps)
                ):
                    budget_event = ode_budget_controller.allocate_next(
                        source_step=step_index,
                        target_step=next_step,
                        difficulty=signal.difficulty,
                    )
                    assert controller is not None
                    controller.current_budget = int(budget_event.assigned_budget)
                    controller.budget_history.append(budget_event.to_dict())
                    trace.add("budget_update", policy="ode_path", causal=True, **budget_event.to_dict())

            latents = pipe.scheduler.step(noise_pred, timestep_scalar, latents, return_dict=False)[0]
            progress_bar.update()

    _cuda_sync()
    denoise_time = time.perf_counter() - denoise_start
    pipe._current_timestep = None

    metadata: dict[str, Any] = {
        "method": config.method,
        "denoise_time_sec": denoise_time,
        "dense_steps": dense_step_count,
        "sparse_steps": sparse_step_count,
        "initial_anchors": initial_anchors,
        "final_anchors": (
            last_sparse_anchors
            if last_sparse_anchors is not None
            else (None if controller is None else list(controller.anchors))
        ),
        "final_budget": None if controller is None else int(controller.current_budget),
        "latent_shape": list(latents.shape),
        "transformer_events": transformer_events,
        "controller": None if controller is None else controller.state_dict(),
        "ode_budget_controller": None if ode_budget_controller is None else ode_budget_controller.state_dict(),
        "config": config.to_dict(),
        "trace": trace.to_dict(),
    }
    if config.trace_path:
        trace.write(config.trace_path)
    return CoFrameGenerationOutput(frames=latents, metadata=metadata)
