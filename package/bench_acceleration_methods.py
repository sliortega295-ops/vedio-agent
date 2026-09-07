#!/usr/bin/env python3
"""Benchmark candidate fusion and approximate search acceleration for SVA.

Methods:
  serial_b1       : one candidate at a time, cond/uncond B1 forwards.
  candidate_b2    : candidate batch B=N, cond and uncond as two batched forwards.
  candidate_cfg_b4: candidate batch fused with CFG, effective batch 2N.
  sparse_cfg      : exact dense warm-up; sparse continuation periodically refreshes
                    unconditional predictions and reuses them between refreshes.
  halving         : exact fused CFG plus successive halving at configured steps.

The exact methods preserve the denoising equations. Sparse CFG and halving are
approximate search policies and must be evaluated with ranking/quality metrics,
not output hashes alone.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from bench_sparse_candidate_concurrency import (
    FRAME_NUM,
    GUIDANCE,
    PATTERNS,
    PROMPT,
    SHIFT,
    SIZE_NAME,
    STEPS,
    WARMUP_STEPS,
    atomic_json,
    make_scheduler,
    scheduler_step,
    target_geometry,
)
from sva_acceleration import (
    ExecutionStats,
    SparseCFGCache,
    exact_predictions_microbatched,
    load_scorer,
    parse_halving_schedule,
    prune_candidate_state,
    repeat_context,
    select_topk,
    sparse_cfg_prediction,
)


EXACT_METHODS = {"serial_b1", "candidate_b2", "candidate_cfg_b4"}
ALL_METHODS = ["serial_b1", "candidate_b2", "candidate_cfg_b4", "sparse_cfg", "halving"]


def clone_noises(noises: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [item.clone() for item in noises]


def compare_tensor_lists(
    reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]
) -> dict[str, float | int]:
    if len(reference) != len(candidate):
        return {"count_match": 0, "reference_count": len(reference), "candidate_count": len(candidate)}
    abs_means: list[float] = []
    abs_maxes: list[float] = []
    cosines: list[float] = []
    for a, b in zip(reference, candidate):
        a32 = a.detach().float().flatten()
        b32 = b.detach().float().flatten()
        diff = (a32 - b32).abs()
        abs_means.append(float(diff.mean().item()))
        abs_maxes.append(float(diff.max().item()))
        denom = float(a32.norm().item() * b32.norm().item())
        cosines.append(float(torch.dot(a32, b32).item() / denom) if denom else 1.0)
    return {
        "count_match": 1,
        "mean_abs": statistics.fmean(abs_means),
        "max_abs": max(abs_maxes),
        "mean_cosine": statistics.fmean(cosines),
    }


def _exact_step(
    *,
    model: Any,
    latents: list[torch.Tensor],
    timestep: torch.Tensor,
    cond_contexts: list[torch.Tensor],
    null_contexts: list[torch.Tensor],
    seq_len: int,
    positions: torch.Tensor,
    guidance: float,
    microbatch_size: int,
    fuse_cfg: bool,
) -> tuple[list[torch.Tensor], ExecutionStats]:
    return exact_predictions_microbatched(
        model,
        latents,
        timestep,
        cond_contexts,
        null_contexts,
        seq_len,
        positions,
        guidance,
        microbatch_size=microbatch_size,
        fuse_cfg=fuse_cfg,
    )


def _sparse_cfg_step_microbatched(
    *,
    model: Any,
    latents: list[torch.Tensor],
    timestep: torch.Tensor,
    cond_contexts: list[torch.Tensor],
    null_contexts: list[torch.Tensor],
    seq_len: int,
    positions: torch.Tensor,
    guidance: float,
    global_step: int,
    refresh_interval: int,
    microbatch_size: int,
    cache: SparseCFGCache,
) -> tuple[list[torch.Tensor], SparseCFGCache, ExecutionStats]:
    predictions: list[torch.Tensor | None] = [None] * len(latents)
    cached_values: list[torch.Tensor | None]
    if cache.unconditional is None:
        cached_values = [None] * len(latents)
    else:
        cached_values = list(cache.unconditional)
    total = ExecutionStats()
    last_refresh = cache.last_refresh_step

    for start in range(0, len(latents), microbatch_size):
        ids = list(range(start, min(start + microbatch_size, len(latents))))
        chunk_cache = SparseCFGCache(
            unconditional=(
                None
                if any(cached_values[i] is None for i in ids)
                else [cached_values[i] for i in ids if cached_values[i] is not None]
            ),
            last_refresh_step=last_refresh,
        )
        pred, updated, stats = sparse_cfg_prediction(
            model,
            [latents[i] for i in ids],
            timestep,
            [cond_contexts[i] for i in ids],
            [null_contexts[i] for i in ids],
            seq_len,
            positions,
            guidance,
            global_step=global_step,
            refresh_interval=refresh_interval,
            cache=chunk_cache,
        )
        for local, i in enumerate(ids):
            predictions[i] = pred[local]
            if updated.unconditional is not None:
                cached_values[i] = updated.unconditional[local]
        last_refresh = updated.last_refresh_step
        total = total.merge(stats)

    if any(item is None for item in predictions):
        raise RuntimeError("sparse-CFG microbatch assembly failed")
    if any(item is None for item in cached_values):
        raise RuntimeError("sparse-CFG cache was not populated")
    return (
        [item for item in predictions if item is not None],
        SparseCFGCache(
            unconditional=[item for item in cached_values if item is not None],
            last_refresh_step=last_refresh,
        ),
        total,
    )


def run_multi_candidate_preview(
    *,
    pipeline: Any,
    model_config: Any,
    base_noises: Sequence[torch.Tensor],
    seeds: Sequence[int],
    cond_context: Any,
    null_context: Any,
    pattern: str,
    method: str,
    candidate_microbatch: int,
    sparse_cfg_interval: int,
    halving_schedule: str,
    scorer_spec: str | None,
    size: tuple[int, int],
) -> tuple[list[int], list[torch.Tensor], ExecutionStats, list[dict[str, Any]]]:
    if method not in ALL_METHODS:
        raise ValueError(method)
    if len(base_noises) != len(seeds):
        raise ValueError("base_noises and seeds length mismatch")
    if candidate_microbatch < 1:
        raise ValueError("candidate_microbatch must be >= 1")

    device = base_noises[0].device
    spec = PATTERNS[pattern]
    key_indices = torch.tensor(spec["key_indices"], dtype=torch.long, device=device)
    sparse_positions = torch.tensor(spec["rope_positions"], dtype=torch.long, device=device)
    _, seq_full, seq_per_frame = target_geometry(pipeline, size)
    seq_sparse = seq_per_frame * len(spec["key_indices"])
    full_positions = torch.arange(base_noises[0].shape[1], device=device, dtype=torch.long)

    candidate_ids = list(seeds)
    latents = clone_noises(base_noises)
    cond_contexts = repeat_context(cond_context, len(latents))
    null_contexts = repeat_context(null_context, len(latents))
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    warm_schedulers = [make_scheduler(model_config, device, torch) for _ in seeds]
    sparse_schedulers = [make_scheduler(model_config, device, torch) for _ in seeds]
    total = ExecutionStats()
    pruning_log: list[dict[str, Any]] = []

    warm_fuse_cfg = method != "candidate_b2" and method != "serial_b1"
    with torch.cuda.amp.autocast(dtype=model_config.param_dtype), torch.no_grad():
        for step_t in warm_schedulers[0].timesteps[:WARMUP_STEPS]:
            pred, stats = _exact_step(
                model=pipeline.model,
                latents=latents,
                timestep=torch.stack([step_t]),
                cond_contexts=cond_contexts,
                null_contexts=null_contexts,
                seq_len=seq_full,
                positions=full_positions,
                guidance=GUIDANCE,
                microbatch_size=(1 if method == "serial_b1" else candidate_microbatch),
                fuse_cfg=warm_fuse_cfg,
            )
            total = total.merge(stats)
            latents = [
                scheduler_step(scheduler, p, step_t, latent, generator)
                for scheduler, p, latent, generator in zip(
                    warm_schedulers, pred, latents, generators
                )
            ]

        latents = [latent[:, key_indices].clone() for latent in latents]
        sparse_cache = SparseCFGCache()
        rounds = {item.step: item.keep for item in parse_halving_schedule(halving_schedule)}
        scorer = load_scorer(scorer_spec)

        for global_step in range(WARMUP_STEPS, STEPS):
            step_t = sparse_schedulers[0].timesteps[global_step]
            if method == "sparse_cfg":
                pred, sparse_cache, stats = _sparse_cfg_step_microbatched(
                    model=pipeline.model,
                    latents=latents,
                    timestep=torch.stack([step_t]),
                    cond_contexts=cond_contexts,
                    null_contexts=null_contexts,
                    seq_len=seq_sparse,
                    positions=sparse_positions,
                    guidance=GUIDANCE,
                    global_step=global_step,
                    refresh_interval=sparse_cfg_interval,
                    microbatch_size=candidate_microbatch,
                    cache=sparse_cache,
                )
            else:
                fuse_cfg = method in {"candidate_cfg_b4", "halving"}
                pred, stats = _exact_step(
                    model=pipeline.model,
                    latents=latents,
                    timestep=torch.stack([step_t]),
                    cond_contexts=cond_contexts,
                    null_contexts=null_contexts,
                    seq_len=seq_sparse,
                    positions=sparse_positions,
                    guidance=GUIDANCE,
                    microbatch_size=(1 if method == "serial_b1" else candidate_microbatch),
                    fuse_cfg=fuse_cfg,
                )
            total = total.merge(stats)
            latents = [
                scheduler_step(scheduler, p, step_t, latent, generator)
                for scheduler, p, latent, generator in zip(
                    sparse_schedulers, pred, latents, generators
                )
            ]

            if method == "halving" and global_step in rounds and len(latents) > 1:
                scores = scorer(candidate_ids, latents, global_step)
                keep_ids = select_topk(candidate_ids, scores, rounds[global_step])
                old_ids = list(candidate_ids)
                index = {cid: i for i, cid in enumerate(candidate_ids)}
                keep_indices = [index[cid] for cid in keep_ids]
                candidate_ids, latents, sparse_schedulers, cond_contexts, null_contexts, _ = prune_candidate_state(
                    candidate_ids,
                    latents,
                    sparse_schedulers,
                    cond_contexts,
                    null_contexts,
                    keep_ids,
                    None,
                )
                generators = [generators[i] for i in keep_indices]
                pruning_log.append(
                    {
                        "global_step": global_step,
                        "active_before": old_ids,
                        "scores": {str(k): float(v) for k, v in scores.items()},
                        "active_after": list(candidate_ids),
                    }
                )

    return candidate_ids, latents, total, pruning_log


def run_method_once(
    *,
    pipeline: Any,
    model_config: Any,
    base_noises: Sequence[torch.Tensor],
    seeds: Sequence[int],
    cond_context: Any,
    null_context: Any,
    pattern: str,
    method: str,
    candidate_microbatch: int,
    sparse_cfg_interval: int,
    halving_schedule: str,
    scorer_spec: str | None,
    size: tuple[int, int],
) -> dict[str, Any]:
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if method == "serial_b1":
        ids: list[int] = []
        outputs: list[torch.Tensor] = []
        total = ExecutionStats()
        pruning_log: list[dict[str, Any]] = []
        for noise, seed in zip(base_noises, seeds):
            one_ids, one_outputs, stats, _ = run_multi_candidate_preview(
                pipeline=pipeline,
                model_config=model_config,
                base_noises=[noise],
                seeds=[seed],
                cond_context=cond_context,
                null_context=null_context,
                pattern=pattern,
                method="serial_b1",
                candidate_microbatch=1,
                sparse_cfg_interval=sparse_cfg_interval,
                halving_schedule="",
                scorer_spec=None,
                size=size,
            )
            ids.extend(one_ids)
            outputs.extend(one_outputs)
            total = total.merge(stats)
    else:
        ids, outputs, total, pruning_log = run_multi_candidate_preview(
            pipeline=pipeline,
            model_config=model_config,
            base_noises=base_noises,
            seeds=seeds,
            cond_context=cond_context,
            null_context=null_context,
            pattern=pattern,
            method=method,
            candidate_microbatch=candidate_microbatch,
            sparse_cfg_interval=sparse_cfg_interval,
            halving_schedule=halving_schedule,
            scorer_spec=scorer_spec,
            size=size,
        )

    end.record()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    cuda_s = start.elapsed_time(end) / 1000.0
    return {
        "method": method,
        "active_candidate_ids": ids,
        "wall_s": wall_s,
        "cuda_s": cuda_s,
        "stats": asdict(total),
        "pruning_log": pruning_log,
        "outputs": [item.detach().cpu() for item in outputs],
    }


def median_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "wall_s_median": statistics.median(float(item["wall_s"]) for item in records),
        "cuda_s_median": statistics.median(float(item["cuda_s"]) for item in records),
        "wall_s_raw": [float(item["wall_s"]) for item in records],
        "cuda_s_raw": [float(item["cuda_s"]) for item in records],
        "stats": records[0]["stats"],
        "active_candidate_ids": records[0]["active_candidate_ids"],
        "pruning_log": records[0]["pruning_log"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patterns", nargs="+", choices=sorted(PATTERNS), default=["N9", "N6"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=["serial_b1", "candidate_b2", "candidate_cfg_b4", "sparse_cfg"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--candidate-microbatch", type=int, default=2)
    parser.add_argument("--sparse-cfg-interval", type=int, default=2)
    parser.add_argument("--halving-schedule", default="15:4,30:2,45:1")
    parser.add_argument(
        "--scorer",
        default=None,
        help=(
            "successive-halving scorer as module:function with signature "
            "(candidate_ids, latents, global_step)->dict[id, score]. "
            "Default latent RMS is systems-only and not quality-valid."
        ),
    )
    args = parser.parse_args()

    if args.blocks < 1:
        raise ValueError("blocks must be >= 1")
    if "halving" in args.methods:
        rounds = parse_halving_schedule(args.halving_schedule)
        if rounds and rounds[0].keep >= len(args.seeds):
            raise ValueError("first halving keep count must be smaller than initial candidate count")
    if any(
        step >= STEPS or step < WARMUP_STEPS
        for step in [r.step for r in parse_halving_schedule(args.halving_schedule)]
    ):
        raise ValueError(f"halving steps must lie in [{WARMUP_STEPS}, {STEPS - 1}]")

    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    if not FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("expected FlashAttention-2 and no FlashAttention-3 for frozen comparison")

    model_config = WAN_CONFIGS["t2v-1.3B"]
    device = torch.device("cuda:0")
    size = SIZE_CONFIGS[SIZE_NAME]
    pipeline = wan.WanT2V(
        config=model_config,
        checkpoint_dir=str(args.checkpoint),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )
    torch.cuda.synchronize()

    with torch.no_grad():
        cond_context = [item.to(device) for item in pipeline.text_encoder([PROMPT], torch.device("cpu"))]
        null_context = [
            item.to(device)
            for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))
        ]
    shape, _, _ = target_geometry(pipeline, size)
    base_noises = [
        torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        for seed in args.seeds
    ]
    a = torch.eye(2, dtype=torch.float32, device=device)
    b = torch.ones((2, 1), dtype=torch.float32, device=device)
    torch.linalg.solve(a, b)
    torch.cuda.synchronize()
    del a, b

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "SVA_CANDIDATE_FUSION_AND_SEARCH_ACCELERATION",
        "workload": {
            "model": "Wan2.1-T2V-1.3B",
            "resolution": SIZE_NAME,
            "frames": FRAME_NUM,
            "steps": STEPS,
            "dense_warmup_steps": WARMUP_STEPS,
            "shift": SHIFT,
            "guidance": GUIDANCE,
            "seeds": args.seeds,
        },
        "method_contract": {
            "exact": sorted(EXACT_METHODS),
            "approximate": ["sparse_cfg", "halving"],
            "candidate_microbatch": args.candidate_microbatch,
            "sparse_cfg_interval": args.sparse_cfg_interval,
            "halving_schedule": args.halving_schedule,
            "halving_scorer": args.scorer or "latent_rms (SYSTEM-ONLY diagnostic)",
        },
        "patterns": {},
    }

    for pattern in args.patterns:
        pattern_result: dict[str, Any] = {"methods": {}}
        reference_outputs: list[torch.Tensor] | None = None
        reference_cuda: float | None = None
        for method in args.methods:
            _ = run_method_once(
                pipeline=pipeline,
                model_config=model_config,
                base_noises=base_noises,
                seeds=args.seeds,
                cond_context=cond_context,
                null_context=null_context,
                pattern=pattern,
                method=method,
                candidate_microbatch=args.candidate_microbatch,
                sparse_cfg_interval=args.sparse_cfg_interval,
                halving_schedule=args.halving_schedule,
                scorer_spec=args.scorer,
                size=size,
            )
            records: list[dict[str, Any]] = []
            for _block in range(args.blocks):
                records.append(
                    run_method_once(
                        pipeline=pipeline,
                        model_config=model_config,
                        base_noises=base_noises,
                        seeds=args.seeds,
                        cond_context=cond_context,
                        null_context=null_context,
                        pattern=pattern,
                        method=method,
                        candidate_microbatch=args.candidate_microbatch,
                        sparse_cfg_interval=args.sparse_cfg_interval,
                        halving_schedule=args.halving_schedule,
                        scorer_spec=args.scorer,
                        size=size,
                    )
                )
            summary = median_summary(records)
            if method == "serial_b1":
                reference_outputs = records[0]["outputs"]
                reference_cuda = float(summary["cuda_s_median"])
                summary["speedup_vs_serial_b1"] = 1.0
                summary["numerical_match_vs_serial_b1"] = {
                    "count_match": 1,
                    "mean_abs": 0.0,
                    "max_abs": 0.0,
                    "mean_cosine": 1.0,
                }
            else:
                if reference_cuda is not None:
                    summary["speedup_vs_serial_b1"] = reference_cuda / float(summary["cuda_s_median"])
                if reference_outputs is not None and method in EXACT_METHODS:
                    summary["numerical_match_vs_serial_b1"] = compare_tensor_lists(
                        reference_outputs, records[0]["outputs"]
                    )
                elif method not in EXACT_METHODS:
                    summary["numerical_match_vs_serial_b1"] = (
                        "not applicable: approximate search policy"
                    )
            pattern_result["methods"][method] = summary
            atomic_json(
                output_root / "partial_result.json",
                final | {"patterns": final["patterns"] | {pattern: pattern_result}},
            )
        final["patterns"][pattern] = pattern_result

    atomic_json(output_root / "result.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
