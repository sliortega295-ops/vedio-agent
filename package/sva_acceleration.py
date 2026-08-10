#!/usr/bin/env python3
"""Reusable acceleration primitives for Sparse Video Agent experiments.

The module deliberately separates exact execution changes from approximate
search policies:

Exact:
  * candidate batching (B=N)
  * candidate x CFG fusion (B=2N)

Approximate:
  * sparse-CFG refresh/reuse
  * successive halving through a pluggable scorer

Wan's model accepts a list of same-shape latent tensors and a list of contexts,
so candidates with the same sparse pattern can be fused without changing model
weights or attention semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Iterable, Sequence

import torch


Tensor = torch.Tensor
ScoreFn = Callable[[Sequence[int], Sequence[Tensor], int], dict[int, float]]


@dataclass(frozen=True)
class ExecutionStats:
    """Lightweight accounting independent of wall-clock profiling."""

    model_calls: int = 0
    model_examples: int = 0
    cfg_refreshes: int = 0

    def add(self, *, calls: int = 0, examples: int = 0, refreshes: int = 0) -> "ExecutionStats":
        return ExecutionStats(
            model_calls=self.model_calls + calls,
            model_examples=self.model_examples + examples,
            cfg_refreshes=self.cfg_refreshes + refreshes,
        )

    def merge(self, other: "ExecutionStats") -> "ExecutionStats":
        return self.add(
            calls=other.model_calls,
            examples=other.model_examples,
            refreshes=other.cfg_refreshes,
        )


@dataclass
class SparseCFGCache:
    unconditional: list[Tensor] | None = None
    last_refresh_step: int | None = None

    def select(self, indices: Sequence[int]) -> "SparseCFGCache":
        if self.unconditional is None:
            return SparseCFGCache(None, self.last_refresh_step)
        return SparseCFGCache([self.unconditional[i] for i in indices], self.last_refresh_step)


@dataclass(frozen=True)
class HalvingRound:
    """After completing `step`, keep the top `keep` active candidates."""

    step: int
    keep: int


def _as_context_tensor(context: Any) -> Tensor:
    """Normalize Wan text_encoder output for one prompt to one context tensor."""
    if isinstance(context, (list, tuple)):
        if len(context) != 1:
            raise ValueError(f"expected one prompt context tensor, got {len(context)}")
        context = context[0]
    if not isinstance(context, torch.Tensor):
        raise TypeError(f"expected Tensor context, got {type(context)!r}")
    return context


def repeat_context(context: Any, count: int) -> list[Tensor]:
    tensor = _as_context_tensor(context)
    return [tensor] * count


def chunk_indices(count: int, microbatch_size: int | None) -> Iterable[list[int]]:
    if count <= 0:
        return
    size = count if microbatch_size is None or microbatch_size <= 0 else microbatch_size
    for start in range(0, count, size):
        yield list(range(start, min(start + size, count)))


def _batched_model_forward(
    model: Any,
    latents: Sequence[Tensor],
    timestep: Tensor,
    contexts: Sequence[Tensor],
    seq_len: int,
    temporal_positions: Tensor,
) -> list[Tensor]:
    if not latents:
        return []
    if len(latents) != len(contexts):
        raise ValueError("latents and contexts must have the same batch size")
    if timestep.numel() != 1:
        raise ValueError("timestep must contain exactly one scalar timestep")
    t = timestep.reshape(1).expand(len(latents))
    outputs = model(
        list(latents),
        t=t,
        context=list(contexts),
        seq_len=seq_len,
        temporal_indices=temporal_positions,
    )
    if len(outputs) != len(latents):
        raise RuntimeError("Wan model returned an unexpected batch size")
    return list(outputs)


def candidate_batched_cfg_prediction(
    model: Any,
    latents: Sequence[Tensor],
    timestep: Tensor,
    cond_contexts: Sequence[Tensor],
    null_contexts: Sequence[Tensor],
    seq_len: int,
    temporal_positions: Tensor,
    guidance: float,
    *,
    fuse_cfg: bool,
) -> tuple[list[Tensor], list[Tensor], ExecutionStats]:
    """Exact CFG for N candidates.

    `fuse_cfg=False`: two model calls of batch N (candidate batch B=N).
    `fuse_cfg=True`: one model call of batch 2N (candidate x CFG fusion).
    """
    n = len(latents)
    if not (n == len(cond_contexts) == len(null_contexts)):
        raise ValueError("candidate/context batch sizes do not match")
    if n == 0:
        return [], [], ExecutionStats()

    if fuse_cfg:
        outputs = _batched_model_forward(
            model,
            list(latents) + list(latents),
            timestep,
            list(cond_contexts) + list(null_contexts),
            seq_len,
            temporal_positions,
        )
        conditional = outputs[:n]
        unconditional = outputs[n:]
        stats = ExecutionStats(model_calls=1, model_examples=2 * n)
    else:
        conditional = _batched_model_forward(
            model, latents, timestep, cond_contexts, seq_len, temporal_positions
        )
        unconditional = _batched_model_forward(
            model, latents, timestep, null_contexts, seq_len, temporal_positions
        )
        stats = ExecutionStats(model_calls=2, model_examples=2 * n)

    guided = [u + guidance * (c - u) for c, u in zip(conditional, unconditional)]
    return guided, unconditional, stats


def sparse_cfg_prediction(
    model: Any,
    latents: Sequence[Tensor],
    timestep: Tensor,
    cond_contexts: Sequence[Tensor],
    null_contexts: Sequence[Tensor],
    seq_len: int,
    temporal_positions: Tensor,
    guidance: float,
    *,
    global_step: int,
    refresh_interval: int,
    cache: SparseCFGCache,
) -> tuple[list[Tensor], SparseCFGCache, ExecutionStats]:
    """Approximate CFG by refreshing the unconditional branch periodically.

    On refresh steps we execute exact fused candidate x CFG. On intervening
    steps only the conditional batch is evaluated and the most recently cached
    unconditional prediction is reused. Dense warm-up should normally stay on
    exact CFG; apply this only to the sparse continuation.
    """
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    refresh = (
        cache.unconditional is None
        or global_step % refresh_interval == 0
    )
    if refresh:
        guided, unconditional, stats = candidate_batched_cfg_prediction(
            model,
            latents,
            timestep,
            cond_contexts,
            null_contexts,
            seq_len,
            temporal_positions,
            guidance,
            fuse_cfg=True,
        )
        return (
            guided,
            SparseCFGCache(unconditional=list(unconditional), last_refresh_step=global_step),
            stats.add(refreshes=1),
        )

    if cache.unconditional is None or len(cache.unconditional) != len(latents):
        raise RuntimeError("sparse-CFG cache does not match active candidate batch")
    conditional = _batched_model_forward(
        model, latents, timestep, cond_contexts, seq_len, temporal_positions
    )
    guided = [
        u + guidance * (c - u)
        for c, u in zip(conditional, cache.unconditional)
    ]
    return guided, cache, ExecutionStats(model_calls=1, model_examples=len(latents))


def exact_predictions_microbatched(
    model: Any,
    latents: Sequence[Tensor],
    timestep: Tensor,
    cond_contexts: Sequence[Tensor],
    null_contexts: Sequence[Tensor],
    seq_len: int,
    temporal_positions: Tensor,
    guidance: float,
    *,
    microbatch_size: int | None,
    fuse_cfg: bool,
) -> tuple[list[Tensor], ExecutionStats]:
    predictions: list[Tensor | None] = [None] * len(latents)
    total = ExecutionStats()
    for ids in chunk_indices(len(latents), microbatch_size):
        pred, _, stats = candidate_batched_cfg_prediction(
            model,
            [latents[i] for i in ids],
            timestep,
            [cond_contexts[i] for i in ids],
            [null_contexts[i] for i in ids],
            seq_len,
            temporal_positions,
            guidance,
            fuse_cfg=fuse_cfg,
        )
        for i, value in zip(ids, pred):
            predictions[i] = value
        total = total.merge(stats)
    if any(value is None for value in predictions):
        raise RuntimeError("prediction microbatch assembly failed")
    return [value for value in predictions if value is not None], total


def parse_halving_schedule(spec: str) -> list[HalvingRound]:
    """Parse `10:4,25:2,40:1` into validated pruning rounds."""
    rounds: list[HalvingRound] = []
    if not spec.strip():
        return rounds
    for token in spec.split(","):
        step_text, keep_text = token.strip().split(":", 1)
        rounds.append(HalvingRound(step=int(step_text), keep=int(keep_text)))
    previous_step = -1
    previous_keep: int | None = None
    for item in rounds:
        if item.step <= previous_step:
            raise ValueError("halving steps must be strictly increasing")
        if item.keep < 1:
            raise ValueError("halving keep counts must be >= 1")
        if previous_keep is not None and item.keep >= previous_keep:
            raise ValueError("halving keep counts must strictly decrease")
        previous_step = item.step
        previous_keep = item.keep
    return rounds


def select_topk(
    candidate_ids: Sequence[int],
    scores: dict[int, float],
    keep: int,
) -> list[int]:
    missing = [cid for cid in candidate_ids if cid not in scores]
    if missing:
        raise ValueError(f"scorer did not return scores for candidates {missing}")
    keep = min(keep, len(candidate_ids))
    return sorted(candidate_ids, key=lambda cid: (-float(scores[cid]), cid))[:keep]


def latent_rms_scorer(
    candidate_ids: Sequence[int], latents: Sequence[Tensor], global_step: int
) -> dict[int, float]:
    """SYSTEM-ONLY diagnostic scorer; not a semantic/video-quality metric."""
    del global_step
    return {
        cid: float(latent.detach().float().square().mean().sqrt().item())
        for cid, latent in zip(candidate_ids, latents)
    }


def load_scorer(spec: str | None) -> ScoreFn:
    """Load `module:function`; default is the diagnostic latent RMS scorer."""
    if not spec:
        return latent_rms_scorer
    module_name, function_name = spec.split(":", 1)
    function = getattr(import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec!r} is not callable")
    return function


def prune_candidate_state(
    candidate_ids: list[int],
    latents: list[Tensor],
    schedulers: list[Any],
    cond_contexts: list[Tensor],
    null_contexts: list[Tensor],
    keep_ids: Sequence[int],
    cache: SparseCFGCache | None = None,
) -> tuple[
    list[int], list[Tensor], list[Any], list[Tensor], list[Tensor], SparseCFGCache | None
]:
    index = {cid: i for i, cid in enumerate(candidate_ids)}
    keep_indices = [index[cid] for cid in keep_ids]
    new_cache = cache.select(keep_indices) if cache is not None else None
    return (
        list(keep_ids),
        [latents[i] for i in keep_indices],
        [schedulers[i] for i in keep_indices],
        [cond_contexts[i] for i in keep_indices],
        [null_contexts[i] for i in keep_indices],
        new_cache,
    )
