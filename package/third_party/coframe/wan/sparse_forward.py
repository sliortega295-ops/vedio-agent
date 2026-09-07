from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from packaging.version import Version

from ..budget import defect_stat, lookup_scheduled_budget, select_budget
from ..config import CoFrameConfig
from ..controller import AdaptiveMeshController
from ..interpolation import (
    frame_token_indices,
    frames_to_tokens,
    leave_one_out_defects,
    reconstruct_sparse_block,
    tokens_to_frames,
)
from ..metrics import (
    frame_gram_matrix,
    headroom_recovery,
    interpolation_interval_costs,
    mesh_reconstruction_metrics,
    one_swap_diagnostics,
    optimal_piecewise_linear_mesh,
    reconstruction_metrics,
)
from ..selection import coverage_interleaved_select, fis_interleaved_select, uniform_select
from ..trace import CoFrameTrace


_PROJECTION_CACHE: dict[tuple[str, int | None, str, int, int, int], torch.Tensor] = {}


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    num_frames: int
    height: int
    width: int

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    @property
    def sequence_length(self) -> int:
        return self.num_frames * self.tokens_per_frame


@dataclass(slots=True)
class TransformerForwardMetadata:
    step_index: int
    block_anchors: dict[int, list[int]]
    refresh_events: list[dict[str, Any]]
    budget_events: list[dict[str, Any]]
    defects: list[dict[str, Any]]
    probes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "block_anchors": {str(key): value for key, value in self.block_anchors.items()},
            "refresh_events": self.refresh_events,
            "budget_events": self.budget_events,
            "defects": self.defects,
            "probes": self.probes,
        }


def require_diffusers_034(strict: bool = True) -> str:
    """Validate the integration target before touching Wan internals."""
    try:
        import diffusers
    except ImportError as exc:  # pragma: no cover - exercised only in GPU env
        raise ImportError("CoFrame's Wan integration requires diffusers==0.34.0") from exc

    found = Version(diffusers.__version__)
    target = Version("0.34.0")
    if found != target:
        message = (
            f"CoFrame currently targets diffusers==0.34.0, but found {found}. "
            "Wan block internals changed after 0.34; install the pinned version or pass "
            "strict_diffusers_version=False only for deliberate porting work."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return str(found)


def _apply_rotary_emb_subset(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # Mirrors diffusers 0.34.0 WanAttnProcessor2_0. hidden_states is
    # [B,H,L,D], freqs is complex [1,1,L,D/2].
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    output = torch.view_as_real(rotated * freqs).flatten(3, 4)
    return output.type_as(hidden_states)


def _active_self_attention(
    attn: Any,
    query_states: torch.Tensor,
    key_value_states: torch.Tensor,
    rotary_q: torch.Tensor,
    rotary_k: torch.Tensor,
) -> torch.Tensor:
    if getattr(attn, "add_k_proj", None) is not None:
        raise NotImplementedError("The initial CoFrame integration targets Wan T2V self-attention only")

    query = attn.to_q(query_states)
    key = attn.to_k(key_value_states)
    value = attn.to_v(key_value_states)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

    query = _apply_rotary_emb_subset(query, rotary_q)
    key = _apply_rotary_emb_subset(key, rotary_k)
    output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    output = output.transpose(1, 2).flatten(2, 3).type_as(query_states)
    output = attn.to_out[0](output)
    output = attn.to_out[1](output)
    return output


def _build_projection(
    hidden_dim: int,
    sketch_dim: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sketch_dim <= 0 or sketch_dim >= hidden_dim:
        return None
    key = (device.type, device.index, str(dtype), int(hidden_dim), int(sketch_dim), int(seed))
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(hidden_dim, sketch_dim, generator=generator, dtype=torch.float32)
    projection /= math.sqrt(float(sketch_dim))
    projection = projection.to(device=device, dtype=dtype)
    _PROJECTION_CACHE[key] = projection
    return projection


def _sparse_block_forward(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    *,
    anchors: list[int],
    geometry: FrameGeometry,
    config: CoFrameConfig,
    compute_defects: bool,
    projection: torch.Tensor | None,
    interpolation_target: str | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Execute one Wan block exactly on anchors and reconstruct all frames."""
    if temb.ndim != 3:
        raise NotImplementedError("CoFrame v0.1 targets Wan2.1's [B,6,D] timestep modulation")

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        block.scale_shift_table + temb.float()
    ).chunk(6, dim=1)

    input_frames = tokens_to_frames(hidden_states, geometry.num_frames, geometry.tokens_per_frame)
    active_indices = frame_token_indices(anchors, geometry.tokens_per_frame, device=hidden_states.device)

    exact_anchor_inputs = hidden_states.index_select(1, active_indices)
    if config.kv_mode == "anchor_only":
        normalized_anchor = (
            block.norm1(exact_anchor_inputs.float()) * (1 + scale_msa) + shift_msa
        ).type_as(exact_anchor_inputs)
        key_value_states = normalized_anchor
        rotary_k = rotary_emb.index_select(2, active_indices)
    elif config.kv_mode == "full_kv":
        normalized_full = (
            block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        normalized_anchor = normalized_full.index_select(1, active_indices)
        key_value_states = normalized_full
        rotary_k = rotary_emb
    else:  # pragma: no cover - config validation guards this
        raise ValueError(f"Unsupported kv_mode: {config.kv_mode}")

    rotary_q = rotary_emb.index_select(2, active_indices)
    attention_output = _active_self_attention(
        block.attn1,
        normalized_anchor,
        key_value_states,
        rotary_q,
        rotary_k,
    )

    exact_anchor_states = (
        exact_anchor_inputs.float() + attention_output * gate_msa
    ).type_as(hidden_states)

    normalized_anchor = block.norm2(exact_anchor_states.float()).type_as(exact_anchor_states)
    attention_output = block.attn2(
        hidden_states=normalized_anchor,
        encoder_hidden_states=encoder_hidden_states,
    )
    exact_anchor_states = exact_anchor_states + attention_output

    normalized_anchor = (
        block.norm3(exact_anchor_states.float()) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(exact_anchor_states)
    feed_forward_output = block.ffn(normalized_anchor)
    exact_anchor_states = (
        exact_anchor_states.float() + feed_forward_output.float() * c_gate_msa
    ).type_as(hidden_states)

    exact_anchor_frames = exact_anchor_states.reshape(
        hidden_states.shape[0], len(anchors), geometry.tokens_per_frame, hidden_states.shape[-1]
    )
    reconstructed_frames, exact_anchor_delta = reconstruct_sparse_block(
        input_frames,
        exact_anchor_frames,
        anchors,
        target=interpolation_target or config.interpolation_target,
    )

    defects: dict[int, torch.Tensor] = {}
    if compute_defects:
        values = exact_anchor_delta if config.defect_target == "delta" else exact_anchor_frames
        defects = leave_one_out_defects(values, anchors, projection=projection)

    return frames_to_tokens(reconstructed_frames), defects


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks with stable tie handling for tiny frame vectors."""
    values_cpu = values.detach().float().cpu()
    order = torch.argsort(values_cpu, stable=True)
    sorted_values = values_cpu.index_select(0, order)
    ranks = torch.empty(values_cpu.numel(), dtype=torch.float32)
    left = 0
    while left < values_cpu.numel():
        right = left + 1
        while right < values_cpu.numel() and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * float(left + right - 1)
        left = right
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-8) -> float | None:
    left = left.float().cpu() - left.float().cpu().mean()
    right = right.float().cpu() - right.float().cpu().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) <= eps:
        return None
    return float(((left * right).sum() / denominator).item())


def _post_observation_risk(
    controller: AdaptiveMeshController,
    defects: dict[int, torch.Tensor],
    anchors: list[int],
) -> torch.Tensor:
    """Preview the risk field after observing the current block's defects."""
    observation = controller.project_defects(defects, anchors)
    dynamic = controller.dynamic_risk.clone() * controller.risk_ema
    mask = observation > 0
    dynamic[mask] += (1.0 - controller.risk_ema) * observation[mask]
    return controller.risk_floor + controller.prior_weight * controller.prior + dynamic


def _shuffle_defects(
    defects: dict[int, torch.Tensor],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    if len(defects) <= 1:
        return dict(defects)
    keys = sorted(defects)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(keys), generator=generator).tolist()
    values = [defects[keys[index]] for index in order]
    return {key: value for key, value in zip(keys, values)}


def _normalize_frame_signal(scores: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    values = scores.detach().float().cpu().clamp_min(0.0)
    if values.numel() <= 2:
        return values
    interior = values[1:-1]
    scale = float(interior.mean().item())
    if scale > eps:
        values = (values / scale).clamp_max(10.0)
    values[0] = 0.0
    values[-1] = 0.0
    return values


def _temporal_curvature_scores(
    frame_values: torch.Tensor,
    projection: torch.Tensor | None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Cheap frame-wise curvature from an already-available [B,F,P,D] state.

    With the default random channel sketch this adds only a small projection and
    elementwise temporal residual; it does not execute another DiT block.
    """
    if frame_values.ndim != 4:
        raise ValueError("frame_values must be [B,F,P,D]")
    frame_count = int(frame_values.shape[1])
    scores = torch.zeros(frame_count, dtype=torch.float32)
    if frame_count < 3:
        return scores
    if projection is not None:
        reduced = torch.matmul(
            frame_values,
            projection.to(device=frame_values.device, dtype=frame_values.dtype),
        ).float()
    else:
        # Deterministic channel subsampling keeps exact-diagnostic mode bounded.
        stride = max(1, int(frame_values.shape[-1]) // 64)
        reduced = frame_values[..., ::stride].float()
    predicted = 0.5 * (reduced[:, :-2] + reduced[:, 2:])
    center = reduced[:, 1:-1]
    residual = (center - predicted).square().mean(dim=(0, 2, 3)).sqrt()
    local_energy = (reduced[:, :-2].square() + center.square() + reduced[:, 2:].square()) / 3.0
    magnitude = local_energy.mean(dim=(0, 2, 3)).sqrt()
    scores[1:-1] = (residual / (magnitude + eps)).detach().cpu()
    return scores


def _shuffle_frame_signal(scores: torch.Tensor, *, seed: int) -> torch.Tensor:
    values = scores.detach().float().cpu().clone()
    if values.numel() <= 3:
        return values
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    interior = values[1:-1].clone()
    order = torch.randperm(interior.numel(), generator=generator)
    values[1:-1] = interior.index_select(0, order)
    return values


def _propagation_diagnostics(
    *,
    blocks: Any,
    block_index: int,
    dense_output: torch.Tensor,
    sparse_output: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_projection: torch.Tensor,
    rotary_emb: torch.Tensor,
    horizons: tuple[int, ...],
    geometry: FrameGeometry,
    chunk_size: int,
) -> dict[str, Any]:
    """Propagate local dense/sparse states through identical dense blocks."""
    requested = sorted({int(value) for value in horizons if int(value) > 0})
    if not requested:
        return {}

    dense_state = dense_output
    sparse_state = sparse_output
    results: dict[str, Any] = {}
    max_horizon = min(max(requested), len(blocks) - block_index - 1)
    for offset in range(1, max_horizon + 1):
        block = blocks[block_index + offset]
        dense_state = block(dense_state, encoder_hidden_states, timestep_projection, rotary_emb)
        sparse_state = block(sparse_state, encoder_hidden_states, timestep_projection, rotary_emb)
        if offset in requested:
            dense_frames = tokens_to_frames(dense_state, geometry.num_frames, geometry.tokens_per_frame)
            sparse_frames = tokens_to_frames(sparse_state, geometry.num_frames, geometry.tokens_per_frame)
            results[str(offset)] = reconstruction_metrics(
                dense_frames,
                sparse_frames,
                chunk_size=chunk_size,
            )
    return results


def _probe_entry(
    *,
    step_index: int,
    block_index: int,
    block_input: torch.Tensor,
    dense_output: torch.Tensor,
    sparse_output: torch.Tensor,
    geometry: FrameGeometry,
    config: CoFrameConfig,
    controller: AdaptiveMeshController,
    anchors: list[int],
    defects: dict[int, torch.Tensor],
    propagation: dict[str, Any],
    block: Any,
    blocks: Any,
    encoder_hidden_states: torch.Tensor,
    timestep_projection: torch.Tensor,
    rotary_emb: torch.Tensor,
    projection: torch.Tensor | None,
    previous_delta_curvature: torch.Tensor | None,
) -> dict[str, Any]:
    input_frames = tokens_to_frames(block_input, geometry.num_frames, geometry.tokens_per_frame)
    dense_frames = tokens_to_frames(dense_output, geometry.num_frames, geometry.tokens_per_frame)
    sparse_frames = tokens_to_frames(sparse_output, geometry.num_frames, geometry.tokens_per_frame)
    dense_delta = dense_frames - input_frames
    sparse_delta = sparse_frames - input_frames

    # Realized sparse-operator error includes both interpolation and any loss of
    # non-anchor K/V context. Delta error is primary because CoFrame's default
    # operator interpolates block updates rather than complete hidden states.
    realized_delta = reconstruction_metrics(
        dense_delta,
        sparse_delta,
        anchors=anchors,
        chunk_size=config.oracle_metric_chunk_size,
    )
    realized_output = reconstruction_metrics(
        dense_frames,
        sparse_frames,
        anchors=anchors,
        chunk_size=config.oracle_metric_chunk_size,
    )
    delta_frame_error = torch.tensor(
        realized_delta["per_frame_global_normalized_rms"],
        dtype=torch.float32,
    )

    anchor_mask = torch.zeros(geometry.num_frames, dtype=torch.bool)
    anchor_mask[torch.tensor(anchors, dtype=torch.long)] = True
    non_anchor_mask = ~anchor_mask
    actual_non_anchor = delta_frame_error[non_anchor_mask]

    prior_base = controller.risk_floor + controller.prior_weight * controller.prior
    prior_predicted = controller.approximation_risk(anchors, prior_base)
    causal_predicted = controller.approximation_risk(anchors, controller.risk)
    current_defect_base = controller.project_defects(defects, anchors)
    current_defect_predicted = controller.approximation_risk(anchors, current_defect_base)

    def correlations(predicted: torch.Tensor) -> tuple[float | None, float | None]:
        values = predicted[non_anchor_mask]
        if values.numel() < 2:
            return None, None
        return _pearson(values, actual_non_anchor), _pearson(_rankdata(values), _rankdata(actual_non_anchor))

    prior_pearson, prior_spearman = correlations(prior_predicted)
    causal_pearson, causal_spearman = correlations(causal_predicted)
    defect_pearson, defect_spearman = correlations(current_defect_predicted)

    # Mesh-only metrics replace each selected anchor with its exact dense block
    # value. This isolates frame-position quality from sparse attention context.
    mesh_target = dense_delta if config.interpolation_target == "delta" else dense_frames
    gram = frame_gram_matrix(mesh_target, chunk_size=config.oracle_metric_chunk_size)
    interval_costs = interpolation_interval_costs(gram)
    total_energy = float(torch.diagonal(gram).sum().item())
    probe_budget = len(anchors) if config.method in {"adaptive_k", "coframe_ode"} else config.num_anchors
    fixed_anchors = uniform_select(
        geometry.num_frames,
        probe_budget,
        config.force_boundaries,
    )
    rhyme_anchors = (
        list(fixed_anchors)
        if config.method in {"adaptive_k", "coframe_ode"}
        else list(getattr(controller, "rhyme_reference_anchors", controller.initial_anchors))
    )
    fis_anchors = fis_interleaved_select(
        geometry.num_frames,
        probe_budget,
        block_index,
        config.sparse_block_start,
        force_boundaries=config.force_boundaries,
        anchor_stride=config.fis_anchor_stride,
    )
    oracle = optimal_piecewise_linear_mesh(
        interval_costs,
        num_anchors=probe_budget,
        total_energy=total_energy,
        min_gap=config.min_anchor_gap,
        force_boundaries=config.force_boundaries,
    )

    meshes = {
        "current": list(anchors),
        "rhyme": rhyme_anchors,
        "fixed": fixed_anchors,
        "fis": fis_anchors,
        "oracle": oracle.anchors,
    }
    metric_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    mesh_metrics: dict[str, dict[str, Any]] = {}
    for name, mesh in meshes.items():
        key = tuple(mesh)
        if key not in metric_cache:
            metric_cache[key] = mesh_reconstruction_metrics(
                mesh_target,
                mesh,
                chunk_size=config.oracle_metric_chunk_size,
            )
        mesh_metrics[name] = metric_cache[key]

    current_mesh_error = float(mesh_metrics["current"]["relative_l2"])
    rhyme_mesh_error = float(mesh_metrics["rhyme"]["relative_l2"])
    oracle_mesh_error = float(oracle.relative_rmse)
    current_mesh_nmse = float(mesh_metrics["current"]["normalized_mse"])
    rhyme_mesh_nmse = float(mesh_metrics["rhyme"]["normalized_mse"])
    oracle_mesh_nmse = float(oracle.normalized_mse)
    recovery = headroom_recovery(
        baseline_error=rhyme_mesh_nmse,
        method_error=current_mesh_nmse,
        oracle_error=oracle_mesh_nmse,
    )

    # Correlation with frame error is not enough: the controller acts by a
    # fixed-budget remove/add swap. Evaluate the action ranking directly.
    post_risk = _post_observation_risk(controller, defects, anchors)
    shuffled_defects = _shuffle_defects(
        defects,
        seed=config.shuffle_defect_seed + step_index * 1009 + block_index * 9176,
    )
    shuffled_post_risk = _post_observation_risk(controller, shuffled_defects, anchors)
    gap_only_risk = torch.ones(controller.num_frames, dtype=torch.float32)
    input_curvature = None
    shuffled_input_curvature = None
    normalized_previous_delta_curvature = None
    shuffled_previous_delta_curvature = None
    if config.probe_curvature_signals:
        input_curvature = _normalize_frame_signal(_temporal_curvature_scores(input_frames, projection))
        shuffled_input_curvature = _shuffle_frame_signal(
            input_curvature,
            seed=config.curvature_shuffle_seed + step_index * 1009 + block_index * 9176,
        )
        if previous_delta_curvature is not None:
            normalized_previous_delta_curvature = _normalize_frame_signal(previous_delta_curvature)
            shuffled_previous_delta_curvature = _shuffle_frame_signal(
                normalized_previous_delta_curvature,
                seed=config.curvature_shuffle_seed + 17 + step_index * 1009 + block_index * 9176,
            )

    swap_diagnostics = {
        "prior": one_swap_diagnostics(
            anchors=anchors,
            interval_costs=interval_costs,
            predicted_risk=prior_base,
            gap_power=controller.gap_power,
            move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain,
            min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        ),
        "causal": one_swap_diagnostics(
            anchors=anchors,
            interval_costs=interval_costs,
            predicted_risk=controller.risk,
            gap_power=controller.gap_power,
            move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain,
            min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        ),
        "post_observation": one_swap_diagnostics(
            anchors=anchors,
            interval_costs=interval_costs,
            predicted_risk=post_risk,
            gap_power=controller.gap_power,
            move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain,
            min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        ),
        "gap_only": one_swap_diagnostics(
            anchors=anchors,
            interval_costs=interval_costs,
            predicted_risk=gap_only_risk,
            gap_power=controller.gap_power,
            move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain,
            min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        ),
        "shuffled_defect": one_swap_diagnostics(
            anchors=anchors,
            interval_costs=interval_costs,
            predicted_risk=shuffled_post_risk,
            gap_power=controller.gap_power,
            move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain,
            min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        ),
    }
    if input_curvature is not None:
        swap_diagnostics["input_curvature"] = one_swap_diagnostics(
            anchors=anchors, interval_costs=interval_costs, predicted_risk=input_curvature,
            gap_power=controller.gap_power, move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        )
        swap_diagnostics["shuffled_input_curvature"] = one_swap_diagnostics(
            anchors=anchors, interval_costs=interval_costs, predicted_risk=shuffled_input_curvature,
            gap_power=controller.gap_power, move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        )
    if normalized_previous_delta_curvature is not None:
        swap_diagnostics["previous_delta_curvature"] = one_swap_diagnostics(
            anchors=anchors, interval_costs=interval_costs, predicted_risk=normalized_previous_delta_curvature,
            gap_power=controller.gap_power, move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        )
        swap_diagnostics["shuffled_previous_delta_curvature"] = one_swap_diagnostics(
            anchors=anchors, interval_costs=interval_costs, predicted_risk=shuffled_previous_delta_curvature,
            gap_power=controller.gap_power, move_penalty=controller.move_penalty,
            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,
            force_boundaries=controller.force_boundaries,
        )

    counterfactual_operator: dict[str, Any] = {}
    reference_meshes = {"rhyme": rhyme_anchors, "fixed": fixed_anchors, "fis": fis_anchors}
    for name in config.probe_counterfactual_methods:
        mesh = list(reference_meshes[name])
        if mesh == anchors:
            candidate_output = sparse_output
            candidate_propagation = propagation
        else:
            candidate_output, _ = _sparse_block_forward(
                block,
                block_input,
                encoder_hidden_states,
                timestep_projection,
                rotary_emb,
                anchors=mesh,
                geometry=geometry,
                config=config,
                compute_defects=False,
                projection=projection,
                interpolation_target="state" if name == "fis" else config.interpolation_target,
            )
            candidate_propagation = _propagation_diagnostics(
                blocks=blocks,
                block_index=block_index,
                dense_output=dense_output,
                sparse_output=candidate_output,
                encoder_hidden_states=encoder_hidden_states,
                timestep_projection=timestep_projection,
                rotary_emb=rotary_emb,
                horizons=tuple(int(value) for value in config.oracle_probe_horizons),
                geometry=geometry,
                chunk_size=config.oracle_metric_chunk_size,
            )
        candidate_frames = tokens_to_frames(candidate_output, geometry.num_frames, geometry.tokens_per_frame)
        candidate_delta = candidate_frames - input_frames
        candidate_realized = reconstruction_metrics(
            dense_delta,
            candidate_delta,
            anchors=mesh,
            chunk_size=config.oracle_metric_chunk_size,
        )
        counterfactual_operator[name] = {
            "anchors": mesh,
            "realized_block_delta": candidate_realized,
            "propagation": candidate_propagation,
        }

    anchor_index = torch.tensor(anchors, device=dense_delta.device, dtype=torch.long)
    anchor_context_delta_relative_l2 = reconstruction_metrics(
        dense_delta.index_select(1, anchor_index),
        sparse_delta.index_select(1, anchor_index),
        chunk_size=config.oracle_metric_chunk_size,
    )["relative_l2"]

    result: dict[str, Any] = {
        "step": step_index,
        "block": block_index,
        "anchors": list(anchors),
        "anchor_budget": probe_budget,
        "actual_frame_error": delta_frame_error.tolist(),
        "actual_delta_frame_error": delta_frame_error.tolist(),
        "actual_output_frame_error": realized_output["per_frame_global_normalized_rms"],
        "prior_expected_error": prior_predicted.tolist(),
        "causal_expected_error": causal_predicted.tolist(),
        "current_defect_expected_error": current_defect_predicted.tolist(),
        "pearson": causal_pearson,
        "spearman": causal_spearman,
        "prior_pearson": prior_pearson,
        "prior_spearman": prior_spearman,
        "defect_pearson": defect_pearson,
        "defect_spearman": defect_spearman,
        "spearman_gain_over_rhyme_prior": (
            None if causal_spearman is None or prior_spearman is None else causal_spearman - prior_spearman
        ),
        "realized_block_delta": realized_delta,
        "realized_block_output": realized_output,
        "mesh_only": {
            **mesh_metrics,
            "oracle": {**mesh_metrics["oracle"], **oracle.to_dict()},
            "headroom_recovery": recovery,
            "current_oracle_excess": current_mesh_error - oracle_mesh_error,
            "current_oracle_nmse_regret": current_mesh_nmse - oracle_mesh_nmse,
            "relative_improvement_over_rhyme": (
                (rhyme_mesh_error - current_mesh_error) / (rhyme_mesh_error + 1.0e-12)
            ),
        },
        "swap_decision": swap_diagnostics,
        "input_curvature_scores": None if input_curvature is None else input_curvature.tolist(),
        "previous_delta_curvature_scores": (
            None if normalized_previous_delta_curvature is None else normalized_previous_delta_curvature.tolist()
        ),
        "counterfactual_operator": counterfactual_operator,
        "propagation": propagation,
        # Flat aliases make batch aggregation simple.
        "block_delta_normalized_mse": realized_delta["normalized_mse"],
        "block_delta_relative_l2": realized_delta["relative_l2"],
        "non_anchor_block_delta_relative_l2": realized_delta["non_anchor_relative_l2"],
        "block_delta_frame_p95": realized_delta["non_anchor_frame_error_p95"],
        "block_delta_frame_cvar10": realized_delta["non_anchor_frame_error_cvar10"],
        "block_delta_frame_max": realized_delta["frame_error_max"],
        "anchor_context_delta_relative_l2": anchor_context_delta_relative_l2,
        "mean_relative_rms": realized_output["frame_error_mean"],
        "non_anchor_mean_relative_rms": realized_output["non_anchor_frame_error_mean"],
        "anchor_context_error_mean": float(delta_frame_error[anchor_mask].mean().item()),
        "max_relative_rms": realized_output["frame_error_max"],
        "mesh_current_relative_l2": current_mesh_error,
        "mesh_rhyme_relative_l2": rhyme_mesh_error,
        "mesh_fixed_relative_l2": float(mesh_metrics["fixed"]["relative_l2"]),
        "mesh_fis_relative_l2": float(mesh_metrics["fis"]["relative_l2"]),
        "mesh_oracle_relative_l2": oracle_mesh_error,
        "mesh_current_nmse": current_mesh_nmse,
        "mesh_rhyme_nmse": rhyme_mesh_nmse,
        "mesh_fixed_nmse": float(mesh_metrics["fixed"]["normalized_mse"]),
        "mesh_fis_nmse": float(mesh_metrics["fis"]["normalized_mse"]),
        "mesh_oracle_nmse": oracle_mesh_nmse,
        "mesh_headroom_recovery": recovery,
        "mesh_nmse_improvement_over_rhyme": rhyme_mesh_nmse - current_mesh_nmse,
        "mesh_current_oracle_excess": current_mesh_error - oracle_mesh_error,
        "mesh_current_oracle_nmse_regret": current_mesh_nmse - oracle_mesh_nmse,
        "swap_prior_spearman": swap_diagnostics["prior"]["spearman"],
        "swap_causal_spearman": swap_diagnostics["causal"]["spearman"],
        "swap_post_observation_spearman": swap_diagnostics["post_observation"]["spearman"],
        "swap_post_observation_gain_recovery": swap_diagnostics["post_observation"]["gain_recovery"],
        "swap_post_observation_regret": swap_diagnostics["post_observation"]["regret"],
        "swap_post_observation_normalized_regret": swap_diagnostics["post_observation"]["normalized_regret"],
        "swap_post_observation_top1_exact": swap_diagnostics["post_observation"]["top1_exact"],
        "swap_gap_only_gain_recovery": swap_diagnostics["gap_only"]["gain_recovery"],
        "swap_gap_only_regret": swap_diagnostics["gap_only"]["regret"],
        "swap_shuffled_gain_recovery": swap_diagnostics["shuffled_defect"]["gain_recovery"],
        "swap_shuffled_regret": swap_diagnostics["shuffled_defect"]["regret"],
    }
    for name, payload in counterfactual_operator.items():
        baseline_nmse = float(payload["realized_block_delta"]["normalized_mse"])
        result[f"counterfactual_{name}_block_delta_nmse"] = baseline_nmse
        result[f"operator_nmse_relative_improvement_over_{name}"] = (
            (baseline_nmse - float(realized_delta["normalized_mse"])) / (baseline_nmse + 1.0e-12)
        )
        for horizon, metrics in payload["propagation"].items():
            result[f"counterfactual_{name}_propagated_relative_l2_h{horizon}"] = metrics["relative_l2"]
            current_metrics = propagation.get(horizon)
            if current_metrics is not None:
                baseline_error = float(metrics["relative_l2"])
                result[f"propagation_relative_improvement_over_{name}_h{horizon}"] = (
                    (baseline_error - float(current_metrics["relative_l2"])) / (baseline_error + 1.0e-12)
                )

    for horizon, metrics in propagation.items():
        result[f"propagated_relative_l2_h{horizon}"] = metrics["relative_l2"]
        result[f"propagated_frame_cvar10_h{horizon}"] = metrics["non_anchor_frame_error_cvar10"]
    return result


def coframe_transformer_forward(
    transformer: Any,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    config: CoFrameConfig,
    controller: AdaptiveMeshController,
    step_index: int,
    replay_block_anchors: dict[int, list[int]] | None = None,
    update_controller: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    trace: CoFrameTrace | None = None,
) -> tuple[torch.Tensor, TransformerForwardMetadata]:
    """Wan2.1 transformer forward with block-conditional sparse frame meshes.

    The conditional CFG branch calls this with ``update_controller=True`` and
    records its per-block schedule. The unconditional branch replays the exact
    same schedule, avoiding branch-dependent token shapes and controller drift.
    """
    require_diffusers_034(config.strict_diffusers_version)
    if attention_kwargs and attention_kwargs.get("scale") is not None:
        raise NotImplementedError("LoRA scale handling is not implemented in CoFrame's custom Wan forward")
    if hidden_states.ndim != 5:
        raise ValueError("hidden_states must be [B,C,F,H,W]")
    if timestep.ndim != 1:
        raise NotImplementedError("CoFrame v0.1 targets Wan2.1 scalar timesteps")

    batch_size, _, num_frames, height, width = hidden_states.shape
    if config.method == "sparse_evolution" and batch_size != 1:
        raise ValueError("sparse_evolution supports one candidate per controller")
    p_t, p_h, p_w = transformer.config.patch_size
    if p_t != 1:
        raise NotImplementedError("CoFrame v0.1 assumes temporal patch_size == 1")
    geometry = FrameGeometry(num_frames, height // p_h, width // p_w)
    config.validate(num_blocks=len(transformer.blocks), num_frames=num_frames)
    if controller.num_frames != num_frames:
        raise ValueError("Controller frame count does not match transformer input")

    rotary_emb = transformer.rope(hidden_states)
    hidden_states = transformer.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
        timestep,
        encoder_hidden_states,
        None,
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    projection = _build_projection(
        hidden_states.shape[-1],
        config.sketch_dim,
        config.sketch_seed,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    metadata = TransformerForwardMetadata(
        step_index=step_index,
        block_anchors={},
        refresh_events=[],
        budget_events=[],
        defects=[],
        probes=[],
    )
    group_defects: dict[int, list[float]] = {}
    sparse_group_index = 0
    previous_delta_curvature: torch.Tensor | None = None
    current_group_anchors: list[int] | None = None
    groups_per_step = math.ceil(
        (config.sparse_block_end - config.sparse_block_start) / config.block_group_size
    )
    if config.method == "adaptive_k" and not config.adaptive_k_carry_across_steps:
        controller.current_budget = int(config.num_anchors)
    if (config.method == "sparse_evolution" and not config.adaptive_k_carry_across_steps
            and update_controller and replay_block_anchors is None):
        controller.current_budget = int(config.num_anchors)

    for block_index, block in enumerate(transformer.blocks):
        if not config.is_sparse_block(block_index):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            continue

        relative_zero = block_index - config.sparse_block_start
        adaptive_group_index = relative_zero // config.block_group_size
        is_group_start = relative_zero % config.block_group_size == 0
        if replay_block_anchors is not None:
            if block_index not in replay_block_anchors:
                raise KeyError(f"Missing replay anchors for sparse block {block_index}")
            anchors = list(replay_block_anchors[block_index])
        elif config.method == "sparse_evolution":
            # One mesh per block group. The preceding group's residual LOO
            # signal sets K; it never chooses spatial/temporal error locations.
            if is_group_start:
                current_group_anchors = coverage_interleaved_select(
                    geometry.num_frames, int(controller.current_budget),
                    step_index * groups_per_step + adaptive_group_index,
                    force_boundaries=True,
                )
                assignment = {
                    "step": step_index, "group": adaptive_group_index,
                    "after_block": block_index - 1,
                    "policy": config.adaptive_k_policy,
                    "assigned_k": int(controller.current_budget),
                    "source": "previous_group_or_initial_budget",
                    "placement": "coverage_interleaved",
                }
                metadata.budget_events.append(assignment)
                if update_controller:
                    controller.budget_history.append(dict(assignment))
            assert current_group_anchors is not None
            anchors = list(current_group_anchors)
        elif config.method == "adaptive_k":
            if is_group_start and config.adaptive_k_policy == "step_block":
                controller.current_budget = lookup_scheduled_budget(
                    config.adaptive_k_schedule,
                    step_index=step_index,
                    group_index=adaptive_group_index,
                    fallback=config.num_anchors,
                )
            anchors = uniform_select(
                geometry.num_frames,
                int(controller.current_budget),
                config.force_boundaries,
            )
            if is_group_start:
                assignment = {
                    "step": step_index,
                    "group": adaptive_group_index,
                    "after_block": block_index - 1,
                    "policy": config.adaptive_k_policy,
                    "assigned_k": int(controller.current_budget),
                    "source": "step_block_schedule" if config.adaptive_k_policy == "step_block" else "previous_group",
                }
                metadata.budget_events.append(assignment)
                controller.budget_history.append(dict(assignment))
        elif config.method == "coframe_ode":
            # K is fixed within this denoising step by the ODE-path controller.
            # The exact frame positions are then updated between block groups
            # using leave-one-out residual defects measured on already-computed
            # interior anchors. No dense reference or extra DiT forward is used.
            if is_group_start or current_group_anchors is None:
                target_budget = int(controller.current_budget)
                if controller.num_anchors != target_budget:
                    controller.set_budget(target_budget)
                current_group_anchors = list(controller.anchors)
            anchors = list(current_group_anchors)
            if is_group_start:
                metadata.budget_events.append(
                    {
                        "step": step_index,
                        "group": adaptive_group_index,
                        "after_block": block_index - 1,
                        "policy": "ode_path",
                        "assigned_k": int(controller.current_budget),
                        "source": "previous_step_trajectory",
                        "placement": "loo_residual_defect",
                    }
                )
        elif config.method == "fis":
            anchors = fis_interleaved_select(
                geometry.num_frames,
                config.num_anchors,
                block_index,
                config.sparse_block_start,
                force_boundaries=config.force_boundaries,
                anchor_stride=config.fis_anchor_stride,
            )
        else:
            anchors = list(controller.anchors)
        metadata.block_anchors[block_index] = anchors

        block_input = hidden_states
        curvature_from_previous_block = previous_delta_curvature
        dense_output = None
        if config.should_probe(step_index, block_index) and replay_block_anchors is None:
            dense_output = block(block_input, encoder_hidden_states, timestep_proj, rotary_emb)

        compute_defects = (
            replay_block_anchors is None
            and (
                (
                    config.method == "coframe"
                    and ((update_controller and config.refresh_signal in {"defect", "shuffled"}) or dense_output is not None)
                )
                or (
                    config.method in {"adaptive_k", "sparse_evolution"}
                    and ((update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}) or dense_output is not None)
                )
                or (
                    config.method == "coframe_ode"
                    and (update_controller or dense_output is not None)
                )
            )
        )
        hidden_states, defects = _sparse_block_forward(
            block,
            block_input,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            anchors=anchors,
            geometry=geometry,
            config=config,
            compute_defects=compute_defects,
            projection=projection,
            interpolation_target="state" if config.method == "fis" else config.interpolation_target,
        )
        if config.probe_curvature_signals:
            if config.should_probe(step_index, block_index + 1):
                before_frames = tokens_to_frames(block_input, geometry.num_frames, geometry.tokens_per_frame)
                after_frames = tokens_to_frames(hidden_states, geometry.num_frames, geometry.tokens_per_frame)
                previous_delta_curvature = _temporal_curvature_scores(after_frames - before_frames, projection)
            else:
                previous_delta_curvature = None

        if defects:
            defect_values = {frame: float(value.detach().float().item()) for frame, value in defects.items()}
            metadata.defects.append(
                {"step": step_index, "block": block_index, "anchors": anchors, "values": defect_values}
            )
            for frame, value in defect_values.items():
                group_defects.setdefault(frame, []).append(value)

        if dense_output is not None:
            propagation = _propagation_diagnostics(
                blocks=transformer.blocks,
                block_index=block_index,
                dense_output=dense_output,
                sparse_output=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_projection=timestep_proj,
                rotary_emb=rotary_emb,
                horizons=tuple(int(value) for value in config.oracle_probe_horizons),
                geometry=geometry,
                chunk_size=config.oracle_metric_chunk_size,
            )
            metadata.probes.append(
                _probe_entry(
                    step_index=step_index,
                    block_index=block_index,
                    block_input=block_input,
                    dense_output=dense_output,
                    sparse_output=hidden_states,
                    geometry=geometry,
                    config=config,
                    controller=controller,
                    anchors=anchors,
                    defects=defects,
                    propagation=propagation,
                    block=block,
                    blocks=transformer.blocks,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_projection=timestep_proj,
                    rotary_emb=rotary_emb,
                    projection=projection,
                    previous_delta_curvature=curvature_from_previous_block,
                )
            )

        relative_in_sparse_region = block_index - config.sparse_block_start + 1
        is_group_boundary = (
            relative_in_sparse_region % config.block_group_size == 0
            or block_index + 1 == config.sparse_block_end
        )
        if is_group_boundary and replay_block_anchors is None:
            sparse_group_index += 1
            if config.method == "coframe" and update_controller:
                aggregated = {
                    frame: sum(values) / len(values)
                    for frame, values in group_defects.items()
                    if values
                }
                if config.refresh_signal == "defect" and aggregated:
                    controller.observe(aggregated, anchors=anchors)
                elif config.refresh_signal == "shuffled" and aggregated:
                    shuffled = _shuffle_defects(
                        {frame: torch.tensor(value) for frame, value in aggregated.items()},
                        seed=config.shuffle_defect_seed + step_index * 1009 + sparse_group_index * 9176,
                    )
                    controller.observe(shuffled, anchors=anchors)
                # gap_only intentionally leaves a uniform risk field; none freezes the mesh.
                if (
                    config.refresh_signal != "none"
                    and sparse_group_index % config.refresh_every_groups == 0
                ):
                    refreshes = controller.refresh()
                    for refresh in refreshes:
                        entry = {
                            "step": step_index,
                            "after_block": block_index,
                            "group": sparse_group_index,
                            "refresh_signal": config.refresh_signal,
                            **refresh.to_dict(),
                        }
                        metadata.refresh_events.append(entry)
                        if trace is not None:
                            trace.add("mesh_refresh", **entry)
            if config.method == "coframe_ode" and update_controller:
                aggregated = {
                    frame: sum(values) / len(values)
                    for frame, values in group_defects.items()
                    if values
                }
                if aggregated:
                    controller.observe(aggregated, anchors=anchors)
                    if sparse_group_index % config.refresh_every_groups == 0:
                        refreshes = controller.refresh()
                        for refresh in refreshes:
                            entry = {
                                "step": step_index,
                                "after_block": block_index,
                                "group": sparse_group_index,
                                "refresh_signal": "loo_residual_defect",
                                "defect_mean": sum(aggregated.values()) / len(aggregated),
                                "defect_max": max(aggregated.values()),
                                **refresh.to_dict(),
                            }
                            metadata.refresh_events.append(entry)
                            if trace is not None:
                                trace.add("mesh_refresh", **entry)
                # The refreshed controller mesh becomes the next group's mesh.
                current_group_anchors = list(controller.anchors)

            if (config.method in {"adaptive_k", "sparse_evolution"} and update_controller
                    and config.adaptive_k_policy in {"mean_defect", "max_defect"}):
                samples = [value for values in group_defects.values() for value in values]
                if config.method == "sparse_evolution" and (
                    not samples or any(not math.isfinite(value) for value in samples)
                ):
                    raise RuntimeError("sparse_evolution requires finite interior residual LOO measurements")
                statistic = "mean" if config.adaptive_k_policy == "mean_defect" else "max"
                risk_value = defect_stat(samples, statistic)
                if risk_value is not None:
                    before_k = int(controller.current_budget)
                    next_k = select_budget(risk_value, config.adaptive_k_thresholds, config.adaptive_k_values)
                    controller.current_budget = int(next_k)
                    update = {
                        "step": step_index,
                        "source_group": adaptive_group_index,
                        "after_block": block_index,
                        "policy": config.adaptive_k_policy,
                        "risk_statistic": statistic,
                        "risk_value": float(risk_value),
                        "before_k": before_k,
                        "next_k": int(next_k),
                        "causal": True,
                    }
                    metadata.budget_events.append(update)
                    controller.budget_history.append(dict(update))
                    if trace is not None:
                        trace.add("budget_update", **update)
            group_defects = {}

    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = transformer.proj_out(hidden_states)

    out_channels = transformer.config.out_channels or transformer.config.in_channels
    hidden_states = hidden_states.reshape(
        batch_size,
        geometry.num_frames,
        geometry.height,
        geometry.width,
        p_t,
        p_h,
        p_w,
        out_channels,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if trace is not None:
        trace.add("transformer_forward", **metadata.to_dict())
    return output, metadata
