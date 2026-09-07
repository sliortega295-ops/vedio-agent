from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .interpolation import piecewise_linear_interpolate, validate_anchors


@dataclass(frozen=True, slots=True)
class OracleMeshResult:
    anchors: list[int]
    squared_error: float
    relative_rmse: float

    @property
    def normalized_mse(self) -> float:
        return self.relative_rmse**2

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchors": list(self.anchors),
            "squared_error": self.squared_error,
            "normalized_mse": self.normalized_mse,
            "relative_rmse": self.relative_rmse,
        }


def _frame_flatten(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 3:
        raise ValueError("values must have shape [B,F,...]")
    return values.detach().movedim(1, 0).contiguous().reshape(values.shape[1], -1)


def frame_error_sums(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    *,
    chunk_size: int = 65_536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame squared error and reference energy.

    Reduction is chunked over non-frame dimensions so a Wan block probe does
    not materialize full FP32 copies of every hidden state.
    """
    if reference.shape != approximation.shape or reference.ndim < 3:
        raise ValueError("reference and approximation must share [B,F,...] shape")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    ref = _frame_flatten(reference)
    app = _frame_flatten(approximation)
    numerator = torch.zeros(ref.shape[0], device=ref.device, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    for start in range(0, ref.shape[1], chunk_size):
        end = min(ref.shape[1], start + chunk_size)
        ref_chunk = ref[:, start:end].float()
        app_chunk = app[:, start:end].float()
        numerator += (ref_chunk - app_chunk).square().sum(dim=1)
        denominator += ref_chunk.square().sum(dim=1)
    return numerator, denominator


def relative_l2(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    *,
    chunk_size: int = 65_536,
) -> float:
    squared_error, reference_energy = frame_error_sums(reference, approximation, chunk_size=chunk_size)
    return math.sqrt(
        max(0.0, float(squared_error.sum().item()) / (float(reference_energy.sum().item()) + 1.0e-12))
    )


def per_frame_global_normalized_rms(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    *,
    chunk_size: int = 65_536,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Per-frame RMS residual normalized by one global reference RMS.

    Per-frame relative errors explode when a particular dense block delta is
    nearly zero. Global normalization is stable while preserving where error is
    concentrated over time. Equal-size frames yield
    ``sqrt(F * frame_sse / total_reference_energy)``.
    """
    squared_error, reference_energy = frame_error_sums(reference, approximation, chunk_size=chunk_size)
    frame_count = int(reference.shape[1])
    return torch.sqrt(float(frame_count) * squared_error / (reference_energy.sum() + eps))


def worst_fraction_mean(values: torch.Tensor, fraction: float = 0.1) -> float:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty one-dimensional tensor")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(math.ceil(float(values.numel()) * fraction)))
    return float(torch.topk(values.float(), k=count, largest=True).values.mean().item())


def temporal_gradient_relative_l2(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    *,
    frame_dim: int,
    chunk_size: int = 65_536,
) -> float:
    """Relative error of first temporal differences.

    This suppresses global latent offsets and emphasizes motion/transition
    distortion. Inputs may use any layout as long as ``frame_dim`` is given.
    """
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have the same shape")
    if reference.shape[frame_dim] < 2:
        return 0.0
    reference_diff = torch.diff(reference, dim=frame_dim)
    approximation_diff = torch.diff(approximation, dim=frame_dim)
    if frame_dim != 1:
        reference_diff = reference_diff.movedim(frame_dim, 1)
        approximation_diff = approximation_diff.movedim(frame_dim, 1)
    return relative_l2(reference_diff, approximation_diff, chunk_size=chunk_size)


def reconstruction_metrics(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    *,
    anchors: Sequence[int] | None = None,
    chunk_size: int = 65_536,
) -> dict[str, Any]:
    """Summarize realized-operator or mesh-only reconstruction error."""
    squared_error, reference_energy = frame_error_sums(reference, approximation, chunk_size=chunk_size)
    frame_count = int(reference.shape[1])
    frame_relative_rms = torch.sqrt(squared_error / (reference_energy + 1.0e-12))
    frame_global_rms = torch.sqrt(float(frame_count) * squared_error / (reference_energy.sum() + 1.0e-12))

    if anchors is None:
        non_anchor_mask = torch.ones(frame_count, dtype=torch.bool, device=reference.device)
    else:
        anchor_list = validate_anchors(anchors, frame_count)
        non_anchor_mask = torch.ones(frame_count, dtype=torch.bool, device=reference.device)
        non_anchor_mask[torch.tensor(anchor_list, device=reference.device, dtype=torch.long)] = False

    non_anchor_squared = squared_error[non_anchor_mask]
    non_anchor_energy = reference_energy[non_anchor_mask]
    non_anchor_global = frame_global_rms[non_anchor_mask]

    total_squared_error = float(squared_error.sum().item())
    total_reference_energy = float(reference_energy.sum().item())
    normalized_mse = total_squared_error / (total_reference_energy + 1.0e-12)
    non_anchor_global_mse = (
        float(non_anchor_squared.sum().item()) / (total_reference_energy + 1.0e-12)
        if non_anchor_squared.numel()
        else 0.0
    )

    return {
        "squared_error": total_squared_error,
        "reference_energy": total_reference_energy,
        "normalized_mse": normalized_mse,
        "relative_l2": math.sqrt(max(0.0, normalized_mse)),
        # This denominator is restricted to non-anchor reference energy and is
        # useful diagnostically, but total normalized MSE is the fair mesh metric.
        "non_anchor_relative_l2": (
            math.sqrt(
                max(
                    0.0,
                    float(non_anchor_squared.sum().item())
                    / (float(non_anchor_energy.sum().item()) + 1.0e-12),
                )
            )
            if non_anchor_squared.numel()
            else 0.0
        ),
        "non_anchor_global_normalized_mse": non_anchor_global_mse,
        "non_anchor_global_normalized_l2": math.sqrt(max(0.0, non_anchor_global_mse)),
        "frame_error_mean": float(frame_global_rms.mean().item()),
        "non_anchor_frame_error_mean": (
            float(non_anchor_global.mean().item()) if non_anchor_global.numel() else 0.0
        ),
        "per_frame_relative_error_mean": float(frame_relative_rms.mean().item()),
        "non_anchor_frame_error_p95": (
            float(torch.quantile(non_anchor_global.float(), 0.95).item()) if non_anchor_global.numel() else 0.0
        ),
        # Default F=21,K=9 leaves 12 skipped frames, so CVaR-10 averages
        # the two worst skipped-frame errors rather than collapsing to max.
        "non_anchor_frame_error_cvar10": (
            worst_fraction_mean(non_anchor_global, 0.1) if non_anchor_global.numel() else 0.0
        ),
        "frame_error_max": float(frame_global_rms.max().item()),
        "per_frame_relative_rms": frame_relative_rms.detach().cpu().tolist(),
        "per_frame_global_normalized_rms": frame_global_rms.detach().cpu().tolist(),
    }


def mesh_reconstruction_metrics(
    exact_values: torch.Tensor,
    anchors: Sequence[int],
    *,
    chunk_size: int = 65_536,
) -> dict[str, Any]:
    """Evaluate positions using exact dense values at selected anchors.

    This isolates frame-position quality from loss of non-anchor K/V context.
    For CoFrame's default operator, ``exact_values`` should be dense block deltas.
    """
    frame_count = int(exact_values.shape[1])
    anchor_list = validate_anchors(anchors, frame_count)
    anchor_tensor = torch.tensor(anchor_list, device=exact_values.device, dtype=torch.long)
    approximation = piecewise_linear_interpolate(
        exact_values.index_select(1, anchor_tensor),
        anchor_list,
        frame_count,
    )
    metrics = reconstruction_metrics(exact_values, approximation, anchors=anchor_list, chunk_size=chunk_size)
    metrics["anchors"] = anchor_list
    return metrics


def frame_gram_matrix(exact_values: torch.Tensor, *, chunk_size: int = 65_536) -> torch.Tensor:
    """Compute a float32 frame Gram matrix with bounded temporary memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    frames = _frame_flatten(exact_values)
    gram = torch.zeros(frames.shape[0], frames.shape[0], device=frames.device, dtype=torch.float32)
    for start in range(0, frames.shape[1], chunk_size):
        end = min(frames.shape[1], start + chunk_size)
        chunk = frames[:, start:end].float()
        gram += chunk @ chunk.transpose(0, 1)
    return gram


def interpolation_interval_costs(gram: torch.Tensor) -> torch.Tensor:
    """Exact linear-interpolation SSE for every interval under ``gram``."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square [F,F]")
    gram_cpu = gram.detach().double().cpu()
    frame_count = int(gram_cpu.shape[0])
    costs = torch.full((frame_count, frame_count), float("inf"), dtype=torch.float64)
    for left in range(frame_count):
        costs[left, left] = 0.0
        for right in range(left + 1, frame_count):
            if right == left + 1:
                costs[left, right] = 0.0
                continue
            gap = float(right - left)
            cost = 0.0
            for frame in range(left + 1, right):
                alpha = float(frame - left) / gap
                wl, wr = 1.0 - alpha, alpha
                residual_sq = (
                    gram_cpu[frame, frame]
                    + wl * wl * gram_cpu[left, left]
                    + wr * wr * gram_cpu[right, right]
                    + 2.0 * wl * wr * gram_cpu[left, right]
                    - 2.0 * wl * gram_cpu[frame, left]
                    - 2.0 * wr * gram_cpu[frame, right]
                )
                cost += max(0.0, float(residual_sq.item()))
            costs[left, right] = cost
    return costs


def mesh_squared_error(anchors: Sequence[int], interval_costs: torch.Tensor) -> float:
    frame_count = int(interval_costs.shape[0])
    mesh = validate_anchors(anchors, frame_count)
    return float(sum(float(interval_costs[left, right].item()) for left, right in zip(mesh[:-1], mesh[1:])))


def optimal_piecewise_linear_mesh(
    interval_costs: torch.Tensor,
    *,
    num_anchors: int,
    total_energy: float,
    min_gap: int = 1,
    force_boundaries: bool = True,
) -> OracleMeshResult:
    """Exact O(KF^2) DP oracle for a fixed-budget 1-D linear mesh."""
    if not force_boundaries:
        raise NotImplementedError("The interpolation oracle currently requires forced boundaries")
    frame_count = int(interval_costs.shape[0])
    if interval_costs.shape != (frame_count, frame_count):
        raise ValueError("interval_costs must be square")
    if not 2 <= num_anchors <= frame_count:
        raise ValueError("num_anchors must be in [2, frame_count]")
    min_gap = max(1, int(min_gap))
    if (num_anchors - 1) * min_gap > frame_count - 1:
        raise ValueError("No valid mesh satisfies num_anchors and min_gap")

    inf = float("inf")
    dp = [[inf for _ in range(frame_count)] for _ in range(num_anchors + 1)]
    parent = [[-1 for _ in range(frame_count)] for _ in range(num_anchors + 1)]
    dp[1][0] = 0.0

    for count in range(2, num_anchors + 1):
        min_right = (count - 1) * min_gap
        max_right = frame_count - 1 - (num_anchors - count) * min_gap
        for right in range(min_right, max_right + 1):
            min_left = (count - 2) * min_gap
            max_left = right - min_gap
            for left in range(min_left, max_left + 1):
                previous = dp[count - 1][left]
                if not math.isfinite(previous):
                    continue
                candidate = previous + float(interval_costs[left, right].item())
                if candidate < dp[count][right]:
                    dp[count][right] = candidate
                    parent[count][right] = left

    final_error = dp[num_anchors][frame_count - 1]
    if not math.isfinite(final_error):
        raise RuntimeError("Failed to construct an oracle mesh")
    anchors = [frame_count - 1]
    right = frame_count - 1
    for count in range(num_anchors, 1, -1):
        right = parent[count][right]
        if right < 0:
            raise RuntimeError("Oracle parent reconstruction failed")
        anchors.append(right)
    anchors.reverse()
    return OracleMeshResult(
        anchors=anchors,
        squared_error=final_error,
        relative_rmse=math.sqrt(final_error / (float(total_energy) + 1.0e-12)),
    )


def headroom_recovery(
    *,
    baseline_error: float,
    method_error: float,
    oracle_error: float,
    eps: float = 1.0e-8,
) -> float | None:
    """Fraction of Rhyme-to-oracle NMSE headroom recovered.

    Rhyme is 0, oracle is 1, and negative values mean worse than Rhyme. ``None``
    means the Rhyme mesh is already numerically indistinguishable from oracle.
    """
    denominator = float(baseline_error) - float(oracle_error)
    if denominator <= eps:
        return None
    return (float(baseline_error) - float(method_error)) / denominator


def _valid_mesh(mesh: Sequence[int], *, frame_count: int, min_gap: int, force_boundaries: bool) -> bool:
    anchors = list(mesh)
    if anchors != sorted(set(anchors)) or not anchors:
        return False
    if anchors[0] < 0 or anchors[-1] >= frame_count:
        return False
    if force_boundaries and frame_count > 1 and (anchors[0] != 0 or anchors[-1] != frame_count - 1):
        return False
    return all(right - left >= min_gap for left, right in zip(anchors[:-1], anchors[1:]))


def enumerate_one_swaps(
    anchors: Sequence[int],
    *,
    frame_count: int,
    min_gap: int = 1,
    force_boundaries: bool = True,
) -> list[tuple[int, int, list[int]]]:
    mesh = validate_anchors(anchors, frame_count)
    boundaries = {0, frame_count - 1} if force_boundaries and frame_count > 1 else set()
    removable = [frame for frame in mesh if frame not in boundaries]
    candidates = [frame for frame in range(frame_count) if frame not in mesh]
    trials: list[tuple[int, int, list[int]]] = []
    for removed in removable:
        base = [frame for frame in mesh if frame != removed]
        for added in candidates:
            trial = sorted(base + [added])
            if _valid_mesh(trial, frame_count=frame_count, min_gap=max(1, int(min_gap)), force_boundaries=force_boundaries):
                trials.append((removed, added, trial))
    return trials


def risk_mesh_cost(anchors: Sequence[int], risk: torch.Tensor, *, gap_power: float = 2.0) -> float:
    frame_count = int(risk.numel())
    mesh = validate_anchors(anchors, frame_count)
    values = risk.detach().float().cpu()
    cost = 0.0
    for left, right in zip(mesh[:-1], mesh[1:]):
        gap = right - left
        if gap <= 1:
            continue
        weighted_sum = 0.0
        weight_sum = 0.0
        for frame in range(left + 1, right):
            unit = float(frame - left) / float(gap)
            envelope = 4.0 * unit * (1.0 - unit)
            weighted_sum += float(values[frame].item()) * envelope
            weight_sum += envelope
        interval_risk = weighted_sum / max(weight_sum, 1.0e-12)
        cost += (float(gap) ** float(gap_power)) * interval_risk
    return cost


def _average_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-12) -> float | None:
    if left.numel() < 2 or right.numel() != left.numel():
        return None
    left = left.double().cpu() - left.double().cpu().mean()
    right = right.double().cpu() - right.double().cpu().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) <= eps:
        return None
    return float(((left * right).sum() / denominator).item())


def one_swap_diagnostics(
    *,
    anchors: Sequence[int],
    interval_costs: torch.Tensor,
    predicted_risk: torch.Tensor,
    gap_power: float,
    move_penalty: float,
    min_gain: float = 0.0,
    min_gap: int = 1,
    force_boundaries: bool = True,
) -> dict[str, Any]:
    """Compare the controller's one-swap decision with dense block truth."""
    frame_count = int(interval_costs.shape[0])
    current = validate_anchors(anchors, frame_count)
    trials = enumerate_one_swaps(
        current,
        frame_count=frame_count,
        min_gap=min_gap,
        force_boundaries=force_boundaries,
    )
    if not trials:
        return {
            "trial_count": 0,
            "spearman": None,
            "gain_recovery": None,
            "regret": None,
            "normalized_regret": None,
            "top1_exact": None,
            "chose_noop": None,
            "oracle_best_swap": None,
            "predicted_best_swap": None,
        }

    true_before = mesh_squared_error(current, interval_costs)
    predicted_before = risk_mesh_cost(current, predicted_risk, gap_power=gap_power)
    true_gains: list[float] = []
    predicted_gains: list[float] = []
    for removed, added, trial in trials:
        true_gains.append(true_before - mesh_squared_error(trial, interval_costs))
        movement = abs(added - removed) / max(1, frame_count - 1)
        predicted_after = risk_mesh_cost(trial, predicted_risk, gap_power=gap_power)
        predicted_gains.append(predicted_before - predicted_after - float(move_penalty) * movement)

    true_tensor = torch.tensor(true_gains, dtype=torch.float64)
    predicted_tensor = torch.tensor(predicted_gains, dtype=torch.float64)
    spearman = _pearson(_average_rank(predicted_tensor), _average_rank(true_tensor))
    oracle_index = int(torch.argmax(true_tensor).item())
    predicted_index = int(torch.argmax(predicted_tensor).item())
    oracle_gain = max(0.0, true_gains[oracle_index])
    choose_noop = predicted_gains[predicted_index] <= float(min_gain)
    chosen_gain = 0.0 if choose_noop else true_gains[predicted_index]
    gain_recovery = chosen_gain / oracle_gain if oracle_gain > 1.0e-12 else None
    regret = oracle_gain - chosen_gain

    def payload(index: int) -> dict[str, Any]:
        removed, added, trial = trials[index]
        return {
            "removed": removed,
            "added": added,
            "anchors": trial,
            "true_gain": true_gains[index],
            "predicted_gain": predicted_gains[index],
        }

    predicted_payload = (
        {
            "removed": None,
            "added": None,
            "anchors": list(current),
            "true_gain": 0.0,
            "predicted_gain": 0.0,
            "noop": True,
        }
        if choose_noop
        else payload(predicted_index)
    )
    return {
        "trial_count": len(trials),
        "spearman": spearman,
        "gain_recovery": gain_recovery,
        "regret": regret,
        "normalized_regret": regret / max(abs(true_before), 1.0e-12),
        "top1_exact": (choose_noop and oracle_gain <= 1.0e-12)
        or (not choose_noop and predicted_index == oracle_index and oracle_gain > 1.0e-12),
        "chose_noop": choose_noop,
        "oracle_best_swap": None if oracle_gain <= 1.0e-12 else payload(oracle_index),
        "predicted_best_swap": predicted_payload,
    }
