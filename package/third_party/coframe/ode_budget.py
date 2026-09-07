from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch


@dataclass(slots=True)
class ODEPathSignal:
    step_index: int
    direction_change: float | None
    endpoint_change: float | None
    temporal_curvature: float
    normalized_direction: float | None
    normalized_endpoint: float | None
    normalized_curvature: float
    difficulty: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ODEBudgetEvent:
    source_step: int
    target_step: int
    difficulty: float
    raw_budget: float
    assigned_budget: int
    remaining_steps_before: int
    remaining_budget_before: int
    running_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _EMAScale:
    def __init__(self, decay: float, clip: float, eps: float) -> None:
        self.decay = float(decay)
        self.clip = float(clip)
        self.eps = float(eps)
        self.value: float | None = None
        self.updates = 0

    def normalize(self, raw: float) -> float:
        raw = max(0.0, float(raw))
        if self.value is None:
            normalized = 1.0
            self.value = raw
        else:
            normalized = raw / max(self.value, self.eps)
            self.value = self.decay * self.value + (1.0 - self.decay) * raw
        self.updates += 1
        lower = 1.0 / self.clip
        return min(self.clip, max(lower, normalized))

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value, "updates": self.updates}


def flow_clean_endpoint(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Convert a flow-prediction model output to its clean endpoint estimate.

    Diffusers' FlowMatchEuler and flow-mode UniPC schedulers both use
    ``x0 = sample - sigma * velocity``. The integration calls this helper only
    after verifying a flow-prediction scheduler contract.
    """
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    while sigma_tensor.ndim < sample.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    return sample.float() - sigma_tensor * velocity.float()


def ode_direction_change(
    current_velocity: torch.Tensor,
    previous_velocity: torch.Tensor,
    eps: float = 1.0e-8,
) -> float:
    current = current_velocity.detach().float()
    previous = previous_velocity.detach().float()
    numerator = (current * previous).sum()
    denominator = current.square().sum().sqrt() * previous.square().sum().sqrt()
    cosine = numerator / denominator.clamp_min(eps)
    return float((1.0 - cosine.clamp(-1.0, 1.0)).item())


def relative_endpoint_change(
    current_endpoint: torch.Tensor,
    previous_endpoint: torch.Tensor,
    eps: float = 1.0e-8,
) -> float:
    current = current_endpoint.detach().float()
    previous = previous_endpoint.detach().float()
    numerator = (current - previous).square().sum().sqrt()
    denominator = previous.square().sum().sqrt().clamp_min(eps)
    return float((numerator / denominator).item())


def temporal_velocity_curvature(
    velocity: torch.Tensor,
    frame_dim: int = 2,
    eps: float = 1.0e-8,
) -> float:
    """Normalized frame-wise second-difference energy at one denoising step."""
    if velocity.ndim < 3:
        raise ValueError("velocity must contain a frame dimension")
    frame_dim = int(frame_dim) % velocity.ndim
    if velocity.shape[frame_dim] < 3:
        return 0.0
    value = velocity.detach().float().movedim(frame_dim, 0)
    second = value[2:] - 2.0 * value[1:-1] + value[:-2]
    numerator = second.square().sum()
    denominator = value.square().sum().clamp_min(eps)
    return float((numerator / denominator).item())


class ODEPathBudgetController:
    """Online step-level exact-frame allocator with exact total-budget control.

    The current step supplies flow direction, clean-endpoint stability, and
    frame-wise temporal curvature. Their EMA-normalized difficulty controls the
    next sparse step's frame budget. A remaining-budget multiplier and a small
    reachability check guarantee that the configured total budget is met exactly.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        total_sparse_steps: int,
        target_average_budget: float,
        min_budget: int,
        max_budget: int,
        allowed_budgets: Sequence[int] = (),
        signal_ema: float = 0.9,
        normalized_clip: float = 4.0,
        direction_weight: float = 0.5,
        endpoint_weight: float = 0.5,
        difficulty_power: float = 1.0 / 3.0,
        eps: float = 1.0e-8,
    ) -> None:
        self.num_frames = int(num_frames)
        self.total_sparse_steps = int(total_sparse_steps)
        self.target_average_budget = float(target_average_budget)
        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.signal_ema = float(signal_ema)
        self.normalized_clip = float(normalized_clip)
        self.direction_weight = float(direction_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.difficulty_power = float(difficulty_power)
        self.eps = float(eps)

        if self.total_sparse_steps < 1:
            raise ValueError("total_sparse_steps must be positive")
        if not 1 <= self.min_budget <= self.max_budget <= self.num_frames:
            raise ValueError("budget bounds must satisfy 1 <= min <= max <= num_frames")
        if not self.min_budget <= self.target_average_budget <= self.max_budget:
            raise ValueError("target average budget must lie within budget bounds")
        if not 0.0 <= self.signal_ema < 1.0:
            raise ValueError("signal_ema must be in [0,1)")
        if self.normalized_clip < 1.0:
            raise ValueError("normalized_clip must be >= 1")
        if self.direction_weight < 0.0 or self.endpoint_weight < 0.0:
            raise ValueError("difficulty weights must be non-negative")
        if self.direction_weight + self.endpoint_weight <= 0.0:
            raise ValueError("at least one difficulty weight must be positive")
        if self.difficulty_power <= 0.0:
            raise ValueError("difficulty_power must be positive")

        if allowed_budgets:
            allowed = sorted({int(value) for value in allowed_budgets})
            if any(value < self.min_budget or value > self.max_budget for value in allowed):
                raise ValueError("allowed budgets must lie within min/max bounds")
        else:
            allowed = list(range(self.min_budget, self.max_budget + 1))
        if not allowed:
            raise ValueError("at least one budget value is required")
        self.allowed_budgets = tuple(allowed)

        self.target_total_budget = int(round(self.target_average_budget * self.total_sparse_steps))
        self._reachable: list[set[int]] = [{0}]
        for _ in range(self.total_sparse_steps):
            previous = self._reachable[-1]
            self._reachable.append({total + budget for total in previous for budget in self.allowed_budgets})
        if self.target_total_budget not in self._reachable[self.total_sparse_steps]:
            raise ValueError(
                "target total budget is unreachable with the configured allowed budgets; "
                "use the default integer support or choose a compatible target"
            )

        self._direction_scale = _EMAScale(signal_ema, normalized_clip, eps)
        self._endpoint_scale = _EMAScale(signal_ema, normalized_clip, eps)
        self._curvature_scale = _EMAScale(signal_ema, normalized_clip, eps)
        self.previous_velocity: torch.Tensor | None = None
        self.previous_endpoint: torch.Tensor | None = None
        self.assigned_steps = 0
        self.spent_budget = 0
        self.signal_history: list[dict[str, Any]] = []
        self.budget_history: list[dict[str, Any]] = []

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        num_frames: int,
        total_sparse_steps: int,
    ) -> "ODEPathBudgetController":
        target = float(config.ode_target_average_k or config.num_anchors)
        boundary_floor = 2 if config.force_boundaries and num_frames > 1 else 1
        minimum = int(config.ode_min_anchors or max(boundary_floor, round(2.0 * target / 3.0)))
        maximum = int(config.ode_max_anchors or num_frames)
        return cls(
            num_frames=num_frames,
            total_sparse_steps=total_sparse_steps,
            target_average_budget=target,
            min_budget=minimum,
            max_budget=maximum,
            allowed_budgets=config.ode_budget_values,
            signal_ema=config.ode_signal_ema,
            normalized_clip=config.ode_signal_clip,
            direction_weight=config.ode_direction_weight,
            endpoint_weight=config.ode_endpoint_weight,
            difficulty_power=config.ode_difficulty_power,
        )

    def observe(
        self,
        *,
        step_index: int,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> ODEPathSignal:
        endpoint = flow_clean_endpoint(sample, velocity, sigma)
        curvature = temporal_velocity_curvature(velocity)
        normalized_curvature = self._curvature_scale.normalize(curvature)

        direction: float | None = None
        endpoint_change: float | None = None
        normalized_direction: float | None = None
        normalized_endpoint: float | None = None
        difficulty: float | None = None
        if self.previous_velocity is not None and self.previous_endpoint is not None:
            direction = ode_direction_change(velocity, self.previous_velocity, self.eps)
            endpoint_change = relative_endpoint_change(endpoint, self.previous_endpoint, self.eps)
            normalized_direction = self._direction_scale.normalize(direction)
            normalized_endpoint = self._endpoint_scale.normalize(endpoint_change)
            weight_sum = self.direction_weight + self.endpoint_weight
            path_term = (
                self.direction_weight * normalized_direction
                + self.endpoint_weight * normalized_endpoint
            ) / weight_sum
            difficulty = max(self.eps, path_term * normalized_curvature)

        self.previous_velocity = velocity.detach()
        self.previous_endpoint = endpoint.detach()
        record = ODEPathSignal(
            step_index=int(step_index),
            direction_change=direction,
            endpoint_change=endpoint_change,
            temporal_curvature=curvature,
            normalized_direction=normalized_direction,
            normalized_endpoint=normalized_endpoint,
            normalized_curvature=normalized_curvature,
            difficulty=difficulty,
        )
        self.signal_history.append(record.to_dict())
        return record

    def allocate_next(
        self,
        *,
        source_step: int,
        target_step: int,
        difficulty: float | None,
    ) -> ODEBudgetEvent:
        if self.assigned_steps >= self.total_sparse_steps:
            raise RuntimeError("all sparse-step budgets have already been assigned")
        remaining_steps = self.total_sparse_steps - self.assigned_steps
        remaining_budget = self.target_total_budget - self.spent_budget
        remaining_average = remaining_budget / float(remaining_steps)
        running_multiplier = remaining_average / max(self.target_average_budget, self.eps)
        effective_difficulty = 1.0 if difficulty is None else max(self.eps, float(difficulty))
        raw_budget = remaining_average * math.pow(effective_difficulty, self.difficulty_power)

        future_steps = remaining_steps - 1
        candidates = [
            budget
            for budget in self.allowed_budgets
            if remaining_budget - budget in self._reachable[future_steps]
        ]
        if not candidates:
            raise RuntimeError("no budget choice can satisfy the remaining total-budget constraint")
        assigned = min(
            candidates,
            key=lambda value: (abs(float(value) - raw_budget), abs(float(value) - remaining_average), value),
        )
        self.assigned_steps += 1
        self.spent_budget += int(assigned)
        event = ODEBudgetEvent(
            source_step=int(source_step),
            target_step=int(target_step),
            difficulty=effective_difficulty,
            raw_budget=float(raw_budget),
            assigned_budget=int(assigned),
            remaining_steps_before=int(remaining_steps),
            remaining_budget_before=int(remaining_budget),
            running_multiplier=float(running_multiplier),
        )
        self.budget_history.append(event.to_dict())
        return event

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "total_sparse_steps": self.total_sparse_steps,
            "target_average_budget": self.target_average_budget,
            "target_total_budget": self.target_total_budget,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "allowed_budgets": list(self.allowed_budgets),
            "assigned_steps": self.assigned_steps,
            "spent_budget": self.spent_budget,
            "direction_scale": self._direction_scale.state_dict(),
            "endpoint_scale": self._endpoint_scale.state_dict(),
            "curvature_scale": self._curvature_scale.state_dict(),
            "signal_history": list(self.signal_history),
            "budget_history": list(self.budget_history),
        }
