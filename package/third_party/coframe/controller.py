from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .selection import normalized_prior, uniform_select


@dataclass(slots=True)
class MeshRefresh:
    before: list[int]
    after: list[int]
    before_cost: float
    after_cost: float
    gain: float
    removed: int | None = None
    added: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "before_cost": self.before_cost,
            "after_cost": self.after_cost,
            "gain": self.gain,
            "removed": self.removed,
            "added": self.added,
        }


class AdaptiveMeshController:
    """Fixed-budget online split/merge controller for CoFrame.

    The latent-frame count is small (21 for Wan's standard 81-frame output),
    so an exhaustive one-swap search is clearer and more reliable than a
    brittle greedy heuristic. Each refresh removes one interior anchor and adds
    one new frame if doing so reduces the risk-weighted interpolation cost.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        num_anchors: int,
        initial_anchors: list[int] | None = None,
        prior_scores: torch.Tensor | None = None,
        force_boundaries: bool = True,
        min_gap: int = 1,
        risk_ema: float = 0.75,
        prior_weight: float = 0.35,
        risk_floor: float = 1.0e-4,
        gap_power: float = 2.0,
        move_penalty: float = 0.02,
        min_refresh_gain: float = 1.0e-4,
        max_swaps_per_refresh: int = 1,
        defect_clip: float = 10.0,
    ) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if not 1 <= num_anchors <= num_frames:
            raise ValueError("num_anchors must lie in [1, num_frames]")
        self.num_frames = int(num_frames)
        self.num_anchors = int(num_anchors)
        self.force_boundaries = bool(force_boundaries)
        self.min_gap = max(1, int(min_gap))
        self.risk_ema = float(risk_ema)
        self.prior_weight = float(prior_weight)
        self.risk_floor = float(risk_floor)
        self.gap_power = float(gap_power)
        self.move_penalty = float(move_penalty)
        self.min_refresh_gain = float(min_refresh_gain)
        self.max_swaps_per_refresh = int(max_swaps_per_refresh)
        self.defect_clip = float(defect_clip)

        if initial_anchors is None:
            initial_anchors = uniform_select(num_frames, num_anchors, force_boundaries)
        self.anchors = self._normalize_mesh(initial_anchors)
        # Keep the original Rhyme/fixed mesh immutable for matched-budget
        # oracle diagnostics after online refresh has moved current anchors.
        self.initial_anchors = list(self.anchors)

        if prior_scores is None:
            prior_scores = torch.zeros(num_frames, dtype=torch.float32)
        if prior_scores.numel() != num_frames:
            raise ValueError("prior_scores length must equal num_frames")
        self.prior = normalized_prior(prior_scores.detach().cpu()).float()
        self.dynamic_risk = torch.zeros(num_frames, dtype=torch.float32)
        self.observation_count = torch.zeros(num_frames, dtype=torch.float32)
        self.refresh_history: list[MeshRefresh] = []
        # Stage-1d budget state is intentionally separate from self.anchors,
        # whose fixed length still serves the original remeshing controller.
        self.current_budget = int(num_anchors)
        self.budget_history: list[dict[str, Any]] = []

    def set_budget(self, num_anchors: int, *, reset_dynamic_risk: bool = False) -> list[int]:
        """Resize the exact-frame mesh while preserving frame-wise risk evidence.

        ODE-path allocation can change K between denoising steps. When K changes,
        start that step from a uniform boundary-preserving mesh; subsequent block
        groups immediately adapt it using leave-one-out residual defects.
        """
        target = int(num_anchors)
        if not 1 <= target <= self.num_frames:
            raise ValueError("num_anchors must lie in [1, num_frames]")
        if self.force_boundaries and self.num_frames > 1 and target < 2:
            raise ValueError("at least two anchors are required when boundaries are forced")
        if target != self.num_anchors:
            self.num_anchors = target
            self.anchors = uniform_select(self.num_frames, target, self.force_boundaries)
        self.current_budget = target
        if reset_dynamic_risk:
            self.dynamic_risk.zero_()
            self.observation_count.zero_()
        return list(self.anchors)

    @property
    def risk(self) -> torch.Tensor:
        return self.risk_floor + self.prior_weight * self.prior + self.dynamic_risk

    def _normalize_mesh(self, anchors: list[int]) -> list[int]:
        result = sorted({int(index) for index in anchors if 0 <= int(index) < self.num_frames})
        if self.force_boundaries and self.num_frames > 1:
            result = sorted(set(result + [0, self.num_frames - 1]))
        if len(result) != self.num_anchors:
            # Uniform fallback is preferable to silently changing the budget.
            result = uniform_select(self.num_frames, self.num_anchors, self.force_boundaries)
        if not self._valid_mesh(result):
            raise ValueError(f"Invalid anchor mesh: {result}")
        return result

    def _valid_mesh(self, anchors: list[int]) -> bool:
        if len(anchors) != self.num_anchors or anchors != sorted(set(anchors)):
            return False
        if anchors[0] < 0 or anchors[-1] >= self.num_frames:
            return False
        if self.force_boundaries and self.num_frames > 1 and (anchors[0] != 0 or anchors[-1] != self.num_frames - 1):
            return False
        return all(right - left >= self.min_gap for left, right in zip(anchors[:-1], anchors[1:]))

    def project_defects(
        self,
        defects: dict[int, float | torch.Tensor],
        anchors: list[int] | None = None,
    ) -> torch.Tensor:
        """Project sparse validator measurements to a frame-wise risk field."""
        mesh = self.anchors if anchors is None else sorted(anchors)
        observation = torch.zeros(self.num_frames, dtype=torch.float32)
        if len(mesh) < 3 or not defects:
            return observation
        weight = torch.zeros(self.num_frames, dtype=torch.float32)
        slot_by_frame = {frame: slot for slot, frame in enumerate(mesh)}

        for validator, raw_value in defects.items():
            validator = int(validator)
            slot = slot_by_frame.get(validator)
            if slot is None or slot == 0 or slot == len(mesh) - 1:
                continue
            value = float(raw_value.detach().float().item() if isinstance(raw_value, torch.Tensor) else raw_value)
            value = max(0.0, min(self.defect_clip, value))
            left, right = mesh[slot - 1], mesh[slot + 1]
            span = max(1, right - left)
            for frame in range(left, right + 1):
                # Triangular support, with a small floor so the full interval
                # receives evidence instead of only the validator location.
                shape = max(0.1, 1.0 - abs(frame - validator) / float(span))
                observation[frame] += value * shape
                weight[frame] += shape

        mask = weight > 0
        observation[mask] /= weight[mask]
        return observation

    def approximation_risk(
        self,
        anchors: list[int] | None = None,
        base_risk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert curvature-like frame risk into expected interpolation error.

        Exact anchor positions receive zero approximation risk. Interior risk is
        modulated by the linear-interpolation error envelope and interval span.
        """
        mesh = self.anchors if anchors is None else sorted(anchors)
        base = self.risk if base_risk is None else base_risk.detach().float().cpu()
        if base.numel() != self.num_frames:
            raise ValueError("base_risk length must equal num_frames")
        expected = torch.zeros(self.num_frames, dtype=torch.float32)
        for left, right in zip(mesh[:-1], mesh[1:]):
            gap = right - left
            if gap <= 1:
                continue
            positions = torch.arange(left + 1, right, dtype=torch.float32)
            unit = (positions - float(left)) / float(gap)
            envelope = 4.0 * unit * (1.0 - unit)
            expected[left + 1 : right] = (
                base[left + 1 : right] * envelope * (float(gap) ** self.gap_power)
            )
        return expected

    def observe(self, defects: dict[int, float | torch.Tensor], anchors: list[int] | None = None) -> None:
        """Spread validator defects over their neighboring temporal interval."""
        observation = self.project_defects(defects, anchors)
        mask = observation > 0
        self.dynamic_risk.mul_(self.risk_ema)
        self.dynamic_risk[mask] += (1.0 - self.risk_ema) * observation[mask]
        self.observation_count[mask] += 1

    def _interval_cost_table(self) -> list[list[float]]:
        risk = self.risk.tolist()
        table = [[float("inf") for _ in range(self.num_frames)] for _ in range(self.num_frames)]
        for left in range(self.num_frames):
            for right in range(left + 1, self.num_frames):
                gap = right - left
                if gap == 1:
                    table[left][right] = 0.0
                    continue
                weighted_sum = 0.0
                weight_sum = 0.0
                for frame in range(left + 1, right):
                    unit = float(frame - left) / float(gap)
                    envelope = 4.0 * unit * (1.0 - unit)
                    weighted_sum += float(risk[frame]) * envelope
                    weight_sum += envelope
                interval_risk = weighted_sum / max(weight_sum, 1.0e-12)
                table[left][right] = (float(gap) ** self.gap_power) * interval_risk
        return table

    @staticmethod
    def _mesh_cost_from_table(mesh: list[int], table: list[list[float]]) -> float:
        return sum(table[left][right] for left, right in zip(mesh[:-1], mesh[1:]))

    def mesh_cost(self, anchors: list[int] | None = None) -> float:
        mesh = self.anchors if anchors is None else anchors
        return self._mesh_cost_from_table(mesh, self._interval_cost_table())

    def refresh(self) -> list[MeshRefresh]:
        events: list[MeshRefresh] = []
        for _ in range(self.max_swaps_per_refresh):
            event = self._best_single_swap()
            events.append(event)
            self.refresh_history.append(event)
            if event.gain <= self.min_refresh_gain or event.after == event.before:
                break
            self.anchors = event.after
        return events

    def _best_single_swap(self) -> MeshRefresh:
        before = list(self.anchors)
        interval_costs = self._interval_cost_table()
        before_cost = self._mesh_cost_from_table(before, interval_costs)
        boundaries = {0, self.num_frames - 1} if self.force_boundaries and self.num_frames > 1 else set()
        removable = [index for index in before if index not in boundaries]
        candidates = [index for index in range(self.num_frames) if index not in before]

        best_mesh = before
        best_cost = before_cost
        best_removed: int | None = None
        best_added: int | None = None

        for removed in removable:
            base = [index for index in before if index != removed]
            for added in candidates:
                trial = sorted(base + [added])
                if not self._valid_mesh(trial):
                    continue
                movement = abs(added - removed) / max(1, self.num_frames - 1)
                penalized_cost = self._mesh_cost_from_table(trial, interval_costs) + self.move_penalty * movement
                if penalized_cost + 1.0e-12 < best_cost:
                    best_cost = penalized_cost
                    best_mesh = trial
                    best_removed = removed
                    best_added = added

        gain = before_cost - best_cost
        if gain <= self.min_refresh_gain:
            best_mesh = before
            best_cost = before_cost
            best_removed = None
            best_added = None
            gain = 0.0

        return MeshRefresh(
            before=before,
            after=best_mesh,
            before_cost=before_cost,
            after_cost=best_cost,
            gain=gain,
            removed=best_removed,
            added=best_added,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "num_anchors": self.num_anchors,
            "initial_anchors": list(self.initial_anchors),
            "anchors": list(self.anchors),
            "prior": self.prior.tolist(),
            "dynamic_risk": self.dynamic_risk.tolist(),
            "risk": self.risk.tolist(),
            "observation_count": self.observation_count.tolist(),
            "refresh_history": [event.to_dict() for event in self.refresh_history],
            "current_budget": self.current_budget,
            "budget_history": list(self.budget_history),
        }
