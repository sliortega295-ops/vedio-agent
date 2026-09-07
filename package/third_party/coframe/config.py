from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal, Sequence

Method = Literal["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode", "sparse_evolution"]
RefreshSignal = Literal["defect", "none", "gap_only", "shuffled"]
BudgetPolicy = Literal["none", "step_block", "mean_defect", "max_defect"]
KVMode = Literal["anchor_only", "full_kv"]
InterpolationTarget = Literal["delta", "state"]
DefectTarget = Literal["delta", "state"]


@dataclass(slots=True)
class CoFrameConfig:
    """Configuration for CoFrame's Wan2.1 validation path.

    The defaults target Wan2.1-T2V-1.3B-Diffusers with 21 latent frames
    (81 decoded frames), 30 transformer blocks, and a 50-step sampler.
    """

    method: Method = "coframe"
    warmup_steps: int = 5
    num_anchors: int = 9
    force_boundaries: bool = True
    min_anchor_gap: int = 1

    # Only middle blocks are sparse by default. Dense boundary blocks act as
    # error-reset/safety layers and keep the first prototype conservative.
    sparse_block_start: int = 3
    sparse_block_end: int = 27  # exclusive
    block_group_size: int = 3

    kv_mode: KVMode = "anchor_only"
    interpolation_target: InterpolationTarget = "delta"
    defect_target: DefectTarget = "delta"

    # CoFrame source-attribution ablations. "none" freezes the initial
    # Rhyme mesh; "gap_only" remeshes with a uniform risk field;
    # "shuffled" preserves defect magnitudes but breaks frame alignment.
    refresh_signal: RefreshSignal = "defect"
    shuffle_defect_seed: int = 20260810

    # FIS-DiT-style interleaved baseline. stride=0 derives ceil(F/K).
    # A dense tail is optional and explicitly counted in latency.
    fis_anchor_stride: int = 0
    fis_dense_tail_steps: int = 0

    # RhymeFlow-style clean-latent prior.
    rhyme_similarity_threshold: float = 0.98
    rhyme_prior_weight: float = 0.35

    # Online risk and mesh refresh.
    risk_ema: float = 0.75
    risk_floor: float = 1.0e-4
    interval_gap_power: float = 2.0
    move_penalty: float = 0.02
    min_refresh_gain: float = 1.0e-4
    max_swaps_per_refresh: int = 1
    refresh_every_groups: int = 1
    defect_clip: float = 10.0

    # Cheap channel sketch for defect measurement. 0 means exact channel RMS.
    sketch_dim: int = 64
    sketch_seed: int = 2026

    # Optional causal diagnostic: recompute a dense block from the same input
    # and correlate CoFrame risk with actual sparse-block error.
    oracle_probe_steps: Sequence[int] = field(default_factory=tuple)
    oracle_probe_blocks: Sequence[int] = field(default_factory=tuple)
    # After a probed block, replay dense dynamics for these many blocks from
    # both dense and sparse states to measure error amplification/correction.
    oracle_probe_horizons: Sequence[int] = field(default_factory=lambda: (1, 3))
    oracle_metric_chunk_size: int = 65_536
    probe_counterfactual_methods: Sequence[str] = field(default_factory=lambda: ("rhyme", "fis", "fixed"))

    # Stage-1c signal screening. These diagnostics never change the deployed
    # mesh; they only rank hypothetical one-swap actions against dense truth.
    probe_curvature_signals: bool = False
    curvature_shuffle_seed: int = 20260811

    # Stage-1d: causal exact-frame budget allocation. Frame placement is
    # deliberately uniform so this experiment isolates "how much to compute"
    # from the rejected defect-localization/remeshing mechanism.
    adaptive_k_policy: BudgetPolicy = "none"
    adaptive_k_values: Sequence[int] = field(default_factory=lambda: (6, 9, 12, 21))
    adaptive_k_thresholds: Sequence[float] = field(default_factory=tuple)
    adaptive_k_schedule: dict[str, int] = field(default_factory=dict)
    adaptive_k_carry_across_steps: bool = True

    # SVA paper path: current-group residual defects budget the next group;
    # deterministic coverage-interleaving chooses positions, not the defects.
    evolution_dense_tail_steps: int = 1

    # Current proposed path: step-level ODE/path-aware budget plus deterministic
    # coverage-aware interleaved placement. Empty ode_budget_values means every
    # integer budget in [ode_min_anchors, ode_max_anchors] is supported.
    ode_target_average_k: float = 0.0  # 0 -> num_anchors
    ode_min_anchors: int = 0  # 0 -> round(2/3 * target), respecting boundaries
    ode_max_anchors: int = 0  # 0 -> all latent frames
    ode_budget_values: Sequence[int] = field(default_factory=tuple)
    ode_signal_ema: float = 0.9
    ode_signal_clip: float = 4.0
    ode_direction_weight: float = 0.5
    ode_endpoint_weight: float = 0.5
    ode_difficulty_power: float = 1.0 / 3.0
    ode_anchor_stride: int = 0
    ode_interleave_penalty: float = 2.0
    ode_interleave_across_steps: bool = True

    trace_path: str | None = None
    strict_diffusers_version: bool = True

    def validate(self, *, num_blocks: int | None = None, num_frames: int | None = None) -> None:
        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode", "sparse_evolution"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.refresh_signal not in {"defect", "none", "gap_only", "shuffled"}:
            raise ValueError(f"Unsupported refresh_signal: {self.refresh_signal}")
        if self.adaptive_k_policy not in {"none", "step_block", "mean_defect", "max_defect"}:
            raise ValueError(f"Unsupported adaptive_k_policy: {self.adaptive_k_policy}")
        if self.method == "adaptive_k" and self.adaptive_k_policy == "none":
            raise ValueError("method=adaptive_k requires an adaptive_k_policy")
        budget_values = [int(value) for value in self.adaptive_k_values]
        if not budget_values or budget_values != sorted(set(budget_values)):
            raise ValueError("adaptive_k_values must be strictly increasing")
        if any(value < 1 for value in budget_values):
            raise ValueError("adaptive_k_values must be positive")
        thresholds = [float(value) for value in self.adaptive_k_thresholds]
        if thresholds != sorted(thresholds):
            raise ValueError("adaptive_k_thresholds must be sorted")
        if self.adaptive_k_policy in {"mean_defect", "max_defect"} and len(thresholds) != len(budget_values) - 1:
            raise ValueError("adaptive defect policies require len(values)-1 thresholds")
        if self.method == "sparse_evolution":
            if self.adaptive_k_policy not in {"mean_defect", "max_defect"}:
                raise ValueError("sparse_evolution requires a mean_defect or max_defect budget policy")
            if self.interpolation_target != "delta" or self.defect_target != "delta":
                raise ValueError("sparse_evolution requires residual (delta) interpolation and defects")
            if self.kv_mode not in {"full_kv", "anchor_only"}:
                raise ValueError("unsupported sparse_evolution kv_mode")
            if self.min_anchor_gap != 1:
                raise ValueError("sparse_evolution coverage placement requires min_anchor_gap=1")
            if not self.force_boundaries or any(value < 3 for value in budget_values):
                raise ValueError("sparse_evolution requires boundary anchors and K>=3 for interior LOO validation")
            if any(type(value) is not int for value in self.adaptive_k_values):
                raise ValueError("sparse_evolution budget values must be integers")
            if self.num_anchors not in budget_values:
                raise ValueError("sparse_evolution num_anchors must belong to adaptive_k_values")
            if any(not math.isfinite(value) or value < 0 for value in thresholds):
                raise ValueError("residual RMS thresholds must be finite and non-negative")
            if self.warmup_steps < 0 or self.evolution_dense_tail_steps < 0:
                raise ValueError("sparse_evolution warmup and dense tail must be non-negative")
            if self.sparse_block_start >= self.sparse_block_end:
                raise ValueError("sparse_evolution requires a non-empty sparse block range")
        if self.adaptive_k_policy == "step_block":
            invalid_schedule = [value for value in self.adaptive_k_schedule.values() if int(value) not in budget_values]
            if invalid_schedule:
                raise ValueError("adaptive_k_schedule contains a budget outside adaptive_k_values")
        if not 0.0 <= self.ode_signal_ema < 1.0:
            raise ValueError("ode_signal_ema must be in [0,1)")
        if self.ode_signal_clip < 1.0:
            raise ValueError("ode_signal_clip must be >= 1")
        if self.ode_direction_weight < 0.0 or self.ode_endpoint_weight < 0.0:
            raise ValueError("ODE difficulty weights must be non-negative")
        if self.ode_direction_weight + self.ode_endpoint_weight <= 0.0:
            raise ValueError("at least one ODE difficulty weight must be positive")
        if self.ode_difficulty_power <= 0.0:
            raise ValueError("ode_difficulty_power must be positive")
        if self.ode_min_anchors < 0 or self.ode_max_anchors < 0:
            raise ValueError("ODE anchor bounds must be non-negative; zero selects the automatic bound")
        if self.ode_anchor_stride < 0:
            raise ValueError("ode_anchor_stride must be >= 0")
        if self.ode_interleave_penalty < 0.0:
            raise ValueError("ode_interleave_penalty must be non-negative")
        ode_values = [int(value) for value in self.ode_budget_values]
        if ode_values and ode_values != sorted(set(ode_values)):
            raise ValueError("ode_budget_values must be strictly increasing when provided")
        if any(value < 1 for value in ode_values):
            raise ValueError("ode_budget_values must be positive")
        if self.fis_anchor_stride < 0:
            raise ValueError("fis_anchor_stride must be >= 0")
        if self.fis_dense_tail_steps < 0:
            raise ValueError("fis_dense_tail_steps must be >= 0")
        invalid_counterfactuals = set(self.probe_counterfactual_methods) - {"rhyme", "fis", "fixed"}
        if invalid_counterfactuals:
            raise ValueError(f"Unsupported probe counterfactuals: {sorted(invalid_counterfactuals)}")
        if self.num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if self.force_boundaries and self.num_anchors < 2 and (num_frames is None or num_frames > 1):
            raise ValueError("At least two anchors are required when force_boundaries=True")
        if num_frames is not None and self.num_anchors > num_frames:
            raise ValueError(f"num_anchors={self.num_anchors} exceeds latent frames={num_frames}")
        if self.method in {"adaptive_k", "sparse_evolution"} and num_frames is not None and any(
            int(value) > num_frames for value in self.adaptive_k_values
        ):
            raise ValueError("adaptive_k_values exceed the latent frame count")
        if (
            self.method == "adaptive_k"
            and self.force_boundaries
            and any(int(value) < 2 for value in self.adaptive_k_values)
            and (num_frames is None or num_frames > 1)
        ):
            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")
        if self.method == "coframe_ode" and num_frames is not None:
            target = float(self.ode_target_average_k or self.num_anchors)
            boundary_floor = 2 if self.force_boundaries and num_frames > 1 else 1
            minimum = int(self.ode_min_anchors or max(boundary_floor, round(2.0 * target / 3.0)))
            maximum = int(self.ode_max_anchors or num_frames)
            if not boundary_floor <= minimum <= target <= maximum <= num_frames:
                raise ValueError("resolved ODE budgets must satisfy boundary_floor <= min <= target <= max <= frames")
            if self.ode_budget_values and any(
                int(value) < minimum or int(value) > maximum for value in self.ode_budget_values
            ):
                raise ValueError("ode_budget_values must lie inside the resolved ODE budget range")
        if self.min_anchor_gap < 1:
            raise ValueError("min_anchor_gap must be >= 1")
        if self.block_group_size < 1:
            raise ValueError("block_group_size must be >= 1")
        if not 0.0 <= self.risk_ema < 1.0:
            raise ValueError("risk_ema must be in [0, 1)")
        if self.interval_gap_power <= 0:
            raise ValueError("interval_gap_power must be positive")
        if self.max_swaps_per_refresh < 0:
            raise ValueError("max_swaps_per_refresh must be non-negative")
        if self.refresh_every_groups < 1:
            raise ValueError("refresh_every_groups must be >= 1")
        if self.sketch_dim < 0:
            raise ValueError("sketch_dim must be >= 0")
        if any(int(horizon) <= 0 for horizon in self.oracle_probe_horizons):
            raise ValueError("oracle_probe_horizons must contain positive integers")
        if self.oracle_metric_chunk_size < 1:
            raise ValueError("oracle_metric_chunk_size must be positive")
        if num_blocks is not None:
            if not 0 <= self.sparse_block_start <= num_blocks:
                raise ValueError("sparse_block_start is out of range")
            if not self.sparse_block_start <= self.sparse_block_end <= num_blocks:
                raise ValueError("sparse_block_end is out of range")

    def is_sparse_block(self, block_index: int) -> bool:
        return self.method != "dense" and self.sparse_block_start <= block_index < self.sparse_block_end

    def should_probe(self, step_index: int, block_index: int) -> bool:
        return step_index in set(self.oracle_probe_steps) and block_index in set(self.oracle_probe_blocks)

    def is_sparse_step(self, step_index: int, total_steps: int) -> bool:
        if self.method == "dense":
            return False
        if self.method == "fis" and self.fis_dense_tail_steps > 0:
            return step_index < max(0, total_steps - self.fis_dense_tail_steps)
        if self.method == "sparse_evolution":
            return step_index < max(0, total_steps - self.evolution_dense_tail_steps)
        return True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["oracle_probe_steps"] = list(self.oracle_probe_steps)
        result["oracle_probe_blocks"] = list(self.oracle_probe_blocks)
        result["oracle_probe_horizons"] = list(self.oracle_probe_horizons)
        result["probe_counterfactual_methods"] = list(self.probe_counterfactual_methods)
        result["adaptive_k_values"] = list(self.adaptive_k_values)
        result["adaptive_k_thresholds"] = list(self.adaptive_k_thresholds)
        result["adaptive_k_schedule"] = dict(self.adaptive_k_schedule)
        result["ode_budget_values"] = list(self.ode_budget_values)
        return result
