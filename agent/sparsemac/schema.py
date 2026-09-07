from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MetricSpec:
    name: str
    weight: float
    target: str
    rationale: str


@dataclass
class Plan:
    original_prompt: str
    prompt_variants: list[str]
    vqa: list[str]
    metrics: list[MetricSpec]
    negative_prompt: str = ""
    notes: str = ""
    semantic_plan: dict[str, Any] = field(default_factory=dict)
    expected_keyframes: list[dict[str, Any]] = field(default_factory=list)
    vqa_groups: dict[str, list[str]] = field(default_factory=dict)
    vqa_scores: dict[str, float] = field(default_factory=dict)
    method_id: str = "legacy"
    method_metadata: dict[str, Any] = field(default_factory=dict)
    primary_skill: str = ""
    must_preserve: list[str] = field(default_factory=list)
    semantic_criteria: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # primary_skill is retained only for backward-compatible legacy plans.
        # New unified planners omit it entirely instead of serializing an empty
        # vestige of the old Skill Router.
        if not self.primary_skill:
            data.pop("primary_skill", None)
        return data


@dataclass
class Candidate:
    candidate_id: str
    round_index: int
    prompt: str
    seed: int
    sparse_video: str
    storyboard: str
    warmup_latent_path: str | None = None
    status: str = "pending"
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evaluation:
    candidate_id: str
    score: float
    passed: bool
    issues: list[str]
    strengths: list[str]
    metric_scores: dict[str, float]
    vqa_results: dict[str, bool | str | dict[str, Any]]
    failure_reports: list[dict[str, Any]] = field(default_factory=list)
    rewrite_hints: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
