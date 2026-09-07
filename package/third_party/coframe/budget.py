from __future__ import annotations

from bisect import bisect_right
from statistics import mean
from typing import Iterable, Mapping, Sequence


def select_budget(risk: float, thresholds: Sequence[float], values: Sequence[int]) -> int:
    """Map a scalar causal risk to one exact-frame budget.

    ``thresholds`` must have one fewer entry than ``values`` and be sorted.
    The mapping is deterministic and monotonic: larger risk never receives a
    smaller exact-frame budget.
    """
    budgets = [int(value) for value in values]
    cuts = [float(value) for value in thresholds]
    if not budgets:
        raise ValueError("adaptive budget values must be non-empty")
    if len(cuts) != len(budgets) - 1:
        raise ValueError("adaptive budget thresholds must have len(values)-1 entries")
    if budgets != sorted(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("adaptive budget values must be strictly increasing")
    if cuts != sorted(cuts):
        raise ValueError("adaptive budget thresholds must be sorted")
    return budgets[bisect_right(cuts, float(risk))]


def schedule_key(step_index: int, group_index: int) -> str:
    return f"{int(step_index)}:{int(group_index)}"


def defect_stat(values: Iterable[float], statistic: str) -> float | None:
    samples = [float(value) for value in values]
    if not samples:
        return None
    if statistic == "mean":
        return float(mean(samples))
    if statistic == "max":
        return float(max(samples))
    raise ValueError(f"unsupported defect statistic: {statistic}")


def lookup_scheduled_budget(
    schedule: Mapping[str, int],
    *,
    step_index: int,
    group_index: int,
    fallback: int,
) -> int:
    return int(schedule.get(schedule_key(step_index, group_index), fallback))
