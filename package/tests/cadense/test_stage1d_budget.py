import pytest

from coframe.budget import defect_stat, lookup_scheduled_budget, select_budget
from coframe.config import CoFrameConfig


def test_budget_mapping_is_monotonic():
    thresholds = (0.2, 0.5, 0.9)
    values = (6, 9, 12, 21)
    assert [select_budget(x, thresholds, values) for x in (0.1, 0.3, 0.7, 1.2)] == [6, 9, 12, 21]


def test_defect_statistics_and_schedule_lookup():
    assert defect_stat([1.0, 2.0, 3.0], "mean") == 2.0
    assert defect_stat([1.0, 2.0, 3.0], "max") == 3.0
    assert lookup_scheduled_budget({"5:2": 12}, step_index=5, group_index=2, fallback=9) == 12
    assert lookup_scheduled_budget({}, step_index=5, group_index=2, fallback=9) == 9


def test_adaptive_k_config_requires_resolved_policy():
    with pytest.raises(ValueError):
        CoFrameConfig(method="adaptive_k").validate(num_frames=21, num_blocks=30)
    config = CoFrameConfig(
        method="adaptive_k",
        adaptive_k_policy="mean_defect",
        adaptive_k_thresholds=(0.1, 0.2, 0.3),
    )
    config.validate(num_frames=21, num_blocks=30)


def test_step_block_schedule_budget_validation():
    config = CoFrameConfig(
        method="adaptive_k",
        adaptive_k_policy="step_block",
        adaptive_k_schedule={"5:0": 12, "5:1": 9},
    )
    config.validate(num_frames=21, num_blocks=30)
