import pytest
import torch

from coframe.ode_budget import (
    ODEPathBudgetController,
    flow_clean_endpoint,
    ode_direction_change,
    relative_endpoint_change,
    temporal_velocity_curvature,
)
from coframe.selection import coverage_interleaved_select
from coframe.wan.pipeline import _require_flow_prediction_scheduler


def test_flow_clean_endpoint_matches_flow_parameterization():
    sample = torch.tensor([[[[[3.0]]]]])
    velocity = torch.tensor([[[[[2.0]]]]])
    endpoint = flow_clean_endpoint(sample, velocity, 0.25)
    assert endpoint.item() == pytest.approx(2.5)


def test_direction_and_endpoint_changes_have_expected_ordering():
    right = torch.tensor([1.0, 0.0])
    same = torch.tensor([2.0, 0.0])
    up = torch.tensor([0.0, 1.0])
    assert ode_direction_change(right, same) == pytest.approx(0.0, abs=1.0e-7)
    assert ode_direction_change(up, right) == pytest.approx(1.0, abs=1.0e-7)
    assert relative_endpoint_change(torch.tensor([2.0]), torch.tensor([1.0])) == pytest.approx(1.0)


def test_temporal_curvature_is_zero_for_linear_frames_and_positive_for_bend():
    linear = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1)
    bent = linear.clone()
    bent[:, :, 2] += 3.0
    assert temporal_velocity_curvature(linear) == pytest.approx(0.0, abs=1.0e-8)
    assert temporal_velocity_curvature(bent) > 0.0


def test_budget_controller_conserves_total_compute_exactly():
    controller = ODEPathBudgetController(
        num_frames=12,
        total_sparse_steps=5,
        target_average_budget=6.0,
        min_budget=3,
        max_budget=10,
    )
    difficulties = [8.0, 0.25, 2.0, 0.5, 1.0]
    budgets = [
        controller.allocate_next(source_step=i, target_step=i + 1, difficulty=value).assigned_budget
        for i, value in enumerate(difficulties)
    ]
    assert sum(budgets) == 30
    assert controller.spent_budget == controller.target_total_budget
    assert budgets[0] > budgets[1]


def test_unreachable_discrete_codebook_fails_closed():
    with pytest.raises(ValueError, match="unreachable"):
        ODEPathBudgetController(
            num_frames=12,
            total_sparse_steps=3,
            target_average_budget=5.0,
            min_budget=4,
            max_budget=8,
            allowed_budgets=(4, 8),
        )


def test_coverage_interleaving_keeps_budget_boundaries_and_changes_phase():
    meshes = [coverage_interleaved_select(21, 9, phase) for phase in range(3)]
    assert all(len(mesh) == 9 for mesh in meshes)
    assert all(mesh[0] == 0 and mesh[-1] == 20 for mesh in meshes)
    assert len({tuple(mesh) for mesh in meshes}) == 3
    assert all(max(right - left for left, right in zip(mesh[:-1], mesh[1:])) <= 3 for mesh in meshes)


def test_flow_scheduler_contract_fails_closed():
    class Scheduler:
        pass

    flow = Scheduler()
    flow.config = type("Config", (), {"prediction_type": "flow_prediction", "use_flow_sigmas": True})()
    _require_flow_prediction_scheduler(flow)

    epsilon = Scheduler()
    epsilon.config = type("Config", (), {"prediction_type": "epsilon", "use_flow_sigmas": False})()
    with pytest.raises(RuntimeError, match="flow-prediction"):
        _require_flow_prediction_scheduler(epsilon)
