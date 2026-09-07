import pytest
import torch

from coframe.metrics import (
    frame_gram_matrix,
    headroom_recovery,
    interpolation_interval_costs,
    mesh_reconstruction_metrics,
    one_swap_diagnostics,
    optimal_piecewise_linear_mesh,
    per_frame_global_normalized_rms,
    temporal_gradient_relative_l2,
    worst_fraction_mean,
)


def test_exact_oracle_recovers_best_three_anchor_mesh() -> None:
    values = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0]).view(1, 5, 1)
    gram = frame_gram_matrix(values)
    costs = interpolation_interval_costs(gram)
    oracle = optimal_piecewise_linear_mesh(
        costs,
        num_anchors=3,
        total_energy=float(torch.diagonal(gram).sum().item()),
    )
    current = mesh_reconstruction_metrics(values, [0, 2, 4])
    rhyme = mesh_reconstruction_metrics(values, [0, 1, 4])

    assert oracle.anchors == [0, 2, 4]
    assert current["normalized_mse"] == pytest.approx(oracle.normalized_mse, abs=1.0e-7)
    assert current["normalized_mse"] < rhyme["normalized_mse"]
    assert headroom_recovery(
        baseline_error=rhyme["normalized_mse"],
        method_error=current["normalized_mse"],
        oracle_error=oracle.normalized_mse,
    ) == pytest.approx(1.0, abs=1.0e-6)


def test_oracle_is_never_worse_than_supplied_meshes() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(2, 9, 3, generator=generator)
    gram = frame_gram_matrix(values)
    costs = interpolation_interval_costs(gram)
    oracle = optimal_piecewise_linear_mesh(
        costs,
        num_anchors=4,
        total_energy=float(torch.diagonal(gram).sum().item()),
    )
    current = mesh_reconstruction_metrics(values, [0, 2, 5, 8])
    rhyme = mesh_reconstruction_metrics(values, [0, 3, 6, 8])

    assert oracle.normalized_mse <= current["normalized_mse"] + 1.0e-7
    assert oracle.normalized_mse <= rhyme["normalized_mse"] + 1.0e-7


def test_headroom_recovery_has_interpretable_endpoints() -> None:
    assert headroom_recovery(baseline_error=0.4, method_error=0.4, oracle_error=0.1) == pytest.approx(0.0)
    assert headroom_recovery(baseline_error=0.4, method_error=0.1, oracle_error=0.1) == pytest.approx(1.0)
    assert headroom_recovery(baseline_error=0.4, method_error=0.5, oracle_error=0.1) < 0.0
    assert headroom_recovery(baseline_error=0.1, method_error=0.1, oracle_error=0.1) is None


def test_one_swap_diagnostic_scores_the_actual_controller_action() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0]).view(1, 7, 1)
    costs = interpolation_interval_costs(frame_gram_matrix(values))
    diagnostics = one_swap_diagnostics(
        anchors=[0, 2, 4, 6],
        interval_costs=costs,
        predicted_risk=torch.tensor([0.0, 0.0, 1.0, 2.0, 10.0, 2.0, 0.0]),
        gap_power=2.0,
        move_penalty=0.0,
        min_gain=0.0,
        min_gap=1,
        force_boundaries=True,
    )

    assert diagnostics["top1_exact"] is True
    assert diagnostics["gain_recovery"] == pytest.approx(1.0)
    assert diagnostics["regret"] == pytest.approx(0.0)
    assert diagnostics["predicted_best_swap"]["anchors"] == [0, 3, 4, 6]


def test_worst_fraction_mean_is_stable_for_small_frame_counts() -> None:
    values = torch.arange(1, 22, dtype=torch.float32)
    # ceil(0.2 * 21) = 5, so CVaR-20 averages errors 17..21.
    assert worst_fraction_mean(values, 0.2) == pytest.approx(19.0)


def test_temporal_gradient_error_ignores_global_offset() -> None:
    reference = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1)
    approximation = reference + 3.0
    assert temporal_gradient_relative_l2(reference, approximation, frame_dim=2) == pytest.approx(0.0)


def test_global_frame_normalization_does_not_explode_on_zero_energy_frame() -> None:
    reference = torch.tensor([0.0, 1.0, 2.0]).view(1, 3, 1)
    approximation = torch.tensor([1.0, 1.0, 2.0]).view(1, 3, 1)
    errors = per_frame_global_normalized_rms(reference, approximation)

    assert torch.isfinite(errors).all()
    assert errors[0] > 0
    assert float(errors[1].item()) == pytest.approx(0.0)
