import torch

from coframe.controller import AdaptiveMeshController


def make_controller(**overrides):
    kwargs = dict(
        num_frames=13,
        num_anchors=4,
        initial_anchors=[0, 4, 8, 12],
        prior_scores=torch.zeros(13),
        force_boundaries=True,
        min_gap=1,
        risk_ema=0.0,
        prior_weight=0.0,
        risk_floor=1.0e-4,
        gap_power=2.0,
        move_penalty=0.0,
        min_refresh_gain=0.0,
        max_swaps_per_refresh=1,
    )
    kwargs.update(overrides)
    return AdaptiveMeshController(**kwargs)


def test_controller_moves_anchor_toward_high_risk_region():
    controller = make_controller()
    controller.dynamic_risk[5:8] = torch.tensor([1.0, 4.0, 1.0])

    events = controller.refresh()

    assert events[0].gain > 0
    assert 6 in controller.anchors
    assert controller.anchors != [0, 4, 8, 12]
    assert controller.anchors[0] == 0
    assert controller.anchors[-1] == 12
    assert len(controller.anchors) == 4


def test_observation_spreads_defect_to_neighbor_interval():
    controller = make_controller(risk_ema=0.5)
    controller.observe({4: 2.0}, anchors=[0, 4, 8, 12])

    assert controller.dynamic_risk[4] > 0
    assert controller.dynamic_risk[2] > 0
    assert controller.dynamic_risk[7] > 0
    assert controller.dynamic_risk[10] == 0


def test_uniform_mesh_is_stable_under_uniform_risk():
    controller = make_controller(risk_floor=1.0)
    event = controller.refresh()[0]
    assert event.after == [0, 4, 8, 12]
    assert event.gain == 0.0


def test_approximation_risk_is_zero_on_exact_anchors():
    controller = make_controller()
    controller.dynamic_risk[:] = 1.0
    expected = controller.approximation_risk([0, 4, 8, 12])
    assert torch.all(expected[torch.tensor([0, 4, 8, 12])] == 0)
    assert torch.all(expected[torch.tensor([1, 2, 3, 5, 6, 7, 9, 10, 11])] > 0)


def test_set_budget_resizes_mesh_and_preserves_risk():
    controller = make_controller()
    controller.dynamic_risk[6] = 3.0
    anchors = controller.set_budget(6)
    assert len(anchors) == 6
    assert anchors[0] == 0 and anchors[-1] == 12
    assert controller.current_budget == 6
    assert controller.num_anchors == 6
    assert controller.dynamic_risk[6] == 3.0
