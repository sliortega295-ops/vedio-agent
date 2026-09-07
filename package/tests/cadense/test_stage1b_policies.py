import torch

from coframe.config import CoFrameConfig
from coframe.controller import AdaptiveMeshController
from coframe.selection import fis_interleaved_select


def test_fis_interleaved_is_exact_budget_and_rotates():
    meshes = [fis_interleaved_select(21, 9, block, 3) for block in (3, 4, 5)]
    assert all(len(mesh) == 9 for mesh in meshes)
    assert all(mesh[0] == 0 and mesh[-1] == 20 for mesh in meshes)
    assert len({tuple(mesh) for mesh in meshes}) == 3
    covered = set().union(*map(set, meshes))
    assert set(range(21)).issubset(covered)


def test_refresh_signal_validation_and_fis_step_gate():
    config = CoFrameConfig(method="fis", fis_dense_tail_steps=2)
    config.validate(num_blocks=30, num_frames=21)
    assert config.is_sparse_step(47, 50)
    assert not config.is_sparse_step(48, 50)
    for signal in ("defect", "none", "gap_only", "shuffled"):
        CoFrameConfig(refresh_signal=signal).validate(num_blocks=30, num_frames=21)


def test_gap_only_uniform_risk_can_regularize_clustered_mesh():
    controller = AdaptiveMeshController(
        num_frames=21,
        num_anchors=9,
        initial_anchors=[0, 1, 2, 3, 4, 5, 6, 7, 20],
        prior_scores=torch.zeros(21),
        prior_weight=0.0,
        risk_floor=1.0,
        move_penalty=0.02,
        max_swaps_per_refresh=1,
    )
    before = list(controller.anchors)
    event = controller.refresh()[0]
    assert event.gain > 0
    assert controller.anchors != before
