import torch

from coframe.interpolation import (
    leave_one_out_defects,
    piecewise_linear_interpolate,
    reconstruct_sparse_block,
)


def test_piecewise_linear_interpolation_is_exact_for_linear_signal():
    frames = 7
    anchors = [0, 3, 6]
    positions = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1)
    full = 2.5 * positions - 3.0
    anchor_values = full[:, anchors]

    reconstructed = piecewise_linear_interpolate(anchor_values, anchors, frames)
    torch.testing.assert_close(reconstructed, full)


def test_delta_reconstruction_preserves_arbitrary_input_state():
    generator = torch.Generator().manual_seed(7)
    input_frames = torch.randn(1, 7, 2, 3, generator=generator)
    positions = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    delta = positions * 0.25 - 0.5
    anchors = [0, 3, 6]
    exact_anchor_outputs = input_frames[:, anchors] + delta[:, anchors]

    reconstructed, anchor_delta = reconstruct_sparse_block(
        input_frames,
        exact_anchor_outputs,
        anchors,
        target="delta",
    )

    torch.testing.assert_close(anchor_delta, delta[:, anchors].expand_as(anchor_delta))
    torch.testing.assert_close(reconstructed, input_frames + delta)


def test_leave_one_out_defect_distinguishes_linear_and_curved_values():
    anchors = [0, 2, 4]
    x = torch.tensor(anchors, dtype=torch.float32).view(1, 3, 1, 1)
    linear = 3.0 * x + 1.0
    quadratic = x.square() + 1.0

    linear_defect = leave_one_out_defects(linear, anchors)[2]
    curved_defect = leave_one_out_defects(quadratic, anchors)[2]

    assert float(linear_defect) < 1.0e-6
    assert float(curved_defect) > 0.1
