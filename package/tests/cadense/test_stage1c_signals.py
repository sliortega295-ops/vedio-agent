import torch

from coframe.wan.sparse_forward import (
    _normalize_frame_signal,
    _shuffle_frame_signal,
    _temporal_curvature_scores,
)


def test_temporal_curvature_is_zero_for_linear_trajectory():
    base = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    scores = _temporal_curvature_scores(base, projection=None)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)


def test_temporal_curvature_detects_local_kink_and_shuffle_preserves_values():
    values = torch.arange(7, dtype=torch.float32)
    values[3] += 4.0
    frames = values.view(1, 7, 1, 1)
    scores = _normalize_frame_signal(_temporal_curvature_scores(frames, projection=None))
    assert int(torch.argmax(scores).item()) == 3
    shuffled = _shuffle_frame_signal(scores, seed=123)
    assert torch.allclose(torch.sort(scores[1:-1]).values, torch.sort(shuffled[1:-1]).values)
    assert shuffled[0] == 0 and shuffled[-1] == 0
