import torch

from coframe.selection import rhyme_select, transition_scores, uniform_select


def test_uniform_select_keeps_budget_and_boundaries():
    anchors = uniform_select(num_frames=21, num_anchors=9, force_boundaries=True)
    assert len(anchors) == 9
    assert anchors[0] == 0
    assert anchors[-1] == 20
    assert anchors == sorted(set(anchors))


def test_rhyme_selector_captures_large_semantic_transition():
    representations = torch.zeros(10, 4)
    representations[:5, 0] = 1.0
    representations[5:, 1] = 1.0

    anchors = rhyme_select(
        representations,
        num_anchors=4,
        similarity_threshold=0.95,
        force_boundaries=True,
    )

    assert 5 in anchors
    assert anchors[0] == 0
    assert anchors[-1] == 9
    assert len(anchors) == 4


def test_transition_scores_peak_at_change_point():
    representations = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    scores = transition_scores(representations)
    assert int(scores.argmax()) == 2
    assert scores[2] > scores[1]
    assert scores[2] > scores[3]
