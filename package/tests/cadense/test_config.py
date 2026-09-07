import pytest

from coframe.config import CoFrameConfig


def test_default_config_matches_wan_1_3b_block_count():
    config = CoFrameConfig()
    config.validate(num_blocks=30, num_frames=21)
    assert config.is_sparse_block(3)
    assert config.is_sparse_block(26)
    assert not config.is_sparse_block(2)
    assert not config.is_sparse_block(27)


def test_anchor_budget_validation():
    with pytest.raises(ValueError):
        CoFrameConfig(num_anchors=22).validate(num_blocks=30, num_frames=21)
