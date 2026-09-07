"""Real Diffusers operators/RoPE, randomly initialized CPU model, no downloads."""
from dataclasses import replace

import pytest
import torch
from diffusers import WanTransformer3DModel

from coframe.config import CoFrameConfig
from coframe.controller import AdaptiveMeshController
from coframe.interpolation import reconstruct_sparse_block, tokens_to_frames
from coframe.wan.sparse_forward import FrameGeometry, _sparse_block_forward, coframe_transformer_forward


@pytest.mark.parametrize("kv_mode", ["full_kv", "anchor_only"])
@torch.inference_mode()
def test_real_wan_full_k_matches_dense_and_sparse_retains_full_state(kv_mode):
    torch.manual_seed(17)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=2, out_channels=2, text_dim=8, freq_dim=8,
        ffn_dim=32, num_layers=4, rope_max_seq_len=32).eval()
    hidden = torch.randn(1, 2, 5, 4, 4)
    timestep = torch.tensor([500.])
    context = torch.randn(1, 3, 8)
    cfg = CoFrameConfig(
        method="sparse_evolution", num_anchors=5, warmup_steps=0,
        sparse_block_start=1, sparse_block_end=3, block_group_size=1,
        kv_mode=kv_mode, adaptive_k_policy="max_defect",
        adaptive_k_values=(5,), adaptive_k_thresholds=(), sketch_dim=0)
    expected = model(hidden, timestep, context, return_dict=False)[0]
    full, meta = coframe_transformer_forward(
        model, hidden, timestep, context, config=cfg,
        controller=AdaptiveMeshController(num_frames=5, num_anchors=5), step_index=0)
    torch.testing.assert_close(full, expected, rtol=2e-5, atol=2e-6)
    assert meta.defects  # full K must still measure LOO, so later groups can shrink

    cfg = replace(cfg, num_anchors=3, adaptive_k_values=(3, 5), adaptive_k_thresholds=(0.,))
    sparse, meta = coframe_transformer_forward(
        model, hidden, timestep, context, config=cfg,
        controller=AdaptiveMeshController(num_frames=5, num_anchors=3), step_index=0)
    assert sparse.shape == hidden.shape and torch.isfinite(sparse).all()
    assert len(meta.block_anchors[1]) == 3 and len(meta.block_anchors[2]) == 5


@torch.inference_mode()
def test_partial_full_kv_uses_original_rope_and_dense_anchor_residuals():
    torch.manual_seed(23)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=2, out_channels=2, text_dim=8, freq_dim=8,
        ffn_dim=32, num_layers=1, rope_max_seq_len=32).eval()
    hidden = torch.randn(1, 2, 5, 4, 4)
    rope = model.rope(hidden)
    tokens = model.patch_embedding(hidden).flatten(2).transpose(1, 2).contiguous()
    _, modulation, context, _ = model.condition_embedder(torch.tensor([500.]), torch.randn(1, 3, 8), None)
    modulation = modulation.unflatten(1, (6, -1))
    cfg = CoFrameConfig(kv_mode="full_kv", sketch_dim=0)
    anchors = [0, 2, 4]
    # Full-KV anchor queries must equal a dense block at those ORIGINAL frame
    # positions; identity-RoPE toy fixtures cannot catch compressed-position bugs.
    dense = model.blocks[0](tokens, context, modulation, rope)
    sparse, defects = _sparse_block_forward(
        model.blocks[0], tokens, context, modulation, rope,
        anchors=anchors, geometry=FrameGeometry(5, 2, 2), config=cfg,
        compute_defects=True, projection=None)
    incoming = tokens_to_frames(tokens, 5, 4)
    dense_frames = tokens_to_frames(dense, 5, 4)
    expected, _ = reconstruct_sparse_block(incoming, dense_frames[:, anchors], anchors)
    torch.testing.assert_close(tokens_to_frames(sparse, 5, 4), expected, rtol=2e-5, atol=2e-6)
    assert set(defects) == {2}
