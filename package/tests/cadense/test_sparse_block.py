import torch
from torch import nn

from coframe.config import CoFrameConfig
from coframe.interpolation import frame_token_indices, tokens_to_frames
from coframe.wan.sparse_forward import FrameGeometry, _sparse_block_forward


class FakeSelfAttention(nn.Module):
    def __init__(self, dim: int = 4, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.add_k_proj = None
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])
        with torch.no_grad():
            for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
                layer.weight.copy_(torch.eye(dim))

    def forward(self, hidden_states, encoder_hidden_states=None, rotary_emb=None):
        key_value = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = self.to_k(key_value).unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = self.to_v(key_value).unflatten(2, (self.heads, -1)).transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).flatten(2, 3)
        return self.to_out[0](output)


class FakeCrossAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        context = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        return 0.1 * hidden_states + context


class FakeBlock(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()
        self.attn1 = FakeSelfAttention(dim=dim)
        self.attn2 = FakeCrossAttention()
        self.ffn = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.ffn.weight.copy_(0.25 * torch.eye(dim))

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb).chunk(6, dim=1)
        normalized = self.norm1(hidden_states) * (1 + scale) + shift
        hidden_states = hidden_states + self.attn1(normalized, rotary_emb=rotary_emb) * gate
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        normalized = self.norm3(hidden_states) * (1 + c_scale) + c_shift
        return hidden_states + self.ffn(normalized) * c_gate


def test_full_kv_sparse_block_matches_dense_at_exact_anchors():
    torch.manual_seed(3)
    geometry = FrameGeometry(num_frames=5, height=1, width=2)
    hidden = torch.randn(1, geometry.sequence_length, 4)
    context = torch.randn(1, 3, 4)
    temb = torch.zeros(1, 6, 4)
    temb[:, 2] = 1.0
    temb[:, 5] = 1.0
    rotary = torch.ones(1, 1, geometry.sequence_length, 1, dtype=torch.complex128)
    block = FakeBlock()
    anchors = [0, 2, 4]
    config = CoFrameConfig(
        num_anchors=3,
        kv_mode="full_kv",
        interpolation_target="delta",
        defect_target="delta",
        sketch_dim=0,
    )

    dense = block(hidden, context, temb, rotary)
    sparse, defects = _sparse_block_forward(
        block,
        hidden,
        context,
        temb,
        rotary,
        anchors=anchors,
        geometry=geometry,
        config=config,
        compute_defects=True,
        projection=None,
    )

    token_indices = frame_token_indices(anchors, geometry.tokens_per_frame, device=hidden.device)
    torch.testing.assert_close(sparse.index_select(1, token_indices), dense.index_select(1, token_indices))
    assert set(defects) == {2}
    assert torch.isfinite(defects[2])


def test_sparse_block_returns_full_frame_sequence():
    torch.manual_seed(4)
    geometry = FrameGeometry(num_frames=5, height=1, width=2)
    hidden = torch.randn(1, geometry.sequence_length, 4)
    context = torch.randn(1, 3, 4)
    temb = torch.zeros(1, 6, 4)
    temb[:, 2] = 1.0
    temb[:, 5] = 1.0
    rotary = torch.ones(1, 1, geometry.sequence_length, 1, dtype=torch.complex128)
    block = FakeBlock()
    config = CoFrameConfig(num_anchors=3, kv_mode="anchor_only", sketch_dim=0)

    sparse, _ = _sparse_block_forward(
        block,
        hidden,
        context,
        temb,
        rotary,
        anchors=[0, 2, 4],
        geometry=geometry,
        config=config,
        compute_defects=False,
        projection=None,
    )

    frames = tokens_to_frames(sparse, geometry.num_frames, geometry.tokens_per_frame)
    assert frames.shape == (1, 5, 2, 4)
    assert torch.isfinite(frames).all()
