from types import SimpleNamespace

import torch
from torch import nn

from coframe.config import CoFrameConfig
from coframe.controller import AdaptiveMeshController
from coframe.wan import sparse_forward as sparse_module


class SelfAttention(nn.Module):
    def __init__(self, dim=4, heads=2):
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
        kv = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = self.to_q(hidden_states).unflatten(2, (self.heads, -1)).transpose(1, 2)
        k = self.to_k(kv).unflatten(2, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(kv).unflatten(2, (self.heads, -1)).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.to_out[0](out.transpose(1, 2).flatten(2, 3))


class CrossAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        return hidden_states * 0.05 + encoder_hidden_states.mean(dim=(1, 2), keepdim=True)


class Block(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()
        self.attn1 = SelfAttention(dim)
        self.attn2 = CrossAttention()
        self.ffn = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.ffn.weight.copy_(0.1 * torch.eye(dim))

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb).chunk(6, dim=1)
        normalized = self.norm1(hidden_states.float()) * (1 + scale) + shift
        hidden_states = hidden_states + self.attn1(normalized.type_as(hidden_states), rotary_emb=rotary_emb) * gate
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        normalized = self.norm3(hidden_states.float()) * (1 + c_scale) + c_shift
        return hidden_states + self.ffn(normalized.type_as(hidden_states)) * c_gate


class Rope(nn.Module):
    def forward(self, hidden_states):
        batch, channels, frames, height, width = hidden_states.shape
        length = frames * height * width
        return torch.ones(1, 1, length, 1, dtype=torch.complex128, device=hidden_states.device)


class ConditionEmbedder(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim

    def forward(self, timestep, encoder_hidden_states, encoder_hidden_states_image=None):
        batch = timestep.shape[0]
        temb = torch.zeros(batch, self.dim, device=timestep.device)
        projected = torch.zeros(batch, 6 * self.dim, device=timestep.device)
        projected.view(batch, 6, self.dim)[:, 2] = 1.0
        projected.view(batch, 6, self.dim)[:, 5] = 1.0
        return temb, projected, encoder_hidden_states, None


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=(1, 1, 1), out_channels=2, in_channels=2)
        self.rope = Rope()
        self.patch_embedding = nn.Conv3d(2, 4, kernel_size=1, bias=False)
        self.condition_embedder = ConditionEmbedder(4)
        self.blocks = nn.ModuleList([Block(4), Block(4)])
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, 4))
        self.norm_out = nn.Identity()
        self.proj_out = nn.Linear(4, 2, bias=False)


def test_conditional_schedule_can_be_replayed_for_cfg(monkeypatch):
    monkeypatch.setattr(sparse_module, "require_diffusers_034", lambda strict=True: "0.34.0")
    torch.manual_seed(11)
    transformer = Transformer()
    hidden = torch.randn(1, 2, 5, 1, 2)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 4)
    config = CoFrameConfig(
        method="coframe",
        num_anchors=3,
        sparse_block_start=0,
        sparse_block_end=2,
        block_group_size=1,
        kv_mode="full_kv",
        sketch_dim=0,
        move_penalty=0.0,
        min_refresh_gain=0.0,
        oracle_probe_steps=(1,),
        oracle_probe_blocks=(0,),
        oracle_probe_horizons=(1,),
    )
    controller = AdaptiveMeshController(
        num_frames=5,
        num_anchors=3,
        initial_anchors=[0, 2, 4],
        prior_scores=torch.zeros(5),
        prior_weight=0.0,
        risk_ema=0.0,
        move_penalty=0.0,
        min_refresh_gain=0.0,
    )

    conditional, cond_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context,
        config=config,
        controller=controller,
        step_index=1,
        update_controller=True,
    )
    unconditional, uncond_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context * 0.0,
        config=config,
        controller=controller,
        step_index=1,
        replay_block_anchors=cond_meta.block_anchors,
        update_controller=False,
    )

    assert conditional.shape == hidden.shape
    assert unconditional.shape == hidden.shape
    assert cond_meta.block_anchors == uncond_meta.block_anchors
    assert set(cond_meta.block_anchors) == {0, 1}
    assert len(cond_meta.probes) == 1
    probe = cond_meta.probes[0]
    assert probe["block_delta_relative_l2"] >= 0.0
    assert probe["mesh_oracle_nmse"] <= probe["mesh_current_nmse"] + 1.0e-7
    assert probe["mesh_oracle_nmse"] <= probe["mesh_rhyme_nmse"] + 1.0e-7
    assert "swap_decision" in probe
    assert "1" in probe["propagation"]
    assert probe["propagated_relative_l2_h1"] >= 0.0
    assert torch.isfinite(conditional).all()
    assert torch.isfinite(unconditional).all()

def test_ode_coframe_uses_group_level_loo_defects(monkeypatch):
    monkeypatch.setattr(sparse_module, "require_diffusers_034", lambda strict=True: "0.34.0")
    torch.manual_seed(17)
    transformer = Transformer()
    hidden = torch.randn(1, 2, 5, 1, 2)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 4)
    config = CoFrameConfig(
        method="coframe_ode",
        num_anchors=3,
        sparse_block_start=0,
        sparse_block_end=2,
        block_group_size=1,
        kv_mode="full_kv",
        sketch_dim=0,
        risk_ema=0.0,
        move_penalty=0.0,
        min_refresh_gain=0.0,
        max_swaps_per_refresh=1,
    )
    controller = AdaptiveMeshController(
        num_frames=5,
        num_anchors=3,
        initial_anchors=[0, 2, 4],
        prior_scores=torch.zeros(5),
        prior_weight=0.0,
        risk_ema=0.0,
        move_penalty=0.0,
        min_refresh_gain=0.0,
        max_swaps_per_refresh=1,
    )
    controller.current_budget = 3

    conditional, cond_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context,
        config=config,
        controller=controller,
        step_index=1,
        update_controller=True,
    )
    replayed, replay_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context * 0.0,
        config=config,
        controller=controller,
        step_index=1,
        replay_block_anchors=cond_meta.block_anchors,
        update_controller=False,
    )

    assert conditional.shape == replayed.shape == hidden.shape
    assert cond_meta.block_anchors == replay_meta.block_anchors
    assert all(len(mesh) == 3 for mesh in cond_meta.block_anchors.values())
    assert len(cond_meta.defects) == 2
    assert all(entry["values"] for entry in cond_meta.defects)
    assert len(cond_meta.refresh_events) >= 1
    assert all(event["refresh_signal"] == "loo_residual_defect" for event in cond_meta.refresh_events)
    assert all(mesh[0] == 0 and mesh[-1] == 4 for mesh in cond_meta.block_anchors.values())
