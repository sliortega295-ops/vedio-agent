"""CPU contracts for the paper's full-length, residual-budget sparse path."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cadense.test_transformer_forward import Block, Transformer
from coframe.config import CoFrameConfig
from coframe.controller import AdaptiveMeshController
from coframe.wan import pipeline as pipeline_module
from coframe.wan import sparse_forward as sparse_module


def config(**changes):
    options = dict(method="sparse_evolution", num_anchors=3, warmup_steps=0,
                   sparse_block_start=1, sparse_block_end=5, block_group_size=2,
                   adaptive_k_policy="max_defect", adaptive_k_values=(3, 5),
                   adaptive_k_thresholds=(1e6,), kv_mode="full_kv", sketch_dim=0)
    options.update(changes)
    return CoFrameConfig(**options)


@pytest.fixture(autouse=True)
def cpu_version_gate(monkeypatch):
    # Only the installed-library gate is stubbed. All sparse operators are real.
    monkeypatch.setattr(sparse_module, "require_diffusers_034", lambda strict=True: "0.34.0")
    monkeypatch.setattr(pipeline_module, "require_diffusers_034", lambda strict=True: "0.34.0")


def test_new_mode_has_explicit_residual_contract():
    config().validate(num_frames=5, num_blocks=6)
    for changes in (dict(adaptive_k_policy="none"), dict(interpolation_target="state"),
                    dict(defect_target="state"), dict(adaptive_k_thresholds=(float("nan"),)),
                    dict(adaptive_k_values=(2, 5)), dict(warmup_steps=-1)):
        with pytest.raises(ValueError):
            config(**changes).validate(num_frames=5, num_blocks=6)


def run_transformer(cfg):
    torch.manual_seed(42)
    model = Transformer()
    model.blocks = nn.ModuleList([Block() for _ in range(6)])
    hidden = torch.randn(1, 2, 5, 1, 2)
    controller = AdaptiveMeshController(num_frames=5, num_anchors=cfg.num_anchors)
    args = (model, hidden, torch.tensor([500.]), torch.ones(1, 3, 4))
    output, meta = sparse_module.coframe_transformer_forward(
        *args, config=cfg, controller=controller, step_index=0)
    return args, controller, output, meta


def test_group_rotation_preserves_full_shape_and_dense_boundaries():
    args, _, output, meta = run_transformer(config())
    assert output.shape == args[1].shape
    assert torch.isfinite(output).all()
    assert set(meta.block_anchors) == {1, 2, 3, 4}
    assert meta.block_anchors[1] == meta.block_anchors[2]
    assert meta.block_anchors[3] == meta.block_anchors[4]
    assert meta.block_anchors[1] != meta.block_anchors[3]
    assert all(len(a) == 3 and a[0] == 0 and a[-1] == 4 for a in meta.block_anchors.values())


@pytest.mark.parametrize("initial,threshold,expected", [(3, 0.0, 5), (5, 1e6, 3)])
def test_defect_changes_next_group_only_including_full_budget_recovery(initial, threshold, expected):
    _, _, _, meta = run_transformer(config(num_anchors=initial, adaptive_k_thresholds=(threshold,)))
    assert len(meta.block_anchors[1]) == len(meta.block_anchors[2]) == initial
    assert len(meta.block_anchors[3]) == len(meta.block_anchors[4]) == expected
    updates = [e for e in meta.budget_events if "next_k" in e]
    assert updates[0]["after_block"] == 2
    assert updates[0]["next_k"] == expected


def test_cfg_replay_does_not_reset_or_update_budget():
    cfg = config(adaptive_k_thresholds=(0.0,), adaptive_k_carry_across_steps=False)
    args, controller, _, meta = run_transformer(cfg)
    state = controller.state_dict()
    assert controller.current_budget == 5
    _, replay = sparse_module.coframe_transformer_forward(
        *args, config=cfg, controller=controller, step_index=0,
        replay_block_anchors=meta.block_anchors, update_controller=False)
    assert replay.block_anchors == meta.block_anchors
    assert not replay.budget_events and not replay.defects
    assert controller.state_dict() == state


class DenseTransformer(Transformer):
    @property
    def dtype(self):
        return torch.float32

    def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        cfg = CoFrameConfig(method="dense", num_anchors=3, sparse_block_start=0, sparse_block_end=2)
        output, _ = sparse_module.coframe_transformer_forward(
            self, hidden_states, timestep, encoder_hidden_states, config=cfg,
            controller=AdaptiveMeshController(num_frames=5, num_anchors=3), step_index=0)
        return (output,)


class TinyPipeline:
    vae_scale_factor_temporal = 4
    _execution_device = torch.device("cpu")

    def __init__(self):
        self.transformer = DenseTransformer()
        self.scheduler = SimpleNamespace(set_timesteps=self.set_timesteps, step=self.step)
        self.seen_shapes = []

    def set_timesteps(self, count, device):
        self.scheduler.timesteps = torch.linspace(1, 0, count, device=device)
        self.scheduler.sigmas = torch.linspace(1, 0, count + 1, device=device)

    def step(self, prediction, timestep, latents, return_dict):
        self.seen_shapes.append(tuple(latents.shape))
        return (latents - 0.1 * prediction,)

    def check_inputs(self, *args):
        pass

    def encode_prompt(self, **kwargs):
        return torch.ones(1, 3, 4), torch.zeros(1, 3, 4)

    def prepare_latents(self, batch, channels, height, width, frames, dtype, device, generator, latents):
        return torch.randn(batch, channels, (frames - 1) // 4 + 1, 1, 2,
                           generator=generator, dtype=dtype, device=device) if latents is None else latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def interrupt(self):
        return self._interrupt

    @contextmanager
    def progress_bar(self, total):
        yield SimpleNamespace(update=lambda: None)


@pytest.mark.parametrize("steps,warmup,tail,sparse,dense", [
    (1, 0, 0, 1, 0), (1, 0, 1, 0, 1), (4, 0, 1, 3, 1), (4, 1, 1, 2, 2)])
def test_full_pipeline_zero_warmup_and_dense_tail(steps, warmup, tail, sparse, dense):
    pipe = TinyPipeline()
    result = pipeline_module.coframe_wan_generate(
        pipe, prompt="test", num_frames=17, num_inference_steps=steps, height=16, width=16,
        config=config(sparse_block_start=0, sparse_block_end=2, block_group_size=1,
                      warmup_steps=warmup, evolution_dense_tail_steps=tail),
        generator=torch.Generator().manual_seed(5))
    assert result.metadata["sparse_steps"] == sparse
    assert result.metadata["dense_steps"] == dense
    assert tuple(result.frames.shape) == (1, 2, 5, 1, 2)
    assert all(shape == (1, 2, 5, 1, 2) for shape in pipe.seen_shapes)


def test_single_candidate_controller_contract():
    with pytest.raises(ValueError, match="one candidate"):
        pipeline_module.coframe_wan_generate(
            TinyPipeline(), prompt=["a", "b"], num_frames=17,
            config=config(sparse_block_start=0, sparse_block_end=2))


def test_non_finite_defects_fail_closed(monkeypatch):
    original = sparse_module._sparse_block_forward
    def invalid(*args, **kwargs):
        output, _ = original(*args, **kwargs)
        return output, {2: torch.tensor(float("nan"))}
    monkeypatch.setattr(sparse_module, "_sparse_block_forward", invalid)
    with pytest.raises(RuntimeError, match="finite interior"):
        run_transformer(config())


def test_final_partial_group_has_causal_budget_update():
    _, _, _, meta = run_transformer(config(sparse_block_end=4, adaptive_k_thresholds=(0.,)))
    assert len(meta.block_anchors[1]) == len(meta.block_anchors[2]) == 3
    assert len(meta.block_anchors[3]) == 5
    assert [e["after_block"] for e in meta.budget_events if "next_k" in e] == [2, 3]
