"""Public entry-point contracts; no checkpoint or GPU required."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import full_length_sparse as backend
from test_sparse_evolution import TinyPipeline, config


def test_cli_defaults_are_full_length_offline_and_explicit(tmp_path):
    argv = ["--prompt", "a horse", "--output-dir", str(tmp_path)]
    args = backend.build_parser().parse_args(argv)
    assert args.method == "sparse_evolution"
    assert args.warmup_steps == 0 and args.evolution_dense_tail_steps == 1
    assert args.kv_mode == "full_kv" and args.decode and args.local_files_only
    with pytest.raises(ValueError, match="thresholds"):
        backend.main(argv)
    assert not list(tmp_path.iterdir())  # no output/model work before validation


def test_candidate_repeatability_fresh_controller_and_explicit_latents():
    pipe = TinyPipeline()
    cfg = config(sparse_block_start=0, sparse_block_end=2, block_group_size=1,
                 adaptive_k_thresholds=(0.,), evolution_dense_tail_steps=0)
    options = dict(prompt="a horse", seed=9, config=cfg, decode=False,
                   num_frames=17, num_inference_steps=2, height=16, width=16)
    first = backend.generate_candidate(pipe, **options)
    second = backend.generate_candidate(pipe, **options)
    torch.testing.assert_close(first.latents, second.latents, rtol=0, atol=0)
    assert first.video is None
    assert first.latents.shape[2] == 5
    for candidate in (first, second):
        events = candidate.metadata["transformer_events"][0]["budget_events"]
        assert events[0]["assigned_k"] == 3
        assert events[1]["next_k"] == 5
    assert cfg.num_anchors == 3  # shared config is not a controller
    assert pipe._current_timestep is None


def test_decode_uses_full_latents_and_vae_denormalization(monkeypatch):
    pipe = TinyPipeline()
    captured = {}
    def decode(latents, return_dict):
        captured["latents"] = latents.clone()
        return (torch.zeros(1, 3, 17, 2, 2),)
    pipe.vae = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, decode=decode,
        config=SimpleNamespace(latents_mean=[2, 2], latents_std=[3, 3], z_dim=2))
    video = np.zeros((1, 17, 2, 2, 3))
    pipe.video_processor = SimpleNamespace(postprocess_video=lambda *a, **k: video)
    result = backend.generate_candidate(
        pipe, prompt="test", seed=2, num_frames=17, num_inference_steps=1,
        config=config(sparse_block_start=0, sparse_block_end=2, evolution_dense_tail_steps=0))
    assert result.video.shape == (17, 2, 2, 3)
    torch.testing.assert_close(captured["latents"], result.latents * 3 + 2)


def test_dense_final_candidate_uses_separate_full_run():
    cfg = replace(config(sparse_block_start=0, sparse_block_end=2), method="dense")
    result = backend.generate_candidate(
        TinyPipeline(), prompt="winner", seed=6, config=cfg, decode=False,
        num_frames=17, num_inference_steps=2)
    assert result.metadata["sparse_steps"] == 0
    assert result.metadata["dense_steps"] == 2
    assert result.prompt == "winner" and result.seed == 6


def test_dense_cli_does_not_require_sparse_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        backend.main(["--method", "dense", "--prompt", "x", "--output-dir", str(tmp_path)])


@pytest.mark.parametrize("method", ["dense", "sparse_evolution"])
def test_public_candidate_api_rejects_multi_video_requests(method):
    with pytest.raises(ValueError, match="one candidate"):
        backend.generate_candidate(
            TinyPipeline(), prompt="test", seed=1, decode=False, num_videos_per_prompt=2,
            config=replace(config(sparse_block_start=0, sparse_block_end=2), method=method))


def test_dense_candidate_rejects_wrong_latent_length():
    with pytest.raises(ValueError, match="requested full latent frame count"):
        backend.generate_candidate(
            TinyPipeline(), prompt="test", seed=1, decode=False, num_frames=17,
            num_inference_steps=1, latents=torch.zeros(1, 2, 4, 1, 2),
            config=replace(config(sparse_block_start=0, sparse_block_end=2), method="dense"))


@pytest.mark.parametrize("options", [{"num_frames":16}, {"num_inference_steps":0}])
@pytest.mark.parametrize("method", ["dense", "sparse_evolution"])
def test_no_silent_frame_rounding_or_empty_sampler(options, method):
    with pytest.raises(ValueError, match="positive steps"):
        backend.generate_candidate(
            TinyPipeline(), prompt="test", seed=0, decode=False,
            config=replace(config(sparse_block_start=0, sparse_block_end=2), method=method), **options)


def test_cli_offline_generation_decode_and_export_route(tmp_path, monkeypatch):
    import diffusers
    from diffusers.schedulers import scheduling_unipc_multistep

    pipe = TinyPipeline()
    pipe.to = lambda device: pipe
    vae = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32,
        config=SimpleNamespace(latents_mean=[0, 0], latents_std=[1, 1], z_dim=2),
        decode=lambda latents, **kw: (torch.zeros(1, 3, 17, 2, 2),))
    pipe.vae = vae
    pipe.video_processor = SimpleNamespace(
        postprocess_video=lambda *a, **k: np.zeros((1, 17, 2, 2, 3)))
    loads, exports = [], []
    def load_vae(*args, **kwargs):
        loads.append(kwargs)
        return vae
    def load_pipe(*args, **kwargs):
        loads.append(kwargs)
        return pipe
    monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", load_vae)
    monkeypatch.setattr(diffusers.WanPipeline, "from_pretrained", load_pipe)
    monkeypatch.setattr(scheduling_unipc_multistep, "UniPCMultistepScheduler", lambda **kw: pipe.scheduler)
    monkeypatch.setattr(diffusers.utils, "export_to_video", lambda video, path, **kw: exports.append((video.shape, path)))
    cpu_generator = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **kw: cpu_generator(device="cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("reset_peak_memory_stats", "synchronize", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)

    # Only checkpoint loading, CUDA plumbing and codec export are replaced;
    # parser, config, denoising, CFG, decoding route and persistence run for real.
    argv = ["--model-id", "/local/wan-diffusers", "--prompt", "test",
            "--output-dir", str(tmp_path), "--run-name", "cli-smoke",
            "--num-frames", "17", "--height", "16", "--width", "16", "--steps", "2",
            "--sparse-block-start", "0", "--sparse-block-end", "2",
            "--block-group-size", "1", "--num-anchors", "3",
            "--adaptive-k-values", "3,5", "--adaptive-k-thresholds", "1000000"]
    assert backend.main(argv) == 0
    summary = json.loads((tmp_path / "cli-smoke/summary.json").read_text())
    assert summary["method"] == "sparse_evolution"
    assert summary["sparse_steps"] == summary["dense_steps"] == 1
    assert len(loads) == 2 and all(item["local_files_only"] for item in loads)
    assert exports[0][0] == (17, 2, 2, 3)
    assert (tmp_path / "cli-smoke/latents.pt").is_file()
    assert (tmp_path / "cli-smoke/trace.json").is_file()
    with pytest.raises(FileExistsError):
        backend.main(argv)
    assert len(loads) == 2  # no reload or overwrite on reused run names
