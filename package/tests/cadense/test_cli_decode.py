from types import SimpleNamespace

import torch

from coframe.cli import _decode_wan_latents


class _FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0], z_dim=2)

    def __init__(self) -> None:
        self.grad_enabled_during_decode = None

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        self.grad_enabled_during_decode = torch.is_grad_enabled()
        return (latents,)


class _FakeVideoProcessor:
    def postprocess_video(self, video: torch.Tensor, output_type: str):
        assert output_type == "np"
        return [video.numpy()]


def test_decode_wan_latents_disables_autograd() -> None:
    vae = _FakeVAE()
    pipe = SimpleNamespace(vae=vae, video_processor=_FakeVideoProcessor())
    result = _decode_wan_latents(pipe, torch.zeros(1, 2, 1, 1, 1, requires_grad=True))
    assert len(result) == 1
    assert vae.grad_enabled_during_decode is False
