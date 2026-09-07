from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_anchors(anchors: Sequence[int], num_frames: int) -> list[int]:
    result = sorted({int(index) for index in anchors})
    if not result:
        raise ValueError("At least one anchor is required")
    if result[0] < 0 or result[-1] >= num_frames:
        raise ValueError(f"Anchors {result} are outside [0, {num_frames})")
    return result


def frame_token_indices(
    frame_indices: Sequence[int],
    tokens_per_frame: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    frames = torch.as_tensor(list(frame_indices), dtype=torch.long, device=device)
    if frames.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    offsets = torch.arange(tokens_per_frame, dtype=torch.long, device=device)
    return (frames[:, None] * tokens_per_frame + offsets[None, :]).reshape(-1)


def tokens_to_frames(tokens: torch.Tensor, num_frames: int, tokens_per_frame: int) -> torch.Tensor:
    """Convert ``[B,F*P,D]`` to ``[B,F,P,D]``."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,L,D], got {tuple(tokens.shape)}")
    expected = num_frames * tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Token length {tokens.shape[1]} != F*P={expected}")
    return tokens.reshape(tokens.shape[0], num_frames, tokens_per_frame, tokens.shape[-1])


def frames_to_tokens(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected [B,F,P,D], got {tuple(frames.shape)}")
    return frames.reshape(frames.shape[0], frames.shape[1] * frames.shape[2], frames.shape[3])


def piecewise_linear_interpolate(
    anchor_values: torch.Tensor,
    anchors: Sequence[int],
    num_frames: int,
) -> torch.Tensor:
    """Interpolate values on an arbitrary 1-D temporal mesh.

    ``anchor_values`` may have any trailing dimensions and must have shape
    ``[B,K,...]``. Extrapolation outside the first/last anchor is constant; in
    the canonical CoFrame setup both boundaries are forced, so this path is
    normally unused.
    """
    if anchor_values.ndim < 2:
        raise ValueError("anchor_values must be [B,K,...]")
    anchor_list = validate_anchors(anchors, num_frames)
    if anchor_values.shape[1] != len(anchor_list):
        raise ValueError("anchor_values K dimension does not match anchors")

    device = anchor_values.device
    anchor_tensor = torch.tensor(anchor_list, device=device, dtype=torch.long)
    positions = torch.arange(num_frames, device=device, dtype=torch.long)

    right_slot = torch.searchsorted(anchor_tensor, positions, right=False)
    right_slot = right_slot.clamp(0, len(anchor_list) - 1)
    left_slot = (right_slot - 1).clamp(0, len(anchor_list) - 1)

    # Exact anchor positions should use that anchor on both sides.
    exact = anchor_tensor.index_select(0, right_slot) == positions
    left_slot = torch.where(exact, right_slot, left_slot)

    left_pos = anchor_tensor.index_select(0, left_slot)
    right_pos = anchor_tensor.index_select(0, right_slot)
    denominator = (right_pos - left_pos).clamp_min(1).to(torch.float32)
    alpha = (positions - left_pos).to(torch.float32) / denominator
    alpha = torch.where(left_pos == right_pos, torch.zeros_like(alpha), alpha)

    left_values = anchor_values.index_select(1, left_slot)
    right_values = anchor_values.index_select(1, right_slot)
    view_shape = [1, num_frames] + [1] * (anchor_values.ndim - 2)
    alpha = alpha.view(*view_shape).to(dtype=anchor_values.dtype)
    return left_values + alpha * (right_values - left_values)


def reconstruct_sparse_block(
    input_frames: torch.Tensor,
    exact_anchor_outputs: torch.Tensor,
    anchors: Sequence[int],
    *,
    target: str = "delta",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct every frame from exact sparse block outputs.

    Returns ``(full_output, exact_anchor_delta)``. Interpolating the block delta
    usually preserves the incoming full-frame state better than interpolating
    the state itself and is therefore CoFrame's default.
    """
    if input_frames.ndim != 4 or exact_anchor_outputs.ndim != 4:
        raise ValueError("Expected input [B,F,P,D] and anchors [B,K,P,D]")
    num_frames = input_frames.shape[1]
    anchor_list = validate_anchors(anchors, num_frames)
    anchor_index = torch.tensor(anchor_list, device=input_frames.device, dtype=torch.long)
    exact_inputs = input_frames.index_select(1, anchor_index)
    exact_delta = exact_anchor_outputs - exact_inputs

    if target == "delta":
        interpolated = piecewise_linear_interpolate(exact_delta, anchor_list, num_frames)
        output = input_frames + interpolated
    elif target == "state":
        output = piecewise_linear_interpolate(exact_anchor_outputs, anchor_list, num_frames)
    else:
        raise ValueError(f"Unknown interpolation target: {target}")

    # Avoid accumulated floating-point drift at exact anchors.
    output = output.clone()
    output.index_copy_(1, anchor_index, exact_anchor_outputs)
    return output, exact_delta


def leave_one_out_defects(
    anchor_values: torch.Tensor,
    anchors: Sequence[int],
    *,
    projection: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> dict[int, torch.Tensor]:
    """Compute normalized hierarchical-surplus defects at interior anchors.

    For each triplet ``left < validator < right``, the validator's exact value
    is compared with linear interpolation of its two neighbors. The returned
    scalar is RMS(residual) / RMS(exact). A fixed random channel projection may
    be supplied to reduce controller overhead.
    """
    if anchor_values.ndim < 3:
        raise ValueError("anchor_values must be [B,K,...,D]")
    anchor_list = validate_anchors(anchors, max(anchors) + 1)
    if anchor_values.shape[1] != len(anchor_list):
        raise ValueError("anchor_values K dimension does not match anchors")
    if len(anchor_list) < 3:
        return {}

    if projection is not None:
        if projection.ndim != 2 or projection.shape[0] != anchor_values.shape[-1]:
            raise ValueError("projection must have shape [D, sketch_dim]")
        # Project in the model dtype first; converting the full [B,K,P,D]
        # tensor to FP32 would erase much of the controller's memory advantage.
        values = torch.matmul(
            anchor_values,
            projection.to(device=anchor_values.device, dtype=anchor_values.dtype),
        ).float()
    else:
        values = anchor_values.float()

    defects: dict[int, torch.Tensor] = {}
    for slot in range(1, len(anchor_list) - 1):
        left, validator, right = anchor_list[slot - 1 : slot + 2]
        alpha = float(validator - left) / float(right - left)
        predicted = values[:, slot - 1] + alpha * (values[:, slot + 1] - values[:, slot - 1])
        exact = values[:, slot]
        reduce_dims = tuple(range(1, exact.ndim))
        residual_rms = (exact - predicted).pow(2).mean(dim=reduce_dims).sqrt().mean()
        exact_rms = exact.pow(2).mean(dim=reduce_dims).sqrt().mean()
        defects[validator] = residual_rms / (exact_rms + eps)
    return defects


def per_frame_relative_rms(reference: torch.Tensor, approximation: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return a ``[F]`` relative RMS error vector for ``[B,F,...]`` tensors."""
    if reference.shape != approximation.shape or reference.ndim < 3:
        raise ValueError("reference and approximation must share [B,F,...] shape")
    reduce_dims = tuple(index for index in range(reference.ndim) if index not in (1,))
    numerator = (reference.float() - approximation.float()).pow(2).mean(dim=reduce_dims).sqrt()
    denominator = reference.float().pow(2).mean(dim=reduce_dims).sqrt()
    return numerator / (denominator + eps)
