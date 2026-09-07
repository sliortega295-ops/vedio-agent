from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F


def frame_representations_from_clean_latents(clean_latents: torch.Tensor) -> torch.Tensor:
    """Return one flattened representation per latent frame.

    Args:
        clean_latents: ``[B, C, F, H, W]`` one-step clean-latent proxy.

    Returns:
        Float tensor ``[F, B*C*H*W]``. Keeping batch in the feature dimension
        makes selection deterministic for the common B=1 validation setting.
    """
    if clean_latents.ndim != 5:
        raise ValueError(f"Expected [B,C,F,H,W], got {tuple(clean_latents.shape)}")
    return clean_latents.detach().float().permute(2, 0, 1, 3, 4).flatten(1)


def transition_scores(frame_representations: torch.Tensor) -> torch.Tensor:
    """Adjacent-frame cosine change used by the RhymeFlow-style prior."""
    if frame_representations.ndim != 2:
        raise ValueError("frame_representations must be [F,D]")
    num_frames = frame_representations.shape[0]
    scores = torch.zeros(num_frames, device=frame_representations.device, dtype=torch.float32)
    if num_frames <= 1:
        return scores
    normalized = F.normalize(frame_representations.float(), dim=1, eps=1.0e-8)
    scores[1:] = 1.0 - (normalized[1:] * normalized[:-1]).sum(dim=1)
    return scores.clamp_min_(0.0)


def uniform_select(num_frames: int, num_anchors: int, force_boundaries: bool = True) -> list[int]:
    """Select a deterministic, approximately uniform fixed-budget mesh."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    if num_anchors == 1:
        return [0 if force_boundaries else num_frames // 2]

    positions = torch.linspace(0, num_frames - 1, num_anchors).round().long().tolist()
    selected = sorted(set(int(x) for x in positions))
    if force_boundaries:
        selected = sorted(set(selected + [0, num_frames - 1]))
    return _fill_budget(
        selected,
        scores=torch.ones(num_frames),
        num_frames=num_frames,
        num_anchors=num_anchors,
        force_boundaries=force_boundaries,
        min_gap=1,
    )


def rhyme_select(
    frame_representations: torch.Tensor,
    num_anchors: int,
    similarity_threshold: float = 0.98,
    force_boundaries: bool = True,
    min_gap: int = 1,
) -> list[int]:
    """RhymeFlow-style sequential cosine selector with an exact budget.

    Frames are scanned in temporal order. A new anchor is opened when its
    cosine similarity to the latest selected anchor falls below ``threshold``.
    Adjacent transition scores then trim/fill the set to the requested budget.
    """
    if frame_representations.ndim != 2:
        raise ValueError("frame_representations must be [F,D]")
    num_frames = int(frame_representations.shape[0])
    if num_frames < 1:
        raise ValueError("At least one frame is required")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")

    min_gap = max(1, int(min_gap))
    normalized = F.normalize(frame_representations.float(), dim=1, eps=1.0e-8)
    scores = transition_scores(normalized)

    selected = [0]
    last = 0
    for index in range(1, num_frames):
        if index - last < min_gap:
            continue
        similarity = torch.dot(normalized[index], normalized[last]).item()
        if similarity < similarity_threshold:
            selected.append(index)
            last = index

    if force_boundaries and num_frames > 1:
        selected.append(num_frames - 1)

    return _fill_budget(
        selected,
        scores=scores,
        num_frames=num_frames,
        num_anchors=num_anchors,
        force_boundaries=force_boundaries,
        min_gap=min_gap,
    )


def normalized_prior(scores: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    values = scores.detach().float().clamp_min(0.0)
    maximum = float(values.max().item()) if values.numel() else 0.0
    if maximum > 0:
        values = values / maximum
    if floor:
        values = values + float(floor)
    return values


def _valid_gap(selected: Sequence[int], candidate: int, min_gap: int) -> bool:
    return all(abs(candidate - existing) >= min_gap for existing in selected)


def _fill_budget(
    selected: Sequence[int],
    *,
    scores: torch.Tensor,
    num_frames: int,
    num_anchors: int,
    force_boundaries: bool,
    min_gap: int,
) -> list[int]:
    selected_set = {int(index) for index in selected if 0 <= int(index) < num_frames}
    boundaries = {0, num_frames - 1} if force_boundaries and num_frames > 1 else set()
    selected_set.update(boundaries)

    if len(selected_set) > num_anchors:
        removable = sorted(
            (index for index in selected_set if index not in boundaries),
            key=lambda index: (float(scores[index]), index),
            reverse=True,
        )
        selected_set = set(boundaries)
        selected_set.update(removable[: max(0, num_anchors - len(boundaries))])

    ranked = sorted(range(num_frames), key=lambda index: (float(scores[index]), -index), reverse=True)
    for candidate in ranked:
        if len(selected_set) >= num_anchors:
            break
        if candidate in selected_set:
            continue
        if _valid_gap(sorted(selected_set), candidate, min_gap):
            selected_set.add(candidate)

    # A strict min-gap can make the exact budget infeasible. Fill the largest
    # temporal holes rather than silently returning fewer anchors.
    while len(selected_set) < num_anchors:
        current = sorted(selected_set)
        candidates = [index for index in range(num_frames) if index not in selected_set]
        if not candidates:
            break
        candidate = max(
            candidates,
            key=lambda index: (
                min(abs(index - anchor) for anchor in current) if current else num_frames,
                float(scores[index]),
                -index,
            ),
        )
        selected_set.add(candidate)

    return sorted(selected_set)[:num_anchors]


def _fit_interleaved_budget(
    selected: Sequence[int],
    *,
    num_frames: int,
    num_anchors: int,
    force_boundaries: bool,
) -> list[int]:
    """Keep the FIS residue pattern while matching an exact frame budget."""
    chosen = sorted({int(i) for i in selected if 0 <= int(i) < num_frames})
    boundaries = [0, num_frames - 1] if force_boundaries and num_frames > 1 else []
    chosen = sorted(set(chosen + boundaries))

    if len(chosen) > num_anchors:
        protected = set(boundaries)
        interior = [i for i in chosen if i not in protected]
        keep_n = max(0, num_anchors - len(protected))
        if keep_n < len(interior):
            slots = torch.linspace(0, len(interior) - 1, keep_n).round().long().tolist() if keep_n else []
            interior = [interior[i] for i in sorted(set(slots))]
        chosen = sorted(set(boundaries + interior))

    while len(chosen) < num_anchors:
        candidates = [i for i in range(num_frames) if i not in chosen]
        if not candidates:
            break
        candidate = max(
            candidates,
            key=lambda i: (
                min(abs(i - anchor) for anchor in chosen) if chosen else num_frames,
                -i,
            ),
        )
        chosen.append(candidate)
        chosen.sort()
    return chosen[:num_anchors]


def _minimum_coverage_mesh(
    num_frames: int,
    num_anchors: int,
    usage: Sequence[int],
    *,
    reuse_penalty: float,
) -> list[int]:
    """Solve one tiny exact-budget coverage problem with dynamic programming."""
    if num_anchors == 1:
        return [0]
    # State maps (selected_count, last_anchor) to (cost, path). Boundaries are
    # fixed; the gap-squared term minimizes large interpolation intervals while
    # the usage term encourages successive groups to cover different frames.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(1, 0): (0.0, (0,))}
    for selected_count in range(2, num_anchors + 1):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        if selected_count == num_anchors:
            right_candidates = (num_frames - 1,)
        else:
            minimum_right = selected_count - 1
            maximum_right = num_frames - 1 - (num_anchors - selected_count)
            right_candidates = range(minimum_right, maximum_right + 1)
        for right in right_candidates:
            best: tuple[float, tuple[int, ...]] | None = None
            for (count, left), (cost, path) in states.items():
                if count != selected_count - 1 or left >= right:
                    continue
                candidate_cost = cost + float((right - left) ** 2)
                if right != num_frames - 1:
                    candidate_cost += float(reuse_penalty) * float(usage[right])
                candidate = (candidate_cost, path + (right,))
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                next_states[(selected_count, right)] = best
        states = next_states
    return list(states[(num_anchors, num_frames - 1)][1])


def coverage_interleaved_select(
    num_frames: int,
    num_anchors: int,
    phase_index: int,
    *,
    force_boundaries: bool = True,
    anchor_stride: int = 0,
    reuse_penalty: float = 2.0,
) -> list[int]:
    """Select an exact-budget, coverage-aware mesh for one block group.

    With temporal boundaries enabled, a small dynamic program minimizes the
    sum of squared anchor gaps plus a reuse penalty accumulated over earlier
    phases in the cycle. This retains near-uniform coverage while rotating
    exact computation across frames. ``anchor_stride`` optionally sets the
    interleaving period; zero derives a short period from ``ceil(F/K)``.
    """
    if num_frames < 1 or num_anchors < 1:
        raise ValueError("num_frames and num_anchors must be positive")
    if reuse_penalty < 0.0:
        raise ValueError("reuse_penalty must be non-negative")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    if not force_boundaries:
        # The residual interpolator is normally used with boundary anchors.
        # Preserve the earlier deterministic rotating-residue behavior for the
        # uncommon boundary-free diagnostic mode.
        stride = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
        phase = int(phase_index) % stride
        selected = [frame for frame in range(num_frames) if (frame - phase) % stride == 0]
        return _fit_interleaved_budget(
            selected,
            num_frames=num_frames,
            num_anchors=num_anchors,
            force_boundaries=False,
        )

    period = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
    phase = int(phase_index) % period
    usage = [0 for _ in range(num_frames)]
    mesh: list[int] = []
    for _ in range(phase + 1):
        mesh = _minimum_coverage_mesh(
            num_frames,
            num_anchors,
            usage,
            reuse_penalty=float(reuse_penalty),
        )
        for frame in mesh[1:-1]:
            usage[frame] += 1
    return mesh


def fis_interleaved_select(
    num_frames: int,
    num_anchors: int,
    block_index: int,
    first_sparse_block: int,
    *,
    force_boundaries: bool = True,
    anchor_stride: int = 0,
) -> list[int]:
    """FIS-DiT-style interleaved anchor schedule with an exact budget.

    The paper uses r_l=(l-l0) mod n and selects frames satisfying
    (f-r_l) mod n=0, always keeping temporal boundaries.  For fair matched-K
    experiments we preserve that rotating residue set and deterministically
    fill/trim only when boundary insertion changes the exact count.
    """
    if num_frames < 1 or num_anchors < 1:
        raise ValueError("num_frames and num_anchors must be positive")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    stride = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
    phase = (int(block_index) - int(first_sparse_block)) % stride
    selected = [frame for frame in range(num_frames) if (frame - phase) % stride == 0]
    return _fit_interleaved_budget(
        selected,
        num_frames=num_frames,
        num_anchors=num_anchors,
        force_boundaries=force_boundaries,
    )
