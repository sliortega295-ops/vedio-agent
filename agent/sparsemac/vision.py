from __future__ import annotations

import math
from pathlib import Path


def load_cv2():
    import cv2
    import numpy as np

    return cv2, np


def make_storyboard(video_path: str, out_path: str, frames: int | list[int] = 12) -> list[str]:
    cv2, np = load_cv2()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot read frames from {video_path}")
    if isinstance(frames, (list, tuple)):
        indices = sorted(list(frames))
    elif frames == 12:
        indices = [0, 1, 2, 3, 8, 9, 10, 11, 17, 18, 19, 20]
    else:
        indices = sorted(set(round(i * (total - 1) / max(1, frames - 1)) for i in range(frames)))
    selected = []
    frame_paths = []
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_path_p.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    idx_set = set(indices)
    cur = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idx_set:
            selected.append(frame)
            fp = frame_dir / f"frame_{len(selected)-1:02d}_src_{cur:03d}.jpg"
            cv2.imwrite(str(fp), frame)
            frame_paths.append(str(fp))
        cur += 1
    cap.release()
    if not selected:
        raise ValueError(f"No frames selected from {video_path}")

    h, w = selected[0].shape[:2]
    thumb_w = 220
    thumb_h = max(1, round(h * thumb_w / w))
    thumbs = [cv2.resize(f, (thumb_w, thumb_h)) for f in selected]
    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    canvas = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = thumb
        cv2.putText(
            canvas,
            str(i),
            (c * thumb_w + 8, r * thumb_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path_p), canvas)
    return frame_paths


def compute_cv_metrics(frame_paths: list[str]) -> dict[str, float]:
    cv2, np = load_cv2()
    if not frame_paths:
        return {"single_frame_quality": 0.0, "keyframe_continuity": 0.0, "dynamic_sufficiency": 0.0}

    frames = [cv2.imread(p) for p in frame_paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return {"single_frame_quality": 0.0, "keyframe_continuity": 0.0, "dynamic_sufficiency": 0.0}

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    sharpness = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in grays]
    sharp_score = _saturate(sum(sharpness) / len(sharpness), 30.0, 260.0)

    diffs = []
    hist_sims = []
    for a, b in zip(grays, grays[1:]):
        a_small = cv2.resize(a, (96, 96))
        b_small = cv2.resize(b, (96, 96))
        diffs.append(float(np.mean(cv2.absdiff(a_small, b_small))))
        ha = cv2.calcHist([a_small], [0], None, [32], [0, 256])
        hb = cv2.calcHist([b_small], [0], None, [32], [0, 256])
        hist_sims.append(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    avg_hist = sum(hist_sims) / len(hist_sims) if hist_sims else 1.0

    dynamic_score = _saturate(avg_diff, 3.0, 28.0)
    continuity_score = max(0.0, min(1.0, 0.65 * ((avg_hist + 1) / 2) + 0.35 * (1 - _saturate(avg_diff, 30.0, 80.0))))
    return {
        "single_frame_quality": sharp_score,
        "keyframe_continuity": continuity_score,
        "dynamic_sufficiency": dynamic_score,
    }


def _saturate(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))

