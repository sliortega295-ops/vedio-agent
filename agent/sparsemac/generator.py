from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .schema import Candidate
from .utils import ensure_dir


def run_sparse_candidate(
    candidate: Candidate,
    generator_root: str,
    ckpt_dir: str,
    python_bin: str,
    gpu_id: int,
    size: str,
    sample_steps: int,
    sample_shift: float,
    guide_scale: float,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 0,
    timeout: int | None = None,
    dry_run: bool = False,
    sparse_stride: int = 2,
) -> Candidate:
    out_path = Path(candidate.sparse_video)
    ensure_dir(out_path.parent)
    warmup_latent_path = out_path.parent / "warmup_latent.pt"
    candidate.warmup_latent_path = str(warmup_latent_path)
    if dry_run:
        candidate.status = "dry_run"
        return candidate
    cmd = [
        python_bin,
        "generate_sparse.py",
        "--ckpt_dir",
        ckpt_dir,
        "--prompt",
        candidate.prompt,
        "--size",
        size,
        "--sample_steps",
        str(sample_steps),
        "--sample_shift",
        str(sample_shift),
        "--guide_scale",
        str(guide_scale),
        "--seed",
        str(candidate.seed),
        "--save_file",
        str(out_path),
        "--sparse_stride",
        str(sparse_stride),
        "--key_only",
        "--save_warmup_latent",
        str(warmup_latent_path),
    ]
    if warmup_steps > 0:
        cmd.extend(["--warmup_steps", str(warmup_steps)])
    elif warmup_ratio > 0:
        cmd.extend(["--warmup_ratio", str(warmup_ratio)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.run(
        cmd,
        cwd=generator_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path = out_path.with_suffix(".log")
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode == 0 and out_path.exists() and warmup_latent_path.exists():
        candidate.status = "ok"
    else:
        candidate.status = "failed"
        candidate.error = (
            f"generate_sparse failed with code {proc.returncode}; "
            f"video_exists={out_path.exists()} warmup_latent_exists={warmup_latent_path.exists()}; "
            f"see {log_path}"
        )
    return candidate


def run_sparse_batch(
    candidates: list[Candidate],
    gpu_ids: list[int],
    **kwargs,
) -> list[Candidate]:
    if not candidates:
        return []
    results: list[Candidate] = []
    max_workers = min(len(candidates), max(1, len(gpu_ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, cand in enumerate(candidates):
            futures.append(pool.submit(run_sparse_candidate, cand, gpu_id=gpu_ids[idx % len(gpu_ids)], **kwargs))
        for fut in as_completed(futures):
            results.append(fut.result())
    return sorted(results, key=lambda c: c.candidate_id)


def run_full_generation(
    prompt: str,
    seed: int,
    generator_root: str,
    ckpt_dir: str,
    python_bin: str,
    save_file: str,
    size: str,
    sample_steps: int,
    sample_shift: float,
    guide_scale: float,
    gpu_id: int,
    init_latent_path: str | None = None,
    dry_run: bool = False,
) -> tuple[bool, str]:
    ensure_dir(Path(save_file).parent)
    if dry_run:
        return True, "dry_run"
    if init_latent_path is None:
        raise ValueError("init_latent_path is required for full generation")
    if not Path(init_latent_path).exists():
        raise FileNotFoundError(f"warmup latent not found: {init_latent_path}")
    cmd = [
        python_bin,
        "generate.py",
        "--task",
        "t2v-1.3B",
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--prompt",
        prompt,
        "--base_seed",
        str(seed),
        "--sample_steps",
        str(sample_steps),
        "--sample_shift",
        str(sample_shift),
        "--sample_guide_scale",
        str(guide_scale),
        "--save_file",
        save_file,
        "--offload_model",
        "True",
        "--t5_cpu",
        "--init_latent_path",
        init_latent_path,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shim_path = str(Path(__file__).resolve().parent / "shims")
    env["PYTHONPATH"] = shim_path + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=generator_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path = str(Path(save_file).with_suffix(".log"))
    Path(log_path).write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0 and Path(save_file).exists(), log_path
