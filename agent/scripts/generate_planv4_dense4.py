#!/usr/bin/env python3
"""Dense-only PlanV4 ablation: four full candidates per prompt, no sparse stage.

Generation deliberately stops before VLM selection.  A separate scorer selects
the highest-scoring full candidate once the visual API is available again.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparsemac.planner import seed_pool
from sparsemac.vision import make_storyboard


DEFAULT_MANIFEST = ROOT / "runs/ablation/vqqa-sparse-lite-hmt12-difficult25/manifest.json"
DEFAULT_PLAN_DIR = ROOT / "logs/plan_eval_mixed_skill_v4/plans"
DEFAULT_RUN_ROOT = ROOT / "runs/ablation/planv4-dense4-difficult175"
DEFAULT_GENERATOR = Path(
    os.environ.get("SPARSEMAC_GENERATOR_ROOT", ROOT / "generation_backend")
)
DEFAULT_CKPT = Path(os.environ.get("SPARSEMAC_WAN_CHECKPOINT", "."))
DEFAULT_PYTHON = Path(os.environ.get("SPARSEMAC_PYTHON", sys.executable))
DEFAULT_FFPROBE = Path(
    os.environ.get("FFPROBE_BIN", shutil.which("ffprobe") or "ffprobe")
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_video(path: Path, *, expected_frames: int = 81) -> bool:
    """Reject interrupted MP4 files instead of trusting file size alone."""
    if not path.is_file() or path.stat().st_size <= 4096:
        return False
    proc = subprocess.run(
        [
            str(DEFAULT_FFPROBE), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    try:
        return int(proc.stdout.strip().splitlines()[-1]) == expected_frames
    except (IndexError, ValueError):
        return False


def plan_for(plan_path: Path) -> dict:
    data = read_json(plan_path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid PlanV4 file: {plan_path}")
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else data
    variants = plan.get("prompt_variants")
    if not isinstance(variants, list) or len([v for v in variants if str(v).strip()]) < 4:
        raise ValueError(f"PlanV4 has fewer than four prompt variants: {plan_path}")
    plan["prompt_variants"] = [str(v).strip() for v in variants if str(v).strip()][:4]
    return plan


def generate_one(*, candidate: dict, gpu: int, args: argparse.Namespace) -> dict:
    video = Path(candidate["video"])
    log = Path(candidate["log"])
    meta = Path(candidate["metadata"])
    if valid_video(video):
        candidate["status"] = "skipped_valid"
        write_json(meta, candidate)
        return candidate
    video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_bin), "generate.py", "--task", "t2v-1.3B",
        "--ckpt_dir", str(args.ckpt_dir), "--prompt", candidate["prompt"],
        "--base_seed", str(candidate["seed"]), "--size", args.size,
        "--frame_num", "81", "--sample_steps", str(args.steps),
        "--sample_shift", str(args.shift), "--sample_guide_scale", str(args.guide_scale),
        "--save_file", str(video), "--offload_model", "True", "--t5_cpu",
    ]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(args.generator_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log.write_text(proc.stdout, encoding="utf-8")
    candidate.update({"gpu": gpu, "elapsed_seconds": round(time.time() - started, 3), "returncode": proc.returncode})
    candidate["status"] = "ok" if proc.returncode == 0 and valid_video(video) else "failed"
    if candidate["status"] == "ok":
        try:
            make_storyboard(str(video), candidate["storyboard"], frames=11)
        except Exception as exc:  # video itself remains usable; scorer can rebuild later
            candidate["storyboard_error"] = f"{type(exc).__name__}: {exc}"
    else:
        candidate["error"] = f"generate.py exit={proc.returncode}; see {log}"
    write_json(meta, candidate)
    return candidate


def run_prompt(item: dict, plan: dict, args: argparse.Namespace) -> dict:
    category, prompt_id = item["category_key"], item["prompt_id"]
    prompt_root = args.run_root / "trajectories" / category / prompt_id
    variants = plan["prompt_variants"]
    seeds = seed_pool(item["prompt"] + json.dumps(variants, ensure_ascii=False), 0, 4)
    candidates: list[dict] = []
    for idx, (prompt, seed) in enumerate(zip(variants, seeds)):
        candidate_dir = prompt_root / "candidates" / f"candidate_{idx:02d}"
        video = candidate_dir / "video.mp4"
        candidates.append({
            "candidate_id": f"candidate_{idx:02d}", "candidate_index": idx,
            "prompt": prompt, "seed": int(seed), "video": str(video),
            "storyboard": str(candidate_dir / "storyboard_11.jpg"),
            "log": str(candidate_dir / "generate.log"), "metadata": str(candidate_dir / "metadata.json"),
            "status": "pending",
        })
    write_json(prompt_root / "plan.json", plan)
    with futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        tasks = [
            pool.submit(generate_one, candidate=c, gpu=args.gpus[i % len(args.gpus)], args=args)
            for i, c in enumerate(candidates)
        ]
        candidates = [task.result() for task in tasks]
    candidates.sort(key=lambda x: x["candidate_index"])
    write_json(prompt_root / "candidates.json", candidates)
    ok = [c for c in candidates if c["status"] in {"ok", "skipped_valid"}]
    result = {
        "prompt_id": prompt_id, "category_key": category, "original_prompt": item["prompt"],
        "status": "generated" if len(ok) == 4 else "incomplete", "candidates": candidates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(prompt_root / "generation_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--generator-root", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not args.gpus: raise SystemExit("at least one GPU is required")
    manifest = read_json(args.source_manifest)
    items = list(manifest["items"])[args.start_index:]
    if args.limit is not None: items = items[:args.limit]
    args.run_root.mkdir(parents=True, exist_ok=True)
    copied = {k: v for k, v in manifest.items() if k != "items"}
    copied.update({"method": "planv4_dense4_full_only", "source_manifest": str(args.source_manifest), "items": manifest["items"], "config": {**manifest.get("config", {}), "candidates_per_round": 4, "dense_only": True, "sparse_generation": False, "selection": "pending_vlm_full_candidate_selection"}})
    write_json(args.run_root / "manifest.json", copied)
    summary = []
    for number, item in enumerate(items, 1):
        plan_path = args.plan_dir / f"{item['prompt_id']}.json"
        try:
            plan = plan_for(plan_path)
            result = run_prompt(item, plan, args)
            print(f"[{number}/{len(items)}] {item['prompt_id']} {result['status']}", flush=True)
            summary.append({"prompt_id": item["prompt_id"], "category": item["category_key"], "status": result["status"]})
        except Exception as exc:
            print(f"[{number}/{len(items)}] {item['prompt_id']} failed: {type(exc).__name__}: {exc}", flush=True)
            summary.append({"prompt_id": item["prompt_id"], "category": item["category_key"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.run_root / "generation_summary.json", summary)
    return 0 if all(x["status"] == "generated" for x in summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())
