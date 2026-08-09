#!/usr/bin/env python3
"""Generate matched 8-seed sparse RoPE candidates for test-time scaling.

The generator has no planner, analyzer, rewrite logic, or agent prompt. It
uses the original benchmark prompt and the same eight seeds as a dense run.
For each prompt/seed, one full-frame warm-up is shared by the selected RoPE
branches; only the temporal RoPE positions differ afterward.

Supported patterns:
  n6_three_block: [0, 1, 2, 9, 10, 11]
  n9_three_block: [3, 4, 5, 8, 9, 10, 13, 14, 15]

Each sparse output is decoded natively by the VAE. No interpolation or frame
repetition is performed. Use ``prepare`` first, then one ``worker`` per GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

PATTERNS = {
    "n6_three_block": [0, 1, 2, 9, 10, 11],
    "n9_three_block": [3, 4, 5, 8, 9, 10, 13, 14, 15],
}
VARIANTS = ("gapped", "compact", "affine")
WAN_MODEL_LABELS = {
    "t2v-1.3B": "Wan2.1-T2V-1.3B",
    "t2v-14B": "Wan2.1-T2V-14B",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def selected_items(items: list[dict[str, Any]], prompts_per_category: int) -> list[dict[str, Any]]:
    if prompts_per_category <= 0:
        return items
    counts: Counter[str] = Counter()
    selected = []
    for item in items:
        category = str(item["category_key"])
        if counts[category] >= prompts_per_category:
            continue
        selected.append(item)
        counts[category] += 1
    categories = {str(item["category_key"]) for item in items}
    if categories != set(counts) or any(counts[c] != prompts_per_category for c in categories):
        raise ValueError(f"could not select {prompts_per_category} prompts/category: {dict(counts)}")
    return selected


def rope_positions(key_indices: list[int], variant: str) -> list[float | int]:
    if variant == "gapped":
        return list(key_indices)
    if variant == "compact":
        return list(range(len(key_indices)))
    if variant == "affine":
        span = key_indices[-1] - key_indices[0]
        return [
            (index - key_indices[0]) * (len(key_indices) - 1) / span
            for index in key_indices
        ]
    raise ValueError(f"unknown RoPE variant: {variant}")


def variant_root(root: Path, pattern: str, variant: str) -> Path:
    return root / pattern / variant


def candidate_relpath(item: dict[str, Any], candidate: dict[str, Any]) -> Path:
    return Path("videos") / str(item["category_key"]) / str(item["prompt_id"]) / str(candidate["candidate_id"]) / "video.mp4"


def make_manifest(source: dict[str, Any], items: list[dict[str, Any]], pattern: str, variant: str, model_key: str) -> dict[str, Any]:
    root_items = []
    for item in items:
        candidates = []
        for candidate in item["candidates"]:
            rel = candidate_relpath(item, candidate)
            candidates.append({
                "candidate_id": candidate["candidate_id"],
                "candidate_index": int(candidate["candidate_index"]),
                "seed": int(candidate["seed"]),
                "video_relpath": str(rel),
                "metadata_relpath": str(rel.with_name("metadata.json")),
                "log_relpath": str(rel.with_name("generate.log")),
            })
        root_items.append({
            "prompt_id": item["prompt_id"],
            "category_key": item["category_key"],
            "category_index": item.get("category_index"),
            "source_file": item.get("source_file"),
            "metadata_file": item.get("metadata_file"),
            "line_number": item["line_number"],
            "eval_video_name": item.get("eval_video_name", f"{int(item['line_number']):04d}.mp4"),
            "prompt": item["prompt"],
            "generation_prompt": item["prompt"],
            "candidates": candidates,
        })
    key_indices = PATTERNS[pattern]
    return {
        "version": 1,
        "created_at": now(),
        "run_root": "",
        "source": "standalone sparse RoPE test-time scaling",
        "source_dense_manifest": str(source.get("run_root", "")),
        "model": WAN_MODEL_LABELS[model_key],
        "baseline": f"{pattern}-{variant}-sparse-random8",
        "config": {
            "pattern": pattern,
            "key_indices": key_indices,
            "rope_variant": variant,
            "rope_positions": rope_positions(key_indices, variant),
            "candidates": 8,
            "frame_num_input": 81,
            "decoded_pixel_frames": (len(key_indices) - 1) * 4 + 1,
            "sample_steps": 50,
            "warmup_steps": 5,
            "sample_shift": 8.0,
            "sample_guide_scale": 6.0,
            "wan_model": model_key,
            "native_sparse_decode": True,
            "interpolation": False,
        },
        "categories": source.get("categories", []),
        "items": root_items,
    }


def prepare(args: argparse.Namespace) -> int:
    source = read_json(args.dense_manifest)
    items = selected_items(source["items"], args.prompts_per_category)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    experiment = {
        "version": 1,
        "created_at": now(),
        "source_dense_manifest": str(args.dense_manifest.resolve()),
        "model": WAN_MODEL_LABELS[args.wan_model],
        "patterns": {},
        "variants": list(args.variants),
        "prompts": len(items),
        "prompt_seed_pairs": sum(len(item["candidates"][: args.seeds_per_prompt or None]) for item in items),
    }
    for pattern in args.patterns:
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern: {pattern}")
        experiment["patterns"][pattern] = {}
        for variant in args.variants:
            variant_items = []
            for item in items:
                limited = dict(item)
                limited["candidates"] = item["candidates"][: args.seeds_per_prompt or None]
                variant_items.append(limited)
            manifest = make_manifest(source, variant_items, pattern, variant, args.wan_model)
            out = variant_root(root, pattern, variant)
            manifest["run_root"] = str(out)
            write_json(out / "manifest.json", manifest)
            experiment["patterns"][pattern][variant] = {
                "run_root": str(out),
                "videos": sum(len(item["candidates"]) for item in variant_items),
            }
    write_json(root / "experiment_manifest.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0


def valid_output(out: Path, item: dict[str, Any], candidate: dict[str, Any], pattern: str, variant: str) -> bool:
    video = out / candidate["video_relpath"]
    metadata = out / candidate["metadata_relpath"]
    if not video.is_file() or video.stat().st_size < 1024 or not metadata.is_file():
        return False
    try:
        row = read_json(metadata)
    except Exception:
        return False
    return (
        row.get("status") == "ok"
        and row.get("pattern") == pattern
        and row.get("rope_variant") == variant
        and row.get("prompt_id") == item["prompt_id"]
        and row.get("candidate_id") == candidate["candidate_id"]
        and int(row.get("seed", -1)) == int(candidate["seed"])
    )


def make_scheduler(config: Any, device: Any, steps: int, shift: float) -> Any:
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device, shift=shift)
    return scheduler


def target_geometry(pipeline: Any, size: tuple[int, int], frame_num: int, latent_frames: int) -> tuple[tuple[int, ...], int, int]:
    lat_h = size[1] // pipeline.vae_stride[1]
    lat_w = size[0] // pipeline.vae_stride[2]
    shape = (pipeline.vae.model.z_dim, latent_frames, lat_h, lat_w)
    tokens_per_frame = math.ceil(lat_h * lat_w / (pipeline.patch_size[1] * pipeline.patch_size[2]))
    return shape, tokens_per_frame * latent_frames, tokens_per_frame


def cfg_prediction(pipeline: Any, latent: Any, timestep: Any, context: Any, context_null: Any, seq_len: int, positions: Any) -> Any:
    kwargs = {"seq_len": seq_len, "temporal_indices": positions}
    conditional = pipeline.model([latent], t=timestep, context=context, **kwargs)[0]
    unconditional = pipeline.model([latent], t=timestep, context=context_null, **kwargs)[0]
    return unconditional + 6.0 * (conditional - unconditional)


def scheduler_step(scheduler: Any, prediction: Any, step_t: Any, latent: Any, generator: Any) -> Any:
    return scheduler.step(
        prediction.unsqueeze(0), step_t, latent.unsqueeze(0), return_dict=False, generator=generator
    )[0].squeeze(0)


def generate_worker(args: argparse.Namespace) -> int:
    import torch
    import torch.cuda.amp as amp
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import cache_video

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    root = args.output_root
    base = variant_root(root, args.pattern, args.variants[0])
    base_manifest = read_json(base / "manifest.json")
    items = base_manifest["items"]
    tasks = [(item, candidate) for item in items for candidate in item["candidates"]]
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    pending = []
    for item, candidate in tasks:
        missing = [
            variant for variant in args.variants
            if not valid_output(variant_root(root, args.pattern, variant), item, candidate, args.pattern, variant)
        ]
        if missing:
            pending.append((item, candidate, missing))
    logging.info("pattern=%s shard=%d/%d assigned=%d pending=%d", args.pattern, args.shard_id, args.num_shards, len(tasks), len(pending))
    if not pending:
        return 0

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    config = WAN_CONFIGS[args.wan_model]
    size = SIZE_CONFIGS[args.size]
    key_indices = PATTERNS[args.pattern]
    latent_frames = (args.frame_num - 1) // 4 + 1
    pipeline = wan.WanT2V(
        config=config,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    shape, seq_full, seq_per_frame = target_geometry(pipeline, size, args.frame_num, latent_frames)
    seq_sparse = seq_per_frame * len(key_indices)
    positions = {
        variant: torch.tensor(rope_positions(key_indices, variant), device=device,
                              dtype=torch.float64 if variant == "affine" else torch.long)
        for variant in args.variants
    }
    load_seconds = 0.0
    completed = failed = 0
    records = []
    for item, candidate, missing in pending:
        prompt = item["prompt"]
        context = pipeline.text_encoder([prompt], device)
        context_null = pipeline.text_encoder([pipeline.sample_neg_prompt], device)
        seed = int(candidate["seed"])
        unit_start = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device=device).manual_seed(seed)
            warm_scheduler = make_scheduler(config, device, args.steps, args.shift)
            latent_full = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
            warm_start = time.perf_counter()
            with amp.autocast(dtype=config.param_dtype), torch.no_grad():
                for step_t in warm_scheduler.timesteps[: args.warmup_steps]:
                    prediction = cfg_prediction(
                        pipeline, latent_full, torch.stack([step_t]), context, context_null, seq_full,
                        torch.arange(latent_frames, device=device, dtype=torch.long),
                    )
                    latent_full = scheduler_step(warm_scheduler, prediction, step_t, latent_full, generator)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warm_start
            for variant in missing:
                variant_start = time.perf_counter()
                sparse_scheduler = make_scheduler(config, device, args.steps, args.shift)
                branch_generator = torch.Generator(device=device).manual_seed(seed)
                latent_sparse = latent_full[:, key_indices].clone()
                with amp.autocast(dtype=config.param_dtype), torch.no_grad():
                    for step_t in sparse_scheduler.timesteps[args.warmup_steps:]:
                        prediction = cfg_prediction(
                            pipeline, latent_sparse, torch.stack([step_t]), context, context_null, seq_sparse,
                            positions[variant],
                        )
                        latent_sparse = scheduler_step(sparse_scheduler, prediction, step_t, latent_sparse, branch_generator)
                torch.cuda.synchronize()
                denoise_seconds = time.perf_counter() - variant_start
                decode_start = time.perf_counter()
                video = pipeline.vae.decode([latent_sparse])[0]
                torch.cuda.synchronize()
                decode_seconds = time.perf_counter() - decode_start
                decoded_frames = int(video.shape[1])
                out = variant_root(root, args.pattern, variant)
                video_path = out / candidate["video_relpath"]
                metadata_path = out / candidate["metadata_relpath"]
                video_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = video_path.with_name("video.tmp.mp4")
                cache_video(tensor=video[None], save_file=str(temporary), fps=config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                os.replace(temporary, video_path)
                record = {
                    "status": "ok",
                    "created_at": now(),
                    "model": WAN_MODEL_LABELS[args.wan_model],
                    "pattern": args.pattern,
                    "key_indices": key_indices,
                    "rope_variant": variant,
                    "rope_positions": rope_positions(key_indices, variant),
                    "prompt_id": item["prompt_id"],
                    "category_key": item["category_key"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_index": candidate["candidate_index"],
                    "seed": seed,
                    "prompt": prompt,
                    "frame_num_input": args.frame_num,
                    "decoded_pixel_frames": decoded_frames,
                    "warmup_steps": args.warmup_steps,
                    "warmup_shared_across_variants": True,
                    "warmup_seconds": warmup_seconds,
                    "sparse_denoise_seconds": denoise_seconds,
                    "vae_decode_seconds": decode_seconds,
                    "native_sparse_decode": True,
                    "interpolation": False,
                    "elapsed_seconds": time.perf_counter() - unit_start,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                write_json(metadata_path, record)
                records.append(record)
                completed += 1
                del latent_sparse, sparse_scheduler, video
            del latent_full, warm_scheduler, generator
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            failed += len(missing)
            logging.exception("failed %s/%s", item["prompt_id"], candidate["candidate_id"])
            for variant in missing:
                write_json(variant_root(root, args.pattern, variant) / candidate["metadata_relpath"], {
                    "status": "failed", "error": repr(exc), "pattern": args.pattern,
                    "rope_variant": variant, "prompt_id": item["prompt_id"],
                    "candidate_id": candidate["candidate_id"], "seed": seed,
                })
        del context, context_null
    write_json(root / args.pattern / f"generation_summary_shard_{args.shard_id:02d}.json", {
        "updated_at": now(), "pattern": args.pattern, "shard_id": args.shard_id,
        "num_shards": args.num_shards, "assigned": len(tasks), "pending_at_start": len(pending),
        "completed": completed, "failed": failed, "records": records,
    })
    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dense-manifest", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--patterns", nargs="+", choices=sorted(PATTERNS), default=list(PATTERNS))
    prep.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    prep.add_argument("--prompts-per-category", type=int, default=0)
    prep.add_argument("--seeds-per-prompt", type=int, default=8)
    prep.add_argument("--wan-model", choices=sorted(WAN_MODEL_LABELS), default="t2v-1.3B")
    run = sub.add_parser("worker")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--pattern", choices=sorted(PATTERNS), required=True)
    run.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    run.add_argument("--ckpt-dir", type=Path, required=True)
    run.add_argument("--wan-model", choices=sorted(WAN_MODEL_LABELS), default="t2v-1.3B")
    run.add_argument("--size", default="832*480")
    run.add_argument("--frame-num", type=int, default=81)
    run.add_argument("--steps", type=int, default=50)
    run.add_argument("--warmup-steps", type=int, default=5)
    run.add_argument("--shift", type=float, default=8.0)
    run.add_argument("--num-shards", type=int, default=8)
    run.add_argument("--shard-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    return generate_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
