#!/usr/bin/env python3
"""Shared-model two-stream baseline for SVA candidate concurrency.

Unlike the frozen benchmark, this experiment keeps only one Wan pipeline/model
resident and gives both candidate workers references to the same read-only
model. It isolates whether duplicate model residency/cache pressure contributed
to the negative two-pipeline result.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
from pathlib import Path
from typing import Any

from bench_sparse_candidate_concurrency import (
    ComputeGuard,
    EXPECTED_GPU_UUID,
    FRAME_NUM,
    GPU_INDEX,
    GUIDANCE,
    PATTERNS,
    PROMPT,
    RequestResources,
    SEEDS,
    SIZE_NAME,
    PersistentWorker,
    atomic_json,
    query_gpu_identity,
    reject_foreign_compute,
    run_condition,
    summarize_pattern,
    target_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patterns", nargs="+", choices=sorted(PATTERNS), default=["N9", "N6"])
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("shared-model baseline currently requires CUDA_VISIBLE_DEVICES=0")
    if args.blocks < 3:
        raise ValueError("use at least three paired blocks")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_gpu = query_gpu_identity()
    if initial_gpu["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError(
            "this reproducibility harness is pinned to the original L40 UUID; "
            f"got {initial_gpu}"
        )
    reject_foreign_compute(set())

    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.cuda.set_device(GPU_INDEX)
    torch.cuda.init()
    torch.cuda.synchronize()
    if not FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("expected the frozen FlashAttention-2 backend")

    model_config = WAN_CONFIGS["t2v-1.3B"]
    device = torch.device("cuda:0")
    size = SIZE_CONFIGS[SIZE_NAME]
    pipeline = wan.WanT2V(
        config=model_config,
        checkpoint_dir=str(args.checkpoint),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )
    # Avoid a first-forward race on WanModel.freqs: forward() lazily moves and
    # reassigns this tensor. Shared-model streams must enter timed concurrency
    # only after that one mutable setup action is complete.
    pipeline.model.freqs = pipeline.model.freqs.to(device)
    torch.cuda.synchronize()

    with torch.no_grad():
        context = [item.to(device) for item in pipeline.text_encoder([PROMPT], torch.device("cpu"))]
        context_null = [
            item.to(device)
            for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))
        ]
    shape, _, _ = target_geometry(pipeline, size)
    base_noises = [
        torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        for seed in SEEDS
    ]
    # Pre-initialize the UniPC CUDA linalg path before worker concurrency.
    a = torch.eye(2, dtype=torch.float32, device=device)
    b = torch.ones((2, 1), dtype=torch.float32, device=device)
    torch.linalg.solve(a, b)
    torch.cuda.synchronize()
    del a, b

    streams = [torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)]
    result_queue: queue.Queue[Any] = queue.Queue()
    workers = [
        PersistentWorker(
            worker_id=i,
            resources=RequestResources(
                pipeline=pipeline,  # intentionally shared
                stream=streams[i],
                context=context,
                context_null=context_null,
                base_noise=base_noises[i],
                seed=SEEDS[i],
            ),
            torch=torch,
            model_config=model_config,
            device=device,
            size=size,
            result_queue=result_queue,
        )
        for i in range(2)
    ]
    for worker in workers:
        worker.start()

    guard = ComputeGuard(
        output_root / "telemetry" / "global_compute_guard.jsonl",
        allowed_pids={os.getpid()},
    )
    guard.start()
    summaries: dict[str, Any] = {}
    records_all: dict[str, list[dict[str, Any]]] = {}
    warmups: list[dict[str, Any]] = []
    try:
        for pattern in args.patterns:
            warmups.append(
                run_condition(
                    torch=torch,
                    workers=workers,
                    result_queue=result_queue,
                    output_root=output_root,
                    pattern=pattern,
                    mode="PARALLEL",
                    block=-1,
                    formal=False,
                    compute_guard=guard,
                )
            )
            records: list[dict[str, Any]] = []
            for block in range(args.blocks):
                order = ("SERIAL", "PARALLEL") if block % 2 == 0 else ("PARALLEL", "SERIAL")
                for mode in order:
                    records.append(
                        run_condition(
                            torch=torch,
                            workers=workers,
                            result_queue=result_queue,
                            output_root=output_root,
                            pattern=pattern,
                            mode=mode,
                            block=block,
                            formal=True,
                            compute_guard=guard,
                        )
                    )
            records_all[pattern] = records
            summaries[pattern] = summarize_pattern(pattern, records)
    finally:
        for worker in workers:
            worker.commands.put({"kind": "STOP"})
        for worker in workers:
            worker.join(timeout=30)
        guard_summary = guard.stop()

    result = {
        "schema_version": 1,
        "experiment": "SVA_SHARED_MODEL_TWO_STREAMS",
        "note": "Both workers share one read-only Wan pipeline/model and use distinct CUDA streams.",
        "gpu": initial_gpu,
        "workload": {
            "model": "Wan2.1-T2V-1.3B",
            "input_frames": FRAME_NUM,
            "resolution": SIZE_NAME,
            "guidance": GUIDANCE,
            "seeds": list(SEEDS),
        },
        "excluded_warmups": warmups,
        "summaries": summaries,
        "global_compute_guard": guard_summary,
    }
    atomic_json(output_root / "result.json", result)
    print(json.dumps({"summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
