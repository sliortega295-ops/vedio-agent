#!/usr/bin/env python3
"""Two-candidate serial-vs-two-stream benchmark for SVA N9/N6 previews.

The benchmark keeps two independent Wan pipelines resident in one process.
Each persistent worker thread owns one pipeline and one non-default CUDA
stream.  Model load, CUDA initialization, text encoding, preparation, and one
shape-specific warm-up are excluded.  Formal timing covers the five dense
warm-up denoising steps plus the remaining 45 sparse denoising steps; VAE
decode, encoding, and VLM work are intentionally excluded to match the SVA
paper's denoising-only efficiency contract.

No process on a GPU other than physical GPU 0 is launched.  A per-condition
nvidia-smi monitor marks the experiment invalid if any foreign compute PID
appears on GPU 0; it never kills an external process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party"))

PATTERNS: dict[str, dict[str, Any]] = {
    "N9": {
        "key_indices": [3, 4, 5, 8, 9, 10, 13, 14, 15],
        "rope_positions": list(range(9)),
        "layout": "three-block compact",
        "decoded_pixel_frames": 33,
    },
    "N6": {
        "key_indices": [0, 1, 2, 9, 10, 11],
        "rope_positions": list(range(6)),
        "layout": "dual-window compact",
        "decoded_pixel_frames": 21,
    },
}
GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-87a0e759-5e45-c617-9c53-e3c9479fd05f"
PROMPT_ID = "p00"
PROMPT = "A blue car drives past a white picket fence on a sunny day."
SEEDS = (0, 1)
SIZE_NAME = "832*480"
FRAME_NUM = 81
STEPS = 50
WARMUP_STEPS = 5
SHIFT = 8.0
GUIDANCE = 6.0
GIB = 1024**3
NVTX_ENABLED = os.environ.get("SVA_ENABLE_NVTX", "0") == "1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_output(argv: list[str], timeout: float = 10.0) -> str:
    result = subprocess.run(
        argv, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {argv!r}\n{result.stdout}")
    return result.stdout


def query_gpu_identity() -> dict[str, Any]:
    line = command_output([
        "nvidia-smi", "-i", str(GPU_INDEX),
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]).strip()
    fields = [part.strip() for part in line.split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected GPU identity row: {line!r}")
    return {
        "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
        "memory_used_mib": float(fields[3]), "memory_total_mib": float(fields[4]),
        "utilization_gpu_pct": float(fields[5]), "pstate": fields[6],
    }


def query_compute_pids() -> list[dict[str, Any]]:
    output = command_output([
        "nvidia-smi", "-i", str(GPU_INDEX),
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows: list[dict[str, Any]] = []
    for raw in output.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "process_name": parts[1],
                         "used_memory_mib": float(parts[2])})
        except ValueError:
            continue
    return rows


def reject_foreign_compute(allowed_pids: set[int]) -> list[dict[str, Any]]:
    rows = query_compute_pids()
    foreign = [row for row in rows if int(row["pid"]) not in allowed_pids]
    if foreign:
        raise RuntimeError(
            "GPU0 has a foreign compute process; aborting without killing it: "
            + json.dumps(foreign, ensure_ascii=False)
        )
    return rows


class Telemetry:
    HEADER = [
        "wall_time", "timestamp", "index", "uuid", "gpu_util_pct",
        "memory_util_pct", "memory_used_mib", "power_w", "sm_clock_mhz",
        "temperature_c", "pstate", "compute_pids", "foreign_pids",
    ]

    def __init__(self, path: Path, allowed_pids: set[int], interval_s: float = 0.5):
        self.path = path
        self.allowed_pids = set(allowed_pids)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu0-telemetry", daemon=True)
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.foreign: list[dict[str, Any]] = []

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.errors.append("telemetry thread did not stop")
        numeric = lambda key: [float(row[key]) for row in self.rows if row.get(key) not in (None, "")]
        def stat(key: str, op: str) -> float | None:
            values = numeric(key)
            if not values:
                return None
            return max(values) if op == "max" else statistics.fmean(values)
        return {
            "path": str(self.path), "samples": len(self.rows), "errors": self.errors,
            "foreign_process_observations": self.foreign,
            "interference_free": not self.foreign and not self.errors,
            "peak_whole_device_vram_mib": stat("memory_used_mib", "max"),
            "mean_gpu_util_pct": stat("gpu_util_pct", "mean"),
            "mean_power_w": stat("power_w", "mean"),
            "mean_sm_clock_mhz": stat("sm_clock_mhz", "mean"),
            "max_temperature_c": stat("temperature_c", "max"),
        }

    def _run(self) -> None:
        try:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.HEADER)
                writer.writeheader()
                while not self.stop_event.is_set():
                    try:
                        gpu_line = command_output([
                            "nvidia-smi", "-i", str(GPU_INDEX),
                            "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,"
                            "memory.used,power.draw,clocks.sm,temperature.gpu,pstate",
                            "--format=csv,noheader,nounits",
                        ], timeout=5).strip()
                        fields = [part.strip() for part in gpu_line.split(",")]
                        apps = query_compute_pids()
                        foreign = [row for row in apps if int(row["pid"]) not in self.allowed_pids]
                        if foreign:
                            self.foreign.extend(foreign)
                        row = {
                            "wall_time": utc_now(),
                            "timestamp": fields[0], "index": fields[1], "uuid": fields[2],
                            "gpu_util_pct": fields[3], "memory_util_pct": fields[4],
                            "memory_used_mib": fields[5], "power_w": fields[6],
                            "sm_clock_mhz": fields[7], "temperature_c": fields[8],
                            "pstate": fields[9],
                            "compute_pids": ";".join(str(app["pid"]) for app in apps),
                            "foreign_pids": ";".join(str(app["pid"]) for app in foreign),
                        }
                        self.rows.append(row)
                        writer.writerow(row)
                        handle.flush()
                    except Exception as exc:  # retain evidence; do not silently pass
                        self.errors.append(repr(exc))
                    self.stop_event.wait(self.interval_s)
        except Exception:
            self.errors.append(traceback.format_exc())


class ComputeGuard:
    """Watch GPU0 for foreign compute PIDs for the entire process lifetime."""

    def __init__(self, path: Path, allowed_pids: set[int], interval_s: float = 1.0):
        self.path = path
        self.allowed_pids = set(allowed_pids)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu0-compute-guard", daemon=True)
        self.observations: list[dict[str, Any]] = []
        self.foreign: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=10)
        return {
            "path": str(self.path), "observations": len(self.observations),
            "foreign_process_observations": self.foreign, "errors": self.errors,
            "interference_free": not self.foreign and not self.errors,
        }

    def assert_clean(self) -> None:
        if self.foreign or self.errors:
            raise RuntimeError(
                "global GPU0 compute guard observed interference: "
                + json.dumps({"foreign": self.foreign, "errors": self.errors}, ensure_ascii=False)
            )

    def _run(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as handle:
                while not self.stop_event.is_set():
                    try:
                        apps = query_compute_pids()
                        foreign = [
                            row for row in apps if int(row["pid"]) not in self.allowed_pids
                        ]
                        row = {"time": utc_now(), "compute_apps": apps, "foreign": foreign}
                        self.observations.append(row)
                        if foreign:
                            self.foreign.extend(foreign)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                    except Exception as exc:
                        self.errors.append(repr(exc))
                    self.stop_event.wait(self.interval_s)
        except Exception:
            self.errors.append(traceback.format_exc())


def make_scheduler(config: Any, device: Any, torch: Any) -> Any:
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(STEPS, device=device, shift=SHIFT)
    return scheduler


def target_geometry(pipeline: Any, size: tuple[int, int]) -> tuple[tuple[int, ...], int, int]:
    latent_frames = (FRAME_NUM - 1) // pipeline.vae_stride[0] + 1
    lat_h = size[1] // pipeline.vae_stride[1]
    lat_w = size[0] // pipeline.vae_stride[2]
    shape = (pipeline.vae.model.z_dim, latent_frames, lat_h, lat_w)
    tokens_per_frame = math.ceil(
        lat_h * lat_w / (pipeline.patch_size[1] * pipeline.patch_size[2])
    )
    return shape, tokens_per_frame * latent_frames, tokens_per_frame


def cfg_prediction(
    pipeline: Any, latent: Any, timestep: Any, context: Any,
    context_null: Any, seq_len: int, positions: Any,
    nvtx_base: str | None = None,
) -> Any:
    kwargs = {"seq_len": seq_len, "temporal_indices": positions}
    if nvtx_base is not None:
        import torch
        torch.cuda.nvtx.range_push(nvtx_base + "/conditional/dit")
    conditional = pipeline.model([latent], t=timestep, context=context, **kwargs)[0]
    if nvtx_base is not None:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(nvtx_base + "/unconditional/dit")
    unconditional = pipeline.model([latent], t=timestep, context=context_null, **kwargs)[0]
    if nvtx_base is not None:
        torch.cuda.nvtx.range_pop()
    return unconditional + GUIDANCE * (conditional - unconditional)


def scheduler_step(scheduler: Any, prediction: Any, step_t: Any, latent: Any, generator: Any) -> Any:
    return scheduler.step(
        prediction.unsqueeze(0), step_t, latent.unsqueeze(0),
        return_dict=False, generator=generator,
    )[0].squeeze(0)


def tensor_summary(tensor: Any, torch: Any, save_path: Path | None) -> dict[str, Any]:
    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw = cpu.numpy().tobytes(order="C")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu, save_path)
    return {
        "shape": list(cpu.shape), "dtype": str(cpu.dtype),
        "sha256_f32le": hashlib.sha256(raw).hexdigest(),
        "mean": float(cpu.mean().item()), "std": float(cpu.std().item()),
        "min": float(cpu.min().item()), "max": float(cpu.max().item()),
        "nan_count": int(torch.isnan(cpu).sum().item()),
        "inf_count": int(torch.isinf(cpu).sum().item()),
        "saved_tensor": str(save_path) if save_path is not None else None,
    }


@dataclass
class RequestResources:
    pipeline: Any
    stream: Any
    context: Any
    context_null: Any
    base_noise: Any
    seed: int


class PersistentWorker(threading.Thread):
    def __init__(self, worker_id: int, resources: RequestResources, torch: Any,
                 model_config: Any, device: Any, size: tuple[int, int],
                 result_queue: queue.Queue[Any]):
        super().__init__(name=f"sva-worker-{worker_id}", daemon=False)
        self.worker_id = worker_id
        self.resources = resources
        self.torch = torch
        self.model_config = model_config
        self.device = device
        self.size = size
        self.result_queue = result_queue
        self.commands: queue.Queue[Any] = queue.Queue()
        self.prepared: dict[str, Any] | None = None

    def run(self) -> None:
        while True:
            command = self.commands.get()
            if command["kind"] == "STOP":
                self.result_queue.put({"kind": "STOPPED", "worker_id": self.worker_id})
                return
            try:
                if command["kind"] == "PREPARE":
                    self._prepare(command)
                elif command["kind"] == "RUN":
                    self._execute(command)
                else:
                    raise RuntimeError(f"unknown worker command: {command['kind']!r}")
            except Exception:
                self.result_queue.put({
                    "kind": "ERROR", "worker_id": self.worker_id,
                    "condition_id": command.get("condition_id"),
                    "traceback": traceback.format_exc(),
                })

    def _prepare(self, command: dict[str, Any]) -> None:
        torch = self.torch
        pattern = command["pattern"]
        spec = PATTERNS[pattern]
        key_indices = torch.tensor(spec["key_indices"], dtype=torch.long, device=self.device)
        positions = torch.tensor(spec["rope_positions"], dtype=torch.long, device=self.device)
        shape, seq_full, seq_per_frame = target_geometry(self.resources.pipeline, self.size)
        if tuple(self.resources.base_noise.shape) != shape:
            raise RuntimeError("base noise shape changed")
        with torch.cuda.stream(self.resources.stream):
            latent_full = self.resources.base_noise.clone()
            warm_scheduler = make_scheduler(self.model_config, self.device, torch)
            sparse_scheduler = make_scheduler(self.model_config, self.device, torch)
        self.resources.stream.synchronize()
        self.prepared = {
            "condition_id": command["condition_id"], "pattern": pattern,
            "key_indices": key_indices, "positions": positions,
            "seq_full": seq_full, "seq_sparse": seq_per_frame * len(spec["key_indices"]),
            "latent_full": latent_full, "warm_scheduler": warm_scheduler,
            "sparse_scheduler": sparse_scheduler,
        }
        self.result_queue.put({
            "kind": "READY", "worker_id": self.worker_id,
            "condition_id": command["condition_id"],
        })

    def _execute(self, command: dict[str, Any]) -> None:
        torch = self.torch
        state = self.prepared
        if state is None or state["condition_id"] != command["condition_id"]:
            raise RuntimeError("worker RUN did not match prepared condition")
        start_event: threading.Event = command["start_event"]
        if not start_event.wait(timeout=120):
            raise TimeoutError("timed out waiting for synchronized release")
        service_start_ns = time.perf_counter_ns()
        stream = self.resources.stream
        cuda_start = torch.cuda.Event(enable_timing=True)
        warm_start = torch.cuda.Event(enable_timing=True)
        warm_end = torch.cuda.Event(enable_timing=True)
        sparse_start = torch.cuda.Event(enable_timing=True)
        sparse_end = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        nvtx_name = (
            f"SVA_{state['pattern']}_{command['mode']}_"
            f"request{self.worker_id}_block{command['block']}"
        )
        with torch.cuda.stream(stream):
            torch.cuda.nvtx.range_push(nvtx_name)
            cuda_start.record(stream)
            warm_start.record(stream)
            latent_full = state["latent_full"]
            generator = torch.Generator(device=self.device).manual_seed(self.resources.seed)
            with torch.cuda.amp.autocast(dtype=self.model_config.param_dtype), torch.no_grad():
                torch.cuda.nvtx.range_push("dense_warmup_5_steps")
                for local_step, step_t in enumerate(
                    state["warm_scheduler"].timesteps[:WARMUP_STEPS]
                ):
                    torch.cuda.nvtx.range_push(f"dense_step_{local_step:02d}")
                    prediction = cfg_prediction(
                        self.resources.pipeline, latent_full, torch.stack([step_t]),
                        self.resources.context, self.resources.context_null,
                        state["seq_full"],
                        torch.arange(latent_full.shape[1], device=self.device, dtype=torch.long),
                        (
                            f"sva/pattern={state['pattern']}/mode={command['mode']}/"
                            f"request={self.worker_id}/step={local_step}"
                            if NVTX_ENABLED else None
                        ),
                    )
                    latent_full = scheduler_step(
                        state["warm_scheduler"], prediction, step_t, latent_full, generator
                    )
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
                warm_end.record(stream)
                latent_sparse = latent_full[:, state["key_indices"]].clone()
                sparse_start.record(stream)
                torch.cuda.nvtx.range_push("sparse_denoise_45_steps")
                for global_step, step_t in enumerate(
                    state["sparse_scheduler"].timesteps[WARMUP_STEPS:],
                    start=WARMUP_STEPS,
                ):
                    if 20 <= global_step <= 24:
                        torch.cuda.nvtx.range_push(f"profile_sparse_step_{global_step:02d}")
                    prediction = cfg_prediction(
                        self.resources.pipeline, latent_sparse, torch.stack([step_t]),
                        self.resources.context, self.resources.context_null,
                        state["seq_sparse"], state["positions"],
                        (
                            f"sva/pattern={state['pattern']}/mode={command['mode']}/"
                            f"request={self.worker_id}/step={global_step}"
                            if NVTX_ENABLED else None
                        ),
                    )
                    latent_sparse = scheduler_step(
                        state["sparse_scheduler"], prediction, step_t, latent_sparse, generator
                    )
                    if 20 <= global_step <= 24:
                        torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
                sparse_end.record(stream)
            cuda_end.record(stream)
            torch.cuda.nvtx.range_pop()
        stream.synchronize()
        service_end_ns = time.perf_counter_ns()
        next_event = command.get("next_event")
        if next_event is not None:
            next_event.set()
        output = tensor_summary(latent_sparse, torch, command.get("save_path"))
        result = {
            "kind": "RESULT", "worker_id": self.worker_id,
            "condition_id": command["condition_id"], "pattern": state["pattern"],
            "mode": command["mode"], "block": command["block"],
            "seed": self.resources.seed,
            "service_start_ns": service_start_ns, "service_end_ns": service_end_ns,
            "service_latency_s": (service_end_ns - service_start_ns) / 1e9,
            "cuda_elapsed_s": cuda_start.elapsed_time(cuda_end) / 1000.0,
            "dense_warmup_cuda_s": warm_start.elapsed_time(warm_end) / 1000.0,
            "sparse_cuda_s": sparse_start.elapsed_time(sparse_end) / 1000.0,
            "stream_id": int(stream.cuda_stream), "output": output,
        }
        self.prepared = None
        del latent_sparse, latent_full, state
        self.result_queue.put(result)


def gather_messages(result_queue: queue.Queue[Any], condition_id: str, kind: str,
                    count: int, timeout_s: float = 1800.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    found: list[dict[str, Any]] = []
    while len(found) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {kind} in {condition_id}")
        message = result_queue.get(timeout=remaining)
        if message.get("kind") == "ERROR":
            raise RuntimeError(
                f"worker {message.get('worker_id')} failed in {condition_id}:\n"
                + str(message.get("traceback"))
            )
        if message.get("condition_id") != condition_id or message.get("kind") != kind:
            raise RuntimeError(f"unexpected worker protocol message: {message!r}")
        found.append(message)
    return sorted(found, key=lambda item: int(item["worker_id"]))


def run_condition(
    *, torch: Any, workers: list[PersistentWorker], result_queue: queue.Queue[Any],
    output_root: Path, pattern: str, mode: str, block: int, formal: bool,
    compute_guard: ComputeGuard, capture_with_cuda_profiler: bool = False,
) -> dict[str, Any]:
    condition_id = f"{pattern}_{mode}_b{block}_{uuid.uuid4().hex[:8]}"
    compute_guard.assert_clean()
    reject_foreign_compute({os.getpid()})
    for worker in workers:
        worker.commands.put({
            "kind": "PREPARE", "condition_id": condition_id, "pattern": pattern,
        })
    gather_messages(result_queue, condition_id, "READY", 2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    telemetry = Telemetry(
        output_root / "telemetry" / f"{condition_id}.csv",
        allowed_pids={os.getpid()},
    )
    telemetry.start()
    time.sleep(0.2)
    release_ns = time.perf_counter_ns()
    if capture_with_cuda_profiler:
        torch.cuda.cudart().cudaProfilerStart()
    parent_nvtx_open = False
    if NVTX_ENABLED:
        torch.cuda.nvtx.range_push(f"sva_capture_{pattern}_{mode}")
        parent_nvtx_open = True
    if mode == "PARALLEL":
        shared = threading.Event()
        for worker in workers:
            save_path = (
                output_root / "outputs" / f"{pattern}_{mode}_b{block}_r{worker.worker_id}.pt"
                if formal and block == 0 else None
            )
            worker.commands.put({
                "kind": "RUN", "condition_id": condition_id, "mode": mode,
                "block": block, "start_event": shared, "next_event": None,
                "save_path": save_path,
            })
        release_ns = time.perf_counter_ns()
        shared.set()
    elif mode == "SERIAL":
        first = threading.Event()
        second = threading.Event()
        for worker, start, following in (
            (workers[0], first, second), (workers[1], second, None),
        ):
            save_path = (
                output_root / "outputs" / f"{pattern}_{mode}_b{block}_r{worker.worker_id}.pt"
                if formal and block == 0 else None
            )
            worker.commands.put({
                "kind": "RUN", "condition_id": condition_id, "mode": mode,
                "block": block, "start_event": start, "next_event": following,
                "save_path": save_path,
            })
        release_ns = time.perf_counter_ns()
        first.set()
    else:
        raise ValueError(mode)
    try:
        results = gather_messages(result_queue, condition_id, "RESULT", 2)
    finally:
        if parent_nvtx_open:
            torch.cuda.nvtx.range_pop()
        if capture_with_cuda_profiler:
            torch.cuda.cudart().cudaProfilerStop()
    pair_end_ns = max(int(item["service_end_ns"]) for item in results)
    telemetry_summary = telemetry.stop()
    torch.cuda.synchronize()
    allocator_peak = int(torch.cuda.max_memory_allocated())
    reserved_peak = int(torch.cuda.max_memory_reserved())
    foreign_after = [
        row for row in query_compute_pids() if int(row["pid"]) != os.getpid()
    ]
    if foreign_after:
        telemetry_summary["foreign_process_observations"].extend(foreign_after)
        telemetry_summary["interference_free"] = False
    record = {
        "schema_version": 1, "condition_id": condition_id, "formal": formal,
        "pattern": pattern, "pattern_spec": PATTERNS[pattern],
        "mode": mode, "block": block, "started_at": utc_now(),
        "release_ns": release_ns, "pair_end_ns": pair_end_ns,
        "pair_makespan_s": (pair_end_ns - release_ns) / 1e9,
        "requests": results,
        "fairness_ratio": (
            max(float(item["service_latency_s"]) for item in results)
            / min(float(item["service_latency_s"]) for item in results)
        ),
        "allocator_peak_bytes": allocator_peak,
        "allocator_peak_gib": allocator_peak / GIB,
        "allocator_reserved_peak_bytes": reserved_peak,
        "allocator_reserved_peak_gib": reserved_peak / GIB,
        "telemetry": telemetry_summary,
        "valid": bool(telemetry_summary["interference_free"]),
        "profiler_enabled": capture_with_cuda_profiler,
    }
    append_jsonl(output_root / "raw_conditions.jsonl", record)
    if not record["valid"]:
        raise RuntimeError(
            "foreign GPU0 compute activity or telemetry failure invalidated condition "
            + condition_id
        )
    return record


def cv_pct(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / statistics.fmean(values) * 100.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_gain_ci(blocks: list[dict[str, Any]], samples: int = 20000) -> list[float]:
    rng = random.Random(20260809)
    gains: list[float] = []
    for _ in range(samples):
        draw = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        serial = statistics.median(float(item["SERIAL"]["pair_makespan_s"]) for item in draw)
        parallel = statistics.median(float(item["PARALLEL"]["pair_makespan_s"]) for item in draw)
        gains.append(1.0 - parallel / serial)
    return [percentile(gains, 0.025), percentile(gains, 0.975)]


def summarize_pattern(pattern: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    by_block: dict[int, dict[str, Any]] = {}
    for record in records:
        by_block.setdefault(int(record["block"]), {})[record["mode"]] = record
    blocks = [
        {"block": index, **modes}
        for index, modes in sorted(by_block.items())
        if set(modes) == {"SERIAL", "PARALLEL"}
    ]
    serial = [float(block["SERIAL"]["pair_makespan_s"]) for block in blocks]
    parallel = [float(block["PARALLEL"]["pair_makespan_s"]) for block in blocks]
    tserial = statistics.median(serial)
    tpair = statistics.median(parallel)
    serial_req = {
        wid: [float(block["SERIAL"]["requests"][wid]["service_latency_s"]) for block in blocks]
        for wid in (0, 1)
    }
    parallel_req = {
        wid: [float(block["PARALLEL"]["requests"][wid]["service_latency_s"]) for block in blocks]
        for wid in (0, 1)
    }
    slowdown = {
        str(wid): statistics.median(parallel_req[wid]) / statistics.median(serial_req[wid])
        for wid in (0, 1)
    }
    hashes = {
        f"{mode}_request{wid}": sorted({
            block[mode]["requests"][wid]["output"]["sha256_f32le"]
            for block in blocks
        })
        for mode in ("SERIAL", "PARALLEL") for wid in (0, 1)
    }
    cross_mode_hash_match = {
        str(wid): set(hashes[f"SERIAL_request{wid}"]) == set(hashes[f"PARALLEL_request{wid}"])
        for wid in (0, 1)
    }
    pair_gains = [
        1.0 - float(block["PARALLEL"]["pair_makespan_s"])
        / float(block["SERIAL"]["pair_makespan_s"])
        for block in blocks
    ]
    peak_vram = max(
        float(block[mode]["telemetry"]["peak_whole_device_vram_mib"] or 0)
        for block in blocks for mode in ("SERIAL", "PARALLEL")
    )
    return {
        "pattern": pattern, "pattern_spec": PATTERNS[pattern],
        "paired_blocks": len(blocks), "raw_serial_makespan_s": serial,
        "raw_parallel_makespan_s": parallel,
        "Tserial2_s": tserial, "Tpair_s": tpair,
        "Rconc": tpair / tserial, "Sconc": tserial / tpair,
        "pair_gain_fraction": 1.0 - tpair / tserial,
        "pair_gain_pct": (1.0 - tpair / tserial) * 100.0,
        "pair_gain_paired_raw": pair_gains,
        "pair_gain_bootstrap_95ci_fraction": bootstrap_gain_ci(blocks),
        "pair_throughput_per_s": 2.0 / tpair,
        "serial_makespan_cv_pct": cv_pct(serial),
        "parallel_makespan_cv_pct": cv_pct(parallel),
        "serial_request_latency_median_s": {
            str(wid): statistics.median(values) for wid, values in serial_req.items()
        },
        "parallel_request_latency_median_s": {
            str(wid): statistics.median(values) for wid, values in parallel_req.items()
        },
        "per_request_slowdown": slowdown,
        "parallel_fairness_ratio_median": statistics.median(
            float(block["PARALLEL"]["fairness_ratio"]) for block in blocks
        ),
        "parallel_fairness_ratio_max": max(
            float(block["PARALLEL"]["fairness_ratio"]) for block in blocks
        ),
        "peak_whole_device_vram_mib": peak_vram,
        "output_hashes": hashes,
        "cross_mode_hash_match": cross_mode_hash_match,
        "correctness_passed": all(cross_mode_hash_match.values()),
        "all_conditions_interference_free": all(
            block[mode]["telemetry"]["interference_free"]
            for block in blocks for mode in ("SERIAL", "PARALLEL")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patterns", nargs="+", choices=sorted(PATTERNS), default=["N9", "N6"])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--max-blocks-on-noise", type=int, default=5)
    parser.add_argument(
        "--profile-once", action="store_true",
        help="after one excluded warm-up, run one cudaProfilerApi-delimited condition",
    )
    parser.add_argument(
        "--profile-mode", choices=("SERIAL", "PARALLEL"), default="PARALLEL",
        help="schedule used only by --profile-once; formal benchmark behavior is unchanged",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this experiment requires exact CUDA_VISIBLE_DEVICES=0")
    if args.blocks < 3 or args.max_blocks_on_noise < args.blocks:
        raise ValueError("need at least three paired blocks")
    if args.profile_once and len(args.patterns) != 1:
        raise ValueError("--profile-once requires exactly one pattern")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_gpu = query_gpu_identity()
    if initial_gpu["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError(f"GPU0 UUID changed: {initial_gpu}")
    initial_compute = reject_foreign_compute(set())
    started_at = utc_now()
    compute_guard = ComputeGuard(
        output_root / "telemetry" / "global_compute_guard.jsonl",
        allowed_pids={os.getpid()},
    )
    compute_guard.start()

    try:
        import torch
        import wan
        from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
        from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    except Exception:
        compute_guard.stop()
        raise

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.synchronize()
    post_cuda_gpu = query_gpu_identity()
    if post_cuda_gpu["uuid"] != EXPECTED_GPU_UUID:
        raise RuntimeError("logical CUDA initialization changed the physical GPU0 identity")
    model_config = WAN_CONFIGS["t2v-1.3B"]
    if model_config.param_dtype is not torch.bfloat16:
        raise RuntimeError("Wan model config is not BF16")
    if not FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError(
            f"frozen FA2 backend not active: FA2={FLASH_ATTN_2_AVAILABLE}, FA3={FLASH_ATTN_3_AVAILABLE}"
        )
    device = torch.device("cuda:0")
    size = SIZE_CONFIGS[SIZE_NAME]
    load_seconds: list[float] = []
    pipelines: list[Any] = []
    for index in range(2):
        start = time.perf_counter()
        pipeline = wan.WanT2V(
            config=model_config, checkpoint_dir=str(args.checkpoint),
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
            use_usp=False, t5_cpu=True,
        )
        torch.cuda.synchronize()
        load_seconds.append(time.perf_counter() - start)
        pipelines.append(pipeline)
        print(f"pipeline {index} loaded in {load_seconds[-1]:.3f}s", flush=True)
    streams = [torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)]
    if int(streams[0].cuda_stream) == int(streams[1].cuda_stream):
        raise RuntimeError("CUDA stream identities unexpectedly match")
    contexts: list[tuple[Any, Any]] = []
    for index, pipeline in enumerate(pipelines):
        start = time.perf_counter()
        with torch.no_grad():
            context = [item.to(device) for item in pipeline.text_encoder([PROMPT], torch.device("cpu"))]
            context_null = [
                item.to(device)
                for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))
            ]
        torch.cuda.synchronize()
        contexts.append((context, context_null))
        print(f"pipeline {index} text contexts prepared in {time.perf_counter()-start:.3f}s", flush=True)
    shape, _, _ = target_geometry(pipelines[0], size)
    base_noises = [
        torch.randn(
            shape, dtype=torch.float32, device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        for seed in SEEDS
    ]
    torch.cuda.synchronize()
    # PyTorch 2.4 lazily initializes the CUDA linalg wrapper on first use.
    # Initialize it once on the main thread so the first parallel UniPC
    # scheduler step cannot race in torch.linalg.solve. This is excluded
    # setup and does not alter the measured denoising workload.
    _linalg_a = torch.eye(2, dtype=torch.float32, device=device)
    _linalg_b = torch.ones((2, 1), dtype=torch.float32, device=device)
    torch.linalg.solve(_linalg_a, _linalg_b)
    torch.cuda.synchronize()
    del _linalg_a, _linalg_b
    result_queue: queue.Queue[Any] = queue.Queue()
    workers = [
        PersistentWorker(
            worker_id=index,
            resources=RequestResources(
                pipeline=pipelines[index], stream=streams[index],
                context=contexts[index][0], context_null=contexts[index][1],
                base_noise=base_noises[index], seed=SEEDS[index],
            ),
            torch=torch, model_config=model_config, device=device, size=size,
            result_queue=result_queue,
        )
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    environment = {
        "started_at": started_at, "hostname": platform.node(), "pid": os.getpid(),
        "python": sys.version, "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_initial": initial_gpu, "initial_compute_processes": initial_compute,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "wan_source": str(Path(wan.__file__).resolve()),
        "flash_attention_2_available": FLASH_ATTN_2_AVAILABLE,
        "flash_attention_3_available": FLASH_ATTN_3_AVAILABLE,
        "model_load_seconds_excluded": load_seconds,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    atomic_json(output_root / "environment.json", environment)
    records_by_pattern: dict[str, list[dict[str, Any]]] = {}
    warmups: list[dict[str, Any]] = []
    try:
        if args.profile_once:
            pattern = args.patterns[0]
            profile_mode = args.profile_mode
            warmup = run_condition(
                torch=torch, workers=workers, result_queue=result_queue,
                output_root=output_root, pattern=pattern, mode=profile_mode,
                block=-1, formal=False, compute_guard=compute_guard,
            )
            profiled = run_condition(
                torch=torch, workers=workers, result_queue=result_queue,
                output_root=output_root, pattern=pattern, mode=profile_mode,
                block=0, formal=False, compute_guard=compute_guard,
                capture_with_cuda_profiler=True,
            )
            guard_summary = compute_guard.stop()
            result = {
                "status": "PROFILE_COMPLETED", "pattern": pattern,
                "excluded_warmup": warmup, "profiled_condition": profiled,
                "global_compute_guard": guard_summary,
                "note": "Profiler timing is diagnostic and is not a formal performance result.",
            }
            atomic_json(output_root / "profile_result.json", result)
            atomic_json(output_root / "status.json", {
                "status": "PROFILE_COMPLETED", "finished_at": utc_now(),
                "result": str(output_root / "profile_result.json"),
            })
            return 0
        for pattern in args.patterns:
            print(f"excluded warm-up: {pattern}", flush=True)
            warmups.append(run_condition(
                torch=torch, workers=workers, result_queue=result_queue,
                output_root=output_root, pattern=pattern, mode="PARALLEL",
                block=-1, formal=False, compute_guard=compute_guard,
            ))
            records: list[dict[str, Any]] = []
            block = 0
            target_blocks = args.blocks
            while block < target_blocks:
                order = ("SERIAL", "PARALLEL") if block % 2 == 0 else ("PARALLEL", "SERIAL")
                for mode in order:
                    print(f"formal {pattern} block={block} mode={mode}", flush=True)
                    records.append(run_condition(
                        torch=torch, workers=workers, result_queue=result_queue,
                        output_root=output_root, pattern=pattern, mode=mode,
                        block=block, formal=True, compute_guard=compute_guard,
                    ))
                    atomic_json(output_root / "partial_result.json", {
                        "status": "RUNNING", "environment": environment,
                        "warmups": warmups, "records": records_by_pattern | {pattern: records},
                    })
                block += 1
                if block == args.blocks:
                    provisional = summarize_pattern(pattern, records)
                    if max(
                        provisional["serial_makespan_cv_pct"],
                        provisional["parallel_makespan_cv_pct"],
                    ) > 5.0:
                        target_blocks = args.max_blocks_on_noise
                        print(
                            f"{pattern} CV>5%; extending to {target_blocks} paired blocks",
                            flush=True,
                        )
            records_by_pattern[pattern] = records
        summaries = {
            pattern: summarize_pattern(pattern, records)
            for pattern, records in records_by_pattern.items()
        }
        final = {
            "schema_version": 1, "status": "COMPLETED",
            "started_at": started_at, "finished_at": utc_now(),
            "experiment": "SVA_N9_N6_TWO_CANDIDATE_CONCURRENCY",
            "track": "SPARSE_PREVIEW_SYSTEM_ORACLE",
            "environment": environment,
            "workload": {
                "model": "Wan2.1-T2V-1.3B", "input_frames": FRAME_NUM,
                "resolution": SIZE_NAME, "denoise_steps": STEPS,
                "dense_warmup_steps": WARMUP_STEPS,
                "guidance_scale": GUIDANCE, "shift": SHIFT,
                "precision": "bfloat16", "attention_backend": "flash_attention_2",
                "prompt_id": PROMPT_ID, "prompt": PROMPT,
                "request_seeds": list(SEEDS), "patterns": PATTERNS,
            },
            "timing_contract": {
                "includes": ["5 dense warm-up DiT steps", "45 sparse DiT steps"],
                "excludes": [
                    "CUDA initialization", "model load", "text encoding",
                    "first shape warm-up", "scheduler/noise preparation",
                    "VAE decode", "video encoding", "VLM",
                ],
                "profiler_enabled": False,
                "serial_and_parallel_use_same_two_pipelines": True,
                "parallel_uses_two_non_default_cuda_streams": True,
            },
            "excluded_warmups": warmups,
            "summaries": summaries,
            "global_compute_guard": compute_guard.stop(),
            "historical_full_reference": {
                "source": "auto_research E013, different prompt-pair contract",
                "Rconc": 0.984497680, "pair_gain_pct": 1.550232,
                "comparison_scope": (
                    "context only; not a paired claim because E013 used p00/p01 "
                    "whereas this SVA experiment uses one prompt with seeds 0/1"
                ),
            },
        }
        atomic_json(output_root / "result.json", final)
        atomic_json(output_root / "status.json", {
            "status": "COMPLETED", "finished_at": final["finished_at"],
            "result": str(output_root / "result.json"),
        })
        print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        guard_summary = compute_guard.stop()
        atomic_json(output_root / "status.json", {
            "status": "FAILED", "finished_at": utc_now(), "error": repr(exc),
            "traceback": traceback.format_exc(), "global_compute_guard": guard_summary,
        })
        raise
    finally:
        for worker in workers:
            worker.commands.put({"kind": "STOP"})
        stopped = 0
        deadline = time.monotonic() + 30
        while stopped < len(workers) and time.monotonic() < deadline:
            try:
                message = result_queue.get(timeout=1)
            except queue.Empty:
                continue
            if message.get("kind") == "STOPPED":
                stopped += 1
        for worker in workers:
            worker.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
