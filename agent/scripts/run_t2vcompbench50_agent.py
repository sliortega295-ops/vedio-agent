#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_REMOTE = os.environ.get("SPARSEMAC_REMOTE", "local")
DEFAULT_PROJECT_DIR = os.environ.get("SPARSEMAC_PROJECT_DIR", str(PROJECT_ROOT))
DEFAULT_RUN_ROOT = os.environ.get("SPARSEMAC_RUN_ROOT", str(PROJECT_ROOT / "runs"))
DEFAULT_PROMPT_DIR = os.environ.get(
    "SPARSEMAC_T2VCOMPBENCH_PROMPT_DIR",
    str(REPOSITORY_ROOT / "package" / "third_party" / "T2V-CompBench" / "prompts"),
)
DEFAULT_PIPELINE = str(Path(__file__).with_name("sparse_to_full_vlm_pipeline.py"))
DEFAULT_LOG_ROOT = os.environ.get("SPARSEMAC_LOG_ROOT", str(PROJECT_ROOT / "logs"))
DEFAULT_VQQA_MANIFEST = os.environ.get(
    "SPARSEMAC_VQQA_MANIFEST",
    str(PROJECT_ROOT / "runs" / "baseline" / "vqqa-dense" / "manifest.json"),
)
CATEGORY_FILES = [
    "1_consistent_attr.txt",
    "2_dynamic_attr.txt",
    "3_spatial_relationship.txt",
    "4_motion_binding.txt",
    "5_action_binding.txt",
    "6_interaction.txt",
    "7_numeracy.txt",
]


class BatchError(RuntimeError):
    pass


def run(cmd: list[str], *, check: bool = True, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode != 0:
        raise BatchError(f"Command failed ({proc.returncode}): {' '.join(shlex.quote(p) for p in cmd)}\n{proc.stdout}")
    return proc


def remote_cmd(args: argparse.Namespace, cmd: str, *, check: bool = True) -> str:
    if args.remote in {"local", "localhost", "127.0.0.1", ""}:
        return run(["/bin/bash", "-lc", cmd], check=check).stdout
    return run(["ssh", args.remote, cmd], check=check).stdout


def remote_json(args: argparse.Namespace, path: str) -> Any:
    return json.loads(remote_cmd(args, f"cat {shlex.quote(path)}"))


def remote_exists(args: argparse.Namespace, path: str) -> bool:
    if args.remote in {"local", "localhost", "127.0.0.1", ""}:
        return run(["test", "-e", path], check=False).returncode == 0
    return run(["ssh", args.remote, f"test -e {shlex.quote(path)}"], check=False).returncode == 0


def remote_rm_rf(args: argparse.Namespace, path: str) -> None:
    if args.remote in {"local", "localhost", "127.0.0.1", ""}:
        run(["rm", "-rf", path], check=True)
    else:
        remote_cmd(args, f"rm -rf {shlex.quote(path)}")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:40] or "prompt"


def load_prompt_file(args: argparse.Namespace, file_name: str) -> list[tuple[int, str]]:
    path = f"{args.prompt_dir.rstrip('/')}/{file_name}"
    text = remote_cmd(args, f"cat {shlex.quote(path)}")
    prompts: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        prompt = line.strip()
        if prompt:
            prompts.append((line_no, prompt))
    if not prompts:
        raise BatchError(f"No prompts found in {path}")
    return prompts


def load_vqqa_subset_manifest(args: argparse.Namespace, batch_id: str) -> list[dict[str, Any]]:
    vqqa_data = remote_json(args, args.vqqa_manifest)
    
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in vqqa_data["items"]:
        cat = item["category_key"]
        by_cat.setdefault(cat, []).append(item)
        
    cat_mapping = {
        "consistent_attr": "1_consistent_attr",
        "dynamic_attr": "2_dynamic_attr",
        "spatial_relationship": "3_spatial_relationship",
        "motion_binding": "4_motion_binding",
        "action_binding": "5_action_binding",
        "interaction": "6_interaction",
        "numeracy": "7_numeracy"
    }
    
    sorted_cats = sorted(by_cat.keys(), key=lambda c: list(cat_mapping.keys()).index(c) if c in cat_mapping else 99)
    
    prompt_index = 0
    records = []
    for cat in sorted_cats:
        items = sorted(by_cat[cat], key=lambda x: x["line_number"])
        subset_start = args.vqqa_subset_start
        subset_end = subset_start + args.vqqa_subset_count
        subset = items[subset_start:subset_end]
        category_name = cat_mapping.get(cat, cat)
        source_file = f"{category_name}.txt"
        for item in subset:
            run_id = f"{batch_id}_p{prompt_index:03d}_{slugify(category_name)}"
            records.append({
                "index": prompt_index,
                "prompt_id": item.get("prompt_id", ""),
                "category": category_name,
                "source_file": source_file,
                "source_line": item["line_number"],
                "prompt": item["prompt"],
                "run_id": run_id,
            })
            prompt_index += 1
            
    return records


def load_input_manifest(args: argparse.Namespace, batch_id: str) -> list[dict[str, Any]]:
    """Normalize an existing benchmark/ablation manifest for this batch.

    This keeps matched subsets (for example the fixed difficult-175 split)
    identical across methods instead of resampling prompts.
    """
    data = remote_json(args, args.input_manifest)
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise BatchError(f"Input manifest has no items: {args.input_manifest}")
    category_prefix = {
        "consistent_attr": "1_consistent_attr",
        "dynamic_attr": "2_dynamic_attr",
        "spatial_relationship": "3_spatial_relationship",
        "motion_binding": "4_motion_binding",
        "action_binding": "5_action_binding",
        "interaction": "6_interaction",
        "numeracy": "7_numeracy",
    }
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not str(item.get("prompt") or "").strip():
            raise BatchError(f"Invalid item {index} in input manifest: {args.input_manifest}")
        category_key = str(item.get("category_key") or item.get("category") or "unknown")
        category = category_prefix.get(category_key, category_key)
        source_file = str(item.get("source_file") or f"{category}.txt")
        source_line = item.get("line_number", item.get("source_line", 0))
        run_id = f"{batch_id}_p{index:03d}_{slugify(category)}"
        records.append(
            {
                "index": index,
                "prompt_id": str(item.get("prompt_id") or ""),
                "category": category,
                "category_key": category_key,
                "source_file": source_file,
                "source_line": int(source_line or 0),
                "prompt": str(item["prompt"]).strip(),
                "run_id": run_id,
                "input_manifest": args.input_manifest,
            }
        )
    return records


def build_manifest(args: argparse.Namespace, batch_id: str) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    prompt_index = 0
    for category_index, file_name in enumerate(CATEGORY_FILES):
        allocation = 8 if category_index == 0 else 7
        prompts = load_prompt_file(args, file_name)
        if len(prompts) < allocation:
            raise BatchError(f"{file_name} has {len(prompts)} prompts, need {allocation}")
        sampled = sorted(rng.sample(prompts, allocation), key=lambda item: item[0])
        category = Path(file_name).stem
        for line_no, prompt in sampled:
            run_id = f"{batch_id}_p{prompt_index:03d}_{slugify(category)}"
            records.append({
                "index": prompt_index,
                "category": category,
                "source_file": file_name,
                "source_line": line_no,
                "prompt": prompt,
                "run_id": run_id,
            })
            prompt_index += 1
    if len(records) != args.prompt_count:
        raise BatchError(f"Expected {args.prompt_count} sampled prompts, got {len(records)}")
    return records


def gpu_snapshot(args: argparse.Namespace) -> list[dict[str, int]]:
    output = remote_cmd(
        args,
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits",
        check=False,
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"index": int(parts[0]), "memory_used_mib": int(parts[1]), "util_percent": int(parts[2])})
    return rows


def wait_for_gpus(args: argparse.Namespace) -> None:
    if not args.wait_for_gpus:
        return
    wanted = {int(item) for item in args.sparse_gpus.split(",") if item.strip()}
    wanted.add(int(args.full_gpu))
    while True:
        rows = gpu_snapshot(args)
        indexed = {row["index"]: row for row in rows}
        ready = True
        for gpu_id in wanted:
            row = indexed.get(gpu_id)
            if row is None or row["memory_used_mib"] > args.max_gpu_memory_mib or row["util_percent"] > args.max_gpu_util:
                ready = False
                break
        if ready:
            return
        print(f"[batch] waiting for GPUs {sorted(wanted)}; snapshot={rows}", flush=True)
        time.sleep(args.gpu_poll_interval)


def pipeline_command(args: argparse.Namespace, record: dict[str, Any], batch_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        args.pipeline,
        "--prompt",
        record["prompt"],
        "--run-id",
        record["run_id"],
        "--remote",
        args.remote,
        "--project-dir",
        args.project_dir,
        "--run-root",
        args.run_root,
        "--candidates",
        str(args.candidates),
        "--max-rounds",
        str(args.max_rounds),
        "--sparse-steps",
        str(args.steps),
        "--full-steps",
        str(args.steps),
        "--sparse-gpus",
        args.sparse_gpus,
        "--full-gpu",
        args.full_gpu,
        "--full-gpus",
        args.full_gpus,
        "--sparse-stride",
        str(args.sparse_stride),
        "--sparse-warmup-ratio",
        str(args.sparse_warmup_ratio),
        "--agent-method",
        args.agent_method,
        "--luna-model",
        args.luna_model,
    ]
    if args.luna_base_url:
        cmd.extend(["--luna-base-url", args.luna_base_url])
    if args.vlm_workers is not None:
        cmd.extend(["--vlm-workers", str(args.vlm_workers)])
    if args.plan_dir:
        prompt_id = str(record.get("prompt_id") or "").strip()
        if prompt_id:
            cmd.extend(["--plan-json", f"{args.plan_dir.rstrip('/')}/{prompt_id}.json"])
    return cmd


def summarize_report(args: argparse.Namespace, record: dict[str, Any], batch_dir: Path, *, status_override: str | None = None, error: str = "") -> dict[str, Any]:
    report_path = f"{args.run_root}/{record['run_id']}/pipeline_report.json"
    timings_path = f"{args.run_root}/{record['run_id']}/timings.json"
    report: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    if remote_exists(args, report_path):
        report = remote_json(args, report_path)
    if remote_exists(args, timings_path):
        timings = remote_json(args, timings_path)
    totals = timings.get("totals", {}) if isinstance(timings, dict) else {}
    rounds = report.get("rounds") or []
    return {
        "index": record["index"],
        "prompt_id": record.get("prompt_id", ""),
        "category": record["category"],
        "source_file": record["source_file"],
        "source_line": record["source_line"],
        "run_id": record["run_id"],
        "prompt": record["prompt"],
        "method_id": report.get("method_id", args.agent_method),
        "method_label": report.get("method_label", ""),
        "status": status_override or report.get("status", "missing_report"),
        "decision": report.get("decision", ""),
        "round_count": len(rounds),
        "sparse_vlm_best_candidate": report.get("sparse_vlm_best_candidate", ""),
        "sparse_vlm_score": report.get("sparse_vlm_score", ""),
        "full_score": report.get("full_score", ""),
        "full_video_path": report.get("full_video_path", ""),
        "planner_wall_seconds": totals.get("planner_wall_seconds", ""),
        "sparse_generation_wall_seconds": totals.get("sparse_generation_wall_seconds", ""),
        "sparse_vlm_judge_wall_seconds": totals.get("sparse_vlm_judge_wall_seconds", ""),
        "deepseek_refine_wall_seconds": totals.get("deepseek_refine_wall_seconds", ""),
        "full_generation_wall_seconds": totals.get("full_generation_wall_seconds", ""),
        "final_vlm_judge_wall_seconds": totals.get("final_vlm_judge_wall_seconds", ""),
        "pipeline_wall_seconds": timings.get("pipeline_wall_seconds", totals.get("pipeline_wall_seconds", "")),
        "batch_prompt_wall_seconds": "",
        "error": error,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(batch_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(batch_dir / "batch_summary.json", rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (batch_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SparseMAC VLM-evaluated agent over 50 balanced T2V-CompBench prompts.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--prompt-dir", default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--batch-id")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--prompt-count", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sparse-gpus", default="0,1,2,3")
    parser.add_argument("--full-gpu", default="0")
    parser.add_argument("--full-gpus", default="0,1")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--keep-partial-runs", action="store_true", help="Do not delete incomplete remote run dirs before retrying.")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--max-gpu-memory-mib", type=int, default=4096)
    parser.add_argument("--max-gpu-util", type=int, default=10)
    parser.add_argument("--gpu-poll-interval", type=int, default=300)
    parser.add_argument("--vlm-workers", type=int, default=None)
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--luna-base-url", default=None)
    parser.add_argument("--sparse-stride", type=int, default=2)
    parser.add_argument("--sparse-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--plan-dir", help="Directory containing precomputed planner JSON files keyed by prompt_id.")
    parser.add_argument(
        "--input-manifest",
        help="Use an existing list/items manifest verbatim instead of sampling prompts.",
    )
    parser.add_argument(
        "--agent-method",
        choices=("legacy", "sparsemac-semantic-v2"),
        default="legacy",
        help="Use sparsemac-semantic-v2 for our explicitly labeled improved SparseMAC method.",
    )
    parser.add_argument("--use-vqqa-subset", action="store_true", help="Use prompts from VQQA manifest as subset.")
    parser.add_argument("--vqqa-manifest", default=DEFAULT_VQQA_MANIFEST)
    parser.add_argument("--vqqa-subset-start", type=int, default=0, help="Per-category start offset inside the VQQA manifest.")
    parser.add_argument("--vqqa-subset-count", type=int, default=25, help="Number of VQQA prompts per category to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.run_root).is_absolute():
        args.run_root = str(Path(args.project_dir) / args.run_root)
    batch_id = args.batch_id or f"t2vcompbench50_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = Path(args.log_root) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    if args.input_manifest:
        manifest = load_input_manifest(args, batch_id)
        args.prompt_count = len(manifest)
    elif args.use_vqqa_subset:
        manifest = load_vqqa_subset_manifest(args, batch_id)
        args.prompt_count = len(manifest)
    else:
        manifest = build_manifest(args, batch_id)
    write_json(batch_dir / "benchmark_manifest.json", manifest)
    print(f"[batch] manifest: {batch_dir / 'benchmark_manifest.json'}", flush=True)
    if args.sample_only:
        write_summary(batch_dir, [])
        return 0

    selected = [record for record in manifest if record["index"] >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]

    rows: list[dict[str, Any]] = []
    summary_path = batch_dir / "batch_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))

    completed_indices = {int(row["index"]) for row in rows if row.get("status") not in {"failed", "missing_report"}}
    for record in selected:
        run_dir = f"{args.run_root}/{record['run_id']}"
        report_path = f"{args.run_root}/{record['run_id']}/pipeline_report.json"
        if int(record["index"]) in completed_indices or remote_exists(args, report_path):
            print(f"[batch] skip completed p{record['index']:03d}: {record['run_id']}", flush=True)
            rows = [row for row in rows if int(row["index"]) != int(record["index"])]
            rows.append(summarize_report(args, record, batch_dir))
            write_summary(batch_dir, sorted(rows, key=lambda row: int(row["index"])))
            continue

        if remote_exists(args, run_dir) and not args.keep_partial_runs:
            print(f"[batch] remove partial run dir before retry: {run_dir}", flush=True)
            remote_rm_rf(args, run_dir)

        wait_for_gpus(args)
        cmd = pipeline_command(args, record, batch_dir)
        log_path = batch_dir / "pipeline_logs" / f"p{record['index']:03d}.log"
        print(f"[batch] run p{record['index']:03d} category={record['category']} run_id={record['run_id']}", flush=True)
        started = time.perf_counter()
        proc = run(cmd, check=False, log_path=log_path)
        wall = round(time.perf_counter() - started, 3)
        rows = [row for row in rows if int(row["index"]) != int(record["index"])]
        if proc.returncode in {0, 2} and remote_exists(args, report_path):
            row = summarize_report(args, record, batch_dir)
            row["batch_prompt_wall_seconds"] = wall
            rows.append(row)
        else:
            rows.append(summarize_report(args, record, batch_dir, status_override="failed", error=f"exit={proc.returncode}; log={log_path}"))
        write_summary(batch_dir, sorted(rows, key=lambda row: int(row["index"])))

    print(f"[batch] summary: {batch_dir / 'batch_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BatchError as exc:
        print(f"[batch] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
