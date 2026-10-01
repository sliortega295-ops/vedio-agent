"""Run `python -m arav.cli --help`. All paid/GPU actions require explicit commands."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .agents import JSONClient, LLMAgents, MockAgents, MockDriver
from .benchmarks import REGISTRY, export_video, normalize, prepare
from .core import Limits, Stage, Task, canonical, run, safe_id, write_json
from .evaluation import assemble, independent_judge
from .evolution import load_policy, propose, validate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_task(path, index=0):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return Task.parse(rows[index])
    return Task.parse(read(path))


def real_agents():
    return LLMAgents(JSONClient.from_env("ARAV_CONTROLLER"), JSONClient.from_env("ARAV_VERIFIER"),
                     audio_capable=os.getenv("ARAV_VERIFIER_AUDIO", "0") == "1")


def make_driver(name, config):
    if name == "mock":
        return MockDriver(config.get("fail_first", 1))
    if name == "longlive":
        from .longlive import LongLiveDriver
        return LongLiveDriver.from_config(config)
    if ":" not in name:
        raise ValueError("Backend must be mock, longlive, or local_module:factory")
    # Only an explicitly selected, user-installed local integration is imported.
    module, factory = name.split(":", 1)
    driver = getattr(importlib.import_module(module), factory)(config)
    for field in ("name", "mock", "native_ar", "joint_av", "reset", "snapshot", "restore", "sample", "commit", "synchronize"):
        if not hasattr(driver, field):
            raise ValueError(f"Driver missing {field}")
    return driver


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verified-prefix AR audio/video research prototype")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="CPU test double only; no video benchmark score")
    demo.add_argument("--out", type=Path, required=True)
    registry = sub.add_parser("bench-list")
    fetch = sub.add_parser("bench-prepare", help="Network downloads official prompts and locks provenance")
    fetch.add_argument("name", choices=REGISTRY)
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--ref", default="main")
    fetch.add_argument("--allow-update", action="store_true")
    imp = sub.add_parser("bench-import", help="Offline normalize already acquired official files")
    imp.add_argument("name", choices=["storyeval", "tempact", "avgen"])
    imp.add_argument("files", type=Path, nargs="+")
    imp.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan", help="Calls configured controller; review and freeze plan before comparisons")
    plan.add_argument("--task", type=Path, required=True)
    plan.add_argument("--index", type=int, default=0)
    plan.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--task", type=Path, required=True)
    execute.add_argument("--index", type=int, default=0)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--backend", required=True)
    execute.add_argument("--config", type=Path)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--method", choices=["vanilla", "retry", "feedback", "evolved"], default="feedback")
    execute.add_argument("--policy", type=Path)
    execute.add_argument("--seed", type=int, default=0)
    execute.add_argument("--attempts", type=int, default=3)
    execute.add_argument("--max-candidates", type=int, default=128)
    execute.add_argument("--wall-seconds", type=float, default=3600)
    execute.add_argument("--threshold", type=float, default=.8)
    export = sub.add_parser("assemble")
    export.add_argument("run_dir", type=Path)
    copy = sub.add_parser("bench-export")
    copy.add_argument("--task", type=Path, required=True)
    copy.add_argument("--index", type=int, default=0)
    copy.add_argument("--video", type=Path, required=True)
    copy.add_argument("--out", type=Path, required=True)
    judge = sub.add_parser("judge-proxy", help="Independent sampled-frame proxy, NOT official scoring")
    judge.add_argument("--task", type=Path, required=True)
    judge.add_argument("--index", type=int, default=0)
    judge.add_argument("--references", type=Path, required=True)
    judge.add_argument("--video", type=Path, required=True)
    judge.add_argument("--out", type=Path, required=True)
    evolve = sub.add_parser("propose-rules")
    evolve.add_argument("runs", type=Path, nargs="+")
    evolve.add_argument("--out", type=Path, required=True)
    gate = sub.add_parser("validate-rules")
    gate.add_argument("--proposal", type=Path, required=True)
    gate.add_argument("--pairs", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    gate.add_argument("--min-pairs", type=int, default=20)
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        result = run(Task("smoke-av", "A hand rings a bell, then rests.", True, "smoke"),
                     [Stage("bell", "Ring bell then rest", "A hand rings a bell, then rests.",
                            "A bell rings once then fades.", nominal_chunks=2, max_chunks=3)],
                     MockDriver(), MockAgents(), args.out)
    elif args.cmd == "bench-list":
        result = REGISTRY
    elif args.cmd == "bench-prepare":
        result = prepare(args.name, args.out, args.ref, args.allow_update)
    elif args.cmd == "bench-import":
        if args.out.exists():
            raise FileExistsError(args.out)
        rows = normalize(args.name, args.files)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(canonical(asdict(t)) + "\n" for t in rows), encoding="utf-8")
        result = {"tasks": len(rows), "out": str(args.out), "official_score": None}
    elif args.cmd == "plan":
        if args.out.exists():
            raise FileExistsError(args.out)
        client = JSONClient.from_env("ARAV_CONTROLLER")
        stages = LLMAgents(client, client).plan(read_task(args.task, args.index))
        result = {"stages": [asdict(s) for s in stages]}
        write_json(args.out, result)
        write_json(args.out.with_suffix(".usage.json"), client.usage)
    elif args.cmd == "run":
        task = read_task(args.task, args.index)
        stages = [Stage.parse(s) for s in read(args.plan)["stages"]]
        if args.policy and args.method != "evolved":
            raise ValueError("--policy requires --method evolved")
        if args.method == "evolved" and not args.policy:
            raise ValueError("Evolved condition requires an independently validated --policy")
        rules = load_policy(args.policy, task.id, task.split) if args.policy else []
        started = time.perf_counter()
        driver = make_driver(args.backend, read(args.config) if args.config else {})
        driver.synchronize()
        load_s = time.perf_counter() - started
        agents = MockAgents() if driver.mock else real_agents()
        result = run(task, stages, driver, agents, args.out, method=args.method, seed=args.seed,
                     limits=Limits(args.attempts, args.max_candidates, args.wall_seconds, args.threshold), rules=rules)
        result["model_load_wall_s"] = load_s
        result["including_model_load_wall_s"] = result["wall_s"] + load_s
        write_json(args.out / "summary.json", result)
        if not driver.mock:
            write_json(args.out / "model_calls.json", {"controller": agents.controller.usage, "verifier": agents.verifier.usage})
    elif args.cmd == "assemble":
        result = {"video": str(assemble(args.run_dir))}
    elif args.cmd == "bench-export":
        result = {"exported": str(export_video(read_task(args.task, args.index), args.video, args.out))}
    elif args.cmd == "judge-proxy":
        result = independent_judge(read_task(args.task, args.index), read(args.references), args.video,
                                   JSONClient.from_env("ARAV_FINAL"), args.out,
                                   audio_capable=os.getenv("ARAV_FINAL_AUDIO", "0") == "1")
    elif args.cmd == "propose-rules":
        result = propose(args.runs, JSONClient.from_env("ARAV_EVOLVER"), args.out)
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        result = validate(read(args.proposal), read(args.pairs), min_pairs=args.min_pairs)
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if isinstance(result, dict) and result.get("status") in {"error", "retry_exhausted", "budget_exhausted", "stage_horizon_exhausted", "controller_stopped"} else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
