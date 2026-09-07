#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sparsemac.semantic_v2 import (  # noqa: E402
    LEGACY_METHOD_ID,
    METHOD_ID as SEMANTIC_V2_METHOD_ID,
    METHOD_LABEL as SEMANTIC_V2_METHOD_LABEL,
    SUPPORTED_METHODS,
    build_analyzer_prompt as build_semantic_v2_analyzer_prompt,
    build_verifier_prompt as build_semantic_v2_verifier_prompt,
    method_metadata,
    normalize_criteria,
    weighted_score as semantic_v2_weighted_score,
)
from sparsemac.luna_client import DEFAULT_LUNA_MODEL  # noqa: E402


DEFAULT_REMOTE = os.environ.get("SPARSEMAC_REMOTE", "local")
DEFAULT_PROJECT_DIR = os.environ.get("SPARSEMAC_PROJECT_DIR", str(PROJECT_ROOT))
DEFAULT_RUN_ROOT = os.environ.get("SPARSEMAC_RUN_ROOT", str(PROJECT_ROOT / "runs"))
DEFAULT_GENERATOR_ROOT = os.environ.get(
    "SPARSEMAC_GENERATOR_ROOT", str(PROJECT_ROOT / "generation_backend")
)
DEFAULT_PYTHON = os.environ.get("SPARSEMAC_PYTHON", sys.executable)
DEFAULT_CKPT_DIR = os.environ.get("SPARSEMAC_WAN_CHECKPOINT", "")
GENERATION_TIMING_PATTERNS = {
    "denoising_seconds": re.compile(r"Denoising:\s*([0-9.]+)s"),
    "vae_decode_seconds": re.compile(r"VAE decode:\s*([0-9.]+)s"),
    "generation_total_seconds": re.compile(r"Total:\s*([0-9.]+)s"),
}

STORYBOARD_API_PROMPT_TEMPLATE = """You are an objective visual evaluator. Judge only visible evidence in the attached storyboard image.
The storyboard has only 11 sampled frames. Do not demand evidence between sampled frames, but do judge whether the visible sampled frames support the required objects, attributes, relations, actions, state changes, and motion trend.
Do not reward style, plausibility, or prior expectations when the storyboard evidence fails the prompt.
Treat the 11-frame storyboard as your sole visual evidence. Do not infer missing objects, attributes, counts, actions, or spatial relations unless they are visually observable in the storyboard.
If evidence is ambiguous from the 11 frames, give a low score and explain that the storyboard evidence is insufficient.

CRITICAL MOTION RULE: If an active motion or action requirement (e.g., dashing, running quickly, jumping, flying, fluttering) is completely absent or if the objects are completely stationary/static, you must assign a vqa_score between 0 and 30.

CRITICAL QUALITY & INTEGRITY RULE: If the storyboard image exhibits severe generative glitches (e.g. flat color-block artifacts, distorted animal/human features, low-resolution pixelation, low-quality canvas, or large foreign shapes/noise blocks covering the subject), you MUST assign a vqa_score below 40 for all impacted VQA checks, regardless of the nominal entity counts.

Here is the original prompt the video was generated from:
"{original_prompt}"

Here are the specific VQA questions to check, along with their weights:
{formatted_vqa_list}

Instructions:
1. Score every VQA question exactly once on a 0-100 integer scale.
2. Compute the final overall 'score' as the weighted average of all vqa_scores divided by 100.

Return strict JSON only. The JSON must contain: score, vqa_results, failure_reasons, evidence, recommendation, success_or_failure_reason.
- The overall 'score' must be a normalized float from 0.0 to 1.0.
- 'vqa_results' must be a list of objects. For each result include: question, vqa_score (0-100 integer), weight, and reason.
- If vqa_score < 82, you MUST provide a concrete reason grounded in visual evidence. If vqa_score >= 82, set reason="".
"""


DEEPSEEK_REFINE_PROMPT_TEMPLATE = """You are SparseMAC's DeepSeek analyzer for VLM-guided prompt refinement.
You have access to the complete history of previous sparse optimization rounds (`round_history`) and the `current_best_candidate` which has the highest score so far.

Your task is divided into two distinct stages:

### STAGE 1: DECISION (Refine or Reject)
Evaluate whether continuing sparse prompt refinement is likely to yield further progress.
If the best candidate's VQA score has plateaued, or if prompt modifications in previous rounds have resulted in no improvement, or if the visual quality is getting worse, you should set `"decision": "reject"`.
`"decision": "reject"` means "Stop sparse refinement and proceed to generate the full video using the current best candidate". This is an early stopping mechanism to prevent wasting candidates on fruitless iterations.
If you believe there is a clear path to resolve the remaining low-score VQA checks by conservatively modifying the prompt, set `"decision": "refine"`.

### STAGE 2: REFINEMENT (Only if decision is "refine")
If `"decision": "refine"`, analyze the `current_best_candidate`'s `vqa_feedback`. Identify checks where `vqa_score` is less than 82 and read the `reason` carefully.
Propose exactly `candidate_budget` small, controlled prompt variations (for `next_prompts`) under the same seed.
Refinement guidelines:
1. Focus only on resolving low-score VQA checks (vqa_score < 82) using their specific reasons.
2. Maintain strict conservative refinement: preserve the original task, core objects, attributes, spatial relations, and motion trends. Do NOT add new unrelated objects, actions, stylistic effects, camera movements, or background scenes.
3. If `"decision": "reject"`, set `"next_prompt"` to null or empty string, and `"next_prompts"` to an empty list.

Return strict JSON only with keys:
- decision: one of ["refine", "reject"]
- reason: explanation of your decision (why you decided to refine or reject)
- failure_focus: list of the most important unresolved visual failures (vqa_score < 82) in the current best candidate
- rewrite_hints: list of concrete prompt rewrite hints to fix these failures (empty if decision is reject)
- next_prompt: one conservative English prompt for the next sparse round (empty if decision is reject)
- next_prompts: exactly candidate_budget distinct English prompt strings for the next sparse candidate set (empty if decision is reject)

Input JSON:
{input_json}"""


class PipelineError(RuntimeError):
    pass


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_normalized_score(value: Any, default: float = 0.0, *, field: str = "score") -> float:
    score = as_float(value, default)
    if score < 0.0 or score > 1.0:
        raise PipelineError(f"VLM {field} must be normalized from 0.0 to 1.0; got {value!r}")
    return score


def semantic_criteria_from_plan(plan_dict: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan_dict.get("semantic_criteria") or plan_dict.get("criteria")
    try:
        return normalize_criteria(raw)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{SEMANTIC_V2_METHOD_LABEL} plan has invalid frozen criteria: {exc}") from exc


def normalize_semantic_v2_judgment(
    plan_dict: dict[str, Any],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    """Make the frozen weighted criteria the sole score source for our method."""
    criteria = semantic_criteria_from_plan(plan_dict)
    try:
        score = semantic_v2_weighted_score(judgment.get("vqa_results"), criteria)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{SEMANTIC_V2_METHOD_LABEL} VLM reply is invalid: {exc}") from exc
    normalized = dict(judgment)
    normalized["score"] = score
    normalized["score_mode"] = "weighted_frozen_original_prompt_semantic_criteria"
    normalized["method_id"] = SEMANTIC_V2_METHOD_ID
    normalized["method_metadata"] = method_metadata(SEMANTIC_V2_METHOD_ID)
    normalized["semantic_criteria"] = criteria
    return normalized


def normalize_candidate_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if isinstance(v, dict)}
    if isinstance(value, list):
        result: dict[str, dict[str, Any]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate_id") or item.get("id")
            if cid is not None:
                result[str(cid)] = item
        return result
    return {}


def sparse_candidate_mapping(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = normalize_candidate_mapping(data.get("candidates", {}))
    if candidates:
        return candidates
    return normalize_candidate_mapping(data)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_local(cmd: list[str], *, check: bool = True, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode != 0:
        raise PipelineError(f"Command failed ({proc.returncode}): {shell_join(cmd)}\n{proc.stdout}")
    return proc


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def elapsed_since(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def parse_generation_timing_text(text: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "denoising_seconds": None,
        "vae_decode_seconds": None,
        "generation_total_seconds": None,
    }
    for key, pattern in GENERATION_TIMING_PATTERNS.items():
        match = pattern.search(text)
        if match:
            result[key] = float(match.group(1))
    return result


def remote_cmd(args: argparse.Namespace, cmd: str, *, check: bool = True) -> str:
    max_retries = 5
    for attempt in range(max_retries):
        if args.remote in {"local", "localhost", "127.0.0.1", ""}:
            proc = run_local(["/bin/bash", "-lc", cmd], check=False)
        else:
            proc = run_local(["ssh", args.remote, cmd], check=False)
        if proc.returncode == 255:
            print(f"[pipeline] SSH connection error (255) during attempt {attempt+1}/{max_retries}. Retrying in 5 seconds...")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
        if check and proc.returncode != 0:
            raise PipelineError(f"Remote command failed ({proc.returncode}): {cmd}\n{proc.stdout}")
        return proc.stdout
    raise PipelineError(f"Remote command failed (max retries reached): {cmd}")


def remote_json(args: argparse.Namespace, path: str) -> Any:
    text = remote_cmd(args, f"cat {shlex.quote(path)}")
    return json.loads(text)


def remote_files_exist(args: argparse.Namespace, paths: list[str]) -> bool:
    tests = " && ".join(f"test -f {shlex.quote(p)}" for p in paths)
    if args.remote in {"local", "localhost", "127.0.0.1", ""}:
        return run_local(["/bin/bash", "-lc", tests], check=False).returncode == 0
    return run_local(["ssh", args.remote, tests], check=False).returncode == 0


def write_remote_file(args: argparse.Namespace, path: str, content: str) -> None:
    code = f"""
from pathlib import Path
Path({path!r}).write_text({content!r}, encoding="utf-8")
"""
    remote_cmd(args, f"{shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")


def write_remote_json(args: argparse.Namespace, path: str, data: Any) -> None:
    write_remote_file(args, path, json.dumps(data, indent=2, ensure_ascii=False))


def remote_text(args: argparse.Namespace, path: str, *, check: bool = True) -> str:
    return remote_cmd(args, f"cat {shlex.quote(path)}", check=check)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def parse_sparse_vlm_result(data: dict[str, Any]) -> tuple[str, float]:
    raw_best = data.get("best_candidate")
    if isinstance(raw_best, dict):
        best_id = raw_best.get("candidate_id") or raw_best.get("id")
    else:
        best_id = raw_best
    candidates = sparse_candidate_mapping(data)

    def score_view(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        nested = value.get("judgment")
        if isinstance(nested, dict):
            return nested
        return value

    if best_id and best_id in candidates:
        best_value = score_view(candidates[best_id])
        if best_value:
            score = as_normalized_score(best_value.get("score", best_value.get("human_reasonable_score", 0.0)), field=f"{best_id}.score")
            return str(best_id), score

    best_pair: tuple[str, float] | None = None
    for cid, value in candidates.items():
        score_data = score_view(value)
        if not score_data:
            continue
        score = as_normalized_score(score_data.get("score", score_data.get("human_reasonable_score", 0.0)), field=f"{cid}.score")
        if best_pair is None or score > best_pair[1]:
            best_pair = (str(cid), score)
    if best_pair:
        return best_pair

    raise PipelineError("Could not find best_candidate in sparse visual judge JSON")


def nested_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get("judgment")
    if isinstance(nested, dict):
        return nested
    return value


def parse_final_vlm_result(data: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
    final_judgment = nested_judgment(data.get("final_video"))
    if not final_judgment:
        final_judgment = nested_judgment(data)

    score_value = final_judgment.get("score", final_judgment.get("human_reasonable_score"))
    if score_value is None:
        raise PipelineError("Final visual judge JSON is missing normalized score")
    score = as_normalized_score(score_value, field="final score")

    decision = str(final_judgment.get("decision") or data.get("decision") or "").strip()
    if not decision:
        raise PipelineError("Final visual judge JSON is missing decision")

    reason = str(
        final_judgment.get("success_or_failure_reason")
        or data.get("success_or_failure_reason")
        or ""
    ).strip()
    if not reason:
        raise PipelineError("Final visual judge JSON is missing success_or_failure_reason")

    return score, decision, final_judgment


def candidate_index(candidate_id: str) -> int | None:
    match = re.search(r"_c(\d+)$", str(candidate_id))
    if not match:
        return None
    return int(match.group(1))


def sparse_vlm_passes(args: argparse.Namespace, sparse_data: dict[str, Any], candidate_id: str, score: float) -> bool:
    decision = str(sparse_data.get("decision", "")).strip().lower()
    return score >= args.sparse_threshold and decision != "sparse_reject"


def best_candidate_context(args: argparse.Namespace, run_dir: str, sparse_data: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates_path = f"{run_dir}/candidates.json"
    candidates = remote_json(args, candidates_path)
    candidate: dict[str, Any] = {}
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
                candidate = item
                break
    judgments = sparse_candidate_mapping(sparse_data)
    judgment: dict[str, Any] = {}
    raw_judgment = judgments.get(candidate_id, {})
    if isinstance(raw_judgment, dict):
        nested = raw_judgment.get("judgment")
        judgment = nested if isinstance(nested, dict) else raw_judgment
    return {
        "candidate": candidate,
        "judgment": judgment,
        "sparse_vlm_decision": sparse_data.get("decision"),
    }


def cleanup_warmup_latents_after_full(
    args: argparse.Namespace,
    parent_run_dir: str,
    keep_run_dir: str,
    keep_candidate_id: str,
) -> None:
    code = f"""
from pathlib import Path
import json

parent_run_dir = Path({parent_run_dir!r})
keep_run_dir = str(Path({keep_run_dir!r}).resolve())
keep_candidate_id = {keep_candidate_id!r}

for candidates_path in parent_run_dir.glob("round_*/candidates.json"):
    round_dir = str(candidates_path.parent.resolve())
    candidates = json.loads(candidates_path.read_text())
    for candidate in candidates:
        if round_dir == keep_run_dir and candidate.get("candidate_id") == keep_candidate_id:
            continue
        path = candidate.get("warmup_latent_path")
        if path:
            Path(path).unlink(missing_ok=True)
"""
    remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")


def collect_sparse_generation_timings(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        video_path = candidate.get("sparse_video") or ""
        log_path = str(Path(video_path).with_suffix(".log")) if video_path else ""
        log_text = remote_text(args, log_path, check=False) if log_path else ""
        parsed = parse_generation_timing_text(log_text)
        items.append({
            "candidate_id": candidate.get("candidate_id"),
            "status": candidate.get("status"),
            "video_path": video_path,
            "log_path": log_path,
            "model_seconds": parsed.get("generation_total_seconds"),
            **parsed,
        })
    return items


def collect_full_generation_timing(args: argparse.Namespace, final_json: dict[str, Any]) -> dict[str, Any]:
    log_path = str(final_json.get("log") or "")
    log_text = remote_text(args, log_path, check=False) if log_path else ""
    parsed = parse_generation_timing_text(log_text)
    return {
        "log_path": log_path,
        "model_seconds": parsed.get("generation_total_seconds"),
        **parsed,
    }


def sum_seconds(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return round(sum(present), 3) if present else None


def build_timing_totals(timings: dict[str, Any]) -> dict[str, Any]:
    rounds = timings.get("rounds") or []
    totals = {
        "planner_wall_seconds": timings.get("planner", {}).get("wall_seconds"),
        "sparse_generation_wall_seconds": sum_seconds([r.get("sparse_generation", {}).get("wall_seconds") for r in rounds]),
        "sparse_vlm_judge_wall_seconds": sum_seconds([r.get("sparse_vlm_judge", {}).get("wall_seconds") for r in rounds]),
        "deepseek_refine_wall_seconds": sum_seconds([r.get("deepseek_refine", {}).get("wall_seconds") for r in rounds]),
        "full_generation_wall_seconds": sum_seconds([r.get("full_generation", {}).get("wall_seconds") for r in rounds]),
        "final_vlm_judge_wall_seconds": sum_seconds([r.get("final_vlm_judge", {}).get("wall_seconds") for r in rounds]),
        "round_total_wall_seconds": sum_seconds([r.get("round_total_wall_seconds") for r in rounds]),
    }
    total_values = [
        totals["planner_wall_seconds"],
        totals["round_total_wall_seconds"],
    ]
    totals["pipeline_wall_seconds"] = sum_seconds(total_values)
    return totals


def write_timings(args: argparse.Namespace, parent_run_dir: str, timings: dict[str, Any]) -> None:
    timings["updated_at"] = utc_now_iso()
    timings["totals"] = build_timing_totals(timings)
    write_remote_json(args, f"{parent_run_dir}/timings.json", timings)


def resolve_run_root(args: argparse.Namespace) -> str:
    run_root = Path(args.run_root)
    if run_root.is_absolute():
        return str(run_root)
    return str(Path(args.project_dir) / run_root)


def remote_make_plan(args: argparse.Namespace, run_id: str, prompt: str, candidates: int) -> dict[str, Any]:
    parent_dir = f"{resolve_run_root(args)}/{run_id}"
    code = f"""
import sys
sys.path.append({args.project_dir!r})
from sparsemac.planner import make_plan
from sparsemac.deepseek_client import DeepSeekClient
from sparsemac.luna_client import LunaClient
from sparsemac.gemini_client import GeminiClient
from sparsemac.utils import ensure_dir
import json
from pathlib import Path

parent_dir = Path({parent_dir!r})
ensure_dir(parent_dir)

# Initialize clients using .env.local or process environment. SemanticV2 uses
# Luna for planning; legacy keeps the historical DeepSeek path.
semantic_v2 = {args.agent_method == SEMANTIC_V2_METHOD_ID!r}
deepseek = None if semantic_v2 else DeepSeekClient(model={args.deepseek_model!r})
luna = LunaClient(model={args.luna_model!r}, base_url={args.luna_base_url!r}) if semantic_v2 else None
gemini = GeminiClient(model={args.gemini_model!r})

# Generate plan
plan = make_plan(
    {prompt!r},
    {candidates},
    gemini=gemini,
    deepseek=deepseek,
    luna=luna,
    strict=True,
    method={args.agent_method!r},
)
plan_dict = plan.to_dict()
if str(plan_dict.get("notes", "")).startswith("Heuristic fallback"):
    raise SystemExit("Planner fell back to heuristic unexpectedly")

# Save plan
(parent_dir / "plan.json").write_text(json.dumps(plan_dict, indent=2, ensure_ascii=False), encoding="utf-8")
print("PLAN_JSON_START")
print(json.dumps(plan_dict, indent=2, ensure_ascii=False))
print("PLAN_JSON_END")
"""
    output = remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")
    if "PLAN_JSON_START" in output:
        lines = output.splitlines()
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if "PLAN_JSON_START" in line:
                start_idx = i
            elif "PLAN_JSON_END" in line:
                end_idx = i
        if start_idx != -1 and end_idx != -1:
            json_text = "\n".join(lines[start_idx+1:end_idx])
            return json.loads(json_text)
    raise PipelineError(f"Remote planner failed to output plan JSON:\n{output}")


def remote_load_external_plan(args: argparse.Namespace, run_id: str, plan_json_path: str) -> dict[str, Any]:
    parent_dir = f"{resolve_run_root(args)}/{run_id}"
    data = remote_json(args, plan_json_path)
    if not isinstance(data, dict):
        raise PipelineError(f"External plan JSON is not an object: {plan_json_path}")
    plan_dict = data.get("plan") if isinstance(data.get("plan"), dict) else data
    if not isinstance(plan_dict, dict):
        raise PipelineError(f"External plan JSON has no usable plan object: {plan_json_path}")
    prompts = plan_dict.get("prompt_variants")
    vqa_scores = plan_dict.get("vqa_scores")
    if not isinstance(prompts, list) or not prompts:
        raise PipelineError(f"External plan missing prompt_variants: {plan_json_path}")
    if not isinstance(vqa_scores, dict) or not vqa_scores:
        raise PipelineError(f"External plan missing vqa_scores: {plan_json_path}")
    if args.agent_method == SEMANTIC_V2_METHOD_ID:
        plan_method = str(plan_dict.get("method_id") or "").strip()
        if plan_method != SEMANTIC_V2_METHOD_ID:
            raise PipelineError(
                f"External plan is not labeled {SEMANTIC_V2_METHOD_ID}: {plan_json_path}. "
                "Generate a fresh SemanticV2 plan instead of relabeling a legacy plan."
            )
        semantic_criteria_from_plan(plan_dict)
    code = f"""
from pathlib import Path
import json
parent_dir = Path({parent_dir!r})
parent_dir.mkdir(parents=True, exist_ok=True)
plan_dict = json.loads({json.dumps(plan_dict, ensure_ascii=False)!r})
(parent_dir / "plan.json").write_text(json.dumps(plan_dict, indent=2, ensure_ascii=False), encoding="utf-8")
print("PLAN_JSON_START")
print(json.dumps(plan_dict, indent=2, ensure_ascii=False))
print("PLAN_JSON_END")
"""
    output = remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")
    if "PLAN_JSON_START" not in output:
        raise PipelineError(f"Failed to persist external plan JSON:\n{output}")
    return plan_dict


def remote_run_sparse_batch(
    args: argparse.Namespace,
    run_id: str,
    round_index: int,
    original_prompt: str,
    prompts: list[str],
    candidates_count: int,
    seed_override: Optional[int] = None,
) -> list[dict[str, Any]]:
    code = f"""
import sys
sys.path.append({args.project_dir!r})
from sparsemac.schema import Candidate
from sparsemac.generator import run_sparse_batch
from sparsemac.gpu import detect_idle_gpus
from sparsemac.planner import seed_pool
from sparsemac.vision import make_storyboard
from sparsemac.utils import ensure_dir, write_json
import json
from pathlib import Path

run_id = {run_id!r}
round_index = {round_index}
original_prompt = {original_prompt!r}
prompts = {json.dumps(prompts, ensure_ascii=False)}
candidates_count = {candidates_count}
generator_root = {args.generator_root!r}
ckpt_dir = {args.ckpt_dir!r}
python_bin = {args.python_bin!r}
size = {args.size!r}
sample_steps = {args.sparse_steps}
sample_shift = {args.sample_shift}
guide_scale = {args.guide_scale}
warmup_ratio = {args.sparse_warmup_ratio}
warmup_steps = {args.sparse_warmup_steps}
timeout = {args.generation_timeout}
gpu_util_limit = {args.gpu_util_limit}
sparse_stride = {args.sparse_stride}

seed_override = {seed_override!r}

# 1. generate seeds
if seed_override is not None:
    seeds = [seed_override] * candidates_count
else:
    seeds = seed_pool(original_prompt + json.dumps(prompts), round_index, candidates_count)

# 2. setup directories and candidates
run_dir = Path({resolve_run_root(args)!r}) / run_id
round_dir = ensure_dir(run_dir / f"round_{{round_index}}")
cand_dir = ensure_dir(round_dir / "candidates")

candidates = []
for idx in range(candidates_count):
    prompt = prompts[idx % len(prompts)]
    seed = seeds[idx % len(seeds)]
    cid = f"r{{round_index:02d}}_c{{idx:02d}}"
    out_dir = ensure_dir(cand_dir / cid)
    candidates.append(
        Candidate(
            candidate_id=cid,
            round_index=round_index,
            prompt=prompt,
            seed=seed,
            sparse_video=str(out_dir / "key_only.mp4"),
            storyboard=str(out_dir / "storyboard_11.jpg"),
        )
    )

# 3. determine gpus to use from config
gpu_ids = [int(g) for g in f"{args.sparse_gpus}".split(",") if g.strip()]
if not gpu_ids:
    gpu_ids = [0]


# 4. run batch
candidates = run_sparse_batch(
    candidates,
    gpu_ids=gpu_ids,
    generator_root=generator_root,
    ckpt_dir=ckpt_dir,
    python_bin=python_bin,
    size=size,
    sample_steps=sample_steps,
    sample_shift=sample_shift,
    guide_scale=guide_scale,
    warmup_ratio=warmup_ratio,
    warmup_steps=warmup_steps,
    timeout=timeout,
    dry_run=False,
    sparse_stride=sparse_stride,
)

# 5. generate storyboards for each completed candidate
for cand in candidates:
    if cand.status == "ok" and Path(cand.sparse_video).exists():
        make_storyboard(cand.sparse_video, cand.storyboard, frames=11)

# 6. write candidates.json
write_json(round_dir / "candidates.json", [c.to_dict() for c in candidates])
print("CANDIDATES_JSON_START")
print(json.dumps([c.to_dict() for c in candidates], indent=2, ensure_ascii=False))
print("CANDIDATES_JSON_END")
"""
    output = remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && CUDA_VISIBLE_DEVICES={shlex.quote(args.sparse_gpus)} {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")
    if "CANDIDATES_JSON_START" in output:
        lines = output.splitlines()
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if "CANDIDATES_JSON_START" in line:
                start_idx = i
            elif "CANDIDATES_JSON_END" in line:
                end_idx = i
        if start_idx != -1 and end_idx != -1:
            json_text = "\n".join(lines[start_idx+1:end_idx])
            return json.loads(json_text)
    raise PipelineError(f"Remote sparse batch run failed:\n{output}")


def prepare_next_round_prompts(refine_plan: dict[str, Any], best_prompt: str, count: int) -> list[str]:
    next_prompts = refine_plan.get("next_prompts", [])
    if not isinstance(next_prompts, list):
        next_prompts = []
    next_prompts = [str(p).strip() for p in next_prompts if str(p).strip()]
    
    next_prompt = str(refine_plan.get("next_prompt") or "").strip()
    
    # 1. 优先 next_prompts
    prompts = list(next_prompts)
    
    # 2. 如果不足，用 next_prompt 补齐 (插在前面)
    if next_prompt and next_prompt not in prompts:
        prompts.insert(0, next_prompt)
        
    # 3. 仍不足时，保守重复 next_prompt 或当前 best prompt，凑够 count
    fallback_prompt = next_prompt or best_prompt
    while len(prompts) < count:
        prompts.append(fallback_prompt)
        
    return prompts[:count]


def run_deepseek_refine(
    args: argparse.Namespace,
    *,
    run_dir: str,
    original_prompt: str,
    plan_dict: dict[str, Any],
    round_history: list[dict[str, Any]],
    current_best_candidate: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "original_prompt": original_prompt,
        "candidate_budget": args.candidates,
        "sparse_threshold": args.sparse_threshold,
        "frozen_semantic_plan": plan_dict.get("semantic_plan", {}),
        "frozen_semantic_criteria": plan_dict.get("semantic_criteria", []),
        "must_preserve": plan_dict.get("must_preserve", []),
        "round_history": round_history,
        "current_best_candidate": current_best_candidate,
    }
    refine_prompt = (
        build_semantic_v2_analyzer_prompt(payload)
        if args.agent_method == SEMANTIC_V2_METHOD_ID
        else DEEPSEEK_REFINE_PROMPT_TEMPLATE.format(
            input_json=json.dumps(payload, indent=2, ensure_ascii=False)
        )
    )
    payload_json = json.dumps(payload, ensure_ascii=False)
    code = f"""
from pathlib import Path
import json
import time

from sparsemac.deepseek_client import DeepSeekClient
from sparsemac.luna_client import LunaClient


payload = json.loads({payload_json!r})
run_dir = Path({run_dir!r})
out_path = run_dir / "vlm_refine_plan.json"

semantic_v2 = {args.agent_method == SEMANTIC_V2_METHOD_ID!r}
client = (
    LunaClient(model={args.luna_model!r}, base_url={args.luna_base_url!r})
    if semantic_v2
    else DeepSeekClient(model={args.analyzer_model!r})
)
if not client.available:
    raise SystemExit(client.last_error or "Analyzer is unavailable")

prompt = {refine_prompt!r}

parsed = None
raw = ""
invalid_replies = []
for attempt in range(3):
    parsed, raw = client.generate_json(
        prompt,
        max_tokens=7000 if semantic_v2 else 4500,
        temperature=0.2 if attempt == 0 else 0.0,
    )
    if isinstance(parsed, dict):
        break
    invalid_replies.append(str(raw or client.last_error or "")[:2000])
    time.sleep(1.5 * (attempt + 1))
if not isinstance(parsed, dict):
    (run_dir / "vlm_refine_plan_invalid_raw.txt").write_text("\\n---ATTEMPT---\\n".join(invalid_replies), encoding="utf-8")
    if semantic_v2:
        raise SystemExit("SparseMAC-SemanticV2 analyzer failed closed after three invalid JSON replies")
    feedback = payload.get("current_best_candidate", {{}}).get("vqa_feedback", []) or []
    failed = []
    for item in feedback:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("vqa_score", 100))
        except Exception:
            score = 100
        if score < 82:
            q = str(item.get("question") or "").strip()
            reason = str(item.get("reason") or "").strip()
            failed.append((q + (": " + reason if reason else "")).strip())
    original = str(payload.get("original_prompt") or "").strip()
    best_prompt = str(payload.get("current_best_candidate", {{}}).get("prompt") or original).strip()
    focus = failed[:3]
    hint = "; make the failed prompt-critical requirement clearly visible: " + "; ".join(focus) if focus else "; make the prompt-critical relation/action clearly visible"
    candidate_budget = max(1, int(payload.get("candidate_budget") or 1))
    fallback_prompts = []
    for base in [original, best_prompt, original + hint, best_prompt + hint]:
        base = base.strip()
        if base and base not in fallback_prompts:
            fallback_prompts.append(base)
    while len(fallback_prompts) < candidate_budget:
        fallback_prompts.append(best_prompt or original)
    parsed = {{
        "decision": "refine",
        "reason": "Analyzer returned invalid JSON after retries; using deterministic conservative refine fallback to keep the completion job running.",
        "failure_focus": focus,
        "rewrite_hints": focus,
        "next_prompt": fallback_prompts[0],
        "next_prompts": fallback_prompts[:candidate_budget],
    }}


def normalize_prompt_list(raw_value):
    if not isinstance(raw_value, list):
        return []
    prompts = []
    for item in raw_value:
        if isinstance(item, dict):
            text = item.get("prompt") or item.get("next_prompt") or item.get("text")
        else:
            text = item
        text = str(text or "").strip()
        if text and text not in prompts:
            prompts.append(text)
    return prompts


def normalize_string_list(raw_value):
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value or "").strip()
    return [text] if text else []


decision = str(parsed.get("decision") or "refine").strip().lower()
if decision not in {{"refine", "reject"}}:
    decision = "refine"
candidate_budget = int(payload.get("candidate_budget") or 1)
candidate_budget = max(1, candidate_budget)

if decision == "reject":
    next_prompts = []
    next_prompt = ""
else:
    next_prompts = normalize_prompt_list(parsed.get("next_prompts"))
    next_prompt = str(parsed.get("next_prompt") or parsed.get("selected_prompt") or "").strip()
    if not next_prompt and next_prompts:
        next_prompt = next_prompts[0]
    if next_prompt and next_prompt not in next_prompts:
        next_prompts = [next_prompt] + next_prompts
    fallback_prompt = next_prompt or str(payload.get("current_best_candidate", {{}}).get("prompt") or payload.get("original_prompt") or "").strip()
    while len(next_prompts) < candidate_budget and fallback_prompt:
        next_prompts.append(fallback_prompt)
    next_prompts = next_prompts[:candidate_budget]
    if not next_prompt and next_prompts:
        next_prompt = next_prompts[0]

if semantic_v2 and decision == "refine":
    distinct_prompts = []
    for item in next_prompts:
        if item not in distinct_prompts:
            distinct_prompts.append(item)
    if len(distinct_prompts) != candidate_budget:
        raise SystemExit(
            f"SparseMAC-SemanticV2 analyzer must return {{candidate_budget}} distinct complete prompts; "
            f"got {{len(distinct_prompts)}}"
        )
    next_prompts = distinct_prompts
    next_prompt = next_prompts[0]

result = {{
    "decision": decision,
    "reason": str(parsed.get("reason") or "").strip(),
    "failure_focus": normalize_string_list(parsed.get("failure_focus")),
    "rewrite_hints": normalize_string_list(parsed.get("rewrite_hints")),
    "next_prompt": next_prompt,
    "next_prompts": next_prompts,
    "source": "sparsemac_semantic_v2_luna_analyzer" if semantic_v2 else "deepseek_vlm_refine",
    "method_id": {args.agent_method!r},
    "method_metadata": {method_metadata(args.agent_method)!r},
    "analyzer_model": client.model,
    "best_candidate_id": payload.get("current_best_candidate", {{}}).get("candidate_id"),
    "sparse_vlm_score": payload.get("current_best_candidate", {{}}).get("score"),
    "input_summary": payload,
    "analyzer_raw": parsed,
}}
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
"""
    remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")
    plan = remote_json(args, f"{run_dir}/vlm_refine_plan.json")
    if not isinstance(plan, dict):
        raise PipelineError(f"Analyzer refine plan is not a JSON object: {run_dir}/vlm_refine_plan.json")
    if plan.get("decision") == "refine" and not str(plan.get("next_prompt") or "").strip():
        raise PipelineError(f"Analyzer did not produce a usable refine prompt despite deciding to refine: {run_dir}/vlm_refine_plan.json")
    return plan


def run_full_from_candidate(args: argparse.Namespace, run_dir: str, candidate_id: str, suffix: str = "", gpu_override: int | None = None) -> None:
    final_dir_name = f"final_{suffix}" if suffix else "final"
    gpu_id = int(gpu_override if gpu_override is not None else args.full_gpu)
    code = f"""
from pathlib import Path
import json
import torch
from sparsemac.generator import run_full_generation
from sparsemac.vision import make_storyboard

run_dir = Path({run_dir!r})
candidate_id = {candidate_id!r}
candidates = json.loads((run_dir / "candidates.json").read_text())
matches = [c for c in candidates if c.get("candidate_id") == candidate_id]
if not matches:
    raise SystemExit(f"candidate not found: {{candidate_id}}")
best = matches[0]
warmup_latent_path = best.get("warmup_latent_path")
if not warmup_latent_path:
    raise SystemExit(f"candidate missing warmup_latent_path: {{candidate_id}}")
warmup_latent_path = str(warmup_latent_path)
if not Path(warmup_latent_path).exists():
    raise SystemExit(f"warmup latent not found: {{warmup_latent_path}}")
warmup_payload = torch.load(warmup_latent_path, map_location="cpu")
if not isinstance(warmup_payload, dict) or not isinstance(warmup_payload.get("metadata"), dict):
    raise SystemExit(f"warmup latent metadata missing or invalid: {{warmup_latent_path}}")
warmup_metadata = warmup_payload["metadata"]
final_dir_name = {final_dir_name!r}
final_dir = run_dir / final_dir_name
final_dir.mkdir(parents=True, exist_ok=True)
final_path = final_dir / "full_wan.mp4"
ok, log = run_full_generation(
    prompt=best["prompt"],
    seed=int(best["seed"]),
    generator_root={args.generator_root!r},
    ckpt_dir={args.ckpt_dir!r},
    python_bin={args.python_bin!r},
    save_file=str(final_path),
    size={args.size!r},
    sample_steps={args.full_steps},
    sample_shift={args.sample_shift},
    guide_scale={args.guide_scale},
    gpu_id={gpu_id},
    init_latent_path=warmup_latent_path,
    dry_run=False,
)
result = {{
    "method_id": {args.agent_method!r},
    "method_metadata": {method_metadata(args.agent_method)!r},
    "ok": ok,
    "path": str(final_path),
    "log": log,
    "prompt": best["prompt"],
    "seed": int(best["seed"]),
    "candidate_id": candidate_id,
    "warmup_reused": True,
    "warmup_start_step": warmup_metadata.get("next_step_index"),
    "warmup_latent_path": warmup_latent_path,
    "warmup_latent_metadata": warmup_metadata,
}}
(final_dir / "final.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
if ok:
    make_storyboard(str(final_path), str(final_dir / "storyboard_11.jpg"), frames=11)
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if ok else 1)
"""
    remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")


def write_pipeline_report(args: argparse.Namespace, run_dir: str, report: dict[str, Any]) -> None:
    code = f"""
from pathlib import Path
import json
run_dir = Path({run_dir!r})
run_dir.mkdir(parents=True, exist_ok=True)
report = {json.dumps(report, ensure_ascii=False)!r}
data = json.loads(report)
(run_dir / "pipeline_report.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def fmt_score(value):
    try:
        return format(float(value), ".2f")
    except Exception:
        return "n/a"

lines = [
    "# SparseMAC VLM Pipeline Report",
    "",
    "- Run ID: `" + str(data.get("run_id", "")) + "`",
    "- Method: `" + str(data.get("method_label", data.get("method_id", "legacy"))) + "`",
    "- Status: `" + str(data.get("status", data.get("decision", "unknown"))) + "`",
    "- Original prompt: " + str(data.get("prompt", "")),
    "- Final sparse prompt: " + str(data.get("final_sparse_prompt", data.get("prompt", ""))),
    "- Sparse VLM best candidate: `" + str(data.get("sparse_vlm_best_candidate", "")) + "`",
    "- Sparse VLM score: `" + fmt_score(data.get("sparse_vlm_score")) + "`",
    "- Full VLM score: `" + fmt_score(data.get("full_score")) + "`",
    "- Decision: `" + str(data.get("decision", "unknown")) + "`",
    "- Full video: `" + str(data.get("full_video_path", "")) + "`",
    "- Final judge: `" + str(data.get("final_judge_json", "")) + "`",
    "- Pre-Full Sparse Status: `" + str(data.get("pre_full_sparse_status", "unknown")) + "`",
    "- Full Trigger: `" + str(data.get("full_trigger", "unknown")) + "`",
    "- Analyzer Decision: `" + str(data.get("analyzer_decision", "unknown")) + "`",
    "- Terminal Sparse Round: `" + str(data.get("terminal_sparse_round", "unknown")) + "`",
]
rounds = data.get("rounds") or []
if rounds:
    lines.extend(["", "## Sparse / Refine Rounds"])
    for item in rounds:
        lines.append(
            "- Round "
            + str(item.get("round_index", ""))
            + ": run=`"
            + str(item.get("run_id", ""))
            + "` status=`"
            + str(item.get("status", ""))
            + "` best=`"
            + str(item.get("sparse_vlm_best_candidate", ""))
            + "` score=`"
            + fmt_score(item.get("sparse_vlm_score"))
            + "`"
        )
        if item.get("seed_override") is not None:
            lines.append("  seed_override: `" + str(item.get("seed_override")) + "`")
        if item.get("refine_plan_json"):
            lines.append("  refine_plan: `" + str(item.get("refine_plan_json")) + "`")
(run_dir / "pipeline_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
"""
    remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PY'\n{code}\nPY")



def remote_rayinai_vlm_eval(
    args,
    original_prompt: str,
    plan_dict: dict[str, Any],
    storyboard_path: str,
    out_json_path: str,
    out_md_path: str,
    *,
    candidate_prompt: str = "",
):
    if args.agent_method == SEMANTIC_V2_METHOD_ID:
        prompt_text = build_semantic_v2_verifier_prompt(
            original_prompt=original_prompt,
            candidate_prompt=candidate_prompt,
            criteria=semantic_criteria_from_plan(plan_dict),
            frame_count=11,
        )
        response_max_tokens = 2500
    else:
        formatted_vqa_local = "".join(
            f"- {question} (weight: {weight})\n"
            for question, weight in plan_dict.get("vqa_scores", {}).items()
        )
        prompt_text = STORYBOARD_API_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            formatted_vqa_list=formatted_vqa_local,
        )
        response_max_tokens = 1500

    semantic_v2 = args.agent_method == SEMANTIC_V2_METHOD_ID
    eval_model = args.luna_model if semantic_v2 else args.legacy_vlm_model
    eval_base_url = args.luna_base_url if semantic_v2 else args.legacy_vlm_base_url
    code = f'''
import json
from pathlib import Path

from sparsemac.luna_client import LunaClient


storyboard_path = {storyboard_path!r}
out_json_path = {out_json_path!r}
out_md_path = {out_md_path!r}
prompt = {prompt_text!r}

client = LunaClient(model={eval_model!r}, base_url={eval_base_url!r})
if not client.available:
    raise SystemExit(client.last_error or "Storyboard evaluator is unavailable")

out_data, raw = client.generate_json(
    prompt,
    image_paths=[storyboard_path],
    max_tokens={response_max_tokens},
    temperature=0.0,
    system_prompt=(
        "You are an expert text-to-video semantic evaluator. Inspect only the "
        "provided ordered storyboard and return exactly one valid JSON object."
    ),
)
if not isinstance(out_data, dict):
    Path(out_json_path + ".invalid.txt").write_text(str(raw), encoding="utf-8")
    raise SystemExit(client.last_error or "Storyboard evaluator returned invalid JSON")

out_data["api_model"] = client.model
out_data["api_base_url"] = client.base_url
Path(out_json_path).write_text(
    json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
)
Path(out_md_path).write_text(
    f"# Evaluation\\n\\nModel: {{client.model}}\\n\\nScore: {{out_data.get('score')}}\\n\\nEvidence: {{out_data.get('evidence')}}\\n",
    encoding="utf-8",
)
print("RAYINAI_EVAL_OK")
'''
    import shlex
    output = remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'EOF_SCRIPT'\n{code}\nEOF_SCRIPT")
    if "RAYINAI_EVAL_OK" not in output:
        raise PipelineError(f"Remote evaluate via Rayinai API failed:\n{output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SparseMAC sparse search, API VLM selection, full video, and final API VLM verification.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--generator-root", default=DEFAULT_GENERATOR_ROOT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--analyzer-model", default="deepseek-v4-flash")
    parser.add_argument("--deepseek-model", default=None)
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    parser.add_argument(
        "--luna-model",
        default=DEFAULT_LUNA_MODEL,
        help="Text and vision model for sparsemac-semantic-v2.",
    )
    parser.add_argument(
        "--luna-base-url",
        default=None,
        help="Optional OpenAI-compatible Luna base URL; otherwise use .env.local.",
    )
    parser.add_argument(
        "--legacy-vlm-model",
        default="gpt-5.4",
        help="Historical storyboard model retained for legacy runs.",
    )
    parser.add_argument(
        "--legacy-vlm-base-url",
        default=None,
        help="Optional legacy storyboard API base URL; otherwise use .env.local.",
    )
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=2, help="Maximum outer VLM-gated sparse/refine rounds")
    parser.add_argument("--sparse-steps", type=int, default=50)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sample-shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--sparse-gpus", default="0,1")
    parser.add_argument("--full-gpu", default="0")
    parser.add_argument("--full-gpus", default="0,1")
    parser.add_argument("--sparse-threshold", type=float, default=0.70)
    parser.add_argument("--full-threshold", type=float, default=0.70)
    parser.add_argument("--generation-timeout", type=int, default=2400)
    parser.add_argument("--gpu-memory-limit-mib", type=int, default=2048)
    parser.add_argument("--gpu-util-limit", type=int, default=10)
    parser.add_argument("--sparse-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--sparse-warmup-steps", type=int, default=0)
    parser.add_argument("--vlm-workers", type=int, default=None)
    parser.add_argument("--sparse-stride", type=int, default=2)
    parser.add_argument("--plan-json", help="Use an existing planner JSON instead of calling the planner.")
    parser.add_argument(
        "--agent-method",
        choices=SUPPORTED_METHODS,
        default=LEGACY_METHOD_ID,
        help=(
            "Prompt/verification policy. Use sparsemac-semantic-v2 for "
            "SparseMAC-SemanticV2 (ours, improved); legacy preserves old runs."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    active_method = method_metadata(args.agent_method)
    if args.full_steps != args.sparse_steps:
        raise PipelineError(
            f"full_steps must equal sparse_steps when reusing warmup latents; "
            f"got full_steps={args.full_steps}, sparse_steps={args.sparse_steps}"
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"sparse_vlm_pipeline_{timestamp}"
    max_rounds = max(1, args.max_rounds)
    original_prompt = args.prompt
    run_root = resolve_run_root(args)
    parent_run_dir = f"{run_root}/{run_id}"
    timings: dict[str, Any] = {
        "run_id": run_id,
        "prompt": original_prompt,
        "method_id": args.agent_method,
        "method_metadata": active_method,
        "started_at": utc_now_iso(),
        "planner": {},
        "rounds": [],
        "totals": {},
    }
    pipeline_start = time.perf_counter()

    print(
        f"[pipeline] run_id={run_id} outer_max_rounds={max_rounds} "
        f"method={active_method.get('method_label', args.agent_method)}"
    )

    # Step 1: Fail fast if parent run directory already exists
    check_dir_cmd = f"test -d {shlex.quote(parent_run_dir)}"
    exists_cmd = ["/bin/bash", "-lc", check_dir_cmd] if args.remote in {"local", "localhost", "127.0.0.1", ""} else ["ssh", args.remote, check_dir_cmd]
    if run_local(exists_cmd, check=False).returncode == 0:
        print(f"[pipeline] ERROR: Parent run directory already exists: {parent_run_dir}. Please specify a new run-id.", file=sys.stderr)
        return 1

    # Step 2: Make plan or load an existing plan (planner only once)
    stage_start = time.perf_counter()
    if args.plan_json:
        print(f"[pipeline] Loading external plan: {args.plan_json}")
        plan_dict = remote_load_external_plan(args, run_id, args.plan_json)
    else:
        print(f"[pipeline] Generating plan for the parent run...")
        plan_dict = remote_make_plan(args, run_id, original_prompt, args.candidates)
    if args.agent_method == SEMANTIC_V2_METHOD_ID:
        if plan_dict.get("method_id") != SEMANTIC_V2_METHOD_ID:
            raise PipelineError(
                f"Planner output is not labeled {SEMANTIC_V2_METHOD_ID}; refusing to mix methods."
            )
        semantic_criteria_from_plan(plan_dict)
    timings["planner"] = {
        "wall_seconds": elapsed_since(stage_start),
        "model_seconds": None,
        "source": "external_plan_json" if args.plan_json else "planner_api",
        "plan_json": args.plan_json or f"{parent_run_dir}/plan.json",
    }
    write_timings(args, parent_run_dir, timings)
    
    # Retrieve the list of prompt variants for Round 0
    prompt_variants = plan_dict.get("prompt_variants", [])
    if not isinstance(prompt_variants, list):
        prompt_variants = []
    prompt_variants = [str(p).strip() for p in prompt_variants if str(p).strip()]
    if not prompt_variants:
        prompt_variants = [original_prompt]
    while len(prompt_variants) < args.candidates:
        prompt_variants.append(prompt_variants[0])
    
    current_prompts = prompt_variants[:args.candidates]
    best_prompt = current_prompts[0]
    
    current_seed_override = None
    rounds: list[dict[str, Any]] = []
    last_run_dir = ""

    passed_sparse = False
    analyzer_decision = "none"
    full_trigger = ""
    pre_full_sparse_status = ""
    terminal_sparse_round = 0

    for round_index in range(max_rounds):
        terminal_sparse_round = round_index
        round_start = time.perf_counter()
        round_run_id = run_id
        round_dir = f"{parent_run_dir}/round_{round_index}"
        last_run_dir = round_dir
        round_timing: dict[str, Any] = {
            "round_index": round_index,
            "sparse_generation": {},
            "sparse_vlm_judge": {},
            "deepseek_refine": {},
            "full_generation": {},
            "final_vlm_judge": {},
        }
        
        print(f"[pipeline] sparse round={round_index} prompts={current_prompts}")
        # Run the sparse batch generation on target-server
        stage_start = time.perf_counter()
        candidates = remote_run_sparse_batch(
            args,
            run_id=run_id,
            round_index=round_index,
            original_prompt=original_prompt,
            prompts=current_prompts,
            candidates_count=args.candidates,
            seed_override=current_seed_override,
        )
        sparse_candidate_timings = collect_sparse_generation_timings(args, candidates)
        round_timing["sparse_generation"] = {
            "wall_seconds": elapsed_since(stage_start),
            "model_seconds": sum_seconds([item.get("model_seconds") for item in sparse_candidate_timings]),
            "candidates": sparse_candidate_timings,
        }
        print(f"[pipeline] sparse generation complete: {round_dir}")

        sparse_json = f"{round_dir}/visual_judge.json"
        sparse_md = f"{round_dir}/visual_judge.md"
        stage_start = time.perf_counter()

        print(f"[pipeline] API VLM evaluation starting for round_{round_index}")
        workers = args.vlm_workers if args.vlm_workers is not None else min(len(candidates), 4)

        def run_cand_judge(cand):
            cid = cand["candidate_id"]
            cand_dir_path = f"{round_dir}/candidates/{cid}"
            cand_json = f"{cand_dir_path}/visual_judge.json"
            cand_md = f"{cand_dir_path}/visual_judge.md"
            remote_rayinai_vlm_eval(
                args,
                original_prompt,
                plan_dict,
                f"{cand_dir_path}/storyboard_11.jpg",
                cand_json,
                cand_md,
                candidate_prompt=str(cand.get("prompt") or ""),
            )
            return cid, cand_json, cand_md

        futures_map = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for cand in candidates:
                fut = executor.submit(run_cand_judge, cand)
                futures_map[fut] = cand["candidate_id"]
            for fut in futures_map:
                cid = futures_map[fut]
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[pipeline] ERROR: Candidate {cid} evaluation failed: {exc}", file=sys.stderr)
                    raise PipelineError(f"Candidate {cid} evaluation failed: {exc}") from exc

        aggregated_candidates = {}
        best_candidate_id = None
        best_score = -1.0
        for cand in candidates:
            cid = cand["candidate_id"]
            cand_dir_path = f"{round_dir}/candidates/{cid}"
            cand_json_path = f"{cand_dir_path}/visual_judge.json"
            cand_data = remote_json(args, cand_json_path)
            if args.agent_method == SEMANTIC_V2_METHOD_ID:
                cand_data = normalize_semantic_v2_judgment(plan_dict, cand_data)
                write_remote_json(args, cand_json_path, cand_data)
            aggregated_candidates[cid] = cand_data
            score = as_normalized_score(cand_data.get("score", 0.0), field=f"candidate {cid} score")
            if score > best_score:
                best_score = score
                best_candidate_id = cid

        aggregated = {
            "method_id": args.agent_method,
            "method_metadata": active_method,
            "best_candidate": best_candidate_id,
            "decision": "sparse_pass" if best_score >= args.sparse_threshold else "sparse_reject",
            "candidates": aggregated_candidates,
        }
        write_remote_json(args, sparse_json, aggregated)

        md_lines = [
            f"# SparseMAC VLM Visual Judge Round {round_index}",
            "",
            f"- Best Candidate: `{best_candidate_id}`",
            f"- Best Score: `{best_score:.4f}`",
            f"- Decision: `{aggregated['decision']}`",
            "",
            "## Candidates Summary",
            "",
        ]
        for cid, cdata in aggregated_candidates.items():
            md_lines.extend([
                f"### Candidate `{cid}` (Score: `{cdata.get('score', 0.0):.4f}`)",
                f"- Recommendation: `{cdata.get('recommendation', 'unknown')}`",
                f"- Success/Failure Reason: {cdata.get('success_or_failure_reason', '')}",
                f"- Evidence: {cdata.get('evidence', '')}",
                "",
            ])
        write_remote_file(args, sparse_md, "\n".join(md_lines) + "\n")

        round_timing["sparse_vlm_judge"] = {
            "wall_seconds": elapsed_since(stage_start),
            "model_seconds": None,
            "output_json": sparse_json,
        }
        sparse_data = remote_json(args, sparse_json)
        best_candidate, sparse_vlm_score = parse_sparse_vlm_result(sparse_data)
        best_context = best_candidate_context(args, round_dir, sparse_data, best_candidate)
        passed_sparse = sparse_vlm_passes(args, sparse_data, best_candidate, sparse_vlm_score)
        candidate = best_context.get("candidate", {})
        judgment = best_context.get("judgment", {})
        
        best_prompt = candidate.get("prompt") or current_prompts[0]

        # Extract vqa_feedback
        vqa_results = judgment.get("vqa_results") or []
        vqa_feedback = []
        for item in vqa_results:
            if isinstance(item, dict):
                feedback_item = {
                    "criterion_id": item.get("criterion_id"),
                    "question": item.get("question"),
                    "vqa_score": item.get("vqa_score"),
                    "weight": item.get("weight"),
                    "reason": item.get("reason"),
                    "evidence_frames": item.get("evidence_frames", []),
                    "confidence": item.get("confidence"),
                }
                vqa_feedback.append(feedback_item)

        round_record: dict[str, Any] = {
            "method_id": args.agent_method,
            "method_metadata": active_method,
            "round_index": round_index,
            "run_id": round_run_id,
            "prompt": current_prompts,
            "run_dir": round_dir,
            "sparse_judge_json": sparse_json,
            "sparse_vlm_best_candidate": best_candidate,
            "sparse_vlm_score": sparse_vlm_score,
            "sparse_vlm_decision": sparse_data.get("decision"),
            "best_candidate_prompt": candidate.get("prompt"),
            "best_candidate_seed": candidate.get("seed"),
            "best_candidate_storyboard": candidate.get("storyboard"),
            "best_candidate_warmup_latent_path": candidate.get("warmup_latent_path"),
            "seed_override": current_seed_override,
            "recommendation": judgment.get("recommendation"),
            "vqa_feedback": vqa_feedback,
            "status": "sparse_pass" if passed_sparse else "needs_refine",
            "refine_plan_json": None,
            "next_prompt": None,
            "next_prompts": [],
            "timings": round_timing,
        }
        rounds.append(round_record)
        print(
            f"[pipeline] sparse VLM round={round_index} best={best_candidate} "
            f"score={sparse_vlm_score:.2f} status={round_record['status']}"
        )

        if passed_sparse:
            full_trigger = "sparse_pass"
            pre_full_sparse_status = "passed"
            round_timing["round_total_wall_seconds"] = elapsed_since(round_start)
            timings["rounds"].append(round_timing)
            break

        if round_index + 1 >= max_rounds:
            full_trigger = "max_rounds_sparse_reject"
            pre_full_sparse_status = "failed_max_rounds"
            round_timing["round_total_wall_seconds"] = elapsed_since(round_start)
            timings["rounds"].append(round_timing)
            break

        stage_start = time.perf_counter()
        
        # Build round history
        history_list = []
        for r in rounds:
            history_list.append({
                "round_index": r["round_index"],
                "prompts_tested": r["prompt"],
                "best_candidate_id": r["sparse_vlm_best_candidate"],
                "best_candidate_prompt": r["best_candidate_prompt"],
                "best_candidate_seed": r["best_candidate_seed"],
                "best_candidate_score": r["sparse_vlm_score"],
                "vqa_feedback": r.get("vqa_feedback", [])
            })
        
        # Absolute best candidate so far
        abs_best_round = max(rounds, key=lambda x: x["sparse_vlm_score"])
        current_best_candidate = {
            "round_index": abs_best_round["round_index"],
            "candidate_id": abs_best_round["sparse_vlm_best_candidate"],
            "prompt": abs_best_round["best_candidate_prompt"],
            "seed": abs_best_round["best_candidate_seed"],
            "score": abs_best_round["sparse_vlm_score"],
            "vqa_feedback": abs_best_round.get("vqa_feedback", []),
        }

        refine_plan = run_deepseek_refine(
            args,
            run_dir=round_dir,
            original_prompt=original_prompt,
            plan_dict=plan_dict,
            round_history=history_list,
            current_best_candidate=current_best_candidate,
        )
        
        analyzer_decision = refine_plan.get("decision", "refine")
        round_timing["deepseek_refine"] = {
            "wall_seconds": elapsed_since(stage_start),
            "model_seconds": None,
            "output_json": f"{round_dir}/vlm_refine_plan.json",
        }
        
        round_record["refine_plan_json"] = f"{round_dir}/vlm_refine_plan.json"
        round_record["refine_reason"] = refine_plan.get("reason")
        round_record["failure_focus"] = refine_plan.get("failure_focus", [])
        round_record["rewrite_hints"] = refine_plan.get("rewrite_hints", [])
        round_record["next_prompt"] = refine_plan.get("next_prompt")
        round_record["next_prompts"] = refine_plan.get("next_prompts", [])
        
        if analyzer_decision == "reject":
            full_trigger = "analyzer_reject"
            pre_full_sparse_status = "failed_refine"
            round_timing["round_total_wall_seconds"] = elapsed_since(round_start)
            timings["rounds"].append(round_timing)
            break
            
        round_timing["round_total_wall_seconds"] = elapsed_since(round_start)
        timings["rounds"].append(round_timing)
        write_timings(args, parent_run_dir, timings)
        
        # Prepare prompts for next round N>0 from analyzer output
        current_prompts = prepare_next_round_prompts(refine_plan, best_prompt, args.candidates)
        print(f"[pipeline] refine prompts for next round: {current_prompts}")
        if sparse_vlm_score >= 0.6:
            current_seed_override = candidate.get("seed")
        else:
            current_seed_override = None

    if not rounds:
        raise PipelineError("No sparse rounds were completed, cannot proceed to Full video.")

    # Find the top 2 candidates across all sparse rounds so far
    all_candidates = []
    for r in rounds:
        if r.get("sparse_judge_json"):
            try:
                data = remote_json(args, r["sparse_judge_json"])
                cands = data.get("candidates", {})
                for cid, cdata in cands.items():
                    score = as_normalized_score(cdata.get("score", 0.0))
                    idx = candidate_index(cid)
                    candidate_prompt = (
                        r["prompt"][idx]
                        if idx is not None
                        and isinstance(r["prompt"], list)
                        and idx < len(r["prompt"])
                        else r["best_candidate_prompt"]
                    )
                    candidate_seed = r["best_candidate_seed"]
                    candidate_warmup_latent_path = r["best_candidate_warmup_latent_path"]
                    try:
                        round_candidates = remote_json(args, f"{r['run_dir']}/candidates.json")
                        if isinstance(round_candidates, list):
                            for item in round_candidates:
                                if isinstance(item, dict) and item.get("candidate_id") == cid:
                                    candidate_prompt = item.get("prompt") or candidate_prompt
                                    candidate_seed = item.get("seed", candidate_seed)
                                    candidate_warmup_latent_path = item.get("warmup_latent_path") or candidate_warmup_latent_path
                                    break
                    except Exception as exc:
                        print(f"Failed to load candidate metadata for {cid}: {exc}")
                    all_candidates.append({
                        "round_index": r["round_index"],
                        "run_dir": r["run_dir"],
                        "candidate_id": cid,
                        "score": score,
                        "prompt": candidate_prompt,
                        "seed": candidate_seed,
                        "warmup_latent_path": candidate_warmup_latent_path,
                    })
            except Exception as e:
                print(f"Failed to load candidates from round {r['round_index']}: {e}")

    if not all_candidates:
        # Fallback to previous logic
        all_candidates = [{"candidate_id": rounds[-1]["sparse_vlm_best_candidate"], "score": rounds[-1]["sparse_vlm_score"], "run_dir": rounds[-1]["run_dir"], "round_index": rounds[-1]["round_index"], "prompt": rounds[-1]["best_candidate_prompt"], "warmup_latent_path": rounds[-1]["best_candidate_warmup_latent_path"]}]
    
    # Sort and pick top 2
    top_candidates = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:2]
    
    print(f"[pipeline] Proceeding to Full Video stage via trigger: '{full_trigger}'")
    print(f"[pipeline] Top sparse candidates:")
    for i, c in enumerate(top_candidates):
        print(f"  {i+1}: round_{c['round_index']}/{c['candidate_id']} with score {c['score']:.2f}")

    best_full_score = -1.0
    best_full_candidate = None
    best_final_json = None
    best_final_judgment = None
    best_decision = None
    best_video_path = None
    best_storyboard_path = None

    last_round_timing = rounds[-1]["timings"]
    last_round_timing["full_generation_candidates"] = []


    full_gpus = [int(x.strip()) for x in args.full_gpus.split(",")] if hasattr(args, 'full_gpus') and args.full_gpus else [int(args.full_gpu)]
    
    import threading
    from concurrent.futures import as_completed
    
    results = []
    
    def process_full_cand(idx, cand):
        cid = cand["candidate_id"]
        c_round_dir = cand["run_dir"]
        suffix = f"cand_{cid}"
        final_out_dir = f"{c_round_dir}/final_{suffix}"
        
        assigned_gpu = full_gpus[idx % len(full_gpus)]
        print(f"[pipeline] full video start for candidate {cid} (score {cand['score']:.2f}) on GPU {assigned_gpu}")
        
        stage_start = time.perf_counter()
        run_full_from_candidate(args, c_round_dir, cid, suffix=suffix, gpu_override=assigned_gpu)
        full_generation_wall = elapsed_since(stage_start)
        
        final_generation_json = remote_json(args, f"{final_out_dir}/final.json")
        timing_info = {
            "candidate_id": cid,
            "wall_seconds": full_generation_wall,
            **collect_full_generation_timing(args, final_generation_json),
        }

        # Run final VLM verification
        final_json = f"{final_out_dir}/final_visual_judge.json"
        final_md = f"{final_out_dir}/final_visual_judge.md"
        stage_start = time.perf_counter()
        
        print(f"[pipeline] API evaluation starting for full storyboard {cid}")
        remote_rayinai_vlm_eval(
            args,
            original_prompt,
            plan_dict,
            f"{final_out_dir}/storyboard_11.jpg",
            final_json,
            final_md,
            candidate_prompt=str(cand.get("prompt") or ""),
        )
        eval_wall = elapsed_since(stage_start)
        
        final_data = remote_json(args, final_json)
        if args.agent_method == SEMANTIC_V2_METHOD_ID:
            final_data = normalize_semantic_v2_judgment(plan_dict, final_data)
        score = as_normalized_score(final_data.get("score", 0.0), field="final score")
        decision = "accept_full_video" if score >= args.full_threshold else "reject_or_retry_full_video"
        final_data["decision"] = decision
        if "success_or_failure_reason" not in final_data:
            final_data["success_or_failure_reason"] = final_data.get("evidence", "Visual evaluation completed.")
        write_remote_json(args, final_json, final_data)

        full_score, f_decision, final_judgment = parse_final_vlm_result(final_data)
        print(f"[pipeline] candidate {cid} full score={full_score:.2f} decision={f_decision}")
        
        return {
            "cand": cand,
            "full_score": full_score,
            "decision": f_decision,
            "final_json": final_json,
            "final_judgment": final_judgment,
            "video_path": f"{final_out_dir}/full_wan.mp4",
            "storyboard_path": f"{final_out_dir}/storyboard_11.jpg",
            "timing_info": timing_info,
            "total_wall": full_generation_wall + eval_wall
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_full_cand, idx, cand) for idx, cand in enumerate(top_candidates)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[pipeline] ERROR in full generation thread: {e}")

    total_added_wall = 0
    for res in results:
        last_round_timing["full_generation_candidates"].append(res["timing_info"])
        total_added_wall = max(total_added_wall, res["total_wall"]) # parallel, so max time
        
        if res["full_score"] > best_full_score:
            best_full_score = res["full_score"]
            best_full_candidate = res["cand"]
            best_final_json = res["final_json"]
            best_final_judgment = res["final_judgment"]
            best_decision = res["decision"]
            best_video_path = res["video_path"]
            best_storyboard_path = res["storyboard_path"]

    last_round_timing["round_total_wall_seconds"] = round(
        last_round_timing.get("round_total_wall_seconds", 0.0) + total_added_wall, 3
    )
    
    if not best_full_candidate:
        raise PipelineError("All full generation candidates failed.")

    accepted = best_decision == "accept_full_video" and best_full_score >= args.full_threshold
    
    report = {
        "method_id": args.agent_method,
        "method_name": active_method.get("method_name"),
        "method_label": active_method.get("method_label"),
        "method_metadata": active_method,
        "status": "success" if accepted else "full_reject",
        "run_id": run_id,
        "prompt": original_prompt,
        "final_sparse_prompt": best_full_candidate["prompt"],
        "run_dir": best_full_candidate["run_dir"],
        "sparse_vlm_best_candidate": best_full_candidate["candidate_id"],
        "sparse_vlm_score": best_full_candidate["score"],
        "sparse_judge_json": f"{best_full_candidate['run_dir']}/visual_judge.json",
        "full_score": best_full_score,
        "decision": best_decision,
        "full_video_path": best_video_path,
        "final_storyboard_path": best_storyboard_path,
        "final_judge_json": best_final_json,
        "final_judge": best_final_judgment,
        "warmup_reused": True,
        "warmup_latent_path": best_full_candidate["warmup_latent_path"],
        
        "full_trigger": full_trigger,
        "pre_full_sparse_status": pre_full_sparse_status,
        "analyzer_decision": analyzer_decision,
        "terminal_sparse_round": terminal_sparse_round,

        "timings": timings,
        "rounds": rounds,
    }
    timings["ended_at"] = utc_now_iso()
    timings["pipeline_wall_seconds"] = elapsed_since(pipeline_start)
    try:
        cleanup_warmup_latents_after_full(
            args,
            parent_run_dir,
            best_full_candidate["run_dir"],
            best_full_candidate["candidate_id"],
        )
    except Exception as exc:
        print(f"[pipeline] WARNING: delayed warmup latent cleanup failed: {exc}")
    write_timings(args, parent_run_dir, timings)
    write_pipeline_report(args, parent_run_dir, report)
    print(f"[pipeline] report: {parent_run_dir}/pipeline_report.json")
    return 0 if accepted else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except PipelineError as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
