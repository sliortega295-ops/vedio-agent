#!/usr/bin/env python3
"""Resume incomplete SparseMAC VLM pipeline runs without deleting existing sparse state.

This script continues the normal agentic flow:
existing sparse/VLM -> analyzer/refine -> more sparse rounds if needed -> full generation.
It does NOT force full directly for sparse-rejected runs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import importlib.util
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT / "scripts" / "sparse_to_full_vlm_pipeline.py"
spec = importlib.util.spec_from_file_location("sparse_to_full_vlm_pipeline", PIPELINE_PATH)
pipe = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pipe)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def norm_score(x: Any) -> float | None:
    try:
        v = float(x)
        if v > 1.0:
            v /= 100.0
        return v
    except Exception:
        return None


def robust_vlm_eval(args: argparse.Namespace, original_prompt: str, plan_vqa: dict, storyboard_path: str, out_json_path: str, out_md_path: str) -> None:
    formatted = "".join(f"- {q} (weight: {w})\n" for q, w in (plan_vqa or {}).items())
    code = f'''
import base64, json, os, time, urllib.request, urllib.error
from pathlib import Path

api_key = os.environ.get("SPARSEMAC_LUNA_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise RuntimeError("SPARSEMAC_LUNA_API_KEY or OPENAI_API_KEY is required")
base_url = os.environ.get("SPARSEMAC_LUNA_BASE_URL", "https://bmapi.020212.xyz/v1").rstrip("/")
model = os.environ.get("SPARSEMAC_LUNA_MODEL", "gpt-5.6-luna")
original_prompt = {original_prompt!r}
formatted_vqa = {formatted!r}
storyboard_path = {storyboard_path!r}
out_json_path = {out_json_path!r}
out_md_path = {out_md_path!r}
err_path = str(Path(out_json_path).with_suffix(".error.txt"))

prompt = """You are an objective visual evaluator. Judge only visible evidence in the attached storyboard image.
The storyboard has only 11 sampled frames. Do not demand evidence between sampled frames, but do judge whether the visible sampled frames support the required objects, attributes, relations, actions, state changes, and motion trend.
Do not reward style, plausibility, or prior expectations when the storyboard evidence fails the prompt.
Treat the 11-frame storyboard as your sole visual evidence. Do not infer missing objects, attributes, counts, actions, or spatial relations unless they are visually observable in the storyboard.
If evidence is ambiguous from the 11 frames, give a low score and explain that the storyboard evidence is insufficient.

CRITICAL MOTION RULE: If an active motion or action requirement is completely absent or if objects are completely stationary/static, assign a vqa_score between 0 and 30 for the affected checks.

Evaluate semantic correctness only. Do not over-penalize ordinary sparse-video blur unless it prevents judging the prompt-critical object/action/relation.

Original prompt:
\""" + original_prompt + """\"

Specific VQA questions and weights:
""" + formatted_vqa + """

Instructions:
1. Score every VQA question exactly once on a 0-100 integer scale.
2. Compute final overall score as weighted average of all vqa_scores divided by 100.
3. Return strict JSON only with keys: score, vqa_results, failure_reasons, evidence, recommendation, success_or_failure_reason.
"""

img_b64 = base64.b64encode(Path(storyboard_path).read_bytes()).decode("utf-8")
messages = [{{"role":"user","content":[{{"type":"text","text":prompt}},{{"type":"image_url","image_url":{{"url":"data:image/jpeg;base64,"+img_b64,"detail":"high"}}}}]}}]
base = {{"model": model, "messages": messages, "max_tokens": 1800, "temperature": 0.0}}
payload_variants = [{{**base, "response_format": {{"type":"json_object"}}}}, base]

def get_content(body):
    ch = (body.get("choices") or [{{}}])[0]
    msg = ch.get("message") or {{}}
    content = msg.get("content")
    if isinstance(content, list):
        parts=[]
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "\\n".join(p for p in parts if p)
    if not content:
        content = msg.get("reasoning_content") or ch.get("text") or body.get("output_text")
    return content

def parse_json_text(text):
    s=(text or "").strip()
    if s.startswith("```"):
        s=s.strip("`").strip()
        if s.lower().startswith("json"):
            s=s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        a=s.find("{{"); b=s.rfind("}}")
        if a>=0 and b>a:
            return json.loads(s[a:b+1])
        raise

last_error = ""
for variant_i, payload in enumerate(payload_variants):
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={{"Authorization":"Bearer "+api_key, "Content-Type":"application/json", "User-Agent":"SparseMAC/1.0"}},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                raw=response.read().decode("utf-8", errors="replace")
            body=json.loads(raw)
            content=get_content(body)
            if not content:
                raise RuntimeError("empty/missing message content; body=" + raw[:1200])
            out=parse_json_text(content)
            score=float(out.get("score", 0.0))
            if score>1.0: out["score"] = score/100.0
            Path(out_json_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            Path(out_md_path).write_text(f"# Evaluation\\n\\nScore: {{out.get('score')}}\\n\\nEvidence: {{out.get('evidence','')}}\\n", encoding="utf-8")
            print("RAYINAI_EVAL_OK")
            raise SystemExit(0)
        except urllib.error.HTTPError as e:
            detail=e.read().decode("utf-8", errors="replace")[:2000]
            last_error=f"variant={{variant_i}} attempt={{attempt+1}} HTTP {{e.code}} {{detail}}"
        except Exception as e:
            last_error=f"variant={{variant_i}} attempt={{attempt+1}} {{type(e).__name__}}: {{e}}"
        print("RAYINAI_EVAL_ATTEMPT_FAILED", last_error)
        time.sleep(min(30, 4*(attempt+1)))
Path(err_path).write_text(last_error, encoding="utf-8")
print("RAYINAI_EVAL_ERROR", last_error)
raise SystemExit(1)
'''
    out = pipe.remote_cmd(args, f"cd {shlex.quote(args.project_dir)} && {shlex.quote(args.python_bin)} - <<'PYREMOTE'\n{code}\nPYREMOTE")
    if "RAYINAI_EVAL_OK" not in out:
        raise pipe.PipelineError(f"robust VLM eval failed for {storyboard_path}:\n{out}")


def candidate_score_from_json(p: Path) -> float | None:
    d = load_json(p)
    if not isinstance(d, dict):
        return None
    return norm_score(d.get("score"))


def initial_prompts(plan: dict[str, Any], original_prompt: str, n: int) -> list[str]:
    prompts = plan.get("prompt_variants") if isinstance(plan, dict) else []
    if not isinstance(prompts, list):
        prompts = []
    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    if not prompts:
        prompts = [original_prompt]
    while len(prompts) < n:
        prompts.append(prompts[0])
    return prompts[:n]


def load_or_generate_sparse(args, run_id: str, round_index: int, original_prompt: str, prompts: list[str], seed_override: int | None):
    round_dir = Path(args.run_root) / run_id / f"round_{round_index}"
    cpath = round_dir / "candidates.json"
    candidates = load_json(cpath)
    ok_existing = False
    if isinstance(candidates, list) and candidates:
        ok_existing = all((not c.get("sparse_video")) or Path(c["sparse_video"]).exists() for c in candidates if isinstance(c, dict))
    if ok_existing:
        print(f"[resume] reuse sparse round_{round_index}: {cpath}", flush=True)
        return candidates
    print(f"[resume] generate sparse round_{round_index} for {run_id}", flush=True)
    return pipe.remote_run_sparse_batch(args, run_id, round_index, original_prompt, prompts, args.candidates, seed_override=seed_override)


def ensure_round_vlm(args, original_prompt: str, plan: dict, run_id: str, round_index: int, candidates: list[dict[str, Any]]):
    round_dir = Path(args.run_root) / run_id / f"round_{round_index}"
    failures=[]
    for cand in candidates:
        cid = cand.get("candidate_id")
        if not cid:
            continue
        cdir = round_dir / "candidates" / cid
        story = cdir / "storyboard_11.jpg"
        out_json = cdir / "visual_judge.json"
        out_md = cdir / "visual_judge.md"
        if candidate_score_from_json(out_json) is not None:
            continue
        if not story.exists():
            failures.append(f"{cid}: missing storyboard {story}")
            continue
        print(f"[resume] VLM judge {run_id} round_{round_index}/{cid}", flush=True)
        try:
            robust_vlm_eval(args, original_prompt, plan.get("vqa_scores", {}), str(story), str(out_json), str(out_md))
        except Exception as exc:
            failures.append(f"{cid}: {exc}")
    if failures:
        raise pipe.PipelineError("candidate VLM incomplete: " + "; ".join(failures[:4]))

    aggregated_candidates = {}
    best_cid = None
    best_score = -1.0
    for cand in candidates:
        cid = cand.get("candidate_id")
        if not cid:
            continue
        jpath = round_dir / "candidates" / cid / "visual_judge.json"
        data = load_json(jpath)
        if not isinstance(data, dict):
            raise pipe.PipelineError(f"missing/invalid candidate judge: {jpath}")
        score = pipe.as_normalized_score(data.get("score", 0.0), field=f"candidate {cid} score")
        aggregated_candidates[cid] = data
        if score > best_score:
            best_score = score
            best_cid = cid
    aggregated = {
        "best_candidate": best_cid,
        "decision": "sparse_pass" if best_score >= args.sparse_threshold else "sparse_reject",
        "candidates": aggregated_candidates,
    }
    sparse_json = round_dir / "visual_judge.json"
    write_json(sparse_json, aggregated)
    md = [f"# SparseMAC VLM Visual Judge Round {round_index}", "", f"- Best Candidate: `{best_cid}`", f"- Best Score: `{best_score:.4f}`", f"- Decision: `{aggregated['decision']}`", ""]
    write_json(round_dir / "visual_judge.summary.json", {"best_candidate": best_cid, "best_score": best_score})
    (round_dir / "visual_judge.md").write_text("\n".join(md), encoding="utf-8")
    return aggregated


def build_round_record(args, run_id: str, round_index: int, prompts: list[str], seed_override: int | None, sparse_data: dict):
    round_dir = f"{args.run_root}/{run_id}/round_{round_index}"
    best_candidate, sparse_score = pipe.parse_sparse_vlm_result(sparse_data)
    best_context = pipe.best_candidate_context(args, round_dir, sparse_data, best_candidate)
    candidate = best_context.get("candidate", {})
    judgment = best_context.get("judgment", {})
    feedback=[]
    for item in judgment.get("vqa_results") or []:
        if isinstance(item, dict):
            feedback.append({"question": item.get("question"), "vqa_score": item.get("vqa_score"), "weight": item.get("weight"), "reason": item.get("reason")})
    passed = pipe.sparse_vlm_passes(args, sparse_data, best_candidate, sparse_score)
    return {
        "round_index": round_index,
        "run_id": run_id,
        "prompt": prompts,
        "run_dir": round_dir,
        "sparse_judge_json": f"{round_dir}/visual_judge.json",
        "sparse_vlm_best_candidate": best_candidate,
        "sparse_vlm_score": sparse_score,
        "sparse_vlm_decision": sparse_data.get("decision"),
        "best_candidate_prompt": candidate.get("prompt"),
        "best_candidate_seed": candidate.get("seed"),
        "best_candidate_storyboard": candidate.get("storyboard"),
        "best_candidate_warmup_latent_path": candidate.get("warmup_latent_path"),
        "seed_override": seed_override,
        "recommendation": judgment.get("recommendation"),
        "vqa_feedback": feedback,
        "status": "sparse_pass" if passed else "needs_refine",
        "refine_plan_json": None,
        "next_prompt": None,
        "next_prompts": [],
        "timings": {"round_index": round_index},
    }, passed


def run_full_stage(args, original_prompt: str, plan: dict, run_id: str, rounds: list[dict[str, Any]], full_trigger: str, pre_full_sparse_status: str, analyzer_decision: str):
    all_candidates=[]
    for r in rounds:
        data = load_json(Path(r["sparse_judge_json"]))
        if not isinstance(data, dict):
            continue
        for cid, cdata in (data.get("candidates") or {}).items():
            score = pipe.as_normalized_score(cdata.get("score", 0.0))
            cand_meta = None
            cands = load_json(Path(r["run_dir"]) / "candidates.json", [])
            if isinstance(cands, list):
                cand_meta = next((c for c in cands if isinstance(c, dict) and c.get("candidate_id") == cid), None)
            if not cand_meta:
                continue
            all_candidates.append({
                "round_index": r["round_index"], "run_dir": r["run_dir"], "candidate_id": cid, "score": score,
                "prompt": cand_meta.get("prompt"), "seed": cand_meta.get("seed"), "warmup_latent_path": cand_meta.get("warmup_latent_path"),
            })
    if not all_candidates:
        raise pipe.PipelineError("no sparse candidates available for full stage")
    top = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:2]
    print(f"[resume] full stage {run_id} trigger={full_trigger} top={[ (c['candidate_id'], round(c['score'],3)) for c in top ]}", flush=True)

    full_gpus=[int(x.strip()) for x in str(args.full_gpus).split(",") if x.strip()]
    if not full_gpus:
        full_gpus=[int(args.full_gpu)]

    results=[]
    def one(idx_c):
        idx, cand = idx_c
        cid=cand["candidate_id"]
        suffix=f"cand_{cid}"
        gpu=full_gpus[idx % len(full_gpus)]
        pipe.run_full_from_candidate(args, cand["run_dir"], cid, suffix=suffix, gpu_override=gpu)
        final_dir=Path(cand["run_dir"]) / f"final_{suffix}"
        final_json=final_dir / "final_visual_judge.json"
        final_md=final_dir / "final_visual_judge.md"
        robust_vlm_eval(args, original_prompt, plan.get("vqa_scores", {}), str(final_dir/"storyboard_11.jpg"), str(final_json), str(final_md))
        final_data=load_json(final_json, {})
        score=pipe.as_normalized_score(final_data.get("score",0.0), field="final score")
        decision="accept_full_video" if score >= args.full_threshold else "reject_or_retry_full_video"
        final_data["decision"]=decision
        write_json(final_json, final_data)
        return {"cand": cand, "full_score": score, "decision": decision, "final_json": str(final_json), "final_judgment": final_data, "video_path": str(final_dir/"full_wan.mp4"), "storyboard_path": str(final_dir/"storyboard_11.jpg")}

    with cf.ThreadPoolExecutor(max_workers=min(2, len(top))) as ex:
        futs=[ex.submit(one, item) for item in enumerate(top)]
        for fut in cf.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"[resume] full candidate failed: {exc}", flush=True)
    if not results:
        raise pipe.PipelineError("all full candidates failed")
    best=max(results, key=lambda x: x["full_score"])
    accepted=best["decision"] == "accept_full_video" and best["full_score"] >= args.full_threshold
    report={
        "status": "success" if accepted else "full_reject",
        "run_id": run_id,
        "prompt": original_prompt,
        "final_sparse_prompt": best["cand"].get("prompt"),
        "run_dir": best["cand"]["run_dir"],
        "sparse_vlm_best_candidate": best["cand"]["candidate_id"],
        "sparse_vlm_score": best["cand"]["score"],
        "sparse_judge_json": f"{best['cand']['run_dir']}/visual_judge.json",
        "full_score": best["full_score"],
        "decision": best["decision"],
        "full_video_path": best["video_path"],
        "final_storyboard_path": best["storyboard_path"],
        "final_judge_json": best["final_json"],
        "final_judge": best["final_judgment"],
        "warmup_reused": True,
        "warmup_latent_path": best["cand"].get("warmup_latent_path"),
        "full_trigger": full_trigger,
        "pre_full_sparse_status": pre_full_sparse_status,
        "analyzer_decision": analyzer_decision,
        "terminal_sparse_round": rounds[-1]["round_index"],
        "rounds": rounds,
        "resume_script": str(Path(__file__).resolve()),
        "resumed_at": dt.datetime.now().isoformat(),
    }
    pipe.write_pipeline_report(args, f"{args.run_root}/{run_id}", report)
    return report


def resume_one(args, record: dict[str, Any]):
    run_id=record["run_id"]
    original_prompt=record["prompt"]
    parent=Path(args.run_root) / run_id
    if (parent / "pipeline_report.json").exists() and list(parent.glob("**/full_wan.mp4")):
        return {"run_id": run_id, "status": "skip_completed", "full_video_path": str(next(parent.glob("**/full_wan.mp4")))}
    parent.mkdir(parents=True, exist_ok=True)
    plan=load_json(parent/"plan.json")
    if not isinstance(plan, dict):
        raise pipe.PipelineError(f"missing plan.json for {run_id}")
    current_prompts=initial_prompts(plan, original_prompt, args.candidates)
    seed_override=None
    rounds=[]
    full_trigger=""
    pre_full=""
    analyzer_decision="none"

    for round_index in range(args.max_rounds):
        candidates=load_or_generate_sparse(args, run_id, round_index, original_prompt, current_prompts, seed_override)
        ensure_round_vlm(args, original_prompt, plan, run_id, round_index, candidates)
        sparse_data=load_json(Path(args.run_root)/run_id/f"round_{round_index}"/"visual_judge.json")
        if not isinstance(sparse_data, dict):
            raise pipe.PipelineError(f"missing round visual_judge.json round_{round_index}")
        rec, passed=build_round_record(args, run_id, round_index, current_prompts, seed_override, sparse_data)
        rounds.append(rec)
        print(f"[resume] {run_id} round_{round_index} best={rec['sparse_vlm_best_candidate']} score={rec['sparse_vlm_score']:.3f} status={rec['status']}", flush=True)
        if passed:
            full_trigger="sparse_pass"; pre_full="passed"; break
        if round_index + 1 >= args.max_rounds:
            full_trigger="max_rounds_sparse_reject"; pre_full="failed_max_rounds"; break

        rdir=Path(args.run_root)/run_id/f"round_{round_index}"
        refine_path=rdir/"vlm_refine_plan.json"
        refine_plan=load_json(refine_path)
        if not isinstance(refine_plan, dict):
            history=[{"round_index": r["round_index"], "prompts_tested": r["prompt"], "best_candidate_id": r["sparse_vlm_best_candidate"], "best_candidate_prompt": r["best_candidate_prompt"], "best_candidate_seed": r["best_candidate_seed"], "best_candidate_score": r["sparse_vlm_score"], "vqa_feedback": r.get("vqa_feedback", [])} for r in rounds]
            abs_best=max(rounds, key=lambda x: x["sparse_vlm_score"])
            current_best={"round_index": abs_best["round_index"], "candidate_id": abs_best["sparse_vlm_best_candidate"], "prompt": abs_best["best_candidate_prompt"], "seed": abs_best["best_candidate_seed"], "score": abs_best["sparse_vlm_score"], "vqa_feedback": abs_best.get("vqa_feedback", [])}
            print(f"[resume] run analyzer {run_id} round_{round_index}", flush=True)
            refine_plan=pipe.run_deepseek_refine(args, run_dir=str(rdir), original_prompt=original_prompt, round_history=history, current_best_candidate=current_best)
        analyzer_decision=str(refine_plan.get("decision") or "refine")
        rec["refine_plan_json"]=str(refine_path)
        rec["refine_reason"]=refine_plan.get("reason")
        rec["failure_focus"]=refine_plan.get("failure_focus", [])
        rec["rewrite_hints"]=refine_plan.get("rewrite_hints", [])
        rec["next_prompt"]=refine_plan.get("next_prompt")
        rec["next_prompts"]=refine_plan.get("next_prompts", [])
        if analyzer_decision == "reject":
            full_trigger="analyzer_reject"; pre_full="failed_refine"; break
        current_prompts=pipe.prepare_next_round_prompts(refine_plan, rec.get("best_candidate_prompt") or current_prompts[0], args.candidates)
        seed_override=rec.get("best_candidate_seed") if rec["sparse_vlm_score"] >= 0.6 else None
    return run_full_stage(args, original_prompt, plan, run_id, rounds, full_trigger, pre_full, analyzer_decision)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default=str(PROJECT/"logs"/"vqqa_all350_hmt12_planv4_agent"))
    ap.add_argument("--run-root", default=str(PROJECT/"runs"/"real-exp"))
    ap.add_argument("--project-dir", default=str(PROJECT))
    ap.add_argument("--generator-root", default=pipe.DEFAULT_GENERATOR_ROOT)
    ap.add_argument("--python-bin", default=pipe.DEFAULT_PYTHON)
    ap.add_argument("--ckpt-dir", default=pipe.DEFAULT_CKPT_DIR)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--sparse-steps", type=int, default=50)
    ap.add_argument("--full-steps", type=int, default=50)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--sample-shift", type=float, default=8.0)
    ap.add_argument("--guide-scale", type=float, default=6.0)
    ap.add_argument("--sparse-gpus", default="0,1,2,3")
    ap.add_argument("--full-gpu", default="0")
    ap.add_argument("--full-gpus", default="0,1")
    ap.add_argument("--sparse-threshold", type=float, default=0.70)
    ap.add_argument("--full-threshold", type=float, default=0.70)
    ap.add_argument("--generation-timeout", type=int, default=2400)
    ap.add_argument("--gpu-memory-limit-mib", type=int, default=2048)
    ap.add_argument("--gpu-util-limit", type=int, default=10)
    ap.add_argument("--sparse-warmup-ratio", type=float, default=0.10)
    ap.add_argument("--sparse-warmup-steps", type=int, default=0)
    ap.add_argument("--sparse-stride", type=int, default=12)
    ap.add_argument("--analyzer-model", default="deepseek-v4-flash")
    ap.add_argument("--only-missing-full", action="store_true", default=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--indices", default="", help="comma-separated manifest indices")
    ap.add_argument("--summary", default=str(PROJECT/"logs"/"resume_hmt12_planv4"/"summary.jsonl"))
    args=ap.parse_args()
    args.remote="local"
    args.dry_run=False
    args.vlm_workers=None

    manifest=load_json(Path(args.batch_dir)/"benchmark_manifest.json", [])
    if args.indices.strip():
        want={int(x) for x in args.indices.split(",") if x.strip()}
        records=[r for r in manifest if int(r["index"]) in want]
    else:
        records=[r for r in manifest if int(r["index"]) >= args.start_index]
    selected=[]
    for r in records:
        parent=Path(args.run_root)/r["run_id"]
        if args.only_missing_full and list(parent.glob("**/full_wan.mp4")) and (parent/"pipeline_report.json").exists():
            continue
        selected.append(r)
    if args.limit:
        selected=selected[:args.limit]
    print(f"[resume] selected {len(selected)} runs", flush=True)
    summary_path=Path(args.summary); summary_path.parent.mkdir(parents=True, exist_ok=True)
    done=0
    for r in selected:
        t=time.time()
        try:
            rep=resume_one(args, r)
            row={"index": r["index"], "run_id": r["run_id"], "status": rep.get("status"), "full_video_path": rep.get("full_video_path"), "wall_seconds": round(time.time()-t,2)}
            done+=1
        except Exception as exc:
            row={"index": r.get("index"), "run_id": r.get("run_id"), "status": "failed", "error": str(exc)[:2000], "wall_seconds": round(time.time()-t,2)}
            print(f"[resume] FAILED {r.get('run_id')}: {exc}", flush=True)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(f"[resume] complete attempted={len(selected)} success_rows={done} summary={summary_path}", flush=True)

if __name__ == "__main__":
    main()
