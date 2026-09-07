from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .analyzer import build_analysis
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .generator import run_full_generation, run_sparse_batch
from .gpu import detect_idle_gpus
from .llava_client import DEFAULT_LLAVA_MODEL, DEFAULT_LLAVA_PYTHON, DEFAULT_LLAVA_ROOT, LlavaClient
from .luna_client import DEFAULT_LUNA_MODEL, LunaClient
from .planner import make_plan, seed_pool
from .schema import Candidate
from .semantic_v2 import LEGACY_METHOD_ID, SUPPORTED_METHODS, method_metadata
from .utils import append_jsonl, ensure_dir, sanitize_slug, write_json
from .verifier import evaluate_candidates


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATOR_ROOT = os.environ.get(
    "SPARSEMAC_GENERATOR_ROOT", str(PACKAGE_ROOT / "generation_backend")
)
DEFAULT_WAN_CHECKPOINT = os.environ.get("SPARSEMAC_WAN_CHECKPOINT", "")
DEFAULT_PYTHON = os.environ.get("SPARSEMAC_PYTHON", sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseMAC Wan sparse-search runner")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generator-root", default=DEFAULT_GENERATOR_ROOT)
    parser.add_argument("--ckpt-dir", default=DEFAULT_WAN_CHECKPOINT)
    parser.add_argument("--sparse-python", default=DEFAULT_PYTHON)
    parser.add_argument("--full-python", default=DEFAULT_PYTHON)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sparse-steps", type=int, default=50)
    parser.add_argument("--sparse-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--sparse-warmup-steps", type=int, default=0)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    parser.add_argument("--deepseek-model", default=None)
    parser.add_argument("--analyzer-deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--luna-model", default=DEFAULT_LUNA_MODEL)
    parser.add_argument("--luna-base-url", default=None)
    parser.add_argument("--llava-verify", action="store_true")
    parser.add_argument("--top-k-llava", type=int, default=2)
    parser.add_argument("--llava-python", default=DEFAULT_LLAVA_PYTHON)
    parser.add_argument("--llava-root", default=DEFAULT_LLAVA_ROOT)
    parser.add_argument("--llava-model-path", default=DEFAULT_LLAVA_MODEL)
    parser.add_argument("--llava-gpu-id", default="0")
    parser.add_argument("--gpu-memory-limit-mib", type=int, default=2048)
    parser.add_argument("--gpu-util-limit", type=int, default=10)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--agent-method",
        choices=SUPPORTED_METHODS,
        default=LEGACY_METHOD_ID,
        help="Use sparsemac-semantic-v2 for our improved semantic agent policy.",
    )
    return parser.parse_args()


def cleanup_non_best_warmup_latents(candidates: list[Candidate], best_candidate_id: str) -> None:
    for candidate in candidates:
        path = candidate.warmup_latent_path
        if not path or candidate.candidate_id == best_candidate_id:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def read_warmup_metadata(path: str) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise RuntimeError(f"Warmup latent is missing metadata: {path}")
    return payload["metadata"]


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"{int(time.time())}_{sanitize_slug(args.prompt)}"
    run_dir = ensure_dir((Path(args.run_root) / run_id).resolve())
    manifest = run_dir / "manifest.jsonl"

    active_method = method_metadata(args.agent_method)
    append_jsonl(
        manifest,
        {
            "event": "start",
            "run_id": run_id,
            "prompt": args.prompt,
            "method_id": args.agent_method,
            "method_metadata": active_method,
        },
    )
    gemini = GeminiClient(model=args.gemini_model)
    deepseek = DeepSeekClient(model=args.deepseek_model)
    analyzer_deepseek = DeepSeekClient(model=args.analyzer_deepseek_model)
    luna = LunaClient(model=args.luna_model, base_url=args.luna_base_url)
    llava = (
        LlavaClient(
            python_bin=args.llava_python,
            llava_root=args.llava_root,
            model_path=args.llava_model_path,
            gpu_id=args.llava_gpu_id,
        )
        if args.llava_verify
        else None
    )
    plan_kwargs = {
        "gemini": gemini,
        "deepseek": deepseek,
        "strict": True,
        "method": args.agent_method,
    }
    if args.agent_method != LEGACY_METHOD_ID:
        plan_kwargs["luna"] = luna
    plan = make_plan(args.prompt, args.candidates, **plan_kwargs)
    write_json(run_dir / "plan.json", plan.to_dict())

    best: Candidate | None = None
    best_eval = None
    prompts = plan.prompt_variants[: args.candidates]
    seeds = seed_pool(args.prompt, 0, args.candidates)
    all_rounds = []

    for round_index in range(args.max_rounds):
        round_dir = ensure_dir(run_dir / f"round_{round_index}")
        cand_dir = ensure_dir(round_dir / "candidates")
        gpu_ids = detect_idle_gpus(
            max_count=args.candidates,
            memory_limit_mib=args.gpu_memory_limit_mib,
            util_limit=args.gpu_util_limit,
        )
        append_jsonl(manifest, {"event": "round_start", "round": round_index, "gpus": gpu_ids})

        candidates = []
        for idx in range(args.candidates):
            prompt = prompts[idx % len(prompts)]
            seed = seeds[idx % len(seeds)]
            cid = f"r{round_index:02d}_c{idx:02d}"
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

        candidates = run_sparse_batch(
            candidates,
            gpu_ids=gpu_ids,
            generator_root=args.generator_root,
            ckpt_dir=args.ckpt_dir,
            python_bin=args.sparse_python,
            size=args.size,
            sample_steps=args.sparse_steps,
            sample_shift=args.sample_shift,
            guide_scale=args.guide_scale,
            warmup_ratio=args.sparse_warmup_ratio,
            warmup_steps=args.sparse_warmup_steps,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        write_json(round_dir / "candidates.json", [c.to_dict() for c in candidates])

        verifier_client = luna if args.agent_method != LEGACY_METHOD_ID else gemini
        evaluations = evaluate_candidates(
            candidates,
            plan,
            gemini=verifier_client,
            top_k_gemini=min(2, args.candidates),
            llava=llava,
            top_k_llava=min(args.top_k_llava, args.candidates) if llava else 0,
        )
        write_json(round_dir / "scores.json", [e.to_dict() for e in evaluations])
        write_json(round_dir / "gemini_eval.json", [e.to_dict() for e in evaluations if e.raw_response])

        round_best_eval = max(evaluations, key=lambda e: e.score) if evaluations else None
        round_best = next((c for c in candidates if round_best_eval and c.candidate_id == round_best_eval.candidate_id), None)
        if round_best and (best_eval is None or round_best_eval.score > best_eval.score):
            best = round_best
            best_eval = round_best_eval
            write_json(run_dir / "best.json", {"candidate": best.to_dict(), "evaluation": best_eval.to_dict()})
        if round_best:
            cleanup_non_best_warmup_latents(candidates, round_best.candidate_id)

        analysis_prompt = round_best.prompt if round_best else (best.prompt if best else prompts[0])
        analysis_seed = round_best.seed if round_best else (best.seed if best else None)
        analysis_kwargs = {
            "deepseek": analyzer_deepseek,
            "current_best_seed": analysis_seed,
        }
        if args.agent_method != LEGACY_METHOD_ID:
            analysis_kwargs["luna"] = luna
        analysis = build_analysis(
            plan,
            evaluations,
            analysis_prompt,
            round_index,
            args.candidates,
            gemini,
            **analysis_kwargs,
        )
        write_json(round_dir / "analysis.json", analysis)
        all_rounds.append(analysis)
        append_jsonl(manifest, {"event": "round_done", "round": round_index, "best": analysis.get("best_candidate_id"), "score": analysis.get("best_score")})

        if analysis.get("stop"):
            append_jsonl(manifest, {"event": "stop", "round": round_index, "reason": analysis.get("decision", "analysis_stop")})
            break
        prompts = analysis["next_prompts"]
        seeds = analysis["next_seeds"]

    if best is None:
        append_jsonl(manifest, {"event": "failed", "reason": "no candidate succeeded"})
        return 2

    last_analysis = all_rounds[-1] if all_rounds else {}
    final_prompt = str(last_analysis.get("selected_prompt") or best.prompt)
    final_seed = int(last_analysis.get("selected_seed") or best.seed)

    if not args.skip_final:
        if not best.warmup_latent_path:
            if not args.dry_run:
                raise RuntimeError(f"Best candidate is missing warmup_latent_path: {best.candidate_id}")
        warmup_metadata = {} if args.dry_run else read_warmup_metadata(best.warmup_latent_path)
        final_path = str(ensure_dir(run_dir / "final") / "full_wan.mp4")
        gpu_ids = detect_idle_gpus(1, args.gpu_memory_limit_mib, args.gpu_util_limit)
        ok, log = run_full_generation(
            prompt=final_prompt,
            seed=final_seed,
            generator_root=args.generator_root,
            ckpt_dir=args.ckpt_dir,
            python_bin=args.full_python,
            save_file=final_path,
            size=args.size,
            sample_steps=args.full_steps,
            sample_shift=args.sample_shift,
            guide_scale=args.guide_scale,
            gpu_id=gpu_ids[0],
            init_latent_path=best.warmup_latent_path,
            dry_run=args.dry_run,
        )
        write_json(run_dir / "final" / "final.json", {
            "ok": ok,
            "path": final_path,
            "log": log,
            "prompt": final_prompt,
            "seed": final_seed,
            "candidate_id": best.candidate_id,
            "warmup_reused": bool(best.warmup_latent_path),
            "warmup_start_step": warmup_metadata.get("next_step_index"),
            "warmup_latent_path": best.warmup_latent_path,
            "warmup_latent_metadata": warmup_metadata,
        })
        append_jsonl(manifest, {"event": "final_done", "ok": ok, "path": final_path, "log": log})

    write_json(run_dir / "summary.json", {
        "run_id": run_id,
        "method_id": args.agent_method,
        "method_metadata": active_method,
        "best": best.to_dict(),
        "prompt": final_prompt,
        "seed": final_seed,
        "rounds": all_rounds
    })

    print(f"SparseMAC run complete: {run_dir}")
    print(f"Best candidate: {best.candidate_id} score={best.score:.3f} seed={best.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
