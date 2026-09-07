from __future__ import annotations

import json
from typing import Any

from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .luna_client import DEFAULT_LUNA_MODEL, LunaClient
from .planner import refine_variants, seed_pool
from .schema import Evaluation, Plan
from .semantic_v2 import (
    LEGACY_METHOD_ID,
    METHOD_ID as SEMANTIC_V2_METHOD_ID,
    build_analyzer_prompt as build_semantic_v2_analyzer_prompt,
    method_metadata,
)


ANALYZER_MODEL = "deepseek-v4-flash"
ANALYZER_PROMPT = """You are SparseMAC's analyzer agent.
Your job is to decide whether the current best sparse/key-only video candidate is good enough to run full generation, or whether SparseMAC should refine prompts and sample another sparse round.

Original prompt:
{original_prompt}

Current best prompt:
{current_best_prompt}

Current best seed:
{current_best_seed}

Round index: {round_index}
Candidate budget for next round: {candidates}

Planner semantic plan:
{semantic_plan}

Planner VQA questions:
{vqa}

Ranked candidate evaluations, including VLM failure_reports and rewrite_hints:
{evaluations}

Return strict JSON only with keys:
- decision: one of ["run_full_generation", "refine"]
- reason: short explanation
- confidence: number 0-1
- selected_prompt: prompt to use if decision is run_full_generation
- selected_seed: integer seed to use if decision is run_full_generation
- failure_focus: list of the most important unresolved issues
- rewrite_hints: list of actionable prompt rewrite hints
- next_prompts: list of {candidates} prompt variants if decision is refine

Decision policy:
- Choose run_full_generation if the best candidate has high score, passes most VQA/semantic checks, and only minor or cosmetic issues remain.
- Choose refine if there are high-severity semantic failures, wrong subject, wrong motion/direction, weak target action, identity drift, scene mismatch, or low confidence.
- Use the VLM evidence. Do not ignore failure_reports.
- Keep prompt variants in English.
- Preserve the original prompt's objects, actions, relations, attributes, and benchmark wording. Do not add unmentioned style/camera/material details.
- If refining, rewrite conservatively: fix the observed failure without changing the task.
"""


def should_stop(evaluations: list[Evaluation], threshold: float = 0.78) -> bool:
    if not evaluations:
        return False
    best = max(evaluations, key=lambda e: e.score)
    if best.passed and best.score >= threshold:
        return True
    passed = [e for e in evaluations if e.passed]
    return bool(passed) and best.score >= threshold


def build_analysis(
    plan: Plan,
    evaluations: list[Evaluation],
    current_best_prompt: str,
    round_index: int,
    candidates: int,
    gemini: GeminiClient | None,
    deepseek: DeepSeekClient | None = None,
    current_best_seed: int | None = None,
    method: str | None = None,
    luna: LunaClient | None = None,
) -> dict:
    method_id = method or plan.method_id or LEGACY_METHOD_ID
    ranked = sorted(evaluations, key=lambda e: e.score, reverse=True)
    best_eval = ranked[0] if ranked else None
    analyzer_feedback = _analysis_feedback(best_eval) if best_eval else ["no successful candidates"]
    next_seeds = seed_pool(plan.original_prompt + current_best_prompt, round_index + 1, candidates)

    llm_result, analyzer_model = _run_analyzer(
        plan=plan,
        ranked=ranked,
        current_best_prompt=current_best_prompt,
        current_best_seed=current_best_seed,
        round_index=round_index,
        candidates=candidates,
        deepseek=deepseek,
        luna=luna,
        method=method_id,
    )
    if llm_result:
        decision = _normalize_decision(llm_result.get("decision"))
        next_prompts = _normalize_prompt_list(llm_result.get("next_prompts"), candidates)
        next_prompt = str(llm_result.get("next_prompt") or "").strip()
        if next_prompt and next_prompt not in next_prompts:
            next_prompts.insert(0, next_prompt)
            next_prompts = next_prompts[:candidates]
        if decision == "refine" and not next_prompts:
            next_prompts = refine_variants(
                plan.original_prompt,
                current_best_prompt,
                analyzer_feedback,
                round_index + 1,
                candidates,
                gemini=gemini,
            )
        elif decision == "run_full_generation":
            next_prompts = [str(llm_result.get("selected_prompt") or current_best_prompt)]
        analysis = _base_analysis(
            round_index=round_index,
            best_eval=best_eval,
            analyzer_feedback=analyzer_feedback,
            next_prompts=next_prompts,
            next_seeds=next_seeds,
        )
        analysis.update(
            {
                "decision": decision,
                "stop": decision == "run_full_generation",
                "reason": str(llm_result.get("reason", "")),
                "confidence": _bounded_float(llm_result.get("confidence"), 0.0),
                "selected_prompt": str(llm_result.get("selected_prompt") or current_best_prompt),
                "selected_seed": _int_or_none(llm_result.get("selected_seed"), current_best_seed),
                "failure_focus": _string_list(llm_result.get("failure_focus")),
                "rewrite_hints": _dedupe(analysis.get("rewrite_hints", []) + _string_list(llm_result.get("rewrite_hints"))),
                "analyzer_model": analyzer_model,
                "analyzer_raw": llm_result,
                "method_id": method_id,
                "method_metadata": method_metadata(method_id),
            }
        )
        return analysis

    decision = "run_full_generation" if should_stop(evaluations) else "refine"
    next_prompts = [current_best_prompt] if decision == "run_full_generation" else refine_variants(
        plan.original_prompt,
        current_best_prompt,
        analyzer_feedback,
        round_index + 1,
        candidates,
        gemini=gemini,
    )
    analysis = _base_analysis(
        round_index=round_index,
        best_eval=best_eval,
        analyzer_feedback=analyzer_feedback,
        next_prompts=next_prompts,
        next_seeds=next_seeds,
    )
    analysis.update(
        {
            "decision": decision,
            "stop": decision == "run_full_generation",
            "reason": "Analyzer unavailable; used threshold fallback.",
            "confidence": best_eval.score if best_eval else 0.0,
            "selected_prompt": current_best_prompt,
            "selected_seed": current_best_seed,
            "failure_focus": analyzer_feedback[:3],
            "analyzer_model": None,
            "analyzer_raw": None,
            "method_id": method_id,
            "method_metadata": method_metadata(method_id),
        }
    )
    return analysis


def _run_analyzer(
    *,
    plan: Plan,
    ranked: list[Evaluation],
    current_best_prompt: str,
    current_best_seed: int | None,
    round_index: int,
    candidates: int,
    deepseek: DeepSeekClient | None,
    luna: LunaClient | None,
    method: str = LEGACY_METHOD_ID,
) -> tuple[dict[str, Any] | None, str | None]:
    client = (
        luna or LunaClient(model=DEFAULT_LUNA_MODEL)
        if method == SEMANTIC_V2_METHOD_ID
        else deepseek or DeepSeekClient(model=ANALYZER_MODEL)
    )
    if not client.available:
        return None, None
    if method == SEMANTIC_V2_METHOD_ID:
        prompt = build_semantic_v2_analyzer_prompt(
            {
                "original_prompt": plan.original_prompt,
                "candidate_budget": candidates,
                "current_best_prompt": current_best_prompt,
                "current_best_seed": current_best_seed,
                "round_index": round_index,
                "frozen_semantic_plan": plan.semantic_plan,
                "frozen_semantic_criteria": plan.semantic_criteria,
                "ranked_candidate_evaluations": [_evaluation_summary(e) for e in ranked],
            }
        )
    else:
        prompt = ANALYZER_PROMPT.format(
            original_prompt=plan.original_prompt,
            current_best_prompt=current_best_prompt,
            current_best_seed=current_best_seed,
            round_index=round_index,
            candidates=candidates,
            semantic_plan=json.dumps(plan.semantic_plan, ensure_ascii=False, indent=2),
            vqa=json.dumps(plan.vqa, ensure_ascii=False, indent=2),
            evaluations=json.dumps([_evaluation_summary(e) for e in ranked], ensure_ascii=False, indent=2),
        )
    max_tokens = 7000 if method == SEMANTIC_V2_METHOD_ID else 3000
    parsed, _ = client.generate_json(prompt, max_tokens=max_tokens, temperature=0.2)
    return (parsed if isinstance(parsed, dict) else None), client.model


def _evaluation_summary(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "candidate_id": evaluation.candidate_id,
        "score": evaluation.score,
        "passed": evaluation.passed,
        "issues": evaluation.issues,
        "strengths": evaluation.strengths,
        "metric_scores": evaluation.metric_scores,
        "vqa_results": evaluation.vqa_results,
        "failure_reports": evaluation.failure_reports,
        "rewrite_hints": evaluation.rewrite_hints,
    }


def _base_analysis(
    *,
    round_index: int,
    best_eval: Evaluation | None,
    analyzer_feedback: list[str],
    next_prompts: list[str],
    next_seeds: list[int],
) -> dict:
    return {
        "round": round_index,
        "stop": False,
        "best_candidate_id": best_eval.candidate_id if best_eval else None,
        "best_score": best_eval.score if best_eval else 0.0,
        "issues": best_eval.issues if best_eval else analyzer_feedback,
        "failure_reports": best_eval.failure_reports if best_eval else [],
        "rewrite_hints": best_eval.rewrite_hints if best_eval else [],
        "analyzer_feedback": analyzer_feedback,
        "next_prompts": next_prompts,
        "next_seeds": next_seeds,
    }


def _analysis_feedback(evaluation: Evaluation | None) -> list[str]:
    if evaluation is None:
        return ["no successful candidates"]
    feedback: list[str] = []
    for report in evaluation.failure_reports:
        category = str(report.get("category", "failure")).strip() or "failure"
        severity = str(report.get("severity", "medium")).strip() or "medium"
        evidence = str(report.get("evidence", "")).strip()
        impact = str(report.get("impact", "")).strip()
        hint = str(report.get("rewrite_hint", "")).strip()
        parts = [f"{category} severity={severity}"]
        if impact:
            parts.append(f"impact: {impact}")
        if evidence:
            parts.append(f"evidence: {evidence}")
        if hint:
            parts.append(f"rewrite_hint: {hint}")
        feedback.append("; ".join(parts))
    feedback.extend(f"rewrite_hint: {hint}" for hint in evaluation.rewrite_hints if hint)
    feedback.extend(evaluation.issues)
    return _dedupe(feedback) or ["no actionable verifier feedback"]


def _normalize_decision(value: object) -> str:
    text = str(value or "").strip().lower()
    return "run_full_generation" if text in {"run_full_generation", "reject"} else "refine"


def _normalize_prompt_list(raw: object, candidates: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    prompts = [str(item).strip() for item in raw if str(item).strip()]
    return _dedupe(prompts)[:candidates]


def _string_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _bounded_float(value: object, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _int_or_none(value: object, default: int | None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
