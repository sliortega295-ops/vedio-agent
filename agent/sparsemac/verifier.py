from __future__ import annotations

from .gemini_client import GeminiClient
from .llava_client import LlavaClient
from .schema import Candidate, Evaluation, Plan
from .semantic_v2 import (
    METHOD_ID as SEMANTIC_V2_METHOD_ID,
    build_verifier_prompt as build_semantic_v2_verifier_prompt,
    weighted_score as semantic_v2_weighted_score,
)
from .vision import compute_cv_metrics, make_storyboard


VERIFY_PROMPT = """You are SparseMAC's video verification agent.
Evaluate the 11-frame storyboard for a Wan key-only sparse candidate.
Original prompt: {original_prompt}
Candidate prompt: {candidate_prompt}
VQA questions: {vqa}

Return strict JSON with keys:
score: number 0-1,
passed: boolean,
issues: list of short strings,
strengths: list of short strings,
vqa_results: object mapping each question to either a boolean or an object with passed/evidence/issue/rewrite_hint,
failure_reports: list of objects with keys category, severity, evidence, impact, rewrite_hint,
rewrite_hints: list of concise prompt-rewrite suggestions for the analyzer.

Rules:
- Do not only answer the VQA questions; explain what is wrong and how to fix it.
- failure_reports should be actionable for the analyzer, e.g. missing_subject, wrong_motion_direction, weak_action, identity_drift, scene_mismatch, temporal_jump, artifact_or_blur.
- evidence must cite visible frame/storyboard evidence, not assumptions.
- rewrite_hint should say what to emphasize, remove, or constrain in the next prompt.
- Be strict about object existence, subject consistency, motion sufficiency, and prompt alignment.
"""


def evaluate_candidates(
    candidates: list[Candidate],
    plan: Plan,
    gemini: GeminiClient | None = None,
    top_k_gemini: int = 2,
    llava: LlavaClient | None = None,
    top_k_llava: int = 0,
) -> list[Evaluation]:
    for cand in candidates:
        if cand.status not in {"ok", "dry_run"}:
            cand.metrics = {}
            cand.score = 0.0
            cand.issues = [cand.error or "candidate generation failed"]
            cand.passed = False
            continue
        if cand.status == "dry_run":
            cand.metrics = {"single_frame_quality": 0.5, "keyframe_continuity": 0.5, "dynamic_sufficiency": 0.5}
        else:
            frame_paths = make_storyboard(cand.sparse_video, cand.storyboard, frames=11)
            cand.metrics = compute_cv_metrics(frame_paths)
        cand.score = _weighted_score(cand.metrics, plan)
        cand.issues = _heuristic_issues(cand.metrics)
        cand.passed = cand.score >= 0.72 and not cand.issues

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    gemini_ids = {c.candidate_id for c in ranked[:top_k_gemini] if c.status in {"ok", "dry_run"}}
    llava_ids = {c.candidate_id for c in ranked[:top_k_llava] if c.status in {"ok", "dry_run"}}
    evaluations = []
    for cand in ranked:
        eval_obj = _heuristic_eval(cand)
        if gemini and gemini.available and cand.candidate_id in gemini_ids and cand.status != "dry_run":
            verify_prompt = (
                build_semantic_v2_verifier_prompt(
                    original_prompt=plan.original_prompt,
                    candidate_prompt=cand.prompt,
                    criteria=plan.semantic_criteria,
                    frame_count=11,
                )
                if plan.method_id == SEMANTIC_V2_METHOD_ID
                else VERIFY_PROMPT.format(
                    original_prompt=plan.original_prompt,
                    candidate_prompt=cand.prompt,
                    vqa=plan.vqa,
                )
            )
            parsed, raw = gemini.generate_json(
                verify_prompt,
                image_paths=[cand.storyboard],
            )
            if parsed:
                eval_obj = (
                    _merge_semantic_v2_eval(cand, plan, parsed, raw)
                    if plan.method_id == SEMANTIC_V2_METHOD_ID
                    else _merge_gemini_eval(cand, parsed, raw)
                )
        if llava and llava.available and cand.candidate_id in llava_ids and cand.status != "dry_run":
            parsed, raw, elapsed = llava.verify_storyboard(cand, plan)
            if parsed:
                eval_obj = _merge_vlm_eval(cand, parsed, raw, "llava-v1.5-13b", elapsed)
        evaluations.append(eval_obj)
    return evaluations


def _weighted_score(metrics: dict[str, float], plan: Plan) -> float:
    base = 0.0
    weight = 0.0
    for spec in plan.metrics:
        value = metrics.get(spec.name)
        if value is None:
            if "object" in spec.name or "relation" in spec.name:
                value = 0.55
            else:
                value = 0.5
        base += spec.weight * float(value)
        weight += spec.weight
    return max(0.0, min(1.0, base / weight if weight else 0.0))


def _heuristic_issues(metrics: dict[str, float]) -> list[str]:
    issues = []
    if metrics.get("single_frame_quality", 0) < 0.35:
        issues.append("low single-frame sharpness or visible collapse")
    if metrics.get("keyframe_continuity", 0) < 0.35:
        issues.append("weak continuity between adjacent keyframes")
    if metrics.get("dynamic_sufficiency", 0) < 0.18:
        issues.append("insufficient visible dynamics across keyframes")
    return issues


def _heuristic_eval(cand: Candidate) -> Evaluation:
    return Evaluation(
        candidate_id=cand.candidate_id,
        score=cand.score,
        passed=cand.passed,
        issues=cand.issues,
        strengths=["heuristic CV score available"] if cand.metrics else [],
        metric_scores=cand.metrics,
        vqa_results={},
        failure_reports=[_issue_to_report(issue) for issue in cand.issues],
        rewrite_hints=[_rewrite_hint_for_issue(issue) for issue in cand.issues],
        raw_response="",
    )


def _merge_gemini_eval(cand: Candidate, parsed: dict, raw: str) -> Evaluation:
    gemini_score = float(parsed.get("score", cand.score))
    score = max(0.0, min(1.0, 0.45 * cand.score + 0.55 * gemini_score))
    failure_reports = _normalize_failure_reports(parsed.get("failure_reports", []))
    rewrite_hints = _normalize_rewrite_hints(parsed.get("rewrite_hints", []), failure_reports)
    issues = [str(x) for x in parsed.get("issues", [])]
    if not issues:
        issues = [_report_to_issue(report) for report in failure_reports] or cand.issues
    strengths = [str(x) for x in parsed.get("strengths", [])]
    vqa_results = parsed.get("vqa_results", {})
    if not failure_reports:
        failure_reports = [_issue_to_report(issue) for issue in issues]
    if not rewrite_hints:
        rewrite_hints = [_rewrite_hint_for_issue(issue) for issue in issues]
    passed = bool(parsed.get("passed", False)) and score >= 0.72
    cand.score = score
    cand.issues = issues
    cand.passed = passed
    return Evaluation(
        candidate_id=cand.candidate_id,
        score=score,
        passed=passed,
        issues=issues,
        strengths=strengths,
        metric_scores=cand.metrics,
        vqa_results=vqa_results if isinstance(vqa_results, dict) else {},
        failure_reports=failure_reports,
        rewrite_hints=rewrite_hints,
        raw_response=raw,
    )


def _merge_semantic_v2_eval(cand: Candidate, plan: Plan, parsed: dict, raw: str) -> Evaluation:
    """Use only the frozen weighted semantic criteria—never CV/quality blending."""
    vqa_results = parsed.get("vqa_results", [])
    score = semantic_v2_weighted_score(vqa_results, plan.semantic_criteria)
    issues = [str(item) for item in parsed.get("failure_reasons", []) if str(item).strip()]
    if not issues:
        issues = [
            str(item.get("reason") or "").strip()
            for item in vqa_results
            if isinstance(item, dict)
            and float(item.get("vqa_score", item.get("score", 100))) < 82
            and str(item.get("reason") or "").strip()
        ]
    failure_reports = [_issue_to_report(issue) for issue in issues]
    rewrite_hints = [_rewrite_hint_for_issue(issue) for issue in issues]
    passed = score >= 0.72
    cand.score = score
    cand.issues = issues
    cand.passed = passed
    return Evaluation(
        candidate_id=cand.candidate_id,
        score=score,
        passed=passed,
        issues=issues,
        strengths=[],
        metric_scores={"semantic_v2_weighted_score": score},
        vqa_results={
            str(item.get("criterion_id") or item.get("question") or index): item
            for index, item in enumerate(vqa_results)
            if isinstance(item, dict)
        },
        failure_reports=failure_reports,
        rewrite_hints=rewrite_hints,
        raw_response=raw,
    )


def _merge_vlm_eval(cand: Candidate, parsed: dict, raw: str, model_name: str, elapsed_sec: float | None = None) -> Evaluation:
    vlm_score = float(parsed.get("score", cand.score))
    score = max(0.0, min(1.0, 0.35 * cand.score + 0.65 * vlm_score))
    failure_reports = _normalize_failure_reports(parsed.get("failure_reports", []))
    rewrite_hints = _normalize_rewrite_hints(parsed.get("rewrite_hints", []), failure_reports)
    issues = [str(x) for x in parsed.get("issues", []) if str(x).strip()]
    if not issues:
        issues = [_report_to_issue(report) for report in failure_reports] or cand.issues
    strengths = [str(x) for x in parsed.get("strengths", []) if str(x).strip()]
    vqa_results = parsed.get("vqa_results", {})
    if not failure_reports:
        failure_reports = [_issue_to_report(issue) for issue in issues]
    if not rewrite_hints:
        rewrite_hints = [_rewrite_hint_for_issue(issue) for issue in issues]
    passed = bool(parsed.get("passed", False)) and score >= 0.72
    cand.score = score
    cand.issues = issues
    cand.passed = passed
    raw_payload = {
        "model": model_name,
        "elapsed_sec": elapsed_sec,
        "response": raw,
    }
    return Evaluation(
        candidate_id=cand.candidate_id,
        score=score,
        passed=passed,
        issues=issues,
        strengths=strengths,
        metric_scores=cand.metrics,
        vqa_results=vqa_results if isinstance(vqa_results, dict) else {},
        failure_reports=failure_reports,
        rewrite_hints=rewrite_hints,
        raw_response=str(raw_payload),
    )



def _normalize_failure_reports(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    reports: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            category = str(item.get("category", "unspecified_failure")).strip() or "unspecified_failure"
            severity = str(item.get("severity", "medium")).strip() or "medium"
            evidence = str(item.get("evidence", "")).strip()
            impact = str(item.get("impact", "")).strip()
            rewrite_hint = str(item.get("rewrite_hint", "")).strip()
            reports.append(
                {
                    "category": category,
                    "severity": severity,
                    "evidence": evidence,
                    "impact": impact,
                    "rewrite_hint": rewrite_hint,
                }
            )
        elif str(item).strip():
            reports.append(_issue_to_report(str(item)))
    return reports


def _normalize_rewrite_hints(raw: object, reports: list[dict[str, object]]) -> list[str]:
    hints: list[str] = []
    if isinstance(raw, list):
        hints.extend(str(item).strip() for item in raw if str(item).strip())
    elif isinstance(raw, str) and raw.strip():
        hints.append(raw.strip())
    for report in reports:
        hint = str(report.get("rewrite_hint", "")).strip()
        if hint:
            hints.append(hint)
    return _dedupe(hints)


def _report_to_issue(report: dict[str, object]) -> str:
    category = str(report.get("category", "failure")).strip() or "failure"
    impact = str(report.get("impact", "")).strip()
    evidence = str(report.get("evidence", "")).strip()
    if impact and evidence:
        return f"{category}: {impact} Evidence: {evidence}"
    if impact:
        return f"{category}: {impact}"
    if evidence:
        return f"{category}: {evidence}"
    return category


def _issue_to_report(issue: str) -> dict[str, object]:
    return {
        "category": "cv_or_semantic_failure",
        "severity": "medium",
        "evidence": issue,
        "impact": issue,
        "rewrite_hint": _rewrite_hint_for_issue(issue),
    }


def _rewrite_hint_for_issue(issue: str) -> str:
    issue_lower = issue.lower()
    if "dynamic" in issue_lower or "motion" in issue_lower:
        return "Make the required action and trajectory explicit, with visible start and end states."
    if "continuity" in issue_lower or "jump" in issue_lower:
        return "Emphasize stable subject identity and smooth temporal progression between key moments."
    if "sharpness" in issue_lower or "blur" in issue_lower or "collapse" in issue_lower:
        return "Add generic quality constraints against blur, deformation, and collapsed frames."
    return "Clarify the missing or weak prompt requirement without adding unmentioned content."


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
