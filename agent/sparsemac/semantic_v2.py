from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


METHOD_ID = "sparsemac-semantic-v2"
METHOD_NAME = "SparseMAC-SemanticV2"
METHOD_LABEL = "SparseMAC-SemanticV2 (Ours, Improved)"
LEGACY_METHOD_ID = "legacy"
SUPPORTED_METHODS = (LEGACY_METHOD_ID, METHOD_ID)
PLANNER_MIN_CRITERIA = 5
MAX_CRITERIA = 10

METHOD_METADATA: dict[str, Any] = {
    "method_id": METHOD_ID,
    "method_name": METHOD_NAME,
    "method_label": METHOD_LABEL,
    "method_family": "SparseMAC",
    "ownership": "ours",
    "variant": "improved_semantic_verification_and_rewrite",
    "base_method": "SparseMAC PlanV4 sparse-to-full agent",
    "inspiration": "VQQA-style semantic question decomposition and concrete prompt expansion",
    "question_policy": "atomic_weighted_semantic_questions",
    "agent_model": "gpt-5.6-luna",
    "verification_scope": "prompt-semantic alignment only",
    "visual_quality_questions": False,
    "frozen_original_prompt_criteria": True,
}

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "semantic_v2"


def method_metadata(method_id: str) -> dict[str, Any]:
    if method_id == METHOD_ID:
        return dict(METHOD_METADATA)
    return {
        "method_id": LEGACY_METHOD_ID,
        "method_name": "SparseMAC-Legacy",
        "method_label": "SparseMAC legacy prompt policy",
        "method_family": "SparseMAC",
        "ownership": "ours",
        "variant": "legacy",
    }


def load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip()


def render_prompt(name: str, **values: object) -> str:
    """Render named placeholders without interpreting JSON braces in prompts."""
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def build_planner_prompt(original_prompt: str, candidates: int) -> str:
    return render_prompt(
        "planner.txt",
        original_prompt=original_prompt,
        candidate_budget=max(1, int(candidates)),
    )


def build_verifier_prompt(
    *,
    original_prompt: str,
    candidate_prompt: str,
    criteria: list[dict[str, Any]],
    frame_count: int = 11,
) -> str:
    return render_prompt(
        "verifier.txt",
        original_prompt=original_prompt,
        candidate_prompt=candidate_prompt,
        semantic_criteria=json.dumps(criteria, ensure_ascii=False, indent=2),
        frame_last_index=max(0, int(frame_count) - 1),
    )


def build_analyzer_prompt(payload: Mapping[str, Any]) -> str:
    return render_prompt(
        "analyzer.txt",
        input_json=json.dumps(dict(payload), ensure_ascii=False, indent=2),
    )


def normalize_criteria(
    raw: object,
    *,
    min_count: int = 2,
    max_count: int = MAX_CRITERIA,
) -> list[dict[str, Any]]:
    """Validate and normalize frozen criteria.

    The current planner is stricter and requests 5-10 atomic questions.  The
    lower default remains two so previously written SemanticV2 plans can still
    be resumed and rescored without silently changing their frozen criteria.
    """
    if min_count < 1 or max_count < min_count:
        raise ValueError("invalid semantic criterion count bounds")
    if not isinstance(raw, list) or not min_count <= len(raw) <= max_count:
        raise ValueError(
            f"SparseMAC-SemanticV2 requires {min_count}-{max_count} semantic criteria"
        )

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"criterion {index} must be an object")
        criterion_id = str(item.get("id") or f"criterion_{index + 1}").strip()
        question = str(item.get("question") or "").strip()
        requirement = str(item.get("requirement") or question).strip()
        criterion_type = str(item.get("type") or "semantic_alignment").strip()
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"criterion {criterion_id} has invalid weight") from exc
        if not criterion_id or criterion_id.lower() in seen_ids:
            raise ValueError(f"criterion id is empty or duplicated: {criterion_id!r}")
        if not question or question.lower() in seen_questions:
            raise ValueError(f"criterion question is empty or duplicated: {question!r}")
        if not question.startswith("On a scale of 0-100, how"):
            raise ValueError(f"criterion {criterion_id} has a non-canonical question")
        if weight <= 0:
            raise ValueError(f"criterion {criterion_id} weight must be positive")
        seen_ids.add(criterion_id.lower())
        seen_questions.add(question.lower())
        result.append(
            {
                "id": criterion_id,
                "type": criterion_type,
                "requirement": requirement,
                "question": question,
                "weight": weight,
            }
        )

    total = sum(float(item["weight"]) for item in result)
    if total <= 0:
        raise ValueError("semantic criterion weights must have a positive sum")
    for item in result:
        item["weight"] = round(float(item["weight"]) / total, 6)
    # Keep the serialized sum stable despite rounding.
    delta = round(1.0 - sum(float(item["weight"]) for item in result), 6)
    result[-1]["weight"] = round(float(result[-1]["weight"]) + delta, 6)
    return result


def criteria_to_vqa_scores(criteria: list[dict[str, Any]]) -> dict[str, float]:
    return {str(item["question"]): float(item["weight"]) for item in criteria}


def criteria_from_vqa_scores(vqa_scores: Mapping[str, object]) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for index, (question, weight) in enumerate(vqa_scores.items()):
        raw.append(
            {
                "id": f"criterion_{index + 1}",
                "type": "semantic_alignment",
                "requirement": str(question),
                "question": str(question),
                "weight": weight,
            }
        )
    return normalize_criteria(raw)


def weighted_score(vqa_results: object, criteria: list[dict[str, Any]]) -> float:
    """Return a normalized 0-1 score from structured criterion answers."""
    if not isinstance(vqa_results, list):
        raise ValueError("vqa_results must be a list")
    by_id = {
        str(item.get("criterion_id") or ""): item
        for item in vqa_results
        if isinstance(item, dict)
    }
    by_question = {
        str(item.get("question") or ""): item
        for item in vqa_results
        if isinstance(item, dict)
    }
    score = 0.0
    for criterion in criteria:
        answer = by_id.get(str(criterion["id"])) or by_question.get(str(criterion["question"]))
        if not answer:
            raise ValueError(f"missing VQA result for criterion {criterion['id']}")
        try:
            value = float(answer.get("vqa_score", answer.get("score")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid score for criterion {criterion['id']}") from exc
        value = max(0.0, min(100.0, value))
        score += value * float(criterion["weight"])
    return round(score / 100.0, 6)
