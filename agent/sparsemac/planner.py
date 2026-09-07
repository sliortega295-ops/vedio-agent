from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from typing import Any

from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .luna_client import LunaClient
from .schema import Plan
from .semantic_v2 import (
    LEGACY_METHOD_ID,
    PLANNER_MIN_CRITERIA,
    MAX_CRITERIA,
    METHOD_ID as SEMANTIC_V2_METHOD_ID,
    build_planner_prompt as build_semantic_v2_planner_prompt,
    criteria_to_vqa_scores,
    method_metadata,
    normalize_criteria,
)


PLANNER_PROMPT_TEMPLATE = """You are SparseMAC's planning agent for text-to-video generation.

Goal:
Create a conservative sparse-search plan for a text-to-video benchmark prompt. The plan will be used to generate key-only sparse Wan candidates, verify 11-frame storyboards, and select the best prompt/seed pair before full generation.

Hard rules:
- Preserve the original prompt semantics exactly.
- Do not add new objects, actions, locations, outcomes, styles, camera moves, or story events that are not implied by the original prompt.
- Do not add unmentioned visual style words such as photorealistic, cinematic, cartoon, polished, dramatic lighting, close-up, side view, medium shot, or static camera unless the original prompt says so.
- Prompt variants should mostly rewrite word order and make required relations/actions explicit; they should not enrich unspecified appearance, material finish, camera, lighting, or genre.
- Preserve benchmark attribute words exactly in prompt variants. For example, keep "cozy" as "cozy" instead of replacing it with "comfortable"; keep color, count, direction, object names, actions, and relation words unchanged unless the original wording is grammatically invalid.
- Keep every generation prompt variant in English.
- Make prompt variants visually concrete but concise enough for a text-to-video model.
- Split verification into compact VQA scoring checks that can be judged from the whole storyboard.
- The VQA checks as a set must cover the entire prompt: required objects, exact counts, attributes, spatial relationships, motion direction/path, interactions, semantic state changes, and any explicitly stated scene/camera/style constraints.
- Assign each VQA check an explicit normalized importance weight. The weights should sum to approximately 1.0 and should reflect the importance of that check for this prompt.
- Do not create per-keyframe verification questions.
- Return strict JSON only. No markdown, no comments, no prose outside JSON.

Return this JSON object:
{{
  "original_prompt": string,
  "semantic_plan": {{
    "entities": [{{"name": string, "count": string, "required_attributes": [string]}}],
    "spatial_relations": [string],
    "motion_events": [string],
    "state_changes": [string],
    "camera_or_composition": [string],
    "failure_risks": [string]
  }},
  "prompt_variants": [string],
  "vqa": [string],
  "vqa_scores": {{"exact VQA question text": number}},
  "negative_prompt": string,
  "notes": string
}}

Field requirements:
- prompt_variants must contain at least {candidates} items.
- vqa should contain 4 to 8 high-impact scoring questions that together cover the complete original prompt.
- vqa_scores must have exactly the same questions as vqa, using the question text as the key.
- vqa_scores values should sum to approximately 1.0.
- Give higher weights to prompt-critical checks: required objects, exact counts, attributes, spatial relations, motion direction/path, interactions, and state changes.
- Give lower weights to generic quality/artifact checks unless the prompt specifically asks for visual style or quality.
- If the original prompt does not mention camera, view angle, composition, lighting, genre, or style, keep camera_or_composition empty and do not add VQA questions about those dimensions.
- Use state_change only for semantic state changes such as transformation, color change, opening/closing, growth, damage, or disappearance. Pure position changes belong in motion_events and motion_trajectory.
- Negative prompt should only include generic quality failures and direct semantic failures. Do not add genre/style negatives unless the original prompt specifies a genre/style.

Original prompt:
{prompt}"""


def make_plan(
    prompt: str,
    candidates: int,
    gemini: GeminiClient | None = None,
    deepseek: DeepSeekClient | None = None,
    strict: bool = False,
    method: str = LEGACY_METHOD_ID,
    luna: LunaClient | None = None,
) -> Plan:
    parsed = None
    raw = ""
    semantic_v2 = method == SEMANTIC_V2_METHOD_ID
    planner_client = (luna or LunaClient()) if semantic_v2 else (deepseek or DeepSeekClient())
    provider_label = "GPT-5.6 Luna" if semantic_v2 else "DeepSeek"
    if planner_client.available:
        parsed, raw = planner_client.generate_json(_planner_prompt(prompt, candidates, method=method))
        if not parsed and strict:
            detail = raw or planner_client.last_error or "planner returned empty response"
            raise RuntimeError(f"{provider_label} planner failed or returned invalid JSON: {detail[:800]}")
    elif strict:
        raise RuntimeError(planner_client.last_error or f"{provider_label} planner is unavailable")
    elif not semantic_v2 and gemini and gemini.available:
        parsed, raw = gemini.generate_json(_planner_prompt(prompt, candidates, method=method))
    if parsed:
        try:
            return _plan_from_json(prompt, parsed, candidates, raw, method=method)
        except (TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError(f"Planner returned an invalid {method} plan: {exc}") from exc
            raw = f"{raw}\nPlan normalization error: {exc}"
    if strict:
        raise RuntimeError(f"{provider_label} planner failed before producing a plan")
    plan = heuristic_plan(prompt, candidates, method=method)
    if raw:
        plan.notes = f"{plan.notes} Planner LLM unavailable or invalid: {raw[:300]}"
    return plan


def _planner_prompt(prompt: str, candidates: int, method: str = LEGACY_METHOD_ID) -> str:
    if method == SEMANTIC_V2_METHOD_ID:
        return build_semantic_v2_planner_prompt(prompt, candidates)
    return PLANNER_PROMPT_TEMPLATE.format(prompt=prompt, candidates=candidates)


def heuristic_plan(prompt: str, candidates: int, method: str = LEGACY_METHOD_ID) -> Plan:
    if method == SEMANTIC_V2_METHOD_ID:
        variants = [
            prompt.strip(),
            f"{prompt.strip()} Keep every named subject, count, attribute, and relation explicit while making the required event visibly readable from beginning to end.",
            f"{prompt.strip()} Show a clear spatial layout and an observable action or motion path without changing any object, role, or outcome from the original request.",
            f"{prompt.strip()} Maintain the same subject identities and bindings throughout, with distinct visible start and end states whenever the requested event changes over time.",
        ]
    else:
        variants = [
            prompt.strip(),
            f"{prompt.strip()}, clear subject identity, coherent natural motion, stable camera framing, detailed lighting and background.",
            f"{prompt.strip()}, emphasize the main objects, their spatial relationship, and a readable motion trajectory across the scene.",
            f"{prompt.strip()}, cinematic realistic video, consistent subject appearance, smooth state changes, no abrupt object disappearance.",
        ]
    while len(variants) < candidates:
        variants.append(f"{prompt.strip()}, variant {len(variants)} with stable composition and visible action.")

    vqa = [
        "Are all main objects mentioned in the prompt visible in the keyframes?",
        "Do the objects keep consistent identity and attributes across keyframes?",
        "Is there enough motion or state change for the prompt?",
        "Do the spatial relationships or interactions match the prompt?",
    ]
    vqa_scores = {
        vqa[0]: 0.30,
        vqa[1]: 0.25,
        vqa[2]: 0.20,
        vqa[3]: 0.25,
    }
    semantic_criteria: list[dict[str, Any]] = []
    primary_skill = ""
    must_preserve: list[str] = []
    if method == SEMANTIC_V2_METHOD_ID:
        must_preserve = [prompt.strip()]
        semantic_criteria = normalize_criteria(
            [
                {
                    "id": "required_subjects",
                    "type": "entity_presence",
                    "requirement": "the required subjects are visibly present",
                    "question": "On a scale of 0-100, how clearly are the required subjects visibly present?",
                    "weight": 0.08,
                },
                {
                    "id": "required_context",
                    "type": "semantic_context",
                    "requirement": "the explicitly requested setting or reference context is present",
                    "question": "On a scale of 0-100, how clearly is the explicitly requested setting or reference context visible?",
                    "weight": 0.07,
                },
                {
                    "id": "semantic_task",
                    "type": "semantic_alignment",
                    "requirement": "the prompt-defining action, relation, interaction, count, or state change",
                    "question": "On a scale of 0-100, how completely is the prompt-defining semantic operation fulfilled?",
                    "weight": 0.42,
                },
                {
                    "id": "temporal_or_spatial_execution",
                    "type": "semantic_progression",
                    "requirement": "the requested direction, ordering, trajectory, or state progression",
                    "question": "On a scale of 0-100, how clearly is the requested spatial or temporal progression established?",
                    "weight": 0.28,
                },
                {
                    "id": "semantic_persistence",
                    "type": "semantic_alignment",
                    "requirement": "required subjects and bindings remain correct while the event unfolds",
                    "question": "On a scale of 0-100, how consistently are the required subjects and bindings maintained through the event?",
                    "weight": 0.15,
                },
            ]
        )
        vqa = [item["question"] for item in semantic_criteria]
        vqa_scores = criteria_to_vqa_scores(semantic_criteria)

    return Plan(
        original_prompt=prompt,
        prompt_variants=variants[:candidates],
        vqa=vqa,
        metrics=list(vqa_scores.keys()),
        negative_prompt=(
            ""
            if method == SEMANTIC_V2_METHOD_ID
            else "blurry, deformed, inconsistent identity, duplicated limbs, abrupt disappearance, low quality"
        ),
        notes=(
            f"Heuristic fallback {method} plan; set SPARSEMAC_LUNA_API_KEY for the SemanticV2 planner."
            if method == SEMANTIC_V2_METHOD_ID
            else f"Heuristic fallback {method} plan; set DEEPSEEK_API_KEY for planner LLM."
        ),
        semantic_plan={
            "entities": [],
            "spatial_relations": [],
            "motion_events": [],
            "state_changes": [],
            "camera_or_composition": [],
            "failure_risks": ["Planner LLM was not available."],
        },
        expected_keyframes=[],
        vqa_groups={"general": vqa},
        vqa_scores=vqa_scores,
        method_id=method,
        method_metadata=method_metadata(method),
        primary_skill=primary_skill,
        must_preserve=must_preserve,
        semantic_criteria=semantic_criteria,
    )


def refine_variants(
    original_prompt: str,
    best_prompt: str,
    issues: list[str],
    round_index: int,
    candidates: int,
    gemini: GeminiClient | None = None,
) -> list[str]:
    prompt = f"""Rewrite the video prompt conservatively for the next SparseMAC round.
Original prompt: {original_prompt}
Current best prompt: {best_prompt}
Observed issues: {issues}
Return JSON: {{"variants": [..]}}.
Keep original objects and relationships. Do not add new semantic content."""
    if gemini and gemini.available:
        parsed, _ = gemini.generate_json(prompt)
        variants = parsed.get("variants") if parsed else None
        if isinstance(variants, list) and variants:
            return [str(v) for v in variants[:candidates]]

    suffix = "; address issues: " + "; ".join(issues[:3]) if issues else "; improve visual clarity and motion readability"
    variants = [best_prompt]
    variants.append(f"{best_prompt}{suffix}.")
    variants.append(f"{original_prompt}, with clearer subject consistency, visible action arc, stable composition.")
    variants.append(f"{best_prompt}, avoid the prior defects, keep the same semantics, improve temporal coherence.")
    while len(variants) < candidates:
        variants.append(f"{best_prompt}, conservative refinement round {round_index}.")
    return variants[:candidates]


def seed_pool(prompt: str, round_index: int, count: int) -> list[int]:
    digest = hashlib.sha256(f"{prompt}:{round_index}".encode("utf-8")).hexdigest()
    base = int(digest[:8], 16)
    rng = random.Random(base)
    return [rng.randint(1, 2_147_483_000) for _ in range(count)]


def _plan_from_json(
    original_prompt: str,
    data: dict,
    candidates: int,
    raw: str,
    method: str = LEGACY_METHOD_ID,
) -> Plan:
    variants = [str(v) for v in data.get("prompt_variants", []) if str(v).strip()]
    if not variants:
        variants = heuristic_plan(original_prompt, candidates, method=method).prompt_variants
    while len(variants) < candidates:
        variants.append(f"{original_prompt}, clear subject identity, stable composition, readable temporal progression.")
    semantic_criteria: list[dict[str, Any]] = []
    if method == SEMANTIC_V2_METHOD_ID:
        semantic_criteria = normalize_criteria(
            data.get("semantic_criteria") or data.get("criteria"),
            min_count=PLANNER_MIN_CRITERIA,
            max_count=MAX_CRITERIA,
        )
    vqa_groups = _normalize_vqa_groups(data.get("vqa_groups"))
    vqa = [str(q) for q in data.get("vqa", []) if str(q).strip()]
    if semantic_criteria:
        vqa = [str(item["question"]) for item in semantic_criteria]
    if not vqa and vqa_groups:
        vqa = [question for questions in vqa_groups.values() for question in questions]
    if not vqa:
        vqa = heuristic_plan(original_prompt, candidates, method=method).vqa
    vqa_scores = (
        criteria_to_vqa_scores(semantic_criteria)
        if semantic_criteria
        else _normalize_vqa_scores(data.get("vqa_scores"), vqa)
    )
    expected_keyframes = _normalize_expected_keyframes(data.get("expected_keyframes"), original_prompt)
    semantic_plan = data.get("semantic_plan") if isinstance(data.get("semantic_plan"), dict) else {}
    primary_skill = (
        ""
        if method == SEMANTIC_V2_METHOD_ID
        else str(data.get("primary_skill") or "").strip()
    )
    must_preserve = [
        str(item).strip()
        for item in data.get("must_preserve", [])
        if str(item).strip()
    ] if isinstance(data.get("must_preserve"), list) else []
    if primary_skill:
        semantic_plan = dict(semantic_plan)
        semantic_plan["primary_skill"] = primary_skill
    return Plan(
        original_prompt=str(data.get("original_prompt", original_prompt)),
        prompt_variants=variants[:candidates],
        vqa=vqa,
        metrics=list(vqa_scores.keys()),
        negative_prompt=str(data.get("negative_prompt", "")),
        notes=str(data.get("notes", "")) or f"Planner raw length={len(raw)}",
        semantic_plan=semantic_plan,
        expected_keyframes=expected_keyframes,
        vqa_groups=vqa_groups,
        vqa_scores=vqa_scores,
        method_id=method,
        method_metadata=method_metadata(method),
        primary_skill=primary_skill,
        must_preserve=must_preserve,
        semantic_criteria=semantic_criteria,
    )


def _normalize_vqa_groups(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    groups: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            questions = [str(item) for item in value if str(item).strip()]
            if questions:
                groups[str(key)] = questions
    return groups


def _normalize_vqa_scores(raw: Any, vqa: list[str]) -> dict[str, float]:
    if not vqa:
        return {}
    scores: dict[str, float] = {}
    if isinstance(raw, dict):
        for question in vqa:
            value = raw.get(question)
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score > 0:
                scores[question] = score
    elif isinstance(raw, list):
        for item in raw:
            question = None
            value = None
            if isinstance(item, dict):
                question = item.get("question") or item.get("q")
                value = item.get("weight") or item.get("score")
                if question is None and len(item) == 1:
                    question, value = next(iter(item.items()))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                question, value = item[0], item[1]
            if question is None:
                continue
            text = str(question).strip()
            if text not in vqa:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score > 0:
                scores[text] = score
    if not scores:
        even = round(1.0 / len(vqa), 6)
        return {question: even for question in vqa}
    total = sum(scores.values())
    if total <= 0:
        even = round(1.0 / len(vqa), 6)
        return {question: even for question in vqa}
    normalized = {question: round(scores.get(question, 0.0) / total, 6) for question in vqa}
    missing = [question for question, value in normalized.items() if value <= 0]
    if missing:
        fallback = round(1.0 / len(vqa), 6)
        return {question: fallback for question in vqa}
    return normalized


def _normalize_expected_keyframes(raw: Any, prompt: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", len(frames)))
            except Exception:
                index = len(frames)
            required = item.get("required_visible_elements")
            frames.append(
                {
                    "index": index,
                    "description": str(item.get("description", "")),
                    "required_visible_elements": [str(x) for x in required] if isinstance(required, list) else [],
                    "expected_change_from_previous": str(item.get("expected_change_from_previous", "")),
                }
            )
    valid = [frame for frame in frames if 0 <= frame["index"] <= 10 and frame["description"]]
    return valid[:3]


def plan_to_dict(plan: Plan) -> dict:
    return asdict(plan)
