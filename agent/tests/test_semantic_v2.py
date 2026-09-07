from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sparsemac.planner import make_plan
from sparsemac.semantic_v2 import (
    METHOD_ID,
    METHOD_LABEL,
    build_analyzer_prompt,
    build_verifier_prompt,
    method_metadata,
    normalize_criteria,
    weighted_score,
)


def sample_criteria():
    return normalize_criteria(
        [
            {
                "id": "action",
                "type": "action_binding",
                "requirement": "the dog visibly runs across the ground",
                "question": "On a scale of 0-100, how clearly does the dog run across the ground?",
                "weight": 3,
            },
            {
                "id": "setting",
                "type": "semantic_context",
                "requirement": "the running occurs on grassy ground",
                "question": "On a scale of 0-100, how clearly does the running occur on grassy ground?",
                "weight": 1,
            },
        ]
    )


def assert_prompt_has_no_method_identity(prompt: str):
    assert "SparseMAC" not in prompt
    assert "VQQA" not in prompt
    assert METHOD_ID not in prompt


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sparse_to_full_vlm_pipeline.py"
    spec = importlib.util.spec_from_file_location("sparsemac_test_pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_is_explicitly_ours_and_not_the_vqqa_baseline():
    metadata = method_metadata(METHOD_ID)
    assert metadata["method_label"] == METHOD_LABEL
    assert metadata["method_family"] == "SparseMAC"
    assert metadata["ownership"] == "ours"
    assert metadata["visual_quality_questions"] is False
    assert "VQQA" not in metadata["method_id"]


def test_semantic_criteria_are_frozen_weighted_and_non_overlapping():
    criteria = sample_criteria()
    assert len(criteria) == 2
    assert sum(item["weight"] for item in criteria) == pytest.approx(1.0)
    assert criteria[0]["weight"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="2-10"):
        normalize_criteria(criteria[:1])
    with pytest.raises(ValueError, match="2-10"):
        normalize_criteria(criteria * 6)


def test_semantic_verifier_ignores_sparse_preview_quality_and_scores_criteria_only():
    criteria = sample_criteria()
    prompt = build_verifier_prompt(
        original_prompt="A dog runs across grassy ground.",
        candidate_prompt="A complete expanded running prompt.",
        criteria=criteria,
    )
    assert "expert sparse-keyframe video semantic fidelity evaluator" in prompt
    assert_prompt_has_no_method_identity(prompt)
    assert "Do not lower semantic scores for ordinary sparse-preview blur" in prompt
    assert "Do not add a separate global or visual-quality score" in prompt
    assert "CRITICAL QUALITY & INTEGRITY RULE" not in prompt
    score = weighted_score(
        [
            {"criterion_id": "action", "vqa_score": 80},
            {"criterion_id": "setting", "vqa_score": 40},
        ],
        criteria,
    )
    assert score == pytest.approx(0.70)


def test_semantic_analyzer_requests_complete_concrete_rewrites():
    prompt = build_analyzer_prompt(
        {
            "original_prompt": "A dog runs across grassy ground.",
            "candidate_budget": 4,
            "frozen_semantic_criteria": sample_criteria(),
            "current_best_candidate": {"score": 0.5},
        }
    )
    compact = " ".join(prompt.split())
    assert "do not make only a tiny local edit" in compact
    assert "do not obey an arbitrary word budget" in compact
    assert "A long prompt is acceptable" in compact
    assert 'Do not turn "a/an" into "exactly one"' in prompt
    assert "exactly candidate_budget complete prompts" in compact
    assert "expert AI video prompt refinement specialist" in compact
    assert_prompt_has_no_method_identity(prompt)


def test_semantic_planner_output_is_labeled_and_questions_are_frozen():
    class FakeLuna:
        available = True
        model = "gpt-5.6-luna"

        def generate_json(self, prompt):
            assert "expert text-to-video semantic planning specialist" in prompt
            assert_prompt_has_no_method_identity(prompt)
            assert "primary_skill" not in prompt
            assert "choose one primary semantic skill" not in prompt
            return (
                {
                    "original_prompt": "A dog runs across grassy ground.",
                    "must_preserve": ["dog", "running", "grassy ground"],
                    "semantic_plan": {"motion_events": ["dog runs across the ground"]},
                    "prompt_variants": [
                        f"Complete semantic generation prompt variant {index} for a dog visibly running across grassy ground."
                        for index in range(4)
                    ],
                    "criteria": [
                        {
                            "id": f"criterion_{index}",
                            "type": "semantic_alignment",
                            "requirement": f"atomic semantic fact {index}",
                            "question": f"On a scale of 0-100, how clearly is atomic semantic fact {index} fulfilled?",
                            "weight": 1,
                        }
                        for index in range(5)
                    ],
                    "negative_prompt": "",
                    "notes": "semantic-v2",
                },
                "{}",
            )

    plan = make_plan(
        "A dog runs across grassy ground.",
        4,
        luna=FakeLuna(),
        strict=True,
        method=METHOD_ID,
    )
    assert plan.method_id == METHOD_ID
    assert plan.method_metadata["ownership"] == "ours"
    assert plan.primary_skill == ""
    assert "primary_skill" not in plan.to_dict()
    assert plan.vqa == [item["question"] for item in plan.semantic_criteria]
    assert len(plan.semantic_criteria) == 5
    assert plan.vqa_scores[plan.vqa[0]] == pytest.approx(0.2)


def test_current_planner_requires_vqqa_style_atomic_question_count():
    class TooCompactLuna:
        available = True
        model = "gpt-5.6-luna"

        def generate_json(self, prompt):
            return (
                {
                    "original_prompt": "A dog runs across grassy ground.",
                    "prompt_variants": ["A dog visibly runs across grassy ground."] * 4,
                    "criteria": sample_criteria(),
                },
                "{}",
            )

    with pytest.raises(RuntimeError, match="5-10"):
        make_plan(
            "A dog runs across grassy ground.",
            4,
            luna=TooCompactLuna(),
            strict=True,
            method=METHOD_ID,
        )


def test_planner_prompt_prefers_atomic_diagnostics_without_word_budget():
    from sparsemac.semantic_v2 import build_planner_prompt

    prompt = build_planner_prompt("A dog runs across grassy ground.", 4)
    compact = " ".join(prompt.split())
    assert "5-10 frozen weighted atomic semantic criteria" in prompt
    assert "Presence and setting questions are allowed" in prompt
    assert "there is no target word count" in compact
    assert 'Do not convert an indefinite singular such as "a child"' in prompt
    assert "40-70" not in prompt
    assert "70-130" not in prompt


def test_pipeline_recomputes_model_score_from_frozen_criteria():
    pipeline = load_pipeline_module()
    criteria = sample_criteria()
    plan = {"semantic_criteria": criteria}
    result = pipeline.normalize_semantic_v2_judgment(
        plan,
        {
            "score": 0.99,
            "vqa_results": [
                {"criterion_id": "action", "vqa_score": 60},
                {"criterion_id": "setting", "vqa_score": 20},
            ],
        },
    )
    assert result["score"] == pytest.approx(0.50)
    assert result["method_id"] == METHOD_ID
    assert result["score_mode"] == "weighted_frozen_original_prompt_semantic_criteria"
