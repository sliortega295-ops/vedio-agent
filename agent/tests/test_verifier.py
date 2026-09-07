from sparsemac.schema import Candidate
from sparsemac.verifier import _heuristic_issues


def test_heuristic_issues_flags_low_scores():
    issues = _heuristic_issues({"single_frame_quality": 0.1, "keyframe_continuity": 0.1, "dynamic_sufficiency": 0.1})
    assert len(issues) == 3


def test_candidate_dict_roundtrip_fields():
    cand = Candidate("c0", 0, "prompt", 123, "a.mp4", "a.jpg")
    data = cand.to_dict()
    assert data["candidate_id"] == "c0"
    assert data["seed"] == 123



def test_merge_gemini_eval_preserves_failure_reports():
    from sparsemac.verifier import _merge_gemini_eval

    cand = Candidate("c0", 0, "prompt", 123, "a.mp4", "a.jpg")
    cand.score = 0.6
    cand.metrics = {"dynamic_sufficiency": 0.2}
    parsed = {
        "score": 0.4,
        "passed": False,
        "strengths": ["cat is visible"],
        "vqa_results": {"Does the cat move left?": {"passed": False, "evidence": "cat stays center"}},
        "failure_reports": [
            {
                "category": "wrong_motion_direction",
                "severity": "high",
                "evidence": "the cat stays near the center across sampled frames",
                "impact": "leftward motion is not visually established",
                "rewrite_hint": "Specify a clear right-to-left trajectory with separated start and end positions.",
            }
        ],
        "rewrite_hints": ["Keep the same cat visible throughout the sequence."],
    }

    evaluation = _merge_gemini_eval(cand, parsed, "{}")

    assert evaluation.failure_reports[0]["category"] == "wrong_motion_direction"
    assert any("right-to-left trajectory" in hint for hint in evaluation.rewrite_hints)
    assert evaluation.vqa_results["Does the cat move left?"]["passed"] is False
    assert evaluation.issues[0].startswith("wrong_motion_direction")


def test_analyzer_feedback_uses_failure_reports():
    from sparsemac.analyzer import _analysis_feedback
    from sparsemac.schema import Evaluation

    evaluation = Evaluation(
        candidate_id="c0",
        score=0.4,
        passed=False,
        issues=["motion is weak"],
        strengths=[],
        metric_scores={},
        vqa_results={},
        failure_reports=[
            {
                "category": "weak_action",
                "severity": "medium",
                "evidence": "sampled frames show a static pose",
                "impact": "slinking action is not legible",
                "rewrite_hint": "Emphasize low crawling body posture and visible step-by-step motion.",
            }
        ],
        rewrite_hints=["Keep the cat identity stable."],
    )

    feedback = _analysis_feedback(evaluation)

    assert any("weak_action" in item for item in feedback)
    assert any("low crawling body posture" in item for item in feedback)
    assert any("cat identity stable" in item for item in feedback)


def test_deepseek_analyzer_refines_from_structured_evidence():
    from sparsemac.analyzer import build_analysis
    from sparsemac.schema import Evaluation, Plan

    class FakeDeepSeek:
        available = True
        model = "deepseek-v4-flash"

        def __init__(self):
            self.prompt = ""

        def generate_json(self, prompt, **kwargs):
            self.prompt = prompt
            return (
                {
                    "decision": "refine",
                    "reason": "The action is too static.",
                    "confidence": 0.72,
                    "failure_focus": ["weak_action"],
                    "rewrite_hints": ["Emphasize the visible action trajectory."],
                    "next_prompts": [
                        "A cat slinks from left to right with a clear low crawling motion.",
                        "A cat maintains identity while visibly slinking across the scene.",
                    ],
                },
                "{}",
            )

    plan = Plan(
        original_prompt="A cat slinks from left to right.",
        prompt_variants=["A cat slinks from left to right."],
        vqa=["Does the cat visibly slink?"],
        metrics=[],
        semantic_plan={"motion_events": ["cat slinks left to right"]},
    )
    evaluation = Evaluation(
        candidate_id="c0",
        score=0.48,
        passed=False,
        issues=["motion is weak"],
        strengths=[],
        metric_scores={},
        vqa_results={"Does the cat visibly slink?": {"passed": False, "evidence": "cat is static"}},
        failure_reports=[
            {
                "category": "weak_action",
                "severity": "high",
                "evidence": "sampled keyframes show almost no displacement",
                "impact": "the slinking action is not legible",
                "rewrite_hint": "Emphasize a clear low crawling trajectory.",
            }
        ],
        rewrite_hints=["Keep the same cat visible."],
    )
    deepseek = FakeDeepSeek()

    analysis = build_analysis(plan, [evaluation], plan.original_prompt, 0, 2, None, deepseek=deepseek, current_best_seed=123)

    assert '"failure_reports"' in deepseek.prompt
    assert analysis["decision"] == "refine"
    assert analysis["stop"] is False
    assert analysis["next_prompts"][0].startswith("A cat slinks")
    assert analysis["analyzer_model"] == "deepseek-v4-flash"
    assert "Emphasize the visible action trajectory." in analysis["rewrite_hints"]


def test_deepseek_analyzer_can_stop_for_full_generation():
    from sparsemac.analyzer import build_analysis
    from sparsemac.schema import Evaluation, Plan

    class FakeDeepSeek:
        available = True
        model = "deepseek-v4-flash"

        def generate_json(self, prompt, **kwargs):
            return (
                {
                    "decision": "run_full_generation",
                    "reason": "The candidate satisfies the semantic checks.",
                    "confidence": 0.91,
                    "selected_prompt": "A red cube moves above a blue sphere.",
                    "selected_seed": 456,
                    "failure_focus": [],
                    "rewrite_hints": [],
                    "next_prompts": [],
                },
                "{}",
            )

    plan = Plan(
        original_prompt="A red cube moves above a blue sphere.",
        prompt_variants=["A red cube moves above a blue sphere."],
        vqa=["Is the red cube above the blue sphere?"],
        metrics=[],
        semantic_plan={"spatial_relations": ["red cube above blue sphere"]},
    )
    evaluation = Evaluation(
        candidate_id="c0",
        score=0.86,
        passed=True,
        issues=[],
        strengths=["objects and relation are visible"],
        metric_scores={},
        vqa_results={"Is the red cube above the blue sphere?": True},
    )

    analysis = build_analysis(plan, [evaluation], plan.original_prompt, 0, 2, None, deepseek=FakeDeepSeek(), current_best_seed=456)

    assert analysis["decision"] == "run_full_generation"
    assert analysis["stop"] is True
    assert analysis["selected_prompt"] == "A red cube moves above a blue sphere."
    assert analysis["selected_seed"] == 456


def test_evaluate_candidates_uses_llava_for_top_k(monkeypatch):
    from sparsemac.schema import MetricSpec, Plan
    from sparsemac import verifier

    class FakeLlava:
        available = True

        def __init__(self):
            self.calls = []

        def verify_storyboard(self, candidate, plan):
            self.calls.append(candidate.candidate_id)
            return (
                {
                    "score": 0.9,
                    "passed": True,
                    "issues": [],
                    "strengths": ["cat and living room are visible"],
                    "vqa_results": {"Is the cat visible?": {"passed": True, "evidence": "cat appears in the storyboard"}},
                    "failure_reports": [],
                    "rewrite_hints": [],
                },
                "{}",
                1.2,
            )

    plan = Plan(
        original_prompt="A cat in a living room.",
        prompt_variants=["A cat in a living room."],
        vqa=["Is the cat visible?"],
        metrics=[MetricSpec("single_frame_quality", 1.0, "sharp", "")],
    )
    candidates = [
        Candidate("c0", 0, "A cat in a living room.", 1, "a.mp4", "a.jpg", status="ok"),
        Candidate("c1", 0, "A cat in a living room.", 2, "b.mp4", "b.jpg", status="ok"),
    ]
    monkeypatch.setattr(verifier, "make_storyboard", lambda *args, **kwargs: ["frame.jpg"])
    monkeypatch.setattr(verifier, "compute_cv_metrics", lambda *args, **kwargs: {"single_frame_quality": 0.8})
    llava = FakeLlava()

    evaluations = verifier.evaluate_candidates(candidates, plan, llava=llava, top_k_llava=1, top_k_gemini=0)

    assert llava.calls == ["c0"]
    assert evaluations[0].score > 0.7
    assert evaluations[0].vqa_results["Is the cat visible?"]["passed"] is True
