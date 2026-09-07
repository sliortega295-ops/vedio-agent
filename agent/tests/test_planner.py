from sparsemac.planner import _plan_from_json, heuristic_plan, make_plan, seed_pool


def test_heuristic_plan_candidate_count():
    plan = heuristic_plan("A red ball rolls left to right", 4)
    assert len(plan.prompt_variants) == 4
    assert plan.vqa
    assert plan.metrics
    assert plan.expected_keyframes == []


def test_seed_pool_stable():
    assert seed_pool("prompt", 0, 3) == seed_pool("prompt", 0, 3)
    assert len(seed_pool("prompt", 0, 3)) == 3


def test_make_plan_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import sparsemac.deepseek_client as dsc
    original_env = dsc._env
    def mock_env(name, default=None):
        if name == "DEEPSEEK_API_KEY":
            return None
        return original_env(name, default)
    monkeypatch.setattr(dsc, "_env", mock_env)
    plan = make_plan("A red ball rolls left to right", 2)
    assert len(plan.prompt_variants) == 2
    assert "Heuristic fallback" in plan.notes


def test_make_plan_uses_deepseek_json():
    class FakeDeepSeek:
        available = True

        def generate_json(self, prompt):
            assert "A red ball rolls left to right" in prompt
            return {
                "original_prompt": "A red ball rolls left to right",
                "expected_keyframes": [{"index": idx, "description": str(idx)} for idx in (0, 5, 10)],
                "prompt_variants": ["A red ball rolls from left to right."],
                "vqa_groups": {"motion_trajectory": ["Does the ball move left to right?"]},
                "metrics": [{"name": "dynamic_sufficiency", "weight": 1.0, "target": "motion", "rationale": "required"}],
                "notes": "deepseek",
            }, "{}"

    plan = make_plan("A red ball rolls left to right", 2, deepseek=FakeDeepSeek())
    assert plan.notes == "deepseek"
    assert plan.prompt_variants[0] == "A red ball rolls from left to right."
    assert len(plan.prompt_variants) == 2
    assert plan.vqa == ["Does the ball move left to right?"]


def test_plan_from_json_extended_fields():
    data = {
        "original_prompt": "A red ball rolls left to right",
        "semantic_plan": {"entities": [{"name": "red ball", "count": "one", "required_attributes": ["red"]}]},
        "expected_keyframes": [
            {
                "index": idx,
                "description": f"frame {idx}",
                "required_visible_elements": ["red ball"],
                "expected_change_from_previous": "start" if idx == 0 else "moves right",
            }
            for idx in (0, 5, 10)
        ],
        "prompt_variants": ["A red ball rolls from left to right across the floor."],
        "vqa_groups": {"motion_trajectory": ["Does the red ball move left to right?"]},
        "metrics": [{"name": "dynamic_sufficiency", "weight": 1.0, "target": "visible motion", "rationale": "required by prompt"}],
        "negative_prompt": "blurry",
        "notes": "ok",
    }
    plan = _plan_from_json("A red ball rolls left to right", data, 3, "{}")
    assert len(plan.prompt_variants) == 3
    assert plan.vqa == ["Does the red ball move left to right?"]
    assert len(plan.expected_keyframes) == 3
    assert plan.semantic_plan["entities"][0]["name"] == "red ball"
    assert plan.vqa_groups["motion_trajectory"]
