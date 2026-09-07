import json
import sys
from pathlib import Path
import pytest
from sparsemac.schema import Candidate, Evaluation, Plan
import sparsemac.run as run

class FakeGemini:
    available = True

class FakeDeepSeek:
    available = True
    model = "deepseek-v4-flash"
    def generate_json(self, prompt, **kwargs):
        return {}, "{}"

def test_run_uses_analyzer_selected_prompt_and_seed(tmp_path, monkeypatch):
    # Mock files & directories
    run_root = tmp_path / "runs"
    run_root.mkdir()

    # Mock command line arguments
    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--prompt", "original prompt",
        "--run-root", str(run_root),
        "--run-id", "test_run_id",
        "--candidates", "2",
        "--max-rounds", "1",
        "--dry-run"
    ])

    # Mock planner
    def mock_make_plan(prompt, candidates, **kwargs):
        return Plan(
            original_prompt=prompt,
            prompt_variants=["variant 1", "variant 2"],
            vqa=[],
            metrics=[],
            semantic_plan={}
        )
    monkeypatch.setattr(run, "make_plan", mock_make_plan)

    # Mock GPU detection
    monkeypatch.setattr(run, "detect_idle_gpus", lambda *args, **kwargs: ["gpu0"])

    # Mock sparse batch generator
    def mock_run_sparse_batch(candidates, **kwargs):
        for c in candidates:
            c.status = "dry_run"
        return candidates
    monkeypatch.setattr(run, "run_sparse_batch", mock_run_sparse_batch)

    # Mock evaluate candidates
    def mock_evaluate_candidates(candidates, plan, **kwargs):
        return [
            Evaluation(
                candidate_id=c.candidate_id,
                score=0.8 if i == 0 else 0.5,
                passed=True,
                issues=[],
                strengths=[],
                metric_scores={},
                vqa_results={}
            )
            for i, c in enumerate(candidates)
        ]
    monkeypatch.setattr(run, "evaluate_candidates", mock_evaluate_candidates)

    # Mock build_analysis to return decision="run_full_generation" with selected_prompt and selected_seed
    def mock_build_analysis(plan, evaluations, current_best_prompt, round_index, candidates, gemini, deepseek, current_best_seed):
        return {
            "round": round_index,
            "stop": True,
            "decision": "run_full_generation",
            "selected_prompt": "analyzer selected prompt",
            "selected_seed": 9999,
            "best_candidate_id": "r00_c00",
            "best_score": 0.8,
            "next_prompts": [],
            "next_seeds": []
        }
    monkeypatch.setattr(run, "build_analysis", mock_build_analysis)

    # Track arguments passed to run_full_generation
    captured_generation = {}
    def mock_run_full_generation(prompt, seed, **kwargs):
        captured_generation["prompt"] = prompt
        captured_generation["seed"] = seed
        return True, "mock_success"
    monkeypatch.setattr(run, "run_full_generation", mock_run_full_generation)

    # Run main!
    exit_code = run.main()
    assert exit_code == 0

    # Verify run_full_generation used the analyzer selected values!
    assert captured_generation["prompt"] == "analyzer selected prompt"
    assert captured_generation["seed"] == 9999

    # Verify final.json has the analyzer selected values!
    final_json_path = run_root / "test_run_id" / "final" / "final.json"
    assert final_json_path.exists()
    with open(final_json_path) as f:
        final_data = json.load(f)
    assert final_data["prompt"] == "analyzer selected prompt"
    assert final_data["seed"] == 9999

    # Verify summary.json has the analyzer selected values!
    summary_json_path = run_root / "test_run_id" / "summary.json"
    assert summary_json_path.exists()
    with open(summary_json_path) as f:
        summary_data = json.load(f)
    assert summary_data["prompt"] == "analyzer selected prompt"
    assert summary_data["seed"] == 9999


def test_run_falls_back_to_best_candidate_on_refine_decision(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_root.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--prompt", "original prompt",
        "--run-root", str(run_root),
        "--run-id", "test_run_refine",
        "--candidates", "2",
        "--max-rounds", "1",
        "--dry-run"
    ])

    # Mock planner
    def mock_make_plan(prompt, candidates, **kwargs):
        return Plan(
            original_prompt=prompt,
            prompt_variants=["variant 1", "variant 2"],
            vqa=[],
            metrics=[],
            semantic_plan={}
        )
    monkeypatch.setattr(run, "make_plan", mock_make_plan)

    monkeypatch.setattr(run, "detect_idle_gpus", lambda *args, **kwargs: ["gpu0"])

    def mock_run_sparse_batch(candidates, **kwargs):
        for c in candidates:
            c.status = "dry_run"
        return candidates
    monkeypatch.setattr(run, "run_sparse_batch", mock_run_sparse_batch)

    def mock_evaluate_candidates(candidates, plan, **kwargs):
        return [
            Evaluation(
                candidate_id=c.candidate_id,
                score=0.8 if i == 0 else 0.5,
                passed=True,
                issues=[],
                strengths=[],
                metric_scores={},
                vqa_results={}
            )
            for i, c in enumerate(candidates)
        ]
    monkeypatch.setattr(run, "evaluate_candidates", mock_evaluate_candidates)

    # Mock build_analysis to return decision="refine" (which doesn't stop, but max-rounds is 1 so it exits loop)
    def mock_build_analysis(plan, evaluations, current_best_prompt, round_index, candidates, gemini, deepseek, current_best_seed):
        return {
            "round": round_index,
            "stop": False,
            "decision": "refine",
            "best_candidate_id": "r00_c00",
            "best_score": 0.8,
            "next_prompts": ["next variant"],
            "next_seeds": [12345]
        }
    monkeypatch.setattr(run, "build_analysis", mock_build_analysis)

    captured_generation = {}
    def mock_run_full_generation(prompt, seed, **kwargs):
        captured_generation["prompt"] = prompt
        captured_generation["seed"] = seed
        return True, "mock_success"
    monkeypatch.setattr(run, "run_full_generation", mock_run_full_generation)

    exit_code = run.main()
    assert exit_code == 0

    # Best candidate prompt and seed (from seeds/prompt selection in Candidates loop)
    # The first candidate was created with prompt_variants[0] = "variant 1" and seeds[0] (which is from seed_pool)
    # Let's inspect what final generation used. It should match the best candidate's prompt/seed.
    best_candidate_path = run_root / "test_run_refine" / "best.json"
    with open(best_candidate_path) as f:
        best_data = json.load(f)
    best_prompt = best_data["candidate"]["prompt"]
    best_seed = best_data["candidate"]["seed"]

    assert captured_generation["prompt"] == best_prompt
    assert captured_generation["seed"] == best_seed

    final_json_path = run_root / "test_run_refine" / "final" / "final.json"
    with open(final_json_path) as f:
        final_data = json.load(f)
    assert final_data["prompt"] == best_prompt
    assert final_data["seed"] == best_seed

    summary_json_path = run_root / "test_run_refine" / "summary.json"
    with open(summary_json_path) as f:
        summary_data = json.load(f)
    assert summary_data["prompt"] == best_prompt
    assert summary_data["seed"] == best_seed

