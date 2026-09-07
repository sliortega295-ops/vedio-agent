import json
import os
import sys
from pathlib import Path
import pytest
import shutil

from sparsemac.schema import Plan
from sparsemac.utils import write_json, read_json
import sparsemac.visual_judge as vj

def test_extract_keyframes_graceful_missing_file(tmp_path):
    # Test that keyframe extraction handles missing file gracefully
    res = vj.extract_keyframes(tmp_path / "non_existent.mp4", tmp_path / "out", num_frames=8)
    assert res == []

def test_generate_markdown_report_unavailable():
    data = {
        "status": "unavailable",
        "reason": "GEMINI_API_KEY is not set",
        "run_id": "test_run",
        "original_prompt": "A cute cat running."
    }
    report = vj.generate_markdown_report(data)
    assert "# SparseMAC Visual Judge Report" in report
    assert "Visual judge is unavailable: GEMINI_API_KEY is not set" in report

def test_generate_markdown_report_success():
    data = {
        "status": "success",
        "run_id": "test_run",
        "original_prompt": "A cute cat running.",
        "candidates": {
            "r00_c00": {
                "prompt": "A cute cat running fast.",
                "seed": 42,
                "judgment": {
                    "human_reasonable_score": 0.85,
                    "prompt_alignment": "Very good alignment.",
                    "object_presence": "Cat is present.",
                    "attribute_consistency": "Consistent.",
                    "motion_or_state_change": "Good running motion.",
                    "temporal_consistency": "Consistent.",
                    "visual_quality": "High resolution.",
                    "major_failures": ["slight leg morphing"],
                    "evidence": "Leg morphs slightly at frame 5.",
                    "recommendation": "Try increasing guide scale."
                },
                "raw_response": "{}"
            }
        },
        "final_video": {
            "prompt": "A cute cat running fast.",
            "seed": 42,
            "judgment": {
                "human_reasonable_score": 0.95,
                "prompt_alignment": "Perfect alignment.",
                "object_presence": "Cat is fully present.",
                "attribute_consistency": "Excellent consistency.",
                "motion_or_state_change": "Incredible motion.",
                "temporal_consistency": "Extremely stable.",
                "visual_quality": "Crisp and realistic.",
                "major_failures": [],
                "evidence": "Consistent throughout all frames.",
                "recommendation": "Ready for production."
            },
            "raw_response": "{}"
        }
    }
    report = vj.generate_markdown_report(data)
    assert "# SparseMAC Visual Judge Report" in report
    assert "Candidate r00_c00" in report
    assert "human_reasonable_score" not in report  # It should be formatted as "Human Reasonable Score: `0.85/1.00`"
    assert "Human Reasonable Score" in report
    assert "slight leg morphing" in report
    assert "Final Wan Video" in report
    assert "Perfect alignment" in report

def test_visual_judge_run_graceful_fallback(tmp_path, monkeypatch):
    # Setup a mock run directory
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    plan_path = run_dir / "plan.json"
    write_json(plan_path, {"original_prompt": "A cat jumping."})
    
    # Ensure GEMINI_API_KEY is unset in this test context
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    # Run the visual judge!
    exit_code = vj.visual_judge_run(run_dir)
    assert exit_code == 0
    
    # Check outputs
    json_path = run_dir / "visual_judge.json"
    md_path = run_dir / "visual_judge.md"
    
    assert json_path.exists()
    assert md_path.exists()
    
    json_data = read_json(json_path)
    assert json_data["status"] == "unavailable"
    assert "GEMINI_API_KEY" in json_data["reason"]
    
    with open(md_path) as f:
        md_content = f.read()
    assert "Visual judge is unavailable" in md_content

def test_visual_judge_run_success(tmp_path, monkeypatch):
    # Setup mock run directory
    run_dir = tmp_path / "test_run_success"
    run_dir.mkdir()
    
    plan_path = run_dir / "plan.json"
    write_json(plan_path, {"original_prompt": "A cat jumping."})
    
    round_dir = run_dir / "round_0"
    round_dir.mkdir()
    
    storyboard_dir = round_dir / "candidates" / "r00_c00"
    storyboard_dir.mkdir(parents=True)
    storyboard_file = storyboard_dir / "storyboard_11.jpg"
    with open(storyboard_file, "w") as f:
        f.write("mock_storyboard_data")
        
    cand_path = round_dir / "candidates.json"
    write_json(cand_path, [{
        "candidate_id": "r00_c00",
        "round_index": 0,
        "prompt": "A cat jumping high.",
        "seed": 100,
        "storyboard": str(storyboard_file),
        "sparse_video": "key_only.mp4"
    }])
    
    # Set fake API key
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_testing")
    
    # Mock GeminiClient
    class MockGeminiClient:
        available = True
        def __init__(self, model, api_key):
            pass
        def generate_json(self, prompt, image_paths):
            assert len(image_paths) == 1
            assert image_paths[0] == str(storyboard_file)
            return {
                "human_reasonable_score": 0.8,
                "prompt_alignment": "Nice jumping action.",
                "object_presence": "Cat is present.",
                "attribute_consistency": "Consistent color.",
                "motion_or_state_change": "High jump.",
                "temporal_consistency": "Reasonably stable.",
                "visual_quality": "Sharp frames.",
                "major_failures": [],
                "evidence": "Jump arc is visible.",
                "recommendation": "None."
            }, "mock_raw_response"
            
    monkeypatch.setattr(vj, "GeminiClient", MockGeminiClient)
    
    # Run!
    exit_code = vj.visual_judge_run(run_dir)
    assert exit_code == 0
    
    # Verify outputs
    json_path = run_dir / "visual_judge.json"
    md_path = run_dir / "visual_judge.md"
    
    assert json_path.exists()
    assert md_path.exists()
    
    json_data = read_json(json_path)
    assert json_data["status"] == "success"
    assert "r00_c00" in json_data["candidates"]
    assert json_data["candidates"]["r00_c00"]["judgment"]["human_reasonable_score"] == 0.8
    assert json_data["candidates"]["r00_c00"]["raw_response"] == "mock_raw_response"
