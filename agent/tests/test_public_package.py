from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent


def load_pipeline_module():
    path = AGENT_ROOT / "scripts" / "sparse_to_full_vlm_pipeline.py"
    spec = importlib.util.spec_from_file_location("sparsemac_public_pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_prompt_asset_set_is_published():
    expected = {
        "mixed_planner_base.txt",
        "mixed_skill_router.txt",
        "planner_deepseek_v1.txt",
        "semantic_v2/analyzer.txt",
        "semantic_v2/planner.txt",
        "semantic_v2/verifier.txt",
    }
    categories = {
        "action_binding",
        "consistent_attr",
        "dynamic_attr",
        "interaction",
        "motion_binding",
        "numeracy",
        "spatial_relationship",
    }
    expected.update(f"primary_skills/{name}.txt" for name in categories)
    expected.update(f"secondary_skills/{name}.txt" for name in categories)

    prompt_root = AGENT_ROOT / "sparsemac" / "prompts"
    actual = {
        str(path.relative_to(prompt_root))
        for path in prompt_root.rglob("*.txt")
    }
    assert actual == expected


def test_generation_backend_entrypoints_are_published():
    backend = AGENT_ROOT / "generation_backend"
    assert (backend / "generate.py").is_file()
    assert (backend / "generate_sparse.py").is_file()


def test_pipeline_defaults_are_portable_and_local():
    pipeline = load_pipeline_module()
    assert pipeline.DEFAULT_REMOTE == "local"
    assert Path(pipeline.DEFAULT_PROJECT_DIR).resolve() == AGENT_ROOT.resolve()
    assert Path(pipeline.DEFAULT_RUN_ROOT).resolve() == (AGENT_ROOT / "runs").resolve()
    assert Path(pipeline.DEFAULT_GENERATOR_ROOT).resolve() == (
        AGENT_ROOT / "generation_backend"
    ).resolve()
    assert pipeline.DEFAULT_PYTHON == sys.executable


def test_public_agent_tree_contains_no_api_token_literal():
    token_pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
    for path in AGENT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".txt", ".example"}:
            continue
        assert token_pattern.search(path.read_text(encoding="utf-8")) is None, path
