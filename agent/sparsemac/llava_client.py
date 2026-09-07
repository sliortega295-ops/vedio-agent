from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .schema import Candidate, Plan


DEFAULT_LLAVA_PYTHON = os.environ.get("SPARSEMAC_LLAVA_PYTHON", "")
DEFAULT_LLAVA_ROOT = os.environ.get("SPARSEMAC_LLAVA_ROOT", "")
DEFAULT_LLAVA_MODEL = os.environ.get("SPARSEMAC_LLAVA_MODEL", "")

LLAVA_VERIFY_QUERY = """The image is an 11-frame storyboard from a generated video candidate, ordered left-to-right and top-to-bottom.

Original prompt:
{original_prompt}

Candidate prompt:
{candidate_prompt}

VQA questions:
{vqa}

Return strict JSON only with keys:
- score: number from 0 to 1
- passed: boolean
- issues: list of short strings
- strengths: list of short strings
- vqa_results: object mapping each VQA question to an object with passed/evidence/issue/rewrite_hint
- failure_reports: list of objects with category, severity, evidence, impact, rewrite_hint
- rewrite_hints: list of concise prompt-rewrite suggestions

Be strict about the requested subject, scene, direction of motion, action type, identity consistency, and visible temporal progression. Evidence must cite visible storyboard frames or frame ranges."""


class LlavaClient:
    def __init__(
        self,
        *,
        python_bin: str = DEFAULT_LLAVA_PYTHON,
        llava_root: str = DEFAULT_LLAVA_ROOT,
        model_path: str = DEFAULT_LLAVA_MODEL,
        gpu_id: str | int | None = None,
        timeout: int = 180,
    ):
        self.python_bin = python_bin
        self.llava_root = llava_root
        self.model_path = model_path
        self.gpu_id = str(gpu_id) if gpu_id is not None else os.environ.get("LLAVA_GPU_ID", "0")
        self.timeout = timeout
        self.model = "llava-v1.5-13b"
        self.available = Path(self.python_bin).exists() and Path(self.llava_root).exists() and Path(self.model_path).exists()
        self.last_error = "" if self.available else "LLaVA python, source root, or model path is missing"

    def verify_storyboard(self, candidate: Candidate, plan: Plan) -> tuple[dict[str, Any] | None, str, float]:
        query = LLAVA_VERIFY_QUERY.format(
            original_prompt=plan.original_prompt,
            candidate_prompt=candidate.prompt,
            vqa=json.dumps(plan.vqa, ensure_ascii=False),
        )
        return self.generate_json(candidate.storyboard, query)

    def generate_json(self, image_path: str, query: str) -> tuple[dict[str, Any] | None, str, float]:
        if not self.available:
            return None, self.last_error, 0.0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        command = [
            self.python_bin,
            "-m",
            "llava.eval.run_llava",
            "--model-path",
            self.model_path,
            "--image-file",
            image_path,
            "--query",
            query,
            "--temperature",
            "0",
            "--max_new_tokens",
            "900",
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                command,
                cwd=self.llava_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except Exception as exc:
            self.last_error = f"LLaVA execution failed: {exc}"
            return None, self.last_error, time.time() - start
        elapsed = time.time() - start
        raw = (proc.stdout or "").strip()
        if proc.returncode != 0:
            self.last_error = f"LLaVA rc={proc.returncode}: {(proc.stderr or raw)[-1200:]}"
            return None, self.last_error, elapsed
        parsed = _extract_json(raw)
        if parsed is None:
            self.last_error = f"LLaVA returned non-JSON output: {raw[-1200:]}"
        return parsed, raw, elapsed


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    parsed = _loads_json_object(text)
    if parsed is not None:
        return parsed
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return _loads_json_object(text[start : end + 1])
    return None


def _loads_json_object(text: str) -> dict[str, Any] | None:
    for candidate in (text, text.replace("\\_", "_")):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except Exception:
            continue
    return None
