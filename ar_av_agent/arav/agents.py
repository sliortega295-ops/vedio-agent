"""Minimal JSON multimodal client; no generic agent framework or hidden memory."""
from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from .core import Candidate, Stage, Task, Verdict, canonical, write_json


def prompt(name: str) -> str:
    return files("arav").joinpath("prompts", name + ".md").read_text(encoding="utf-8")


class JSONClient:
    """For a user-configured Chat Completions-compatible endpoint.

    No automatic HTTP retry: duplicate paid calls and retry costs must be visible.
    The endpoint MUST support image_url and, for AV, input_audio in the same turn.
    An image-only model is deliberately not treated as an audio evaluator.
    """
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not model:
            raise ValueError("Valid explicit base URL and model required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.usage: list[dict] = []

    def ask(self, system: str, payload: dict, images: tuple[str, ...] = (), audio: str | None = None) -> dict:
        content: list[dict] = [{"type": "text", "text": canonical(payload)}]
        if len(images) > 64:
            raise ValueError("More than 64 evidence images; use an explicit evaluation window")
        for filename in images:
            path = Path(filename)
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Evidence image exceeds 8 MiB")
            mime = mimetypes.guess_type(filename)[0]
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                raise ValueError("JPEG/PNG/WebP evidence required")
            data = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        if audio:
            path = Path(audio)
            if path.suffix.lower() != ".wav" or path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("A PCM WAV evidence file of at most 20 MiB is required")
            with wave.open(str(path), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError("Uncompressed WAV required")
            content.append({"type": "input_audio", "input_audio": {
                "data": base64.b64encode(path.read_bytes()).decode(), "format": "wav"}})
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                {"role": "user", "content": content}], "response_format": {"type": "json_object"},
                "temperature": 0, "max_tokens": 4096}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url, canonical(body).encode(), headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
            parsed = json.loads(raw["choices"][0]["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            self.usage.append({"model": self.model, "elapsed_s": time.perf_counter() - started,
                               "status": "error", "error_type": type(exc).__name__})
            raise RuntimeError(f"JSON model request failed ({type(exc).__name__}); no candidate accepted") from None
        self.usage.append({"model": self.model, "elapsed_s": time.perf_counter() - started,
                           "status": "ok", "usage": raw.get("usage")})
        return parsed

    @classmethod
    def from_env(cls, prefix: str = "ARAV_LLM") -> "JSONClient":
        return cls(os.environ[prefix + "_BASE_URL"], os.environ[prefix + "_MODEL"],
                   os.environ.get(prefix + "_API_KEY", ""))


class LLMAgents:
    def __init__(self, controller: JSONClient, verifier: JSONClient, audio_capable: bool = False):
        self.controller, self.verifier = controller, verifier
        self.audio_capable = audio_capable

    def plan(self, task: Task) -> list[Stage]:
        answer = self.controller.ask(prompt("planner"), {"task": asdict(task)})
        if set(answer) != {"stages"} or not isinstance(answer["stages"], list):
            raise ValueError("Planner must return only stages")
        stages = [Stage.parse(row) for row in answer["stages"]]
        if not stages or len({s.id for s in stages}) != len(stages):
            raise ValueError("Plan must have nonempty, unique stage IDs")
        return stages

    def verify(self, task: Task, stage: Stage, prefix: list[dict], candidate: Candidate) -> Verdict:
        if candidate.observation:
            raise ValueError("Mock oracle observations cannot be used by a real verifier")
        if task.audio_required and not self.audio_capable:
            raise ValueError("Actual audio-capable verifier required")
        # Include the preceding boundary images, not only isolated candidate frames.
        boundary = tuple(prefix[-1]["candidate"]["frame_paths"][-2:]) if prefix else ()
        result = self.verifier.ask(prompt("verifier"), {
            "original_task": asdict(task), "current_stage": asdict(stage),
            "completed_history": [{"stage_id": x["stage_id"], "start_s": x["candidate"]["start_s"],
                                   "duration_s": x["candidate"]["video_frames"] / x["candidate"]["fps"]} for x in prefix],
            "boundary_image_count": len(boundary), "candidate_image_count": len(candidate.frame_paths),
            "candidate_start_s": candidate.start_s, "candidate_duration_s": candidate.duration_s,
            "has_audio_evidence": bool(candidate.audio_path),
        }, images=boundary + candidate.frame_paths, audio=candidate.audio_path if task.audio_required else None)
        verdict = Verdict.parse(result)
        if any(e["time_s"] > candidate.duration_s + 1e-6 for e in verdict.evidence):
            raise ValueError("Evidence timestamps must be within current candidate")
        return verdict

    def repair(self, task: Task, stage: Stage, verdict: Verdict, control: dict, rules: list[str]) -> dict:
        return self.controller.ask(prompt("controller"), {
            "original_task": asdict(task), "stage": asdict(stage), "failed_control": control,
            "verifier_feedback": asdict(verdict), "validated_rules": rules})


class MockDriver:
    """CPU state-machine test double. It does not synthesize realistic video."""
    name, mock, native_ar, joint_av = "mock-av-state-machine", True, True, True

    def __init__(self, fail_first: int = 1):
        self.fail_first = fail_first
        self.calls = 0
        self.reset()

    def reset(self) -> None:
        self.state = {"video_cache": [], "audio_cache": [], "position": 0, "stage": None, "stage_chunks": 0}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def restore(self, state: dict) -> None:
        self.state = copy.deepcopy(state)

    def synchronize(self) -> None:
        return None

    def sample(self, stage: Stage, control: dict, seed: int, out: Path, start_s: float) -> Candidate:
        self.calls += 1
        if self.state["stage"] != stage.id:
            self.state["stage"], self.state["stage_chunks"] = stage.id, 0
        self.state["stage_chunks"] += 1
        self.state["position"] += 1
        self.state["video_cache"].append(seed)
        self.state["audio_cache"].append(seed + 1)
        observation = {"video_ok": self.calls > self.fail_first, "audio_ok": True, "sync_ok": True,
                       "stage_complete": self.state["stage_chunks"] >= stage.nominal_chunks}
        evidence = out / "mock-observation.json"
        write_json(evidence, {"TEST_DOUBLE_ONLY": True, "seed": seed, **observation})
        audio = out / "mock-silence.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
        return Candidate(str(seed), None, (), str(audio), (str(evidence), str(audio)), start_s,
                         16, 16.0, 16000, 16000, start_s, observation)

    def commit(self, candidate: Candidate) -> None:
        return None


class MockAgents:
    audio_capable = True

    def verify(self, task: Task, stage: Stage, prefix: list[dict], candidate: Candidate) -> Verdict:
        obs = candidate.observation
        return Verdict.parse({**obs, "confidence": 1.0, "feedback": "TEST_DOUBLE: scripted result",
                              "evidence": [{"modality": "av", "time_s": 0.0, "observation": "synthetic fixture"}]})

    def repair(self, task: Task, stage: Stage, verdict: Verdict, control: dict, rules: list[str]) -> dict:
        return {"action": "reprompt", "video_prompt": control["video_prompt"] + " [REPAIRED]",
                "audio_prompt": control["audio_prompt"], "reason": "TEST_DOUBLE_ONLY"}
