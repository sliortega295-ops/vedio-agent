"""Append-only, fail-before-commit execution. No committed-prefix backtracking.

`accepted` means accepted by a fallible verifier, NOT provably correct.
Drivers own tensor/KV/decoder state; the engine owns committed media and policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Atomic manifest replacement; never overwrite committed media."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(canonical(data) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def safe_id(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:80]
    return (stem or "task") + "-" + digest(value)[:10]


def require_bool(value: Any, name: str, nullable: bool = False) -> bool | None:
    if value is None and nullable:
        return None
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean" + (" or null" if nullable else ""))
    return value


@dataclass(frozen=True)
class Stage:
    id: str
    goal: str
    video_prompt: str
    audio_prompt: str = ""
    checks: tuple[str, ...] = ()
    min_chunks: int = 1
    max_chunks: int = 8
    nominal_chunks: int = 4

    @classmethod
    def parse(cls, row: dict) -> "Stage":
        allowed = set(cls.__dataclass_fields__)
        if set(row) - allowed:
            raise ValueError(f"Unknown stage keys: {set(row) - allowed}")
        data = dict(row)
        data["checks"] = tuple(data.get("checks", ()))
        result = cls(**data)
        for name in ("id", "goal", "video_prompt"):
            if not isinstance(getattr(result, name), str) or not getattr(result, name).strip():
                raise ValueError(f"Nonempty {name} required")
        if not isinstance(result.audio_prompt, str) or not all(isinstance(x, str) for x in result.checks):
            raise ValueError("Prompts and checks must be strings")
        if any(type(getattr(result, key)) is not int for key in ("min_chunks", "max_chunks", "nominal_chunks")):
            raise ValueError("Chunk counts must be integers")
        if not 1 <= result.min_chunks <= result.nominal_chunks <= result.max_chunks <= 10000:
            raise ValueError("Expected 1 <= min_chunks <= nominal_chunks <= max_chunks <= 10000")
        return result


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    audio_required: bool = False
    split: str = "test"
    source: str = "custom"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, row: dict) -> "Task":
        keys = set(cls.__dataclass_fields__)
        result = cls(**{k: v for k, v in row.items() if k in keys})
        require_bool(result.audio_required, "audio_required")
        if not isinstance(result.id, str) or not result.id.strip() or not isinstance(result.prompt, str) or not result.prompt.strip():
            raise ValueError("Task id and original prompt are required")
        if result.split not in {"train", "dev", "test", "smoke"}:
            raise ValueError("Invalid split")
        return result


@dataclass(frozen=True)
class Candidate:
    id: str
    video_path: str | None
    frame_paths: tuple[str, ...]
    audio_path: str | None
    artifacts: tuple[str, ...]
    start_s: float
    video_frames: int
    fps: float
    audio_samples: int = 0
    sample_rate: int = 0
    audio_start_s: float | None = None
    observation: dict = field(default_factory=dict)  # mock only; never sent to real judge
    driver_costs: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.video_frames / self.fps

    def validate(self, task: Task, prefix: list[dict], root: Path, mock: bool) -> None:
        if type(self.video_frames) is not int or self.video_frames <= 0:
            raise ValueError("Candidate must contain actual video frames")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        expected = sum(x["candidate"]["video_frames"] / x["candidate"]["fps"] for x in prefix)
        if not math.isfinite(self.start_s) or abs(self.start_s - expected) > 1e-6:
            raise ValueError("Non-contiguous video timestamp")
        if not self.artifacts:
            raise ValueError("No media artifacts")
        files = set(self.artifacts)
        for value in [self.video_path, self.audio_path, *self.frame_paths]:
            if value and value not in files:
                raise ValueError("Every evidence/media file must be integrity-tracked")
        for name in files:
            path = Path(name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Invalid or out-of-run artifact: {name}")
        if not mock and (not self.video_path or not self.frame_paths):
            raise ValueError("Real candidate requires video and decoded image evidence")
        if task.audio_required:
            if type(self.audio_samples) is not int or type(self.sample_rate) is not int:
                raise ValueError("Audio counts and rates must be integers")
            if not self.audio_path or self.audio_samples <= 0 or self.sample_rate <= 0:
                raise ValueError("Audio-required task has missing audio")
            if self.audio_start_s is None or not math.isfinite(self.audio_start_s):
                raise ValueError("Audio presentation timestamp missing")
            if abs(self.audio_start_s - self.start_s) > 1 / self.sample_rate:
                raise ValueError("Audio/video presentation starts disagree")
            with wave.open(self.audio_path, "rb") as wav:
                if wav.getnframes() != self.audio_samples or wav.getframerate() != self.sample_rate:
                    raise ValueError("WAV header disagrees with candidate metadata")
            tolerance = 1 / self.fps + 1 / self.sample_rate
            if abs(self.audio_samples / self.sample_rate - self.duration_s) > tolerance:
                raise ValueError("Audio/video duration mismatch")


@dataclass(frozen=True)
class Verdict:
    video_ok: bool | None
    audio_ok: bool | None
    sync_ok: bool | None
    stage_complete: bool
    confidence: float
    feedback: str
    evidence: tuple[dict, ...] = ()

    @classmethod
    def parse(cls, row: dict) -> "Verdict":
        expected = set(cls.__dataclass_fields__)
        if set(row) != expected:
            raise ValueError(f"Verdict must have exactly {sorted(expected)}")
        for name in ("video_ok", "audio_ok", "sync_ok"):
            require_bool(row[name], name, nullable=True)
        require_bool(row["stage_complete"], "stage_complete")
        confidence = row["confidence"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite in [0,1]")
        if not isinstance(row["feedback"], str) or not isinstance(row["evidence"], list):
            raise ValueError("Malformed verifier evidence")
        for evidence in row["evidence"]:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("observation"), str):
                raise ValueError("Evidence requires an observation")
            if evidence.get("modality") not in {"video", "audio", "av"}:
                raise ValueError("Evidence modality must be video/audio/av")
            if type(evidence.get("time_s")) not in (int, float) or not math.isfinite(evidence["time_s"]) or evidence["time_s"] < 0:
                raise ValueError("Evidence time_s must be nonnegative and finite")
        return cls(**{**row, "evidence": tuple(row["evidence"])})

    def accepted(self, audio_required: bool, threshold: float) -> bool:
        modalities = {x["modality"] for x in self.evidence}
        visual_evidence = bool(modalities & {"video", "av"})
        audio_evidence = bool(modalities & {"audio", "av"})
        return bool(self.video_ok is True and visual_evidence and self.confidence >= threshold
                    and (not audio_required or (self.audio_ok is True and self.sync_ok is True and audio_evidence)))


@dataclass(frozen=True)
class Limits:
    attempts_per_chunk: int = 3
    max_candidates: int = 128
    wall_seconds: float = 3600
    threshold: float = 0.8

    def __post_init__(self) -> None:
        if type(self.attempts_per_chunk) is not int or self.attempts_per_chunk < 1:
            raise ValueError("attempts_per_chunk must be positive")
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive and finite")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0,1]")


class Driver(Protocol):
    """snapshot MUST include both modalities, KV/cross-KV, positions and decoders.

    No snapshot may alias mutable live state. Model weights are NOT snapshots.
    commit promotes only the tentative state. restore never rewrites past media.
    """
    name: str
    mock: bool
    native_ar: bool
    joint_av: bool
    def reset(self) -> None: ...
    def snapshot(self) -> Any: ...
    def restore(self, state: Any) -> None: ...
    def sample(self, stage: Stage, control: dict, seed: int, out: Path, start_s: float) -> Candidate: ...
    def commit(self, candidate: Candidate) -> None: ...
    def synchronize(self) -> None: ...


class Agents(Protocol):
    audio_capable: bool
    def verify(self, task: Task, stage: Stage, prefix: list[dict], candidate: Candidate) -> Verdict: ...
    def repair(self, task: Task, stage: Stage, verdict: Verdict, control: dict, rules: list[str]) -> dict: ...


def check_prefix(prefix: list[dict]) -> None:
    for chunk in prefix:
        for name, expected in chunk["hashes"].items():
            if file_hash(name) != expected:
                raise RuntimeError(f"Committed artifact was changed: {name}")


def run(task: Task, stages: list[Stage], driver: Driver, agents: Agents, output: Path,
        method: str = "feedback", seed: int = 0, limits: Limits | None = None,
        rules: list[str] | None = None) -> dict:
    limits = limits or Limits()
    if method not in {"vanilla", "retry", "feedback", "evolved"}:
        raise ValueError("Unknown method")
    if not stages or len({s.id for s in stages}) != len(stages):
        raise ValueError("Nonempty plan with unique stage IDs required")
    if not driver.native_ar:
        raise ValueError("This experiment requires a native autoregressive driver")
    if task.audio_required and not driver.joint_av:
        raise ValueError("This driver is video-only; dubbing is NOT joint autoregressive AV")
    if task.audio_required and method != "vanilla" and not agents.audio_capable:
        raise ValueError("Verifier cannot inspect actual audio")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prefix: list[dict] = []
    events: list[dict] = []
    completed: list[str] = []
    status, error = "execution_complete", None
    rules = rules or []
    if method != "evolved" and rules:
        raise ValueError("Rules only permitted in evolved condition")
    metadata = {"task": asdict(task), "stages": [asdict(x) for x in stages], "seed": seed,
                "method": method, "limits": asdict(limits), "driver": driver.name,
                "mock": driver.mock, "rules_hash": digest(rules), "rules": rules}
    write_json(output / "run.json", metadata)
    driver.reset()
    write_json(output / "prefix.json", prefix)

    def log(event: dict) -> None:
        events.append(event)
        with (output / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(event) + "\n")
            stream.flush()

    class StopRun(Exception):
        pass

    try:
        for stage in stages:
            control = {"video_prompt": stage.video_prompt, "audio_prompt": stage.audio_prompt}
            for stage_chunk in range(stage.max_chunks):
                accepted = False
                for attempt in range(1 if method == "vanilla" else limits.attempts_per_chunk):
                    if len(events) >= limits.max_candidates or time.perf_counter() - started >= limits.wall_seconds:
                        raise StopRun("budget_exhausted")
                    check_prefix(prefix)
                    # Seed is stable across runs and independent of Python's randomized hash().
                    candidate_seed = int(digest([seed, task.id, stage.id, stage_chunk, attempt])[:15], 16)
                    candidate_dir = output / "candidates" / f"{len(events):05d}"
                    candidate_dir.mkdir(parents=True)
                    event = {"index": len(events), "task_id": task.id, "split": task.split,
                             "stage_id": stage.id, "stage_chunk": stage_chunk, "attempt": attempt,
                             "seed": candidate_seed, "control": dict(control), "status": "started"}
                    driver.synchronize()
                    timer = time.perf_counter()
                    snapshot = driver.snapshot()
                    driver.synchronize()
                    event["snapshot_s"] = time.perf_counter() - timer
                    keep = False
                    stop_after = None
                    try:
                        timer = time.perf_counter()
                        try:
                            candidate = driver.sample(stage, control, candidate_seed, candidate_dir,
                                                      sum(x["candidate"]["video_frames"] / x["candidate"]["fps"] for x in prefix))
                        finally:
                            driver.synchronize()
                            event["sample_decode_s"] = time.perf_counter() - timer
                        candidate.validate(task, prefix, output, driver.mock)
                        check_prefix(prefix)
                        event["candidate"] = asdict(candidate)
                        timer = time.perf_counter()
                        try:
                            verdict = None if method == "vanilla" else agents.verify(task, stage, prefix, candidate)
                        finally:
                            event["verify_s"] = time.perf_counter() - timer
                        event["verdict"] = asdict(verdict) if verdict else None
                        passed = method == "vanilla" or verdict.accepted(task.audio_required, limits.threshold)
                        if time.perf_counter() - started >= limits.wall_seconds:
                            passed, stop_after = False, "budget_exhausted"
                        if passed:
                            driver.commit(candidate)
                            check_prefix(prefix)
                            chunk = {"candidate": asdict(candidate), "stage_id": stage.id,
                                     "accepted_by_verifier": verdict is not None,
                                     "hashes": {name: file_hash(name) for name in candidate.artifacts}}
                            next_prefix = [*prefix, chunk]
                            write_json(output / "prefix.json", next_prefix)
                            prefix = next_prefix
                            keep, accepted = True, True
                            event["status"] = "committed"
                            stage_done = ((stage_chunk + 1 >= stage.nominal_chunks) if verdict is None else
                                          (verdict.stage_complete and stage_chunk + 1 >= stage.min_chunks))
                        else:
                            event["status"] = "rejected"
                            timer = time.perf_counter()
                            if stop_after is None and attempt + 1 < limits.attempts_per_chunk and method in {"feedback", "evolved"}:
                                try:
                                    proposal = agents.repair(task, stage, verdict, control, rules)
                                finally:
                                    event["controller_s"] = time.perf_counter() - timer
                                if proposal.get("action") == "stop":
                                    stop_after = "controller_stopped"
                                elif proposal.get("action") == "resample":
                                    pass
                                elif proposal.get("action") == "reprompt":
                                    if not isinstance(proposal.get("video_prompt"), str) or not proposal["video_prompt"].strip():
                                        raise ValueError("Reprompt requires a nonempty video_prompt")
                                    if not isinstance(proposal.get("audio_prompt"), str):
                                        raise ValueError("Reprompt requires audio_prompt string")
                                    control = {k: proposal[k] for k in ("video_prompt", "audio_prompt")}
                                else:
                                    raise ValueError("Unsupported repair action; prefix rollback is forbidden")
                                event["repair"] = proposal
                            event["controller_s"] = time.perf_counter() - timer
                    except Exception as exc:
                        event["status"] = "error"
                        event["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        if not keep:
                            timer = time.perf_counter()
                            driver.restore(snapshot)
                            driver.synchronize()
                            event["restore_s"] = time.perf_counter() - timer
                        log(event)
                    if stop_after:
                        raise StopRun(stop_after)
                    if accepted:
                        break
                if not accepted:
                    raise StopRun("retry_exhausted")
                if stage_done:
                    completed.append(stage.id)
                    break
            else:
                raise StopRun("stage_horizon_exhausted")
    except StopRun as exc:
        status = str(exc)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    check_prefix(prefix)
    summary = {"status": status, "error": error, "mock": driver.mock,
               "completed_stage_ids": completed, "committed_chunks": len(prefix),
               "attempted_candidates": len(events), "rejected_candidates": sum(e["status"] == "rejected" for e in events),
               "wall_s": time.perf_counter() - started,
               "task_success": None, "official_benchmark_score": None,
               "cost_s": {key: sum(e.get(key, 0.0) for e in events) for key in
                          ("snapshot_s", "sample_decode_s", "verify_s", "controller_s", "restore_s")}}
    write_json(output / "summary.json", summary)
    return summary
