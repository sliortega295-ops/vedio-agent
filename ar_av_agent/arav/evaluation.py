"""Independent project proxy + media export. Never manufactures official scores."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
import wave
from pathlib import Path

from .agents import JSONClient, prompt
from .core import Task, check_prefix, file_hash, require_bool, write_json


def assemble(run_dir: Path) -> Path:
    meta = json.loads((run_dir / "run.json").read_text())
    if meta["mock"]:
        raise ValueError("Mock traces have no generated video")
    prefix = json.loads((run_dir / "prefix.json").read_text())
    check_prefix(prefix)
    if not prefix:
        raise ValueError("No committed media to assemble")
    out = run_dir / "final.mp4"
    if out.exists():
        raise FileExistsError("Refusing to replace an existing final output")
    started = time.perf_counter()
    listing = run_dir / "concat.txt"
    # ffmpeg concat syntax: quote and escape paths, never invoke a shell.
    listing.write_text("".join("file '" + str(Path(x["candidate"]["video_path"]).resolve()).replace("'", "'\\''") + "'\n" for x in prefix))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if meta["task"]["audio_required"]:
        merged = run_dir / "final.wav"
        if merged.exists():
            raise FileExistsError("Use a fresh export directory after a failed export")
        expected = None
        with wave.open(str(merged), "wb") as writer:
            for chunk in prefix:
                with wave.open(chunk["candidate"]["audio_path"], "rb") as reader:
                    shape = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype())
                    if expected is None:
                        expected = shape
                        writer.setparams(reader.getparams())
                    elif shape != expected:
                        raise ValueError("Audio format changed across committed chunks")
                    writer.writeframes(reader.readframes(reader.getnframes()))
        command += ["-i", str(merged), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac"]
    else:
        command += ["-map", "0:v:0", "-c:v", "copy", "-an"]
    command += [str(out)]
    subprocess.run(command, check=True, timeout=600)
    check_prefix(prefix)
    write_json(run_dir / "export.json", {"path": str(out), "sha256": file_hash(out),
                                         "wall_s": time.perf_counter() - started,
                                         "complete_execution": json.loads((run_dir / "summary.json").read_text())["status"] == "execution_complete"})
    return out


def score_events(task: Task, event_ids: list[str], answer: dict) -> dict:
    if set(answer) != {"events", "order_ok", "audio_ok", "sync_ok"}:
        raise ValueError("Unexpected final judge schema")
    for field in ("order_ok", "audio_ok", "sync_ok"):
        require_bool(answer[field], field, nullable=True)
    events = answer["events"]
    if not isinstance(events, list) or len(events) != len(event_ids):
        raise ValueError("Missing/extra final-event judgments")
    if not event_ids or len(set(event_ids)) != len(event_ids):
        raise ValueError("Reference event IDs must be unique and nonempty")
    if {e["id"] for e in events} != set(event_ids):
        raise ValueError("Final judge changed the reference events")
    known, passed = 0, 0
    for event in events:
        value = require_bool(event["fulfilled"], "fulfilled", nullable=True)
        if not isinstance(event.get("evidence"), str) or not event["evidence"].strip():
            raise ValueError("Independent judge must provide evidence")
        start, end = event.get("start_s"), event.get("end_s")
        if value is True:
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
                raise ValueError("Passed events require finite ordered evidence timestamps")
        known += value is not None
        passed += value is True
    audio = not task.audio_required or (answer["audio_ok"] is True and answer["sync_ok"] is True)
    return {"protocol": "project_event_proxy_v0_NOT_OFFICIAL", "official_benchmark_score": None,
            "event_completion_lower_bound": passed / len(event_ids), "event_judgment_coverage": known / len(event_ids),
            "task_success_lower_bound": passed == len(event_ids) and answer["order_ok"] is True and audio,
            "judgment": answer}


def independent_judge(task: Task, references: list[dict], media: Path, client: JSONClient,
                      output: Path, audio_capable: bool = False) -> dict:
    """Uniform sampled-image proxy, limited to <=64 frames. Not official video scoring.

    Source reference events must be frozen BEFORE generation, not copied from the
    online planner or verifier. Use upstream full protocols for paper main tables.
    """
    if output.exists():
        raise FileExistsError(output)
    if task.audio_required and not audio_capable:
        raise ValueError("Final AV evaluation requires real audio support")
    info = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media)], timeout=30)
    duration = float(json.loads(info)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid media duration")
    evidence_dir = output.parent / (output.stem + "-evidence")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(media),
                    "-vf", f"fps={min(2, 63/duration)},scale=512:-2", "-frames:v", "64", str(evidence_dir / "frame-%04d.jpg")], check=True, timeout=180)
    images = tuple(str(p) for p in sorted(evidence_dir.glob("frame-*.jpg")))
    if not images:
        raise ValueError("No frames extracted")
    audio = None
    if task.audio_required:
        audio = str(evidence_dir / "audio.wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio], check=True, timeout=180)
    answer = client.ask(prompt("final_judge"), {"original_task": task.prompt, "reference_events": references,
                        "duration_s": duration, "sampling_fps": min(2, 63/duration)}, images, audio)
    result = score_events(task, [r["id"] for r in references], answer)
    result.update(task_id=task.id, source=task.source, media_sha256=file_hash(media), evaluator_model=client.model)
    write_json(output, result)
    return result


def aggregate(records: list[dict], expected_ids: list[str]) -> dict:
    """All assigned tasks remain in denominator, including failed/missing outputs."""
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Nonempty unique expected task IDs required")
    by_id = {}
    for row in records:
        if row.get("mock") is not False:
            raise ValueError("Mock/unknown provenance is not an experiment result")
        if row["task_id"] in by_id or row["task_id"] not in expected_ids:
            raise ValueError("Duplicate/unassigned result")
        require_bool(row["success"], "success", nullable=True)
        if type(row["total_wall_s"]) not in (int, float) or not math.isfinite(row["total_wall_s"]) or row["total_wall_s"] < 0:
            raise ValueError("Invalid measured total cost")
        by_id[row["task_id"]] = row
    successes = sum(row["success"] is True for row in by_id.values())
    total = sum(row["total_wall_s"] for row in by_id.values())
    missing = set(expected_ids) - set(by_id)
    return {"n_expected": len(expected_ids), "n_observed": len(by_id), "n_missing": len(missing),
            "success_rate_lower_bound": successes / len(expected_ids), "observed_total_wall_s": total,
            "amortized_wall_per_success_s": total / successes if successes and not missing else None,
            "official_score": None}
