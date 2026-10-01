"""Official prompt acquisition + provenance-preserving normalization.

No upstream scorer is reimplemented here. Imported public evaluation prompts
remain TEST ONLY, never controller-evolution training examples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .core import Task, canonical, digest, file_hash, write_json

REGISTRY = {
    "storyeval": {"repo": "ypwang61/StoryEval", "paths": ["prompts/all_prompts.json"],
                  "purpose": "ordered event coverage; video only", "status": "public_source_verified"},
    "tempact": {"repo": "jingw193/TempAct", "paths": ["dataset/temporal_eval_simple_100.csv", "dataset/temporal_eval_combined_100.csv"],
                "purpose": "temporal order/state transitions; video only", "status": "public_source_verified"},
    "avgen": {"repo": "microsoft/AVGen-Bench", "directory": "prompts",
              "purpose": "joint AV, synchronization and task semantics", "status": "public_source_verified"},
    "cinescope": {"paper": "https://arxiv.org/abs/2608.29621", "status": "blocked_release_not_verified",
                  "purpose": "long-story causal state + narrative quality; no invented official data"},
    "vbench_long": {"repo": "Vchitect/VBench", "status": "manual_optional",
                    "purpose": "auxiliary visual quality; not event completion"},
}
# Git blob hashes of reviewed files, not SHA256 and not repository commits.
REVIEWED_BLOBS = {
    "prompts/all_prompts.json": "6b4877a3dc889b1104a7ab76b3bdda329396e7ce",
    "dataset/temporal_eval_simple_100.csv": "5618fe90f8962e5eba922c18a19e0e2567d9b8b4",
    "dataset/temporal_eval_combined_100.csv": "b40a648194e42a904bae83512d5b608f61b42d16",
}


def request_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "verified-prefix-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Unexpectedly large benchmark text file")
    return data


def prepare(name: str, root: Path, ref: str = "main", allow_update: bool = False) -> dict:
    """Download text only. No model/code execution, credential or pip side effects.

    Resolve ref ONCE, pin all requests to that exact SHA. Re-runs require a new
    output directory; a partial failed download remains visibly incomplete.
    """
    spec = REGISTRY[name]
    if spec["status"] != "public_source_verified":
        raise RuntimeError(f"{name}: {spec['status']}; do not fabricate a release")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh dataset directory")
    root.mkdir(parents=True, exist_ok=True)
    repo = spec["repo"]
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", ref) or ref.startswith("-"):
        raise ValueError("Invalid ref")
    commit = json.loads(request_bytes(f"https://api.github.com/repos/{repo}/commits/{ref}"))["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("GitHub did not return a full commit SHA")
    paths = list(spec.get("paths", []))
    if "directory" in spec:
        listing = json.loads(request_bytes(f"https://api.github.com/repos/{repo}/contents/{spec['directory']}?ref={commit}"))
        paths = sorted(row["path"] for row in listing if row["type"] == "file" and row["path"].endswith(".json"))
    if not paths:
        raise ValueError("No official data files found")
    manifest = {"benchmark": name, "repo": repo, "commit": commit, "complete": False,
                "data_files": {}, "official_score_computed": False, "checked_on": "2026-10-01"}
    write_json(root / "manifest.json", manifest)
    for path in paths:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Unsafe upstream path")
        content = request_bytes(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}")
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        expected = REVIEWED_BLOBS.get(path)
        if expected and expected != blob and not allow_update:
            raise RuntimeError(f"Upstream changed: {path}; review before --allow-update")
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        manifest["data_files"][path] = {"sha256": file_hash(dest), "git_blob_sha1": blob}
    # Keep license and official instructions next to data; never execute them.
    for path in ("LICENSE", "README.md"):
        try:
            (root / path).write_bytes(request_bytes(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"))
        except Exception as exc:
            manifest.setdefault("metadata_warnings", []).append(f"{path}: {type(exc).__name__}")
    tasks = normalize(name, [root / path for path in paths])
    with (root / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(canonical(asdict(task)) + "\n")
    manifest.update(complete=True, task_count=len(tasks), tasks_sha256=file_hash(root / "tasks.jsonl"))
    manifest["evaluator_checkout"] = f"git clone https://github.com/{repo}.git; git checkout {commit}"
    write_json(root / "manifest.json", manifest)
    return manifest


def av_filename(content: str) -> str:
    """Matches reviewed AVGen generation/runner.py naming, with safe path output."""
    name = re.sub(r'[/\\:*?"<>|\n\r\t]', "_", str(content).strip())
    name = re.sub(r"\s+", " ", name).strip() or "untitled"
    return name[:180].rstrip() + ".mp4"


def normalize(name: str, paths: list[Path]) -> list[Task]:
    rows, seen = [], set()
    for path in paths:
        checksum = file_hash(path)
        if name == "storyeval":
            source = json.loads(path.read_text(encoding="utf-8"))
            iterator = [(key, item, key) for key, item in source.items()]
        elif name == "tempact":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "prompt" not in reader.fieldnames:
                    raise ValueError("TempAct CSV must have a prompt column")
                iterator = [(str(i), item, None) for i, item in enumerate(reader)]
        elif name == "avgen":
            source = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(source, list):
                raise ValueError("AVGen JSON must be a list")
            iterator = [(str(i), item, av_filename(item["content"])) for i, item in enumerate(source)]
        else:
            raise ValueError("No verified importer for this benchmark")
        for key, item, filename in iterator:
            if filename and Path(filename).name != filename:
                raise ValueError("Unsafe official filename")
            if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError("Missing original prompt")
            task_id = f"{name}/{path.stem}/{key}"
            if task_id in seen:
                raise ValueError("Duplicate task ID; do not double-count input files")
            seen.add(task_id)
            meta = {"source_file": path.name, "source_sha256": checksum, "source_index": key,
                    "category": item.get("category", path.stem), "original_row": item,
                    "official_filename": filename, "prompt_hash": digest(item["prompt"])}
            # Preserve reference event list; never replace it with agent-generated plan.
            if "event_list" in item:
                meta["reference_events"] = item["event_list"]
            rows.append(Task(task_id, item["prompt"], name == "avgen", "test", name, meta))
    if not rows:
        raise ValueError("Empty benchmark")
    return rows


def export_video(task: Task, video: Path, destination: Path) -> Path:
    """Caller assembles the FINAL accepted output first; no partial-success filtering."""
    filename = task.metadata.get("official_filename")
    if not filename:
        raise ValueError("This benchmark needs its native evaluator-specific manifest, not filename guessing")
    if Path(filename).name != filename:
        raise ValueError("Unsafe filename")
    parent = destination.resolve()
    if task.source == "avgen":
        category = task.metadata["category"]
        if Path(category).name != category or category in {".", ".."}:
            raise ValueError("Unsafe category")
        parent = parent / category
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / filename
    if out.exists():
        raise FileExistsError("Filename collision: preserve all examples, do not overwrite")
    shutil.copyfile(video, out)
    return out
