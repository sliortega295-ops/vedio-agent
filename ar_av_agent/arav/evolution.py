"""Bounded text-rule evolution. Train evidence -> independent paired dev gate.

No generated patch is executable Python. Test runs cannot enter this pipeline.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from .agents import JSONClient, prompt
from .core import digest, file_hash, require_bool, write_json


def propose(run_dirs: list[Path], client: JSONClient, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    evidence, train_ids = [], []
    for root in run_dirs:
        meta = json.loads((root / "run.json").read_text())
        if meta["task"]["split"] != "train" or meta["mock"]:
            raise ValueError("Only real TRAIN trajectories may propose rules")
        task_id = meta["task"]["id"]
        train_ids.append(task_id)
        events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
        evidence.append({"task_id": task_id, "task": meta["task"]["prompt"],
                         "events": [{k: e.get(k) for k in ("stage_id", "attempt", "control", "verdict", "repair", "status", "sample_decode_s", "verify_s")} for e in events]})
    if not evidence:
        raise ValueError("No train evidence")
    answer = client.ask(prompt("evolver"), {"trajectories": evidence})
    patches = answer.get("rules")
    if not isinstance(patches, list) or not patches or len(patches) > 16:
        raise ValueError("Expected 1..16 bounded rules")
    for patch in patches:
        if not isinstance(patch.get("rule"), str) or not 1 <= len(patch["rule"]) <= 1200:
            raise ValueError("Rule text must be 1..1200 characters")
        if not patch.get("supporting_task_ids") or not set(patch["supporting_task_ids"]) <= set(train_ids):
            raise ValueError("Invented rule evidence")
    result = {"status": "proposed_not_active", "rules": [p["rule"] for p in patches], "patches": patches,
              "train_task_ids": sorted(set(train_ids)), "policy_hash": digest([p["rule"] for p in patches]),
              "evolution_usage": client.usage}
    write_json(output, result)
    return result


def validate(proposal: dict, pairs: list[dict], min_pairs: int = 20, tolerance: float = 0.02,
             require_faster: bool = True) -> dict:
    """Exploratory predeclared paired bootstrap gate, NOT a convergence guarantee.

    Pair schema includes task_id, seed, split='dev', mock=false, old/new policy
    hashes, old/new independent scores [0,1], old/new total costs, and judge ID.
    Validation task IDs must not overlap proposal-training IDs.
    """
    if digest(proposal["rules"]) != proposal.get("policy_hash"):
        raise ValueError("Proposal content/hash mismatch")
    if min_pairs < 2 or len(pairs) < min_pairs or not 0 <= tolerance <= 0.1:
        raise ValueError("Insufficient paired DEV data or invalid preregistered gate")
    seen, qualities, costs, old_hashes = set(), {}, {}, set()
    for p in pairs:
        if p.get("split") != "dev" or p.get("mock") is not False:
            raise ValueError("Only real held-out DEV evaluations can activate rules")
        if p["task_id"] in proposal["train_task_ids"]:
            raise ValueError("Train/dev leakage")
        key = (p["task_id"], p["seed"])
        if key in seen:
            raise ValueError("Repeated pair")
        seen.add(key)
        if not p.get("independent_judge") or p["new_policy_hash"] != proposal["policy_hash"]:
            raise ValueError("Unverified judge or wrong candidate policy")
        old_hashes.add(p["old_policy_hash"])
        for name in ("old_score", "new_score", "old_total_s", "new_total_s"):
            x = p[name]
            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                raise ValueError("Finite measured paired values required")
        if max(p["old_score"], p["new_score"]) > 1:
            raise ValueError("Paired scores must be prespecified on [0,1]")
        qualities.setdefault(p["task_id"], []).append(p["new_score"] - p["old_score"])
        costs.setdefault(p["task_id"], []).append(p["new_total_s"] - p["old_total_s"])
    if len(old_hashes) != 1 or proposal["policy_hash"] in old_hashes:
        raise ValueError("Mixed baseline versions or unchanged policy")
    # Cluster by task; repeated seeds are not independent evidence of transfer.
    if len(qualities) < max(5, min_pairs // 2):
        raise ValueError("Too few distinct DEV tasks")
    q = [statistics.mean(qualities[k]) for k in sorted(qualities)]
    c = [statistics.mean(costs[k]) for k in sorted(costs)]
    rng = random.Random(20261001)
    samples = [([rng.randrange(len(q)) for _ in q]) for _ in range(2000)]
    qb = sorted(statistics.mean(q[i] for i in indices) for indices in samples)
    cb = sorted(statistics.mean(c[i] for i in indices) for indices in samples)
    qlower, cupper = qb[49], cb[1949]
    accepted = qlower >= -tolerance and (not require_faster or cupper < 0)
    return {"status": "validated" if accepted else "rejected", "rules": proposal["rules"],
            "policy_hash": proposal["policy_hash"], "old_policy_hash": next(iter(old_hashes)),
            "train_task_ids": proposal["train_task_ids"], "dev_task_ids": sorted(qualities),
            "paired_records_hash": digest(pairs), "pairs": len(pairs),
            "quality_delta_ci95": [qb[49], qb[1949]], "cost_delta_ci95": [cb[49], cupper],
            "gate": {"tolerance": tolerance, "min_pairs": min_pairs, "require_faster": require_faster}}


def load_policy(path: Path, task_id: str, split: str) -> list[str]:
    record = json.loads(path.read_text())
    if record.get("status") != "validated" or digest(record["rules"]) != record.get("policy_hash"):
        raise ValueError("Policy was not validated or was edited after validation")
    if split == "test" and task_id in set(record["train_task_ids"] + record["dev_task_ids"]):
        raise ValueError("Held-out test task was used to develop this policy")
    return list(record["rules"])
