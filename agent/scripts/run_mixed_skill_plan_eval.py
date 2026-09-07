#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from sparsemac.deepseek_client import DeepSeekClient
from sparsemac.planner import _normalize_vqa_scores


SKILLS = {
    "consistent_attr",
    "dynamic_attr",
    "spatial_relationship",
    "motion_binding",
    "action_binding",
    "interaction",
    "numeracy",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_skill(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "n/a"}:
        return None
    return text if text in SKILLS else None


def validate_plan(data: Any, candidates: int) -> tuple[bool, str, dict[str, float]]:
    if not isinstance(data, dict):
        return False, "not_dict", {}
    variants = data.get("prompt_variants")
    vqa = data.get("vqa")
    if not isinstance(variants, list) or len([x for x in variants if str(x).strip()]) < candidates:
        return False, "bad_variants", {}
    if not isinstance(vqa, list) or not (3 <= len(vqa) <= 6):
        return False, "bad_vqa_count", {}
    questions = [str(q).strip() for q in vqa if str(q).strip()]
    if len(questions) != len(vqa):
        return False, "bad_vqa_question", {}
    scores = _normalize_vqa_scores(data.get("vqa_scores"), questions)
    if set(scores) != set(questions):
        return False, "score_question_mismatch", scores
    if any(value <= 0 for value in scores.values()):
        return False, "non_positive_weight", scores
    total = sum(scores.values())
    if not 0.95 <= total <= 1.05:
        return False, f"weight_sum_{total:.3f}", scores
    return True, "", scores


class MixedSkillPlanner:
    def __init__(self, *, model: str, candidates: int):
        self.model = model
        self.candidates = candidates
        self.prompt_root = PROJECT_ROOT / "sparsemac" / "prompts"
        self.router_template = read_text(self.prompt_root / "mixed_skill_router.txt")
        self.base_template = read_text(self.prompt_root / "mixed_planner_base.txt")
        self.primary_cards = {
            skill: read_text(self.prompt_root / "primary_skills" / f"{skill}.txt")
            for skill in SKILLS
        }
        self.secondary_cards = {
            skill: read_text(self.prompt_root / "secondary_skills" / f"{skill}.txt")
            for skill in SKILLS
        }
        self.local = threading.local()

    def client(self) -> DeepSeekClient:
        client = getattr(self.local, "client", None)
        if client is None:
            client = DeepSeekClient(model=self.model, timeout=90)
            self.local.client = client
        return client

    def route(self, prompt: str) -> tuple[str, str | None, dict[str, Any]]:
        raw_prompt = self.router_template.format(prompt=prompt)
        parsed, raw = self.client().generate_json(raw_prompt, max_tokens=700, temperature=0.0)
        primary = normalize_skill(parsed.get("primary_skill") if isinstance(parsed, dict) else None)
        secondary = normalize_skill(parsed.get("secondary_skill") if isinstance(parsed, dict) else None)
        if primary is None:
            return "motion_binding", None, {
                "ok": False,
                "raw": raw[:1000] if raw else "",
                "error": self.client().last_error,
            }
        if secondary == primary:
            secondary = None
        return primary, secondary, {"ok": True, "raw": raw[:1000] if raw else ""}

    def build_plan_prompt(self, *, prompt: str, primary: str, secondary: str | None) -> str:
        secondary_text = self.secondary_cards.get(secondary, "Secondary constraint: none.") if secondary else "Secondary constraint: none."
        base = self.base_template.format(candidates=self.candidates)
        return "\n\n".join(
            [
                base,
                self.primary_cards[primary],
                secondary_text,
                "Original prompt:\n" + prompt,
            ]
        )

    def plan(self, *, prompt: str, primary: str, secondary: str | None) -> tuple[dict[str, Any] | None, str]:
        plan_prompt = self.build_plan_prompt(prompt=prompt, primary=primary, secondary=secondary)
        parsed, raw = self.client().generate_json(plan_prompt, max_tokens=4200, temperature=0.2)
        if isinstance(parsed, dict) and isinstance(parsed.get("vqa"), list):
            scores = _normalize_vqa_scores(parsed.get("vqa_scores"), [str(q) for q in parsed["vqa"]])
            parsed["vqa_scores"] = scores
        return parsed, raw[:2000] if raw else ""


def load_items(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)
    return list(manifest["items"])


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    generic_presence = re.compile(r"^\s*(is there|are there|do you see|can you see)\b", re.I)
    quality = re.compile(r"\b(sharp|blur|lighting|aesthetic|beautiful|high quality|visual quality|polish|resolution)\b", re.I)
    q_counts = []
    weight_sums = []
    generic_by_cat: Counter[str] = Counter()
    quality_by_cat: Counter[str] = Counter()
    total_q_by_cat: Counter[str] = Counter()
    for record in results:
        plan = record.get("plan")
        if not record.get("ok") or not isinstance(plan, dict):
            continue
        vqa = plan.get("vqa") or []
        scores = plan.get("vqa_scores") or {}
        if isinstance(vqa, list):
            q_counts.append(len(vqa))
            for question in vqa:
                text = str(question)
                total_q_by_cat[record.get("source_category", "")] += 1
                if generic_presence.search(text):
                    generic_by_cat[record.get("source_category", "")] += 1
                if quality.search(text):
                    quality_by_cat[record.get("source_category", "")] += 1
        if isinstance(scores, dict):
            try:
                weight_sums.append(sum(float(value) for value in scores.values()))
            except Exception:
                pass
    return {
        "vqa_question_count_mean": round(sum(q_counts) / len(q_counts), 3) if q_counts else None,
        "vqa_question_count_median": statistics.median(q_counts) if q_counts else None,
        "weight_sum_mean": round(sum(weight_sums) / len(weight_sums), 4) if weight_sums else None,
        "generic_presence_question_counts_by_source_category": dict(generic_by_cat),
        "quality_like_question_counts_by_source_category": dict(quality_by_cat),
        "total_questions_by_source_category": dict(total_q_by_cat),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek mixed-skill SparseMAC planner on a manifest without generating video.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "runs/baseline/vqqa-dense/manifest.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "logs/plan_eval_mixed_skill_v2")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reuse-routing-csv", type=Path)
    parser.add_argument("--plan-retries", type=int, default=1)
    args = parser.parse_args()

    out_dir = args.out_dir
    plan_dir = out_dir / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "plan_results.jsonl"
    csv_path = out_dir / "plan_results.csv"
    summary_path = out_dir / "summary.json"
    jsonl_path.write_text("", encoding="utf-8")

    items = load_items(args.manifest)
    if args.limit is not None:
        items = items[: args.limit]

    reused_routes: dict[str, dict[str, str]] = {}
    if args.reuse_routing_csv and args.reuse_routing_csv.exists():
        with args.reuse_routing_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reused_routes[row["prompt_id"]] = row

    planner = MixedSkillPlanner(model=args.model, candidates=args.candidates)
    lock = threading.Lock()

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        route = reused_routes.get(item["prompt_id"])
        route_meta: dict[str, Any] = {"ok": True, "raw": "", "reused": bool(route)}
        if route:
            primary = normalize_skill(route.get("primary_skill")) or normalize_skill(route.get("primary")) or item.get("category_key")
            secondary = normalize_skill(route.get("secondary_skill"))
            if primary not in SKILLS:
                primary = item.get("category_key") if item.get("category_key") in SKILLS else "motion_binding"
            if secondary == primary:
                secondary = None
        else:
            primary, secondary, route_meta = planner.route(item["prompt"])
        parsed = None
        raw = ""
        ok = False
        error = "not_run"
        scores: dict[str, float] = {}
        attempts = 0
        for attempt in range(max(0, args.plan_retries) + 1):
            attempts = attempt + 1
            parsed, raw = planner.plan(prompt=item["prompt"], primary=primary, secondary=secondary)
            ok, error, scores = validate_plan(parsed, args.candidates)
            if ok or error != "not_dict":
                break
        if isinstance(parsed, dict) and scores:
            parsed["vqa_scores"] = scores
        plan_path = plan_dir / f"{item['prompt_id']}.json"
        record = {
            "prompt_id": item["prompt_id"],
            "source_category": item.get("category_key"),
            "prompt": item["prompt"],
            "primary_skill": primary,
            "secondary_skill": secondary or "",
            "route_ok": bool(route_meta.get("ok")),
            "route_reused": bool(route_meta.get("reused")),
            "ok": ok,
            "error": error,
            "plan_attempts": attempts,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "plan_path": str(plan_path),
        }
        write_json(plan_path, {"record": record, "plan": parsed, "raw": raw, "route": route_meta})
        with lock:
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {**record, "plan": parsed}

    print(f"plan-only items={len(items)} out={out_dir}", flush=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in items]
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                failures += 1
                record = {"prompt_id": "<exception>", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(record)
            completed += 1
            if completed % 25 == 0 or completed == len(items):
                ok = sum(1 for result in results if result.get("ok"))
                print(f"progress {completed}/{len(items)} ok={ok} failures={failures} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    by_id = {record["prompt_id"]: record for record in results if record.get("prompt_id") and record["prompt_id"] != "<exception>"}
    ordered = [by_id[item["prompt_id"]] for item in items if item["prompt_id"] in by_id]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "prompt_id",
            "source_category",
            "primary_skill",
            "secondary_skill",
            "route_ok",
            "route_reused",
            "ok",
            "error",
            "plan_attempts",
            "elapsed_seconds",
            "plan_path",
            "prompt",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in ordered:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_category[str(record.get("source_category", ""))].append(record)

    summary = {
        "model": args.model,
        "strategy": "mixed skill planner; primary full card + secondary short card; normalized vqa_scores",
        "total": len(items),
        "results": len(ordered),
        "ok": sum(1 for record in ordered if record.get("ok")),
        "failed": sum(1 for record in ordered if not record.get("ok")),
        "errors": dict(Counter(record.get("error", "") for record in ordered if not record.get("ok"))),
        "primary_counts": dict(Counter(record.get("primary_skill", "") for record in ordered)),
        "secondary_counts": dict(Counter(record.get("secondary_skill") or "null" for record in ordered)),
        "ok_by_category": {cat: sum(1 for record in records if record.get("ok")) for cat, records in sorted(by_category.items())},
        "total_by_category": {cat: len(records) for cat, records in sorted(by_category.items())},
        "output_csv": str(csv_path),
        "output_jsonl": str(jsonl_path),
        "plan_dir": str(plan_dir),
    }
    summary.update(analyze(ordered))
    write_json(summary_path, summary)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    print("SAMPLE_PLANS", flush=True)
    for record in ordered[:5]:
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        print(f"\n--- {record.get('prompt_id')} {record.get('source_category')} primary={record.get('primary_skill')} secondary={record.get('secondary_skill') or 'null'} ok={record.get('ok')}", flush=True)
        print(record.get("prompt"), flush=True)
        scores = plan.get("vqa_scores") if isinstance(plan, dict) else {}
        for question in (plan.get("vqa") if isinstance(plan, dict) else []) or []:
            print(f"  {scores.get(question) if isinstance(scores, dict) else None} {question}", flush=True)
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
