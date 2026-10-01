from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from arav.agents import MockAgents, MockDriver, prompt
from arav.benchmarks import REGISTRY, av_filename, normalize, prepare
from arav.core import Candidate, Limits, Stage, Task, Verdict, check_prefix, digest, run, write_json
from arav.evaluation import aggregate, score_events
from arav.evolution import load_policy, validate


def task(audio=True):
    return Task("unit-test", "Ring a bell once.", audio, "smoke")


def stages(count=2):
    return [Stage("ring", "Ring a bell", "A hand rings a bell.", "A bell rings.",
                  nominal_chunks=count, max_chunks=count + 1)]


def test_rejected_chunk_never_contaminates_either_cache(tmp_path):
    driver = MockDriver(fail_first=1)
    result = run(task(), stages(), driver, MockAgents(), tmp_path)
    trace = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [x["status"] for x in trace] == ["rejected", "committed", "committed"]
    assert driver.state["position"] == 2
    assert driver.state["video_cache"] == [x["seed"] for x in trace[1:]]
    assert driver.state["audio_cache"] == [x["seed"] + 1 for x in trace[1:]]
    assert result["status"] == "execution_complete"
    assert result["task_success"] is None and result["official_benchmark_score"] is None
    check_prefix(json.loads((tmp_path / "prefix.json").read_text()))


def test_partial_progress_is_committed_without_finishing_stage(tmp_path):
    result = run(task(), stages(3), MockDriver(0), MockAgents(), tmp_path)
    assert result["committed_chunks"] == 3
    assert result["completed_stage_ids"] == ["ring"]


def test_exhaustion_does_not_accept_best_failed_candidate(tmp_path):
    driver = MockDriver(99)
    result = run(task(), stages(), driver, MockAgents(), tmp_path, limits=Limits(attempts_per_chunk=2))
    assert result["status"] == "retry_exhausted"
    assert driver.state["position"] == 0
    assert json.loads((tmp_path / "prefix.json").read_text()) == []


def test_exception_restores_both_modalities(tmp_path):
    class Broken(MockDriver):
        def sample(self, *args, **kwargs):
            super().sample(*args, **kwargs)
            raise RuntimeError("injected model failure")
    driver = Broken()
    result = run(task(), stages(), driver, MockAgents(), tmp_path)
    assert result["status"] == "error"
    assert result["cost_s"]["sample_decode_s"] > 0
    assert driver.state["video_cache"] == driver.state["audio_cache"] == []


@pytest.mark.parametrize("key", ["video_ok", "audio_ok", "sync_ok", "stage_complete"])
def test_verdict_strings_are_not_booleans(key):
    row = {"video_ok": True, "audio_ok": True, "sync_ok": True, "stage_complete": True,
           "confidence": .9, "feedback": "x", "evidence": []}
    row[key] = "false"
    with pytest.raises(ValueError):
        Verdict.parse(row)


@pytest.mark.parametrize("key", ["audio_ok", "sync_ok"])
def test_audio_or_sync_failure_rejects_joint_candidate(key, tmp_path):
    class BadAV(MockAgents):
        def verify(self, *args):
            row = asdict(super().verify(*args))
            row["evidence"] = list(row["evidence"])
            row[key] = False
            return Verdict.parse(row)
    result = run(task(), stages(), MockDriver(0), BadAV(), tmp_path)
    assert result["committed_chunks"] == 0


def test_video_driver_cannot_masquerade_as_joint_av(tmp_path):
    driver = MockDriver()
    driver.joint_av = False
    with pytest.raises(ValueError, match="video-only"):
        run(task(), stages(), driver, MockAgents(), tmp_path)


def test_audio_clock_and_header_validated(tmp_path):
    driver = MockDriver()
    candidate = driver.sample(stages()[0], {}, 42, tmp_path, 0)
    with pytest.raises(ValueError):
        replace(candidate, audio_start_s=.2).validate(task(), [], tmp_path, True)
    with pytest.raises(ValueError, match="WAV header"):
        replace(candidate, audio_samples=1).validate(task(), [], tmp_path, True)


def test_budget_stops_without_committing_new_failure(tmp_path):
    result = run(task(), stages(), MockDriver(99), MockAgents(), tmp_path,
                 limits=Limits(attempts_per_chunk=9, max_candidates=2))
    assert result["status"] == "budget_exhausted" and result["attempted_candidates"] == 2


def test_original_task_is_unchanged_by_repair(tmp_path):
    original = task()
    run(original, stages(), MockDriver(1), MockAgents(), tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["task"]["prompt"] == original.prompt


def test_existing_directory_is_never_clobbered(tmp_path):
    (tmp_path / "valuable.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        run(task(), stages(), MockDriver(), MockAgents(), tmp_path)
    assert (tmp_path / "valuable.txt").read_text() == "keep"


def test_hash_detects_changed_committed_prefix(tmp_path):
    run(task(), stages(1), MockDriver(0), MockAgents(), tmp_path)
    prefix = json.loads((tmp_path / "prefix.json").read_text())
    Path(prefix[0]["candidate"]["artifacts"][0]).write_text("changed")
    with pytest.raises(RuntimeError, match="Committed"):
        check_prefix(prefix)


def test_vanilla_does_not_call_online_verifier(tmp_path):
    class Unused(MockAgents):
        def verify(self, *args):
            raise AssertionError("must not call")
    result = run(task(), stages(), MockDriver(99), Unused(), tmp_path, method="vanilla")
    assert result["status"] == "execution_complete" and result["task_success"] is None


def test_official_schema_import_and_test_isolation(tmp_path):
    story = tmp_path / "story.json"
    story.write_text(json.dumps({"example.mp4": {"prompt": "Do A, then B.", "event_list": ["A", "B"], "class": ["human"]}}))
    s = normalize("storyeval", [story])[0]
    assert s.split == "test" and s.metadata["reference_events"] == ["A", "B"]
    assert s.metadata["official_filename"] == "example.mp4"
    csv = tmp_path / "simple.csv"
    csv.write_text('prompt,category\n"Do A, then B.",human\n')
    t = normalize("tempact", [csv])[0]
    assert t.prompt == "Do A, then B." and t.audio_required is False
    av = tmp_path / "ads.json"
    av.write_text(json.dumps([{"content": "Bell: one", "prompt": "A bell rings audibly."}]))
    a = normalize("avgen", [av])[0]
    assert a.audio_required and a.metadata["official_filename"] == "Bell_ one.mp4"
    assert a.metadata["category"] == "ads"


def test_cinescope_unavailable_is_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="blocked"):
        prepare("cinescope", tmp_path)


def test_unknown_score_is_not_counted_as_success():
    answer = {"events": [{"id": "a", "fulfilled": None, "start_s": None, "end_s": None, "evidence": "unclear"}],
              "order_ok": True, "audio_ok": None, "sync_ok": None}
    result = score_events(task(False), ["a"], answer)
    assert result["event_completion_lower_bound"] == 0
    assert result["task_success_lower_bound"] is False
    assert result["official_benchmark_score"] is None


def test_missing_outputs_remain_in_denominator():
    result = aggregate([{"task_id": "a", "success": True, "total_wall_s": 2.0, "mock": False}], ["a", "b"])
    assert result["success_rate_lower_bound"] == .5
    assert result["amortized_wall_per_success_s"] is None
    with pytest.raises(ValueError):
        aggregate([{"task_id": "a", "success": True, "total_wall_s": 2.0, "mock": True}], ["a"])


def proposal_and_pairs():
    rules = ["Describe the next observable state explicitly."]
    proposal = {"rules": rules, "policy_hash": digest(rules), "train_task_ids": ["train-1"]}
    pairs = [{"task_id": f"dev-{i}", "seed": 0, "split": "dev", "mock": False,
              "old_policy_hash": digest([]), "new_policy_hash": digest(rules), "independent_judge": "fixture-only",
              "old_score": .5, "new_score": .6, "old_total_s": 10., "new_total_s": 8.} for i in range(20)]
    return proposal, pairs


def test_evolution_gate_math_and_heldout_guard(tmp_path):
    # Synthetic numbers validate gate arithmetic only, never reported as outcomes.
    proposal, pairs = proposal_and_pairs()
    result = validate(proposal, pairs)
    assert result["status"] == "validated"
    path = tmp_path / "policy.json"
    write_json(path, result)
    assert load_policy(path, "test-unseen", "test") == proposal["rules"]
    with pytest.raises(ValueError, match="Held-out"):
        load_policy(path, "dev-0", "test")


@pytest.mark.parametrize("bad", ["test", "mock", "overlap", "duplicate"])
def test_evolution_rejects_contamination(bad):
    proposal, pairs = proposal_and_pairs()
    if bad == "test":
        pairs[0]["split"] = "test"
    elif bad == "mock":
        pairs[0]["mock"] = True
    elif bad == "overlap":
        pairs[0]["task_id"] = "train-1"
    else:
        pairs[1] = pairs[0]
    with pytest.raises(ValueError):
        validate(proposal, pairs)


def test_prompt_assets_packaged():
    for name in ("planner", "verifier", "controller", "evolver", "final_judge", "cinescope_proxy"):
        assert len(prompt(name)) > 100


def test_longlive_snapshot_clones_kv_tensors_without_cloning_weights():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from arav.longlive import LongLiveDriver
    model = SimpleNamespace(block_mask="before")
    pipe = SimpleNamespace(num_frame_per_block=3, frame_seq_length=1560,
                           kv_cache1=[{"k": torch.zeros(2), "v": torch.ones(2)}],
                           crossattn_cache=[{"k": torch.zeros(2), "is_init": False}],
                           generator=SimpleNamespace(model=model))
    driver = LongLiveDriver(pipe, device="cpu")
    driver.history = torch.empty(1, 0, 16, 60, 104)
    snapshot = driver.snapshot()
    pipe.kv_cache1[0]["k"].add_(7)
    pipe.crossattn_cache[0]["is_init"] = True
    model.block_mask = "failed"
    driver.restore(snapshot)
    assert torch.equal(pipe.kv_cache1[0]["k"], torch.zeros(2))
    assert pipe.crossattn_cache[0]["is_init"] is False
    assert model.block_mask == "before"
