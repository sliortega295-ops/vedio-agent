# Full-length sparse candidate generation

## Scope and status

`package/full_length_sparse.py` adds the paper's residual-feedback sparse path as
`method="sparse_evolution"`. It is an **approximate candidate-generation backend**,
not an exact batching optimization. It keeps every latent frame throughout
denoising and decodes the full requested video; it does not warm up and then
drop all but N9/N6 keyframes.

The existing native-Wan N9/N6 benchmark, acceleration methods, and published
results are unchanged. This new path uses a **separate Diffusers 0.34.0 Wan2.1
T2V 1.3B pipeline and checkpoint**. Do not pass a native `WanT2V` object or reuse
the historical environment/checkpoint directory without verifying its format.

CPU contract tests include a randomly initialized, real Diffusers Wan
transformer, original RoPE, full-K/dense equivalence, conditional/unconditional
schedule replay, causal budget changes, zero warmup, dense tail, independent
candidates, and VAE latent denormalization. No model weights are downloaded by
the tests. **GPU latency, VRAM, generated-video quality, verifier ranking,
best-of-N gains, and calibrated budget thresholds: NOT_RUN.** Existing N9/N6,
ODE-path, or upstream CADense numbers are not results for this new method.

## Architecture-to-code mapping

| Component | Implemented contract | Location |
|---|---|---|
| Candidate interface | One prompt/seed, fresh controller and RNG per call; decoded full video available to an external verifier | `package/full_length_sparse.py:generate_candidate` |
| Dense / sparse / dense blocks | Default 30-block model: blocks `[0,3)` and `[27,30)` dense, `[3,27)` sparse, three blocks per group | `coframe/config.py` and `wan/sparse_forward.py` |
| Interleaved anchors | Fixed within a group; next group uses an exact-K coverage-aware rotating mesh, always retaining temporal endpoints | `coframe/selection.py:coverage_interleaved_select` |
| Residual reconstruction | Compute anchor block outputs; interpolate their residual updates onto **all** frames; retain the incoming full hidden state | `coframe/interpolation.py:reconstruct_sparse_block` |
| Residual feedback (figure c) | Interior-anchor leave-one-out residual defect in group g selects K for group g+1; never retroactively changes group g | `wan/sparse_forward.py`, `coframe/budget.py` |
| CFG | Conditional branch records anchors; unconditional branch replays them without measuring defects or mutating the controller | `wan/pipeline.py` |
| Final dense regeneration | Invoke the same candidate API with a dense config and the selected prompt/seed, starting from noise again | Example below |

The outer proposal/verification/search loop is **caller-owned**. This repository
does not contain a newly implemented VLM verifier, prompt-revision policy, or
end-to-end search evaluation. The existing halving utility's latent-RMS scorer
is not a substitute for a semantic video verifier.

For each sparse **block** in a group, with incoming full state H and anchors S:

```text
delta_S = block(H)_S - H_S          # anchor-query computation, not a full block run
H_next  = H + interpolate(delta_S, S)
H_next[S] = exact_anchor_outputs   # preserve exact anchor outputs numerically
```

Default `full_kv` uses anchor queries with K/V from the full incoming sequence;
cross-attention and FFN run on anchors. `anchor_only` is a separate supported
ablation using sparse K/V too; its anchor outputs are not claimed equal to
dense attention. Both paths retain **original temporal/spatial RoPE indices**,
not compressed or reset frame positions. “Full” here means full sequence length,
not that the reconstructed frames are exact dense outputs.

The placement routine minimizes squared temporal gaps plus a reuse penalty
over a short deterministic phase cycle. It is not a learned selector, global
coverage guarantee, or defect-localization/remeshing rule.

## Residual budget rule and upstream distinction

At each interior anchor i, remove its already computed block residual and
predict it from neighboring anchors. The measurement is:

```text
d_i = RMS(delta_i - interpolate(delta_left, delta_right; i))
      / (RMS(delta_i) + 1e-8)
r_g = max(d_i over all interior anchors and blocks in group g)  # default
K_next = budget_values[bisect_right(thresholds, r_g)]
```

`mean_defect` replaces the max with a mean. No extra dense pass or future-group
information is used unless diagnostic oracle probes are explicitly enabled.
This is **relative RMS, not squared normalized error** and not a certified video
quality bound. `--sketch-dim 64` (default) uses a fixed random channel sketch;
`--sketch-dim 0` measures exact channel RMS. Calibrate thresholds with the same
sketch, K/V mode, group size, sampler, and model as the intended evaluation.

Budgets must be strictly increasing integers in `[3,F]`, include initial K,
and have one fewer nonnegative finite threshold than budget values. Endpoint
anchors plus at least one interior validator are mandatory. Full K still
computes LOO, so a subsequent group can shrink. Non-finite/missing defects fail
closed. There is no promised fixed average compute budget or proven quality
threshold. A one-value budget list is a fixed-K interleaving ablation.

With default carry enabled, the last sparse group's budget is available to the
next sparse step. `--no-adaptive-k-carry` resets K before each conditional step,
never during unconditional replay. Warmup defaults to **0**, and the last
**1 denoising step** defaults to dense (`--evolution-dense-tail-steps`); these are
distinct from the dense head/tail **blocks**. A one-step run with tail=1 is all
dense; tail=0 is required to exercise sparse execution in that smoke test.

Source is pinned to CADense commit
[`ceddd6f`](https://github.com/sliortega295-ops/CADense/tree/ceddd6f08ec4821acac1210da8044d3015f1601b).
Its Python package remains named `coframe`. Upstream `coframe_ode` uses ODE-path
signals for the next step's budget and LOO defects for anchor remeshing; it is
**not** the same feedback loop as figure c. It and other upstream modes remain
available for explicit comparison, rather than being renamed to imply equivalence.

## Run in an isolated environment

Install a suitable CUDA PyTorch build in a new environment on the intended
experiment host, then:

```bash
python -m pip install -r package/requirements/full_length_sparse.txt
export MODEL_DIR=/persistent/models/Wan2.1-T2V-1.3B-Diffusers
export PROMPT='A horse runs along a beach.'
export OUTPUT_DIR=/persistent/experiments/sva-full-length
export BUDGET_THRESHOLDS='<three calibrated relative-RMS cuts, comma separated>'
# Defaults: K=9, budgets 6/9/12/21, zero warmup, full K/V, final dense step.
bash scripts/run_full_length_sparse.sh --run-name candidate-000
```

The threshold placeholder must be replaced; no measured default is invented.
The new entry is **local-files-only by default** and decodes by default.
`--allow-download` explicitly permits checkpoint fetching; `--no-decode` saves
latents only. Use a new run name for each invocation; existing run directories
are rejected. Outputs: `video.mp4`, full `latents.pt`, `trace.json`, `summary.json`.
Only load trusted local `.pt` outputs. Oracle probes add dense computation and
must be disabled for deployment latency measurements.

Fixed-K functional smoke without calibrated thresholds (not a quality result):

```bash
python package/full_length_sparse.py --model-id "$MODEL_DIR" \
  --prompt "$PROMPT" --output-dir "$OUTPUT_DIR" --run-name fixed-k-smoke \
  --steps 2 --num-anchors 9 --adaptive-k-values 9
```

Programmatic integration from a caller with `package/` on `PYTHONPATH`:

```python
from dataclasses import replace
from full_length_sparse import CoFrameConfig, generate_candidate

# calibrated_cuts comes from held-out calibration, not evaluation-set tuning.
cfg = CoFrameConfig(
    method="sparse_evolution", warmup_steps=0, kv_mode="full_kv",
    adaptive_k_policy="max_defect", adaptive_k_values=(6, 9, 12, 21),
    adaptive_k_thresholds=calibrated_cuts,
)
candidate = generate_candidate(pipe, prompt=proposed_prompt, seed=seed, config=cfg)
score = verifier(original_prompt, candidate.video)  # caller-supplied verifier
# After the caller selects its winning prompt/seed:
final = generate_candidate(
    pipe, prompt=winning_prompt, seed=winning_seed,
    config=replace(cfg, method="dense", trace_path=None),
)
```

One adaptive candidate per call is enforced. Calls sharing a pipeline must be
serial because scheduler/pipeline state is mutable. Native-Wan `candidate_b2`,
`candidate_cfg_b4`, sparse-CFG cache, and halving resume state are **not wired
into this backend**; combining independent adaptive schedules requires separate
work and tests. Reusing the same integer seed across the two different backends
does not establish identical initial noise or comparable scheduler trajectories.

## Validation and rollout gate

```bash
python -m pip install pytest
OMP_NUM_THREADS=1 python -m pytest package/tests -q
bash -n scripts/run_full_length_sparse.sh
```

Before reporting paper results: run same-Diffusers dense/full-K/variable-K
comparisons on an explicitly assigned GPU; check full latent/video frame counts,
finite outputs, `block_anchors`, `budget_events`, and CFG replay; then calibrate
on held-out prompts and evaluate verifier rank/selection quality on separate
prompts/seeds. Report denoising and end-to-end/decode time separately. Stop on
non-finite signals, shape loss, or replay divergence and revert the new method
to the same-backend dense run. Do not overwrite the frozen N9/N6 results.
