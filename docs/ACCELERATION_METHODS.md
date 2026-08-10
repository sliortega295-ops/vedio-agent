# SVA acceleration experiments

This branch adds implementations for the next concurrency/fusion experiments without changing the frozen `formal_n9_n6_v3` baseline.

## Implemented methods

| Method | Exact? | Purpose |
|---|---|---|
| `shared-model streams` | yes | Re-run two-stream concurrency with one resident Wan model instead of two private pipelines. |
| `serial_b1` | yes | Matched one-candidate-at-a-time reference. |
| `candidate_b2` | yes | Fuse same-shape candidates into batch `N`; conditional and unconditional CFG are two batched forwards. |
| `candidate_cfg_b4` | yes | Fuse candidates and CFG branches into one effective batch `2N`. With two candidates this is B4. |
| `sparse_cfg` | approximate | Keep dense warm-up exact; in the sparse phase refresh the unconditional branch every `k` steps and reuse it between refreshes. |
| `halving` | approximate | Run candidates in fused batches and prune low-scoring candidates at configured sparse denoising steps. |

The exact methods should be compared with output numerical error and wall/CUDA time. The approximate methods must additionally be evaluated with SVA ranking metrics (Spearman, pairwise agreement, Top-1/Recall@k/Regret@k) and the final T2V-CompBench/VBench score.

## Files

- `package/sva_acceleration.py`: reusable batching, CFG fusion/reuse, and halving primitives.
- `package/bench_shared_model_streams.py`: one-model/two-stream diagnostic baseline.
- `package/bench_acceleration_methods.py`: unified exact/approximate benchmark.
- `scripts/run_acceleration_methods.sh`: example launch sequence on the L40 setup.

## Recommended experiment order

First run `shared-model streams` to isolate whether duplicate model residency caused part of the negative concurrency result. Then compare `serial_b1`, `candidate_b2`, and `candidate_cfg_b4`; these are exact and directly answer whether fusion is better than independent streams. Only after establishing the best exact executor should `sparse_cfg` and `halving` be evaluated.

```bash
export PYTHON_BIN=/path/to/wan/python
export CKPT_DIR=/data/models/Wan2.1-T2V-1.3B
export GPU=0
export RUN_ROOT=$PWD/runs/sva-acceleration
bash scripts/run_acceleration_methods.sh
```

## Successive-halving scorer contract

`bench_acceleration_methods.py --scorer module:function` loads a Python callable with this signature:

```python
def score(candidate_ids, latents, global_step):
    return {candidate_id: score, ...}
```

The built-in `latent_rms` scorer exists only to validate scheduler mechanics. **Do not use it as a quality result.** For the paper, plug in the real sparse-preview/VLM scoring path (or a frozen offline score adapter) and report which candidates are pruned at every round.

A typical N=8 policy is:

```text
8 candidates -> step 15 -> keep 4 -> step 30 -> keep 2 -> step 45 -> keep 1
```

The pruning checkpoints are deliberately configurable; they should be chosen from a quality-vs-cost sweep rather than fixed by the runtime.

## Important interpretation

The current negative result remains valid: two independent sparse requests on separate CUDA streams do not produce extra throughput on the tested L40 shapes. These new methods test a different hypothesis: **can multiple SVA candidates be fused into a hardware-friendlier workload, or can search-specific approximation reduce candidate-step work?**
