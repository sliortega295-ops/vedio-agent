# Reproducibility

## Hardware and fixed workload

- NVIDIA L40, original physical GPU index 0
- Wan2.1-T2V-1.3B at `/data/models/Wan2.1-T2V-1.3B`
- 81 input frames, 832x480, 50 denoising steps
- five dense warmup steps followed by 45 sparse steps
- BF16, FlashAttention-2, guidance 6.0, shift 8.0
- prompt: `A blue car drives past a white picket fence on a sunny day.`
- seeds: 0 and 1

Model weights, CUDA installation, Nsight Systems, and Conda environments are not
redistributed.

## Environment

The original benchmark command used:

```bash
PY=/data/jiayu/yongyan_liu/auto_research/.conda/bin/python
CKPT=/data/models/Wan2.1-T2V-1.3B
RUN=/path/to/output

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/package/third_party"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$PWD/.cache/torch"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

"$PY" package/bench_sparse_candidate_concurrency.py \
  --checkpoint "$CKPT" \
  --output-root "$RUN" \
  --patterns N9 N6 \
  --blocks 3 \
  --max-blocks-on-noise 5
```

The harness deliberately hard-codes the original physical GPU index and UUID as
an admission guard. Exact reproduction on the original server needs no change.
Porting to another machine requires updating that identity guard and recording
the new GPU; it does not authorize changing the scientific workload.

## Fair timing contract

- Two independent resident Wan pipelines and request states.
- One persistent worker thread and one non-default CUDA stream per pipeline.
- Same allocator/warmup state for serial and parallel legs.
- Barrier-synchronized parallel start.
- One excluded warmup; at least three SP/PS-interleaved blocks.
- Extend to five only when makespan CV exceeds 5%.
- All reported performance numbers are profiler-off.
- A whole-process GPU monitor invalidates any condition if a foreign GPU0 PID
  appears. It never terminates another process.

## Nsight Systems

The exact launcher is `scripts/run_nsys_profiles.sh`. Its essential command is:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o <output> \
  "$PY" package/bench_sparse_candidate_concurrency.py \
    --checkpoint "$CKPT" \
    --output-root <profile-run> \
    --patterns N9 \
    --profile-once
```

`SVA_ENABLE_NVTX=1` must be set. The harness captures denoising steps 20--24.
Use the raw `.nsys-rep`/SQLite files from the release to audit the parsed JSON.
Profiler makespan must not be substituted for the formal result.

## Integrity

The release contains:

- a complete frozen source/result tree archive;
- a SHA256 for the archive;
- a 147-file internal SHA256 manifest.

Run:

```bash
sha256sum -c sva_candidate_concurrency_20260809_full.tar.zst.sha256
unzstd -c sva_candidate_concurrency_20260809_full.tar.zst | tar -xf -
cd sva_candidate_concurrency
sha256sum -c ../sva_candidate_concurrency_20260809_FILES.sha256
```
