# SVA Sparse-Candidate Concurrency Preregistration

## Question

Does temporal sparsity make two independent test-time-search trajectories
meaningfully more concurrency-friendly after the algorithmic N9/N6 speedup?

The already completed dense/Full experiment is not repeated.

## Frozen workload

- GPU: physical L40 GPU 0 only
- Model: Wan2.1-T2V-1.3B
- Prompt: auto_research p00
- Independent trajectories: seeds 0 and 1
- Input geometry: 81 frames, 832x480
- Solver: UniPC, 50 steps, shift 8.0, guidance 6.0
- Precision/backend: BF16, FlashAttention-2
- Dense warm-up: first 5 steps
- N9: retained latent indices [3,4,5,8,9,10,13,14,15], compact RoPE 0..8
- N6: retained latent indices [0,1,2,9,10,11], compact RoPE 0..5

## Timing and fairness

Two independent resident pipelines and request states share one process. Each
owns a persistent worker thread and a distinct non-default CUDA stream. The
candidate-specific serial and parallel legs use the same pipelines, allocator
state, prompt, seeds, tensor shapes, and math. Model load, CUDA init, text
encoding, scheduler/noise preparation, first shape warm-up, VAE decode, video
encoding, and VLM are excluded. Timing covers the five dense DiT warm-up steps
and 45 sparse DiT steps, matching the paper's denoising-only contract.

Run one excluded warm-up per pattern, then at least three paired blocks with
SP/PS alternation. If either makespan CV exceeds 5%, extend to five blocks.
Profiler-off results are authoritative.

Primary metrics:

- Tserial2
- Tpair
- Rconc = Tpair / Tserial2
- Sconc = Tserial2 / Tpair
- pair gain = 1 - Rconc
- per-request parallel slowdown relative to the matched serial service time
- fairness, whole-device peak VRAM, power, clock, temperature
- deterministic output hashes and finite-value checks

The sparse concurrency gain is reported separately from the sparse algorithmic
speedup. It must not be computed by dividing against the dense serial time.

## Admission and interference

GPU0 must be idle before start. During every condition, any compute PID other
than this benchmark PID invalidates the condition and aborts the run. The
monitor never kills an external process. CUDA_VISIBLE_DEVICES must equal 0;
GPUs 1-3 are forbidden.

The historical dense E013 Rconc=0.984497680 is context only because it used the
p00/p01 prompt pair, not the same-prompt seed-0/seed-1 contract used here.
