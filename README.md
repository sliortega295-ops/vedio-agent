# Sparse Video Agent candidate concurrency on one NVIDIA L40

## New full-length sparse backend

This repository also includes a separate Diffusers Wan
backend with **interleaved anchor groups, full-length residual reconstruction,
and causal residual-to-budget feedback**. See
[`docs/FULL_LENGTH_SPARSITY.md`](docs/FULL_LENGTH_SPARSITY.md) for the paper-figure
mapping, CLI/API, CADense provenance, and validation limits. This is not the old
warmup-then-keyframe-only N9/N6 method. Its CPU contracts are tested; GPU speed
and video quality are **not yet measured**. The historical results below remain
unchanged and do not apply to this new backend.

This repository contains the code, structured results, telemetry, failure records,
and profiler analysis for a focused systems experiment:

> After temporal sparsity has already reduced a Video-DiT request, does it also
> make two independent test-time-search trajectories more concurrency-friendly?

The experiment compares candidate-specific **serial-two** and synchronized
**parallel-two** execution for the N9 and N6 sparse preview policies on one
NVIDIA L40. The previously completed dense/Full experiment was deliberately not
rerun and is not used as either sparse candidate's denominator.

## Headline result

Temporal sparsity did **not** create an additional two-candidate throughput gain
for these fixed shapes:

| Policy | `Tserial2` (s) | `Tpair` (s) | `Rconc` | Throughput speedup | Pair gain | 95% bootstrap CI for pair gain |
|---|---:|---:|---:|---:|---:|---:|
| N9 | 231.8510 | 231.8252 | 0.999888 | 1.000112x | +0.011% | [-0.479%, +0.732%] |
| N6 | 161.0061 | 166.7881 | 1.035912 | 0.965333x | -3.591% | [-3.901%, -3.127%] |

For N9, each concurrent request takes about twice its matched serial service
time, so two requests finish in essentially the same time as serial execution.
For N6, co-running is modestly worse than serial. Both conditions used about
19,914 MiB peak whole-device VRAM and passed deterministic output-hash and
finite-value checks.

The profiler-off timing run observed GPU0 for its entire lifetime: 2,791 guard
observations contained no foreign compute process. The result is therefore not
explained by another GPU workload.

## Nsight Systems evidence

A narrow CUDA/NVTX capture covers denoising steps 20--24 of parallel N9 and N6:

| Policy | Cross-request kernel overlap | Physical GPU idle | Top-10 kernel coverage | FlashAttention share |
|---|---:|---:|---:|---:|
| N9 | 15.621% | 5.778% | 85.449% | 29.025% |
| N6 | 23.214% | 6.439% | 88.474% | 24.449% |

Some kernels overlap, but the major BF16 CUTLASS `128x256` GEMM has zero
same-name cross-request overlap in both captures. Sparse trajectories therefore
remain dominated by long, resource-heavy kernels rather than turning into two
lightweight jobs that scale nearly 2x through streams alone.

The N6 `.nsys-rep` and SQLite export are valid and the parser admission passed;
a post-capture `nvidia-smi` timeout made the harness-level profile condition
invalid. This caveat does not affect the separate profiler-off timing result.
Serial Nsys reference traces and NCU counters are explicitly **Not Run** in this
snapshot because GPU0 was occupied again by external work; no external process
was killed. Consequently, the repository does not claim a serial-to-parallel
per-kernel inflation ratio or SM/Tensor/DRAM SOL from these traces.

## Repository layout

- `package/`: frozen benchmark harness, sparse generator, and vendored Wan code.
- `scripts/`: exact launch and Nsight Systems scripts used on the server.
- `results/`: formal JSON/JSONL results, telemetry, invalid attempts, and parsed
  Nsys analysis (populated from the frozen snapshot).
- `docs/PREREGISTRATION.md`: fixed workload, timing, fairness, and admission rules.
- `docs/RESULTS.md`: detailed interpretation and claim boundaries.
- `docs/REPRODUCIBILITY.md`: exact environment and commands.
- `docs/SNAPSHOT_STATUS.md`: completed, invalid, and Not Run inventory.
- `manifests/`: SHA256 manifests for source and released artifacts.
- `third_party_licenses/`: upstream license texts for vendored components.

Large raw artifacts (`.nsys-rep`, `.sqlite`, `.pt`, and the complete launcher
logs) are attached to the [tagged GitHub Release](https://github.com/sliortega295-ops/vedio-agent/releases/tag/sva-concurrency-2026-08-09) rather than committed as ordinary Git blobs. The release also includes one complete frozen 147-file snapshot archive
and its SHA256 manifest.

## Reproduce

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Model weights and Conda
environments are intentionally excluded; the original run used
`/data/models/Wan2.1-T2V-1.3B` and a user-owned environment.

## Claim boundary

This is a single-model, single-GPU, single-prompt/two-seed, denoising-only
systems result. N9/N6 retain their algorithmic sparse speedup versus Full, but
that speedup is separate from the concurrency question tested here. The data do
not show an extra concurrency multiplier for two candidates at these shapes.
