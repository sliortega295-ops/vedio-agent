# Results and interpretation

## Frozen question

Does an already-sparse N9 or N6 Video-DiT trajectory release enough compute
headroom that two independent test-time-search candidates co-run more
 efficiently than they execute serially?

The only valid denominator is each policy's own matched serial-two makespan:

```text
Rconc = Tpair / Tserial2
Sconc = Tserial2 / Tpair
pair gain = 1 - Rconc
```

Dense/Full numbers are context only and are not mixed into these ratios.

## Formal profiler-off run (`formal_n9_n6_v3`)

| Metric | N9 | N6 |
|---|---:|---:|
| Serial-two raw makespans (s) | 232.150220, 231.422422, 231.851023 | 161.006089, 161.416841, 160.958417 |
| Parallel-two raw makespans (s) | 230.451547, 232.530471, 231.825169 | 166.788116, 166.463634, 167.237151 |
| Median `Tserial2` (s) | 231.851023 | 161.006089 |
| Median `Tpair` (s) | 231.825169 | 166.788116 |
| `Rconc` | 0.999888491 | 1.035911852 |
| `Sconc` | 1.000111521x | 0.965333101x |
| Pair gain | +0.011151% | -3.591185% |
| Pair-gain bootstrap 95% CI | [-0.478799%, +0.731713%] | [-3.900843%, -3.126559%] |
| Per-request parallel slowdown | approximately 2.00x | approximately 2.07x |
| Peak whole-device VRAM | 19,914 MiB | 19,914 MiB |
| Correctness | hashes match; finite | hashes match; finite |

Protocol: one excluded warmup per pattern, three SP/PS-alternating paired blocks,
fixed prompt, seeds 0 and 1, two resident private pipelines, distinct CUDA
streams, and a synchronized parallel start. CV did not trigger the preregistered
extension to five blocks. Model loading, CUDA initialization, text encoding,
noise/scheduler preparation, VAE decode, video encoding, and VLM work are
excluded. The timed scope contains five dense warmup denoising steps plus 45
sparse denoising steps.

### Admission and correctness

- Physical GPU: NVIDIA L40, GPU0 only.
- Whole-process guard: 2,791 observations; zero foreign compute observations.
- Fixed workload: 81 input frames, 832x480, 50 steps, BF16, FlashAttention-2,
  guidance 6.0, shift 8.0.
- N9 retained latent indices: `[3,4,5,8,9,10,13,14,15]`.
- N6 retained latent indices: `[0,1,2,9,10,11]`.
- Output tensor hashes for serial and parallel paths match for both seeds; no
  NaN or Inf was observed.

## Timeline evidence

Nsight Systems captured only the fixed middle denoising window, steps 20--24.
Profiler time is not used for the formal makespans.

| Metric | N9 parallel | N6 parallel |
|---|---:|---:|
| Request/kernel overlap fraction | 15.6210% | 23.2140% |
| GPU idle fraction in analysis window | 5.7779% | 6.4394% |
| Top-10 kernel-time coverage | 85.4489% | 88.4739% |
| FlashAttention share of kernel sum | 29.0255% | 24.4492% |
| Main CUTLASS 128x256 same-name overlap | none | none |

Interpretation:

1. Sparse execution does permit some genuine kernel overlap.
2. That overlap does not translate into more completed trajectories per unit
   time: N9 is statistically indistinguishable from serial, while N6 regresses.
3. Low-but-nonzero idle time means host gaps exist, but eliminating all measured
   idle cannot approach a second-request-for-free result.
4. The dominant attention/GEMM shapes remain large enough to contend or queue.
   A serial trace/counter comparison is required before assigning the slowdown
   specifically to Tensor Core, SM, L2, or DRAM saturation.

## Negative and incomplete evidence retained

- `formal_n9_n6_v1`: INVALID before warmup because an external GPU0 workload
  appeared.
- `formal_n9_n6_v2`: INVALID before timing because concurrent lazy
  `torch.linalg.solve` initialization raced; the minimal pre-initialization fix
  is retained as a patch.
- `formal_n9_n6_v3`: valid formal result.
- N9/N6 parallel Nsys: captured and exported. N6 has a post-capture telemetry
  timeout caveat, but the report and parser admission are valid.
- Serial Nsys: attempted only after external work returned; INVALID/Not Run as a
  scientific reference, with no report.
- NCU: Not Run; hardware counters required privileged access and the GPU was no
  longer exclusively available.

## Conclusion

For this exact workload, temporal sparsity reduces the cost of an individual
trajectory but does not independently move the two-candidate concurrency
boundary. Treat sparse speedup and concurrency speedup as separate factors;
measured concurrency contributes approximately 1.000x for N9 and 0.965x for N6.
