# Sparse Video Agent N9/N6 L40 concurrency artifact

Complete artifact for the two-candidate serial-vs-concurrent experiment on one
NVIDIA L40.

## Measured result

- N9: `Rconc=0.999888491`, throughput `1.000111521x`, pair gain `+0.011151%`,
  bootstrap 95% CI `[-0.478799%, +0.731713%]`.
- N6: `Rconc=1.035911852`, throughput `0.965333101x`, pair gain `-3.591185%`,
  bootstrap 95% CI `[-3.900843%, -3.126559%]`.
- Correctness hashes match and all outputs are finite.
- Peak whole-device VRAM: 19,914 MiB.
- The formal run was interference-free across 2,791 whole-process GPU0 guard
  observations.

## Assets

- `sva_candidate_concurrency_20260809_full.tar.zst`: complete frozen 147-file
  source/result tree, including raw Nsys reports/SQLite, `.pt` correctness
  outputs, complete logs, telemetry, and invalid attempts.
- `sva_candidate_concurrency_20260809_full.tar.zst.sha256`: relative archive
  checksum.
- `sva_candidate_concurrency_20260809_FILES.sha256`: file-level manifest for the
  extracted snapshot.

Archive SHA256:

```text
8c0674892d26258215f5e11c7b87da814d753744581aa39f57fc6a4afe2cff6e
```

## Claim boundary

The formal metrics are profiler-off and denoising-only. Parallel N9/N6 Nsys
captures are included. Valid serial Nsys references and NCU counters were Not
Run before snapshot freeze because GPU0 was occupied again by external work;
no external process was terminated. The release therefore does not claim
Tensor/SM/DRAM saturation or a measured serial-to-parallel per-kernel inflation
factor.
