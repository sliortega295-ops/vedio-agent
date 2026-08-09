# Frozen snapshot status

Snapshot frozen on 2026-08-09 after stopping only this study's idle-waiting
serial-profile wrapper. External `/data/anqian` GPU processes were not touched.

## Complete

- N9/N6 serial-vs-parallel preregistration.
- Benchmark harness and sparse Wan source package.
- Formal profiler-off run `formal_n9_n6_v3`.
- Three paired blocks per policy with raw condition records and telemetry.
- Correctness hashes/finite checks.
- N9 parallel Nsight Systems report, SQLite export, and parsed analysis.
- N6 parallel Nsight Systems report, SQLite export, and parsed analysis.
- Full raw snapshot, file manifest, and archive SHA256.

## Invalid attempts retained

- `formal_n9_n6_v1`: foreign GPU0 workload appeared before warmup.
- `formal_n9_n6_v2`: pre-timing lazy linear-algebra initialization race.
- `profile_n9_serial`: external GPU0 workload; no valid trace produced.

## Not Run / pending at snapshot

- Valid N9 and N6 serial Nsight Systems reference traces.
- Nsight Compute hardware-counter collection.
- Dense/Full rerun (explicitly excluded by user because it was already complete
  under a separate experiment).

These missing rows are not silently inferred. In particular, the parallel Nsys
captures do not support a claim about Tensor/SM/DRAM saturation or a precise
serial-to-parallel kernel-duration inflation factor.
