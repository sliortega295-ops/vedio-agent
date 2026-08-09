# INVALID attempt: formal_n9_n6_v1

- Admission: GPU0 was idle for 120 consecutive seconds.
- Failure point: before the first excluded N9 shape warm-up; no formal timing sample was collected.
- Cause: external /data/anqian/scripts/filter_sparse_by_ssim_lpips.py workers appeared on physical GPU0 after both private pipelines had loaded.
- Guard action: the benchmark aborted rather than mixing foreign load into a result.
- Classification: INVALID_EXTERNAL_GPU0_INTERFERENCE.
- Reuse: no latency or throughput number from this attempt is admissible.
- Follow-up: v2 requires 600 consecutive seconds of GPU0 idleness and absence of the known external GPU-worker commands. No external process was killed.
