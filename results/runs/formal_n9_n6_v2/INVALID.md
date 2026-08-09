# INVALID attempt: formal_n9_n6_v2

- GPU0 exclusivity: PASS (236 guard observations; zero foreign-process observations).
- Failure point: first excluded N9 parallel shape warm-up; no formal timing sample was collected.
- Cause: PyTorch 2.4 CUDA linalg lazy-wrapper initialization raced between the two persistent worker threads at the first UniPC scheduler torch.linalg.solve call.
- Error: RuntimeError: lazy wrapper should be called at most once.
- Classification: INVALID_PRE_TIMING_HARNESS_RACE.
- Repair: one tiny torch.linalg.solve call is now initialized on the main thread before worker startup and excluded from all timings.
- Scientific workload, shapes, seeds, schedules, streams, and timing boundary are unchanged.
