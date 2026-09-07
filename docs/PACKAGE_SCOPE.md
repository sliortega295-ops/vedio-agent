# Source-package scope note

`package/` remains the focused candidate-concurrency snapshot. Its README was
retained from a broader test-time-scaling archive and is not the reproduction
guide for the concurrency result.

The repository now also contains `agent/`, a separate public source package for
the complete SparseMAC sparse-to-full workflow. The agent includes its planner,
analyzer, VLM clients, prompt templates, generation launchers, and tests. Its
presence does not change the scope or provenance of the frozen concurrency
measurements in `package/`, `results/`, and `docs/RESULTS.md`.

For this repository, the operative files are:

- `package/bench_sparse_candidate_concurrency.py`;
- `package/scripts/generate_sparse_rope8.py`;
- `package/third_party/wan/`;
- `package/config.json`;
- `docs/PREREGISTRATION.md`;
- `scripts/launch_when_gpu0_idle_v3.sh`;
- `scripts/run_nsys_profiles.sh`.

No populated API credential, model weight, generated video collection, or
external user environment is included. `agent/config/env.example` contains
variable names and placeholder values only. Use `docs/REPRODUCIBILITY.md` for
the concurrency artifact and `agent/README.md` for the agent.
