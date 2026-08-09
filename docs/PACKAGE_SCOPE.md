# Source-package scope note

`package/README.md` is retained unchanged from the broader user-provided
source archive for provenance. It describes components that are not present in
this focused concurrency snapshot (for example VLM selection and the full
T2V-CompBench evaluator). It is not the reproduction guide for this experiment.

For this repository, the operative files are:

- `package/bench_sparse_candidate_concurrency.py`;
- `package/scripts/generate_sparse_rope8.py`;
- `package/third_party/wan/`;
- `package/config.json`;
- `docs/PREREGISTRATION.md`;
- `scripts/launch_when_gpu0_idle_v3.sh`;
- `scripts/run_nsys_profiles.sh`.

No `config/vlm.env`, API credential, model weight, generated video collection,
or external user environment is included. Use `docs/REPRODUCIBILITY.md` rather
than the retained broad package README.
