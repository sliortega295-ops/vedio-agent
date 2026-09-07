# Vendored CADense / CoFrame

- Source: https://github.com/sliortega295-ops/CADense
- Revision: `ceddd6f08ec4821acac1210da8044d3015f1601b` (main snapshot)
- Source package: `coframe/`; tests: upstream `tests/*.py`, placed in
  `package/tests/cadense/` without modifications.
- License: Apache-2.0, declared by upstream `pyproject.toml` and `NOTICE`.
  Upstream NOTICE is retained verbatim; LICENSE contains the standard Apache
  2.0 text. No checkpoints, results, credentials, environments, or Git metadata
  are included.
- `upstream_sha256.json` records hashes **before local modifications**, with
  paths relative to the upstream repository; it is not a current-tree manifest.

Local modifications to upstream Python files:

| File | Change |
|---|---|
| `config.py` | Explicit `sparse_evolution` method, residual-budget validation, dense sampler tail |
| `wan/sparse_forward.py` | Group-stable coverage interleaving, causal residual-to-budget feedback, non-finite guard, replay-safe budget reset, one-candidate guard |
| `wan/pipeline.py` | New controller routing, exact zero-warmup/tail behavior, full-frame and single-candidate checks |
| `cli.py` | New mode/options, optional parser injection for SVA defaults, validate before model load/output creation, refuse existing run directories |

Other Python files and upstream tests retain their pinned contents. New SVA
entry/tests/documentation live outside this vendor directory. The original
upstream modes are retained as comparisons; `sparse_evolution` is a separately
named integration, not a claim that CADense's existing `coframe_ode` already
implements the paper figure's residual-budget feedback.
