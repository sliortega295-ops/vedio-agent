#!/usr/bin/env bash
set -euo pipefail

PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../package" && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
CKPT_DIR=${CKPT_DIR:-/data/models/Wan2.1-T2V-1.3B}
RUN_ROOT=${RUN_ROOT:-$PWD/runs/sva-acceleration}
GPU=${GPU:-0}

mkdir -p "$RUN_ROOT"

# 1) Does sharing one model help the existing two-stream design?
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$PKG/bench_shared_model_streams.py" \
  --checkpoint "$CKPT_DIR" \
  --output-root "$RUN_ROOT/shared-model-streams"

# 2) Compare exact fusion and approximate search policies.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$PKG/bench_acceleration_methods.py" \
  --checkpoint "$CKPT_DIR" \
  --output-root "$RUN_ROOT/fusion" \
  --patterns N9 N6 \
  --methods serial_b1 candidate_b2 candidate_cfg_b4 sparse_cfg \
  --seeds 0 1 \
  --candidate-microbatch 2 \
  --sparse-cfg-interval 2 \
  --blocks 3

# 3) Successive halving is intended for N=8. The built-in latent-RMS scorer is
# only a systems smoke test. For a paper result pass --scorer module:function
# that performs the actual preview/VLM scoring policy.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$PKG/bench_acceleration_methods.py" \
  --checkpoint "$CKPT_DIR" \
  --output-root "$RUN_ROOT/halving-system-smoke" \
  --patterns N6 \
  --methods serial_b1 halving \
  --seeds 0 1 2 3 4 5 6 7 \
  --candidate-microbatch 2 \
  --halving-schedule '15:4,30:2,45:1' \
  --blocks 1
