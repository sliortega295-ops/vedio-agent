#!/usr/bin/env bash
# Run only on the user's selected GPU with a local Diffusers checkpoint.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to a Wan2.1-T2V-1.3B-Diffusers checkpoint}"
: "${PROMPT:?Set PROMPT to the candidate prompt}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to persistent experiment storage}"
: "${BUDGET_THRESHOLDS:?Set calibrated relative-RMS thresholds, e.g. three cuts for four budgets}"
exec "${PYTHON:-python}" "$ROOT/package/full_length_sparse.py" \
  --model-id "$MODEL_DIR" --prompt "$PROMPT" --output-dir "$OUTPUT_DIR" \
  --seed "${SEED:-0}" --num-anchors "${INITIAL_K:-9}" \
  --adaptive-k-values "${BUDGET_VALUES:-6,9,12,21}" \
  --adaptive-k-thresholds "$BUDGET_THRESHOLDS" "$@"
