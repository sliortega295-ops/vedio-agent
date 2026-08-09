#!/usr/bin/env bash
set -euo pipefail
BASE=/data/jiayu/yongyan_liu/auto_research/research/sva_candidate_concurrency
PKG="$BASE/package"
AUTO=/data/jiayu/yongyan_liu/auto_research
PY="$AUTO/.conda/bin/python"
NSYS="$AUTO/.tools/nsys"
CKPT=/data/models/Wan2.1-T2V-1.3B
FORMAL="$BASE/runs/formal_n9_n6_v3/result.json"

[[ -s "$FORMAL" ]] || { echo "formal result missing: $FORMAL" >&2; exit 2; }
for pattern in N9 N6; do
  lower="${pattern,,}"
  OUT="$BASE/profiles/nsys_${lower}_parallel"
  RUN="$BASE/runs/profile_${lower}"
  mkdir -p "$BASE/profiles" "$RUN" "$BASE/cache/huggingface" "$BASE/cache/torch" \
    "$BASE/cache/triton" "$BASE/cache/cuda"
  apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
  [[ -z "${apps//[[:space:]]/}" ]] || {
    echo "GPU0 not idle before $pattern profile; refusing to interfere" >&2
    printf "%s\n" "$apps" >&2
    exit 3
  }
  (
    cd "$PKG"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$PKG/third_party"
    export SVA_ENABLE_NVTX=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$BASE/cache/huggingface"
    export TRANSFORMERS_CACHE="$BASE/cache/huggingface"
    export TORCH_HOME="$BASE/cache/torch"
    export TRITON_CACHE_DIR="$BASE/cache/triton"
    export CUDA_CACHE_PATH="$BASE/cache/cuda"
    "$NSYS" profile \
      --trace=cuda,nvtx \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      -o "$OUT" \
      "$PY" "$PKG/bench_sparse_candidate_concurrency.py" \
        --checkpoint "$CKPT" \
        --output-root "$RUN" \
        --patterns "$pattern" \
        --blocks 3 \
        --max-blocks-on-noise 5 \
        --profile-once
  ) >"$RUN/nsys_stdout.log" 2>"$RUN/nsys_stderr.log"
  [[ -s "$OUT.nsys-rep" ]] || { echo "missing $OUT.nsys-rep" >&2; exit 4; }
  "$NSYS" export --type sqlite --force-overwrite=true \
    --output "$OUT.sqlite" "$OUT.nsys-rep" \
    >"$RUN/export_stdout.log" 2>"$RUN/export_stderr.log"
  [[ -s "$OUT.sqlite" ]] || { echo "missing $OUT.sqlite" >&2; exit 5; }
  sha256sum "$OUT.nsys-rep" "$OUT.sqlite" >"$RUN/profile_sha256.txt"
done

"$PY" "$AUTO/research/profiles/analyze_nsys_sqlite.py" \
  "$BASE/profiles/nsys_n9_parallel.sqlite" \
  "$BASE/profiles/nsys_n6_parallel.sqlite" \
  --labels N9_PARALLEL N6_PARALLEL \
  --expected-request-counts 2 2 \
  --first-step 20 --last-step 24 \
  --capture-name-pattern '^sva_capture_(?:N9|N6)_PARALLEL$' \
  --output "$BASE/profiles/nsys_analysis.json"
