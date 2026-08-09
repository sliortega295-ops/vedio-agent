#!/usr/bin/env bash
set -u
BASE=/data/jiayu/yongyan_liu/auto_research/research/sva_candidate_concurrency
PKG="$BASE/package"
AUTO=/data/jiayu/yongyan_liu/auto_research
PY="$AUTO/.conda/bin/python"
NSYS="$AUTO/.tools/nsys"
CKPT=/data/models/Wan2.1-T2V-1.3B
wait_exclusive() {
  local needed="$1" idle_since=0 deadline=$(( $(date +%s) + 14400 ))
  while true; do
    local now apps workers
    now=$(date +%s)
    (( now < deadline )) || { echo "exclusive wait timeout"; exit 75; }
    apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    workers=$(ps -eo pid=,args= | grep -E '/data/anqian/.*(geneval_generate|filter_sparse_by_ssim_lpips|sd-cli)' | grep -v grep || true)
    if [[ -z "${apps//[[:space:]]/}" && -z "${workers//[[:space:]]/}" ]]; then
      (( idle_since != 0 )) || { idle_since=$now; echo "idle_start=$(date -Is) needed=${needed}s"; }
      (( now-idle_since >= needed )) && return
    else
      idle_since=0
      echo "waiting_external=$(date -Is)"
      printf "%s\n%s\n" "$apps" "$workers"
    fi
    sleep 10
  done
}
first=1
for pattern in N9 N6; do
  if (( first )); then wait_exclusive 300; first=0; else wait_exclusive 120; fi
  lower="${pattern,,}"
  OUT="$BASE/profiles/nsys_${lower}_serial_v2"
  RUN="$BASE/runs/profile_${lower}_serial_v2"
  mkdir -p "$RUN" "$BASE/profiles"
  (
    cd "$PKG"
    export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PKG/third_party" SVA_ENABLE_NVTX=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_HOME="$BASE/cache/huggingface" TRANSFORMERS_CACHE="$BASE/cache/huggingface"
    export TORCH_HOME="$BASE/cache/torch" TRITON_CACHE_DIR="$BASE/cache/triton" CUDA_CACHE_PATH="$BASE/cache/cuda"
    "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none       --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true       -o "$OUT"       "$PY" "$PKG/bench_sparse_candidate_concurrency.py"         --checkpoint "$CKPT" --output-root "$RUN" --patterns "$pattern"         --blocks 3 --max-blocks-on-noise 5 --profile-once --profile-mode SERIAL
  ) >"$RUN/nsys_stdout.log" 2>"$RUN/nsys_stderr.log"
  rc=$?
  echo "$rc" >"$RUN/nsys_return_code.txt"
  if [[ ! -s "$OUT.nsys-rep" ]]; then echo "missing report $pattern rc=$rc"; exit 76; fi
  "$NSYS" export --type sqlite --force-overwrite=true --output "$OUT.sqlite" "$OUT.nsys-rep"     >"$RUN/export_stdout.log" 2>"$RUN/export_stderr.log" || exit 77
  sha256sum "$OUT.nsys-rep" "$OUT.sqlite" >"$RUN/profile_sha256.txt"
  echo "$pattern serial trace complete target_rc=$rc"
done
"$PY" "$AUTO/research/profiles/analyze_nsys_sqlite.py"   "$BASE/profiles/nsys_n9_serial_v2.sqlite" "$BASE/profiles/nsys_n6_serial_v2.sqlite"   --labels N9_SERIAL N6_SERIAL --expected-request-counts 2 2   --first-step 20 --last-step 24   --capture-name-pattern '^sva_capture_(?:N9|N6)_SERIAL$'   --output "$BASE/profiles/nsys_serial_analysis.json"
