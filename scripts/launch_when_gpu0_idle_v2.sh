#!/usr/bin/env bash
set -euo pipefail
BASE=/data/jiayu/yongyan_liu/auto_research/research/sva_candidate_concurrency
PKG="$BASE/package"
PY=/data/jiayu/yongyan_liu/auto_research/.conda/bin/python
CKPT=/data/models/Wan2.1-T2V-1.3B
RUN="$BASE/runs/formal_n9_n6_v2"
LOG="$RUN/launcher.log"
mkdir -p "$RUN"
exec >>"$LOG" 2>&1
echo "launcher_started=$(date -Is)"
echo "launcher_pid=$$"
echo "policy=wait for 600 consecutive seconds with physical GPU0 idle and known external GPU workers absent; never kill external work"
if ! mkdir "$BASE/.formal_launch_lock" 2>/dev/null; then
  echo "ERROR: launch lock already exists"
  exit 70
fi
printf "%s\n" "$$" > "$BASE/.formal_launch_lock/pid"
cleanup_lock() { rm -f "$BASE/.formal_launch_lock/pid"; rmdir "$BASE/.formal_launch_lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
idle_since=0
deadline=$(( $(date +%s) + 14400 ))
while true; do
  now=$(date +%s)
  if (( now >= deadline )); then
    echo "ERROR: timed out waiting for GPU0 idle at $(date -Is)"
    exit 75
  fi
  apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  workers=$(ps -eo pid=,args= | grep -E '/data/anqian/scripts/(geneval_generate|filter_sparse_by_ssim_lpips)[.]py|[ /]sd-cli( |$)' | grep -v grep || true)
  if [[ -z "${apps//[[:space:]]/}" && -z "${workers//[[:space:]]/}" ]]; then
    if (( idle_since == 0 )); then
      idle_since=$now
      echo "idle_window_started=$(date -Is)"
    fi
    if (( now - idle_since >= 600 )); then
      break
    fi
  else
    if (( idle_since != 0 )); then
      echo "idle_window_reset=$(date -Is)"
    fi
    idle_since=0
    echo "waiting_for_exclusive_gpu0=$(date -Is)"
    if [[ -n "${apps//[[:space:]]/}" ]]; then
      echo "gpu0_compute_apps:"
      printf "%s\n" "$apps"
    fi
    if [[ -n "${workers//[[:space:]]/}" ]]; then
      echo "known_external_gpu_workers:"
      printf "%s\n" "$workers"
    fi
  fi
  sleep 10
done
echo "admitted_at=$(date -Is)"
nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader > "$RUN/gpu0_preflight.csv"
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RUN/gpu0_compute_preflight.csv" 2>&1 || true
if [[ -s "$RUN/gpu0_compute_preflight.csv" ]]; then
  echo "ERROR: GPU0 admission raced with a new compute process"
  cat "$RUN/gpu0_compute_preflight.csv"
  exit 76
fi
(
  cd "$PKG"
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="$PKG/third_party"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HOME="$BASE/cache/huggingface"
  export TRANSFORMERS_CACHE="$BASE/cache/huggingface"
  export TORCH_HOME="$BASE/cache/torch"
  export TRITON_CACHE_DIR="$BASE/cache/triton"
  export CUDA_CACHE_PATH="$BASE/cache/cuda"
  mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
  echo "benchmark_command=$PY $PKG/bench_sparse_candidate_concurrency.py --checkpoint $CKPT --output-root $RUN --patterns N9 N6 --blocks 3 --max-blocks-on-noise 5"
  "$PY" "$PKG/bench_sparse_candidate_concurrency.py" \
    --checkpoint "$CKPT" \
    --output-root "$RUN" \
    --patterns N9 N6 \
    --blocks 3 \
    --max-blocks-on-noise 5
)
rc=$?
echo "$rc" > "$RUN/return_code.txt"
echo "launcher_finished=$(date -Is) rc=$rc"
nvidia-smi -i 0 --query-gpu=index,uuid,memory.used,utilization.gpu,pstate --format=csv,noheader > "$RUN/gpu0_post.csv" || true
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RUN/gpu0_compute_post.csv" 2>&1 || true
exit "$rc"
