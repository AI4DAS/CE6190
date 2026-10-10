#!/usr/bin/env bash
set -euo pipefail

# Usage: run_with_vram_monitor OUTPUT_DIR LOG_FILE command [args...]
run_with_vram_monitor() {
  local output_dir="$1" log_file="$2"
  shift 2
  local start_seconds finish_seconds train_pid status current peak_mib=0 descendants
  start_seconds=$(date +%s)
  "$@" > >(tee "${log_file}") 2>&1 &
  train_pid=$!
  while kill -0 "${train_pid}" 2>/dev/null; do
    descendants=$(pstree -p "${train_pid}" 2>/dev/null | grep -oE '\([0-9]+\)' | tr -d '()' | paste -sd, - || true)
    if [[ -n "${descendants}" ]]; then
      current=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', *' -v ids=",${descendants}," 'index(ids, ","$1",") {if ($2+0>m) m=$2+0} END {print m+0}')
      (( current > peak_mib )) && peak_mib=${current}
    fi
    sleep 1
  done
  set +e
  wait "${train_pid}"
  status=$?
  set -e
  finish_seconds=$(date +%s)
  echo "finish=$(date --iso-8601=seconds)" | tee -a "${output_dir}/runtime.txt"
  echo "training_time_seconds=$((finish_seconds-start_seconds))" | tee -a "${output_dir}/runtime.txt"
  echo "peak_vram_mib=${peak_mib}" | tee "${output_dir}/peak_vram.txt"
  awk -v mib="${peak_mib}" 'BEGIN {printf "peak_vram_gb=%.4f\n", mib/1024}' | tee -a "${output_dir}/peak_vram.txt"
  return "${status}"
}

