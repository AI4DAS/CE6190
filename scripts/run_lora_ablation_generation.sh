#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zwu/anaconda3/envs/personalization-benchmark/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/ablations/generation"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
variants=(r4_s800 r16_s400 r16_s800 r16_s1200 r32_s800)
migs=("${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")
mkdir -p "${LOG_DIR}"

for offset in 0 4; do
  pids=()
  names=()
  limit=4
  (( offset == 4 )) && limit=1
  for ((lane=0; lane<limit; lane++)); do
    index=$((offset + lane))
    variant="${variants[$index]}"
    CUDA_VISIBLE_DEVICES="${migs[$lane]}" \
      "${PYTHON}" "${PROJECT_ROOT}/src/inference/generate_lora_ablation.py" \
      --project-root "${PROJECT_ROOT}" --variant "${variant}" \
      >"${LOG_DIR}/${variant}.log" 2>&1 &
    pids+=("$!")
    names+=("${variant}")
    echo "STARTED variant=${variant} pid=$!"
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "SUCCESS variant=${names[$index]}"
    else
      code=$?
      echo "FAILED variant=${names[$index]} code=${code}" >&2
      exit "${code}"
    fi
  done
done
