#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zwu/anaconda3/envs/personalization-benchmark/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/main_evaluation/ip_adapter_scale03"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"

subjects=(dog2 cat pet_dog1 bear_plushie robot_toy teapot colorful_sneaker luggage_backpack1)
migs=("${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")
mkdir -p "${LOG_DIR}"

echo "IP_ADAPTER_REGENERATION_START timestamp=$(date --iso-8601=seconds) scale=0.3"
for offset in 0 4; do
  pids=()
  batch_subjects=()
  for lane in 0 1 2 3; do
    index=$((offset + lane))
    subject="${subjects[$index]}"
    log="${LOG_DIR}/${subject}.log"
    echo "START subject=${subject} device=${migs[$lane]} log=${log}"
    CUDA_VISIBLE_DEVICES="${migs[$lane]}" \
      "${PYTHON}" "${PROJECT_ROOT}/src/inference/generate_main_benchmark.py" \
      --project-root "${PROJECT_ROOT}" \
      --subject "${subject}" \
      --method ip_adapter_plus \
      --overwrite >"${log}" 2>&1 &
    pids+=("$!")
    batch_subjects+=("${subject}")
  done

  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "SUCCESS subject=${batch_subjects[$index]}"
    else
      code=$?
      echo "FAILED subject=${batch_subjects[$index]} exit_code=${code}" >&2
      exit "${code}"
    fi
  done
done
echo "IP_ADAPTER_REGENERATION_COMPLETED timestamp=$(date --iso-8601=seconds) scale=0.3"
