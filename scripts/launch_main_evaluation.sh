#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
mkdir -p "${PROJECT_ROOT}/logs/main_evaluation"

subjects=(dog2 cat pet_dog1 bear_plushie robot_toy teapot colorful_sneaker luggage_backpack1)
migs=("${MIG0}" "${MIG1}" "${MIG2}" "${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")

for index in "${!subjects[@]}"; do
  subject="${subjects[$index]}"
  log="${PROJECT_ROOT}/logs/main_evaluation/${subject}.log"
  CUDA_VISIBLE_DEVICES="${migs[$index]}" \
    nohup bash "${PROJECT_ROOT}/scripts/jobs/generate_main_subject.sh" "${subject}" \
    > "${log}" 2>&1 &
  echo "$!" > "${PROJECT_ROOT}/logs/main_evaluation/${subject}.pid"
  echo "LAUNCHED subject=${subject} pid=$! log=${log}"
  sleep 2
done
