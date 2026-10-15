#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
LOG_DIR="${PROJECT_ROOT}/logs/main_class_images"
mkdir -p "${LOG_DIR}"

SUBJECTS=(cat bear_plushie robot_toy teapot colorful_sneaker luggage_backpack1)
MIGS=("${MIG0}" "${MIG1}" "${MIG2}" "${MIG3}" "${MIG4}" "${MIG5}")

for index in "${!SUBJECTS[@]}"; do
  subject="${SUBJECTS[$index]}"
  mig="${MIGS[$index]}"
  log="${LOG_DIR}/${subject}.launcher.log"
  pidfile="${LOG_DIR}/${subject}.pid"
  [[ ! -e "${pidfile}" ]] || { echo "ERROR: pid file already exists: ${pidfile}" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="${mig}" nohup bash "${PROJECT_ROOT}/scripts/jobs/prepare_main_class_images.sh" "${subject}" > "${log}" 2>&1 &
  pid=$!
  echo "${pid}" > "${pidfile}"
  echo "LAUNCHED_CLASS_PREP subject=${subject} mig=${mig} pid=${pid} log=${log}"
done
