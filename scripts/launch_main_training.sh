#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
LOG_DIR="${PROJECT_ROOT}/logs/main_training"
mkdir -p "${LOG_DIR}"

SUBJECTS=(cat pet_dog1 bear_plushie robot_toy teapot colorful_sneaker luggage_backpack1)
MIGS=("${MIG0}" "${MIG1}" "${MIG2}" "${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}")
PIDS=()

for index in "${!SUBJECTS[@]}"; do
  subject="${SUBJECTS[$index]}"
  mig="${MIGS[$index]}"
  log="${LOG_DIR}/${subject}.lane.log"
  pidfile="${LOG_DIR}/${subject}.pid"
  [[ ! -e "${pidfile}" ]] || { echo "ERROR: pid file already exists: ${pidfile}" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="${mig}" bash "${PROJECT_ROOT}/scripts/jobs/run_main_subject_lane.sh" "${subject}" > "${log}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  echo "${pid}" > "${pidfile}"
  echo "LAUNCHED subject=${subject} mig=${mig} pid=${pid} log=${log}"
done

status=0
for index in "${!SUBJECTS[@]}"; do
  subject="${SUBJECTS[$index]}"
  if wait "${PIDS[$index]}"; then
    touch "${LOG_DIR}/${subject}.lane.SUCCESS"
    rm -f "${LOG_DIR}/${subject}.lane.FAILED"
    echo "LANE_COMPLETED subject=${subject}"
  else
    code=$?
    status=1
    touch "${LOG_DIR}/${subject}.lane.FAILED"
    rm -f "${LOG_DIR}/${subject}.lane.SUCCESS"
    echo "LANE_FAILED subject=${subject} exit_code=${code}" >&2
  fi
done
exit "${status}"
