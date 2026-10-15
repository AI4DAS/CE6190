#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
MIGS=("${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")
TASKS=(
  "dreambooth dog2" "dreambooth teapot" "dreambooth bear_plushie"
  "lora dog2" "lora teapot" "lora bear_plushie"
  "textual_inversion dog2" "textual_inversion teapot" "textual_inversion bear_plushie"
)
LOG_ROOT="${PROJECT_ROOT}/logs/ablations/task_c_steps"
STATUS_ROOT="${PROJECT_ROOT}/outputs/ablations/task_c_steps_training"
mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"
unlink "${STATUS_ROOT}/FAILED" 2>/dev/null || true
unlink "${STATUS_ROOT}/SUCCESS" 2>/dev/null || true
echo "TASK_C_START=$(date --iso-8601=seconds) JOBS=${#TASKS[@]} ETA_MINUTES=70-120"
start=$(date +%s)

worker() {
  local lane="$1" device="$2" index method subject
  for ((index=lane; index<${#TASKS[@]}; index+=${#MIGS[@]})); do
    read -r method subject <<< "${TASKS[$index]}"
    echo "START lane=${lane} method=${method} subject=${subject} device=${device}"
    CUDA_VISIBLE_DEVICES="${device}" bash "${PROJECT_ROOT}/scripts/jobs/train_task_c_steps_lane.sh" \
      "${method}" "${subject}" > "${LOG_ROOT}/${method}_${subject}.launcher.log" 2>&1
    echo "DONE lane=${lane} method=${method} subject=${subject}"
  done
}

pids=()
for lane in "${!MIGS[@]}"; do worker "${lane}" "${MIGS[$lane]}" & pids+=("$!"); done
code=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then code=1; fi; done
finish=$(date +%s)
echo "runtime_seconds=$((finish-start))" > "${LOG_ROOT}/runtime.txt"
if (( code != 0 )); then touch "${STATUS_ROOT}/FAILED"; exit 1; fi
touch "${STATUS_ROOT}/SUCCESS"
echo "TASK_C_COMPLETE=$(date --iso-8601=seconds)"
