#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
source "${ROOT}/scripts/mig_devices.sh"
MIGS=("${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}")
TASKS=("dreambooth dog2" "dreambooth cat" "lora dog2" "lora cat")
LOG="${ROOT}/logs/ablations/task_d_face_corruption"
STATUS="${ROOT}/outputs/experiments/task_d_face_corruption"
mkdir -p "${LOG}" "${STATUS}"
unlink "${STATUS}/FAILED" 2>/dev/null || true
unlink "${STATUS}/SUCCESS" 2>/dev/null || true
echo "TASK_D_START=$(date --iso-8601=seconds) EXPECTED_IMAGES=96 ETA_MINUTES=12-25"
started=$(date +%s)
pids=()
for index in "${!TASKS[@]}"; do
  read -r method subject <<<"${TASKS[$index]}"
  CUDA_VISIBLE_DEVICES="${MIGS[$index]}" \
    "${PYTHON}" \
    "${ROOT}/src/inference/generate_face_corruption_checks.py" \
    --project-root "${ROOT}" --method "${method}" --subject "${subject}" --resume \
    >"${LOG}/${method}_${subject}.log" 2>&1 &
  pids+=("$!")
done
code=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then code=1; fi; done
if ((code)); then touch "${STATUS}/FAILED"; exit 1; fi
count=$(find "${STATUS}" -type f -name 'seed_*.png' | wc -l)
[[ "${count}" -eq 96 ]] || { echo "Expected 96 PNG, found ${count}" >&2; touch "${STATUS}/FAILED"; exit 2; }
touch "${STATUS}/SUCCESS"
finished=$(date +%s)
echo "runtime_seconds=$((finished-started))" >"${LOG}/runtime.txt"
echo "TASK_D_COMPLETE=$(date --iso-8601=seconds)"
