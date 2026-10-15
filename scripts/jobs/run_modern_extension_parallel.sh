#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"

DREAMO_LOG="${PROJECT_ROOT}/logs/modern_extension/dreamo_extension.log"
UNO_LOG="${PROJECT_ROOT}/logs/modern_extension/uno_extension.log"
mkdir -p "${PROJECT_ROOT}/logs/modern_extension"

echo "MODERN_EXTENSION_PARALLEL_START $(date --iso-8601=seconds)"
echo "DreamO MIG: ${MIG0}"
echo "UNO MIG: ${MIG1}"
{
  echo
  echo "===== RESUMABLE RUN $(date --iso-8601=seconds) ====="
} >> "${DREAMO_LOG}"
{
  echo
  echo "===== RESUMABLE RUN $(date --iso-8601=seconds) ====="
} >> "${UNO_LOG}"

env CUDA_VISIBLE_DEVICES="${MIG0}" \
  bash "${PROJECT_ROOT}/scripts/jobs/generate_dreamo_extension.sh" \
  >> "${DREAMO_LOG}" 2>&1 &
dreamo_pid=$!

env CUDA_VISIBLE_DEVICES="${MIG1}" \
  bash "${PROJECT_ROOT}/scripts/jobs/generate_uno_extension.sh" \
  >> "${UNO_LOG}" 2>&1 &
uno_pid=$!

echo "DreamO wrapper PID: ${dreamo_pid}"
echo "UNO wrapper PID: ${uno_pid}"

terminate_children() {
  echo "Coordinator received termination signal; stopping only its two child jobs." >&2
  kill "${dreamo_pid}" "${uno_pid}" 2>/dev/null || true
  wait "${dreamo_pid}" "${uno_pid}" 2>/dev/null || true
  exit 143
}
trap terminate_children INT TERM

if wait "${dreamo_pid}"; then
  dreamo_code=0
else
  dreamo_code=$?
fi
if wait "${uno_pid}"; then
  uno_code=0
else
  uno_code=$?
fi

echo "DreamO exit code: ${dreamo_code}"
echo "UNO exit code: ${uno_code}"
echo "MODERN_EXTENSION_PARALLEL_FINISH $(date --iso-8601=seconds)"

if [[ "${dreamo_code}" -ne 0 || "${uno_code}" -ne 0 ]]; then
  exit 1
fi
