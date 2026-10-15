#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBJECT="${1:?usage: run_main_subject_lane.sh SUBJECT}"
LOG_DIR="${PROJECT_ROOT}/logs/main_training"
mkdir -p "${LOG_DIR}"

status=0
for method in textual_inversion dreambooth lora; do
  echo "LANE_JOB_START subject=${SUBJECT} method=${method} timestamp=$(date --iso-8601=seconds)"
  if bash "${PROJECT_ROOT}/scripts/jobs/train_main_subject.sh" "${method}" "${SUBJECT}"; then
    echo "LANE_JOB_SUCCESS subject=${SUBJECT} method=${method} timestamp=$(date --iso-8601=seconds)"
  else
    code=$?
    status=1
    echo "LANE_JOB_FAILED subject=${SUBJECT} method=${method} exit_code=${code} timestamp=$(date --iso-8601=seconds)" >&2
  fi
done
exit "${status}"
