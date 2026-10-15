#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/zwu/anaconda3/envs/environment_uno/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/modern_extension/uno"
LOG_DIR="${PROJECT_ROOT}/logs/modern_extension"
SEED="42,123,456,789"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set one MIG UUID" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/FAILED" "${OUTPUT_DIR}/SUCCESS"
on_exit() {
  code=$?
  trap - EXIT
  if [[ ${code} -eq 0 ]]; then
    touch "${OUTPUT_DIR}/SUCCESS"
  else
    touch "${OUTPUT_DIR}/FAILED"
  fi
  echo "exit_code=${code}"
  date
  exit "${code}"
}
trap on_exit EXIT
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
start=$(date +%s)
"${PYTHON}" "${PROJECT_ROOT}/src/inference/generate_modern_uno.py" \
  --project-root "${PROJECT_ROOT}" \
  --uno-repo "${PROJECT_ROOT}/third_party/modern/UNO" \
  --config "${PROJECT_ROOT}/configs/modern_extension.yaml" \
  --resume
png_count=$(find "${OUTPUT_DIR}" -type f -name 'seed_*.png' | wc -l)
json_count=$(find "${OUTPUT_DIR}" -type f -name 'seed_*.json' | wc -l)
[[ "${png_count}" -eq 192 && "${json_count}" -eq 192 ]] || {
  echo "ERROR: expected 192 PNG and 192 JSON files; found ${png_count} PNG and ${json_count} JSON" >&2
  exit 4
}
finish=$(date +%s)
echo "runtime_seconds=$((finish-start))"
echo "UNO_EXTENSION_COMPLETED_SUCCESSFULLY"
