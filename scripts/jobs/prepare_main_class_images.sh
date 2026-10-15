#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
LOG_DIR="${PROJECT_ROOT}/logs/main_class_images"
SEED=10000
SUBJECT="${1:?usage: prepare_main_class_images.sh SUBJECT}"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
CONFIG="${PROJECT_ROOT}/configs/main_subjects.yaml"
READER="${PROJECT_ROOT}/src/utils/read_main_subject.py"

mapfile -t FIELDS < <("${PYTHON}" "${READER}" --config "${CONFIG}" --subject "${SUBJECT}")
[[ "${#FIELDS[@]}" -eq 11 ]] || { echo "ERROR: invalid subject record" >&2; exit 2; }
CLASS_PROMPT="${FIELDS[6]}"
CLASS_DIR="${PROJECT_ROOT}/${FIELDS[7]}"
EXPECTED=200
SAFE_NAME="${SUBJECT//[^a-zA-Z0-9_-]/_}"
LOG_FILE="${LOG_DIR}/${SAFE_NAME}.log"
SUCCESS_MARKER="${LOG_DIR}/${SAFE_NAME}.SUCCESS"
FAILED_MARKER="${LOG_DIR}/${SAFE_NAME}.FAILED"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set one MIG UUID" >&2; exit 2; }
[[ -f "${MODEL_DIR}/model_index.json" ]] || { echo "ERROR: missing model" >&2; exit 3; }
mkdir -p "${CLASS_DIR}" "${LOG_DIR}"
NON_IMAGE_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f ! \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
[[ "${NON_IMAGE_COUNT}" -eq 0 ]] || { echo "ERROR: class directory contains non-images" >&2; exit 4; }
COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
if [[ "${COUNT}" -eq "${EXPECTED}" ]]; then
  echo "CLASS_IMAGES_ALREADY_COMPLETE subject=${SUBJECT} count=${COUNT}"
  touch "${SUCCESS_MARKER}"
  rm -f "${FAILED_MARKER}"
  exit 0
fi
[[ "${COUNT}" -lt "${EXPECTED}" ]] || { echo "ERROR: found ${COUNT} images, expected at most ${EXPECTED}" >&2; exit 5; }

on_exit() {
  local code=$?
  if (( code != 0 )); then
    touch "${FAILED_MARKER}"
    rm -f "${SUCCESS_MARKER}"
  fi
}
trap on_exit EXIT
rm -f "${FAILED_MARKER}"
date
hostname
echo "subject=${SUBJECT}"
echo "prompt=${CLASS_PROMPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
"${PYTHON}" "${PROJECT_ROOT}/src/data/generate_class_images.py" \
  --model "${MODEL_DIR}" \
  --output-dir "${CLASS_DIR}" \
  --prompt "${CLASS_PROMPT}" \
  --num-images "${EXPECTED}" \
  --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"

COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
NON_IMAGE_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f ! \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
[[ "${COUNT}" -eq "${EXPECTED}" && "${NON_IMAGE_COUNT}" -eq 0 ]] || {
  echo "ERROR: final class-image validation failed" >&2
  exit 6
}
touch "${SUCCESS_MARKER}"
rm -f "${FAILED_MARKER}"
echo "CLASS_PREPARATION_SUCCESS subject=${SUBJECT} count=${COUNT}"
date
