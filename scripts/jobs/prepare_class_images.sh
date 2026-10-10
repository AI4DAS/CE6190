#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
OUTPUT_DIR="${DATA_DIR}"
LOG_DIR="${PROJECT_ROOT}/logs"
SEED=10000
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set CUDA_VISIBLE_DEVICES to one MIG UUID" >&2; exit 2; }
[[ -f "${MODEL_DIR}/model_index.json" ]] || { echo "ERROR: incomplete model: ${MODEL_DIR}" >&2; exit 3; }
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
date; hostname; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi
"${PYTHON}" "${PROJECT_ROOT}/src/data/generate_class_images.py" --model "${MODEL_DIR}" --output-dir "${DATA_DIR}" --prompt "a photo of a dog" --num-images 200 --seed "${SEED}" 2>&1 | tee "${LOG_DIR}/prepare_class_images.log"

