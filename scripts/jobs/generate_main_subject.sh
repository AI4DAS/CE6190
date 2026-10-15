#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBJECT="${1:?usage: generate_main_subject.sh SUBJECT}"
PYTHON="/home/zwu/anaconda3/envs/personalization-benchmark/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/main_evaluation"
SEED="42,123,456,789"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/main_evaluation"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set one MIG UUID" >&2; exit 2; }
mkdir -p "${LOG_DIR}"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SEEDS=${SEED}"
if ! nvidia-smi; then
  echo "WARNING: nvidia-smi information query failed; Python CUDA isolation check remains authoritative" >&2
fi

for method in dreambooth lora textual_inversion ip_adapter_plus; do
  echo "GENERATION_START subject=${SUBJECT} method=${method} timestamp=$(date --iso-8601=seconds)"
  "${PYTHON}" "${PROJECT_ROOT}/src/inference/generate_main_benchmark.py" \
    --project-root "${PROJECT_ROOT}" --subject "${SUBJECT}" --method "${method}"
  echo "GENERATION_SUCCESS subject=${SUBJECT} method=${method} timestamp=$(date --iso-8601=seconds)"
done

echo "MAIN_SUBJECT_GENERATION_COMPLETED subject=${SUBJECT} timestamp=$(date --iso-8601=seconds)"
