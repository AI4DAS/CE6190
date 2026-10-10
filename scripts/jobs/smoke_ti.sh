#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoke/textual_inversion/run_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}"
SEED=42
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
ENV_DIR="$(cd "$(dirname "${ENV_PYTHON}")/.." && pwd)"
ACCELERATE="${ENV_DIR}/bin/accelerate"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/textual_inversion/textual_inversion.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one MIG UUID" >&2; exit 2; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${SCRIPT}" "${CONFIG}" "${ACCELERATE}"; do
  [[ -e "${path}" ]] || { echo "ERROR: required path missing: ${path}" >&2; exit 3; }
done
mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"

COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${SCRIPT}"
  --pretrained_model_name_or_path="${MODEL_DIR}" --train_data_dir="${DATA_DIR}"
  --learnable_property=object --placeholder_token='<sks-dog>' --initializer_token=dog
  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1
  --max_train_steps=5 --learning_rate=5e-4 --lr_scheduler=constant --lr_warmup_steps=0
  --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=5
  --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/training.log" "${COMMAND[@]}"
touch "${OUTPUT_DIR}/SUCCESS"
echo "SMOKE_TI_COMPLETED_SUCCESSFULLY"

