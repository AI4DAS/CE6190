#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/debug/dreambooth_20steps"
LOG_DIR="${OUTPUT_DIR}"
SEED=42
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
ENV_DIR="$(cd "$(dirname "${ENV_PYTHON}")/.." && pwd)"
ACCELERATE="${ENV_DIR}/bin/accelerate"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set one MIG UUID" >&2; exit 2; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${CLASS_DIR}" "${SCRIPT}" "${CONFIG}"; do [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 3; }; done
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${OUTPUT_DIR}/SUCCESS" ]] || { echo "ERROR: debug run already passed" >&2; exit 4; }
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"
COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${SCRIPT}" --pretrained_model_name_or_path="${MODEL_DIR}" --instance_data_dir="${DATA_DIR}" --instance_prompt='a photo of sks dog' --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog' --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200 --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20 --learning_rate=5e-6 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=200 --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4 --validation_steps=200 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/training.log" "${COMMAND[@]}"
touch "${OUTPUT_DIR}/SUCCESS"
echo REPRO_DREAMBOOTH_20STEPS_COMPLETED_SUCCESSFULLY
