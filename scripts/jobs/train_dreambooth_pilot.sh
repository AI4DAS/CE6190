#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
OUTPUT_DIR="${PROJECT_ROOT}/models/personalized/dreambooth/dreambooth/dog2/s800_seed42"
LOG_DIR="${PROJECT_ROOT}/logs"
SEED=42
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set CUDA_VISIBLE_DEVICES to one MIG UUID" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 3; }
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${CLASS_DIR}"
echo "start=$(date --iso-8601=seconds)" | tee "${OUTPUT_DIR}/runtime.txt"
hostname | tee "${OUTPUT_DIR}/hostname.txt"
nvidia-smi | tee "${OUTPUT_DIR}/nvidia_smi_start.txt"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee "${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null > "${OUTPUT_DIR}/git_commit.txt" || echo UNCOMMITTED > "${OUTPUT_DIR}/git_commit.txt"
cp "${PROJECT_ROOT}/configs/dreambooth.yaml" "${OUTPUT_DIR}/config.yaml"

COMMAND=(conda run -n "${ENV_NAME}" accelerate launch --num_processes 1 "${SCRIPT}"
  --pretrained_model_name_or_path="${MODEL_DIR}" --instance_data_dir="${DATA_DIR}"
  --instance_prompt='a photo of sks dog' --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog'
  --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing
  --max_train_steps=800 --learning_rate=5e-6 --lr_scheduler=constant --lr_warmup_steps=0
  --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=200
  --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4 --validation_steps=200
  --output_dir="${OUTPUT_DIR}" --report_to=none)
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/train_dreambooth_pilot.log" "${COMMAND[@]}"

