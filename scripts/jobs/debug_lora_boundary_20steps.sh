#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/debug/lora_boundary_20steps"
ENV_DIR="$(cd "$(dirname "$(conda run -n "${ENV_NAME:-personalization-benchmark}" python -c 'import sys; print(sys.executable)')")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"
mkdir -p "${OUTPUT_DIR}"
COMMAND=("${ENV_DIR}/bin/accelerate" launch --config_file "${PROJECT_ROOT}/configs/accelerate_single_mig.yaml" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py" --pretrained_model_name_or_path="${MODEL_DIR}" --instance_data_dir="${DATA_DIR}" --instance_prompt='a photo of sks dog' --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog' --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200 --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --rank=16 --max_train_steps=20 --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed=42 --checkpointing_steps=20 --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4 --validation_epochs=1 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${OUTPUT_DIR}/training.log" "${COMMAND[@]}"
touch "${OUTPUT_DIR}/SUCCESS"
