#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
OUTPUT_DIR="${PROJECT_ROOT}/models/personalized/textual_inversion/dreambooth/dog2/s3000_seed42"
LOG_DIR="${PROJECT_ROOT}/logs"
SEED=42
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/textual_inversion/textual_inversion.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set CUDA_VISIBLE_DEVICES to one MIG UUID" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 3; }
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
echo "start=$(date --iso-8601=seconds)" | tee "${OUTPUT_DIR}/runtime.txt"
hostname | tee "${OUTPUT_DIR}/hostname.txt"
nvidia-smi | tee "${OUTPUT_DIR}/nvidia_smi_start.txt"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee "${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null > "${OUTPUT_DIR}/git_commit.txt" || echo UNCOMMITTED > "${OUTPUT_DIR}/git_commit.txt"
cp "${PROJECT_ROOT}/configs/textual_inversion.yaml" "${OUTPUT_DIR}/config.yaml"

COMMAND=(conda run -n "${ENV_NAME}" accelerate launch --num_processes 1 "${SCRIPT}"
  --pretrained_model_name_or_path="${MODEL_DIR}" --train_data_dir="${DATA_DIR}"
  --learnable_property=object --placeholder_token='<sks-dog>' --initializer_token=dog
  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4
  --max_train_steps=3000 --learning_rate=5e-4 --lr_scheduler=constant --lr_warmup_steps=0
  --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=500
  --validation_prompt='a photo of <sks-dog> on a beach' --num_validation_images=4 --validation_steps=500
  --output_dir="${OUTPUT_DIR}" --report_to=none)
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/train_ti_pilot.log" "${COMMAND[@]}"

