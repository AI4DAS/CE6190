#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/training/lora/${RUN_ID}"
LOG_DIR="${OUTPUT_DIR}"
SEED=42
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
ENV_DIR="$(cd "$(dirname "${ENV_PYTHON}")/.." && pwd)"
PYTHON="${ENV_DIR}/bin/python"
ACCELERATE="${ENV_DIR}/bin/accelerate"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: CUDA_VISIBLE_DEVICES must contain one MIG UUID" >&2; exit 2; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${SCRIPT}" "${CONFIG}" "${PYTHON}" "${ACCELERATE}"; do [[ -e "${path}" ]] || { echo "ERROR: required path missing: ${path}" >&2; exit 3; }; done
IMAGE_COUNT=$(find "${DATA_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
mkdir -p "${CLASS_DIR}"
CLASS_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
[[ "${IMAGE_COUNT}" -eq 6 ]] || { echo "ERROR: expected 6 subject images, found ${IMAGE_COUNT}" >&2; exit 4; }
[[ "${CLASS_COUNT}" -ge 200 ]] || { echo "ERROR: expected 200 class images in ${CLASS_DIR}; run scripts/jobs/prepare_class_images.sh first" >&2; exit 5; }
[[ ! -f "${OUTPUT_DIR}/SUCCESS" ]] || { echo "ERROR: run already successful: ${OUTPUT_DIR}" >&2; exit 6; }
mkdir -p "${OUTPUT_DIR}"; [[ -w "${OUTPUT_DIR}" ]] || { echo "ERROR: output not writable: ${OUTPUT_DIR}" >&2; exit 7; }
date; hostname; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"; echo "${CUDA_VISIBLE_DEVICES}" > "${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"; echo "${PYTHON}" > "${OUTPUT_DIR}/python_executable.txt"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/package_versions.txt"
import torch, diffusers, transformers, accelerate, peft
for x in (torch, diffusers, transformers, accelerate, peft): print(x.__name__, x.__version__)
print("torch_cuda", torch.version.cuda)
PY
cp "${PROJECT_ROOT}/configs/lora.yaml" "${OUTPUT_DIR}/config.yaml"
RESUME_ARGS=(); if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null; then RESUME_ARGS=(--resume_from_checkpoint=latest); fi
COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${SCRIPT}" --pretrained_model_name_or_path="${MODEL_DIR}" --instance_data_dir="${DATA_DIR}" --instance_prompt='a photo of sks dog' --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog' --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200 --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --rank=16 --max_train_steps=800 --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=200 --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4 --validation_epochs=10 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard "${RESUME_ARGS[@]}")
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/training.log" "${COMMAND[@]}"
touch "${OUTPUT_DIR}/SUCCESS"
echo "TRAINING_COMPLETED_SUCCESSFULLY"
date
