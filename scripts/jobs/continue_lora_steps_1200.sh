#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/training/lora/dog2_lora_r16_s800_seed42/checkpoint-800"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/ablations/lora_steps/dog2_lora_r16_s1200_seed42"
LOG_DIR="${OUTPUT_DIR}"
SEED=42
PYTHON="${PYTHON:-$(command -v python)}"
ACCELERATE="${ACCELERATE:-$(command -v accelerate)}"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set exactly one MIG UUID" >&2; exit 3; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${CLASS_DIR}" "${CONFIG}" "${SCRIPT}" "${SOURCE_CHECKPOINT}/pytorch_lora_weights.safetensors"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 4; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: output already exists: ${OUTPUT_DIR}" >&2; exit 5; }
mkdir -p "${OUTPUT_DIR}"
cp -a "${SOURCE_CHECKPOINT}" "${OUTPUT_DIR}/checkpoint-800"
trap 'code=$?; if (( code != 0 )); then rm -f "${OUTPUT_DIR}/SUCCESS"; touch "${OUTPUT_DIR}/FAILED"; fi' EXIT

date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "start=$(date --iso-8601=seconds)" >"${OUTPUT_DIR}/runtime.txt"
echo "resumed_from=${SOURCE_CHECKPOINT}" >>"${OUTPUT_DIR}/runtime.txt"
echo "${CUDA_VISIBLE_DEVICES}" >"${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${OUTPUT_DIR}/git_commit.txt"
"${PYTHON}" - <<'PY' >"${OUTPUT_DIR}/package_versions.txt"
import torch, diffusers, transformers, accelerate, peft
for module in (torch, diffusers, transformers, accelerate, peft):
    print(module.__name__, module.__version__)
print("torch_cuda", torch.version.cuda)
PY

COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1
  --mixed_precision fp16 --dynamo_backend no "${SCRIPT}"
  --pretrained_model_name_or_path="${MODEL_DIR}"
  --instance_data_dir="${DATA_DIR}" --instance_prompt='a photo of sks dog'
  --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog'
  --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1
  --gradient_checkpointing --rank=16 --max_train_steps=1200
  --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0
  --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=400
  --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4
  --validation_epochs=10 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard
  --resume_from_checkpoint=latest)
printf '%q ' "${COMMAND[@]}" >"${OUTPUT_DIR}/training_command.txt"
echo >>"${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/training.log" "${COMMAND[@]}"
[[ -s "${OUTPUT_DIR}/pytorch_lora_weights.safetensors" ]] || { echo "ERROR: final LoRA weights missing" >&2; exit 6; }
[[ -s "${OUTPUT_DIR}/checkpoint-1200/pytorch_lora_weights.safetensors" ]] || { echo "ERROR: checkpoint-1200 missing" >&2; exit 7; }
touch "${OUTPUT_DIR}/SUCCESS"
rm -f "${OUTPUT_DIR}/FAILED"
echo "TRAINING_COMPLETED_SUCCESSFULLY steps=1200 resumed_from=800"
