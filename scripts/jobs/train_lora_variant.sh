#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBJECT="${1:?subject}"
RANK="${2:?rank}"
REF_COUNT="${3:?ref count, 0 means all}"
PRIOR="${4:?on or off}"
OUTPUT_REL="${5:?output directory relative to project}"
[[ "${PRIOR}" == "on" || "${PRIOR}" == "off" ]] || exit 2
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set one MIG" >&2; exit 2; }
PYTHON="${PYTHON:-$(command -v python)}"
ACCELERATE="${ACCELERATE:-$(command -v accelerate)}"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
MODEL="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_REL}"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"
mapfile -t F < <("${PYTHON}" "${PROJECT_ROOT}/src/utils/read_main_subject.py" --config "${PROJECT_ROOT}/configs/main_subjects.yaml" --subject "${SUBJECT}")
DATA_DIR="${PROJECT_ROOT}/${F[3]}"; INSTANCE_PROMPT="${F[5]}"; CLASS_PROMPT="${F[6]}"; CLASS_DIR="${PROJECT_ROOT}/${F[7]}"
if [[ "${REF_COUNT}" -gt 0 ]]; then DATA_DIR="${PROJECT_ROOT}/data/processed/ablations/refcount/${SUBJECT}/k${REF_COUNT}/images"; fi
expected="${REF_COUNT}"; [[ "${expected}" -eq 0 ]] && expected="${F[4]}"
actual=$(find "${DATA_DIR}" -maxdepth 1 -type f | wc -l); [[ "${actual}" -eq "${expected}" ]] || { echo "Reference count mismatch" >&2; exit 3; }
if [[ -f "${OUTPUT_DIR}/SUCCESS" && -s "${OUTPUT_DIR}/pytorch_lora_weights.safetensors" ]]; then echo "SKIP completed ${OUTPUT_DIR}"; exit 0; fi
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing incomplete existing output ${OUTPUT_DIR}" >&2; exit 4; }
mkdir -p "${OUTPUT_DIR}"
trap 'c=$?; if ((c)); then touch "${OUTPUT_DIR}/FAILED"; fi' EXIT
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"
echo "rank=${RANK} alpha=${RANK} refs=${expected} prior=${PRIOR}" > "${OUTPUT_DIR}/variant_config.txt"
COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${SCRIPT}"
  --pretrained_model_name_or_path="${MODEL}" --instance_data_dir="${DATA_DIR}" --instance_prompt="${INSTANCE_PROMPT}"
  --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing
  --rank="${RANK}" --max_train_steps=800 --learning_rate=1e-4 --checkpointing_steps=801
  --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed=42
  --validation_prompt="${INSTANCE_PROMPT} on a beach" --num_validation_images=4 --validation_epochs=10
  --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
if [[ "${PRIOR}" == "on" ]]; then
  COMMAND+=(--class_data_dir="${CLASS_DIR}" --class_prompt="${CLASS_PROMPT}" --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200)
fi
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${OUTPUT_DIR}/training.log" "${COMMAND[@]}"
[[ -s "${OUTPUT_DIR}/pytorch_lora_weights.safetensors" ]] || exit 5
touch "${OUTPUT_DIR}/SUCCESS"; unlink "${OUTPUT_DIR}/FAILED" 2>/dev/null || true
echo "finish=$(date --iso-8601=seconds)" >> "${OUTPUT_DIR}/runtime.txt"
echo "LORA_VARIANT_COMPLETE subject=${SUBJECT} rank=${RANK} refs=${expected} prior=${PRIOR}"
