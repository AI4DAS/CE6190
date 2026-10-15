#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
DATA_DIR="${PROJECT_ROOT}/data/processed/dreambooth/dog2/images"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/training/dreambooth/${RUN_ID}"
LOG_DIR="${OUTPUT_DIR}"
SEED=42
CLASS_DIR="${PROJECT_ROOT}/data/processed/class_images/dog_sd15_200"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
ENV_DIR="$(cd "$(dirname "${ENV_PYTHON}")/.." && pwd)"
PYTHON="${ENV_DIR}/bin/python"
ACCELERATE="${ENV_DIR}/bin/accelerate"
CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"
mkdir -p "${OUTPUT_DIR}"
RUN_ACTIVE=0
on_exit() {
  local code=$?
  if (( code != 0 && RUN_ACTIVE == 1 )); then
    rm -f "${OUTPUT_DIR}/SUCCESS"
    touch "${OUTPUT_DIR}/FAILED"
    {
      echo "exit_code=${code}"
      echo "timestamp=$(date --iso-8601=seconds)"
      if [[ -f "${OUTPUT_DIR}/training.log" ]]; then
        echo "first_actual_exception:"
        grep -m1 -E 'UnidentifiedImageError:|FileNotFoundError:|PermissionError:|RuntimeError:|ValueError:|TypeError:|CUDA out of memory|SIGKILL|Killed' "${OUTPUT_DIR}/training.log" || true
        echo "last_relevant_log_lines:"
        tail -n 20 "${OUTPUT_DIR}/training.log"
      fi
    } > "${OUTPUT_DIR}/failure_summary.txt"
    echo "TRAINING_FAILED exit_code=${code}; see ${OUTPUT_DIR}/failure_summary.txt" >&2
  fi
}
trap on_exit EXIT
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: CUDA_VISIBLE_DEVICES must contain one MIG UUID" >&2; exit 2; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${SCRIPT}" "${CONFIG}" "${PYTHON}" "${ACCELERATE}"; do [[ -e "${path}" ]] || { echo "ERROR: required path missing: ${path}" >&2; exit 3; }; done
IMAGE_COUNT=$(find "${DATA_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
mkdir -p "${CLASS_DIR}"
CLASS_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
NON_IMAGE_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f ! \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
[[ "${IMAGE_COUNT}" -eq 6 ]] || { echo "ERROR: expected 6 subject images, found ${IMAGE_COUNT}" >&2; exit 4; }
[[ "${CLASS_COUNT}" -eq 200 ]] || { echo "ERROR: expected exactly 200 JPG class images in ${CLASS_DIR}; found ${CLASS_COUNT}" >&2; exit 5; }
[[ "${NON_IMAGE_COUNT}" -eq 0 ]] || { echo "ERROR: class_data_dir must contain images only; found ${NON_IMAGE_COUNT} other files" >&2; exit 5; }
[[ ! -f "${OUTPUT_DIR}/SUCCESS" ]] || { echo "ERROR: run already successful: ${OUTPUT_DIR}" >&2; exit 6; }
[[ -w "${OUTPUT_DIR}" ]] || { echo "ERROR: output not writable: ${OUTPUT_DIR}" >&2; exit 7; }
if [[ -f "${OUTPUT_DIR}/FAILED" ]]; then
  ARCHIVE_STAMP=$(date +%Y%m%d_%H%M%S)
  [[ ! -f "${OUTPUT_DIR}/training.log" ]] || mv "${OUTPUT_DIR}/training.log" "${OUTPUT_DIR}/training.failed_${ARCHIVE_STAMP}.log"
  [[ ! -f "${OUTPUT_DIR}/failure_summary.txt" ]] || mv "${OUTPUT_DIR}/failure_summary.txt" "${OUTPUT_DIR}/failure_summary_${ARCHIVE_STAMP}.txt"
fi
rm -f "${OUTPUT_DIR}/FAILED"
RUN_ACTIVE=1
date; hostname; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"; echo "${CUDA_VISIBLE_DEVICES}" > "${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"; echo "${PYTHON}" > "${OUTPUT_DIR}/python_executable.txt"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/package_versions.txt"
import torch, diffusers, transformers, accelerate, peft
for x in (torch, diffusers, transformers, accelerate, peft): print(x.__name__, x.__version__)
print("torch_cuda", torch.version.cuda)
PY
cp "${PROJECT_ROOT}/configs/dreambooth.yaml" "${OUTPUT_DIR}/config.yaml"
RESUME_ARGS=(); if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null; then RESUME_ARGS=(--resume_from_checkpoint=latest); fi
COMMAND=("${ACCELERATE}" launch --config_file "${CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "${SCRIPT}" --pretrained_model_name_or_path="${MODEL_DIR}" --instance_data_dir="${DATA_DIR}" --instance_prompt='a photo of sks dog' --class_data_dir="${CLASS_DIR}" --class_prompt='a photo of a dog' --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200 --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=800 --learning_rate=5e-6 --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}" --checkpointing_steps=200 --validation_prompt='a photo of sks dog on a beach' --num_validation_images=4 --validation_steps=200 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard "${RESUME_ARGS[@]}")
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"; echo >> "${OUTPUT_DIR}/training_command.txt"
run_with_vram_monitor "${OUTPUT_DIR}" "${LOG_DIR}/training.log" "${COMMAND[@]}"
touch "${OUTPUT_DIR}/SUCCESS"
rm -f "${OUTPUT_DIR}/FAILED"
RUN_ACTIVE=0
echo "TRAINING_COMPLETED_SUCCESSFULLY"
date
