#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
SEED=42
METHOD="${1:?usage: train_main_subject.sh METHOD SUBJECT [--validate-only]}"
SUBJECT="${2:?usage: train_main_subject.sh METHOD SUBJECT [--validate-only]}"
MODE="${3:-}"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
ENV_DIR="$(cd "$(dirname "${ENV_PYTHON}")/.." && pwd)"
PYTHON="${ENV_DIR}/bin/python"
ACCELERATE="${ENV_DIR}/bin/accelerate"
ACCEL_CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
SUBJECT_CONFIG="${PROJECT_ROOT}/configs/main_subjects.yaml"
READER="${PROJECT_ROOT}/src/utils/read_main_subject.py"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

case "${METHOD}" in
  dreambooth|lora|textual_inversion) ;;
  *) echo "ERROR: METHOD must be dreambooth, lora, or textual_inversion" >&2; exit 2 ;;
esac
[[ -z "${MODE}" || "${MODE}" == "--validate-only" ]] || { echo "ERROR: unknown mode ${MODE}" >&2; exit 2; }
mapfile -t FIELDS < <("${PYTHON}" "${READER}" --config "${SUBJECT_CONFIG}" --subject "${SUBJECT}")
[[ "${#FIELDS[@]}" -eq 11 ]] || { echo "ERROR: invalid subject record" >&2; exit 2; }
DATASET="${FIELDS[0]}"
CATEGORY="${FIELDS[1]}"
CLASS_NAME="${FIELDS[2]}"
DATA_DIR="${PROJECT_ROOT}/${FIELDS[3]}"
EXPECTED_IMAGES="${FIELDS[4]}"
INSTANCE_PROMPT="${FIELDS[5]}"
CLASS_PROMPT="${FIELDS[6]}"
CLASS_DIR="${PROJECT_ROOT}/${FIELDS[7]}"
PLACEHOLDER_TOKEN="${FIELDS[8]}"
INITIALIZER_TOKEN="${FIELDS[9]}"
COMPLETED_PILOT="${FIELDS[10]}"

if [[ "${COMPLETED_PILOT}" == "true" ]]; then
  echo "ERROR: ${SUBJECT} is a completed pilot subject and must not be retrained" >&2
  exit 3
fi

case "${METHOD}" in
  dreambooth)
    RUN_ID="${SUBJECT}_db_s800_seed42"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/training/dreambooth/${RUN_ID}"
    METHOD_CONFIG="${PROJECT_ROOT}/configs/dreambooth.yaml"
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth.py"
    ;;
  lora)
    RUN_ID="${SUBJECT}_lora_r16_s800_seed42"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/training/lora/${RUN_ID}"
    METHOD_CONFIG="${PROJECT_ROOT}/configs/lora.yaml"
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
    ;;
  textual_inversion)
    RUN_ID="${SUBJECT}_ti_s3000_seed42"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/training/textual_inversion/${RUN_ID}"
    METHOD_CONFIG="${PROJECT_ROOT}/configs/textual_inversion.yaml"
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/textual_inversion/textual_inversion.py"
    ;;
esac

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "ERROR: set one MIG UUID" >&2; exit 4; }
for path in "${MODEL_DIR}/model_index.json" "${DATA_DIR}" "${SCRIPT}" "${ACCEL_CONFIG}" "${PYTHON}" "${ACCELERATE}"; do
  [[ -e "${path}" ]] || { echo "ERROR: required path missing: ${path}" >&2; exit 5; }
done
IMAGE_COUNT=$(find "${DATA_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)
[[ "${IMAGE_COUNT}" -eq "${EXPECTED_IMAGES}" ]] || { echo "ERROR: ${SUBJECT} expected ${EXPECTED_IMAGES} references, found ${IMAGE_COUNT}" >&2; exit 6; }

if [[ "${METHOD}" != "textual_inversion" ]]; then
  [[ -d "${CLASS_DIR}" ]] || { echo "ERROR: missing class directory: ${CLASS_DIR}" >&2; exit 7; }
  CLASS_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
  NON_IMAGE_COUNT=$(find "${CLASS_DIR}" -maxdepth 1 -type f ! \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  [[ "${CLASS_COUNT}" -eq 200 ]] || { echo "ERROR: expected 200 class images, found ${CLASS_COUNT}" >&2; exit 7; }
  [[ "${NON_IMAGE_COUNT}" -eq 0 ]] || { echo "ERROR: class directory must be image-only" >&2; exit 7; }
fi

if [[ "${METHOD}" == "lora" ]]; then
  grep -q 'pipeline = pipeline.to(accelerator.device)$' "${SCRIPT}" || {
    echo "ERROR: validated LoRA FP16 dtype-safety fix is missing" >&2
    exit 8
  }
fi

if [[ "${MODE}" == "--validate-only" ]]; then
  echo "VALIDATION_OK method=${METHOD} subject=${SUBJECT} dataset=${DATASET} category=${CATEGORY}"
  echo "data_dir=${DATA_DIR}"
  echo "class_dir=${CLASS_DIR}"
  echo "output_dir=${OUTPUT_DIR}"
  exit 0
fi

[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to reuse existing output: ${OUTPUT_DIR}" >&2; exit 9; }
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
      grep -m1 -E 'Traceback|RuntimeError:|ValueError:|TypeError:|CUDA out of memory|SIGKILL|Killed' "${OUTPUT_DIR}/training.log" || true
      tail -n 30 "${OUTPUT_DIR}/training.log" || true
    } > "${OUTPUT_DIR}/failure_summary.txt"
  fi
}
trap on_exit EXIT

date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/runtime.txt"
echo "${CUDA_VISIBLE_DEVICES}" > "${OUTPUT_DIR}/mig_uuid.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
echo "${PYTHON}" > "${OUTPUT_DIR}/python_executable.txt"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/package_versions.txt"
import torch, diffusers, transformers, accelerate, peft
for package in (torch, diffusers, transformers, accelerate, peft):
    print(package.__name__, package.__version__)
print("torch_cuda", torch.version.cuda)
PY
cp "${PROJECT_ROOT}/configs/base.yaml" "${OUTPUT_DIR}/base_config.yaml"
cp "${METHOD_CONFIG}" "${OUTPUT_DIR}/method_config.yaml"
cp "${SUBJECT_CONFIG}" "${OUTPUT_DIR}/subjects_config.yaml"

COMMON=("${ACCELERATE}" launch --config_file "${ACCEL_CONFIG}" --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no)
case "${METHOD}" in
  dreambooth)
    COMMAND=("${COMMON[@]}" "${SCRIPT}"
      --pretrained_model_name_or_path="${MODEL_DIR}"
      --instance_data_dir="${DATA_DIR}" --instance_prompt="${INSTANCE_PROMPT}"
      --class_data_dir="${CLASS_DIR}" --class_prompt="${CLASS_PROMPT}"
      --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
      --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1
      --gradient_checkpointing --max_train_steps=800 --learning_rate=5e-6
      --checkpointing_steps=801
      --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}"
      --validation_prompt="${INSTANCE_PROMPT} on a beach" --num_validation_images=4
      --validation_steps=200 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
    ;;
  lora)
    COMMAND=("${COMMON[@]}" "${SCRIPT}"
      --pretrained_model_name_or_path="${MODEL_DIR}"
      --instance_data_dir="${DATA_DIR}" --instance_prompt="${INSTANCE_PROMPT}"
      --class_data_dir="${CLASS_DIR}" --class_prompt="${CLASS_PROMPT}"
      --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
      --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1
      --gradient_checkpointing --rank=16 --max_train_steps=800 --learning_rate=1e-4
      --checkpointing_steps=801
      --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}"
      --validation_prompt="${INSTANCE_PROMPT} on a beach" --num_validation_images=4
      --validation_epochs=10 --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
    ;;
  textual_inversion)
    COMMAND=("${COMMON[@]}" "${SCRIPT}"
      --pretrained_model_name_or_path="${MODEL_DIR}" --train_data_dir="${DATA_DIR}"
      --learnable_property=object --placeholder_token="${PLACEHOLDER_TOKEN}"
      --initializer_token="${INITIALIZER_TOKEN}" --resolution=512 --train_batch_size=1
      --gradient_accumulation_steps=4 --max_train_steps=3000 --learning_rate=5e-4
      --checkpointing_steps=3001
      --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed="${SEED}"
      --validation_prompt="a photo of ${PLACEHOLDER_TOKEN} on a beach"
      --num_validation_images=4 --validation_steps=500
      --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
    ;;
esac

printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/training_command.txt"
echo >> "${OUTPUT_DIR}/training_command.txt"
RUN_ACTIVE=1
run_with_vram_monitor "${OUTPUT_DIR}" "${OUTPUT_DIR}/training.log" "${COMMAND[@]}"

case "${METHOD}" in
  dreambooth) [[ -f "${OUTPUT_DIR}/model_index.json" && -d "${OUTPUT_DIR}/unet" ]] ;;
  lora) [[ -s "${OUTPUT_DIR}/pytorch_lora_weights.safetensors" ]] ;;
  textual_inversion) [[ -s "${OUTPUT_DIR}/learned_embeds.safetensors" ]] ;;
esac
if find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
  echo "ERROR: storage policy violation: intermediate checkpoint found" >&2
  exit 10
fi
touch "${OUTPUT_DIR}/SUCCESS"
rm -f "${OUTPUT_DIR}/FAILED"
RUN_ACTIVE=0
echo "TRAINING_COMPLETED_SUCCESSFULLY method=${METHOD} subject=${SUBJECT}"
date
