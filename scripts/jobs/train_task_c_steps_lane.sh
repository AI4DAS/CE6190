#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD="${1:?usage: train_task_c_steps_lane.sh METHOD SUBJECT}"
SUBJECT="${2:?usage: train_task_c_steps_lane.sh METHOD SUBJECT}"
[[ "${METHOD}" =~ ^(dreambooth|lora|textual_inversion)$ ]] || { echo "Invalid method" >&2; exit 2; }
[[ "${SUBJECT}" =~ ^(dog2|teapot|bear_plushie)$ ]] || { echo "Invalid Task C subject" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set exactly one MIG UUID" >&2; exit 3; }

PYTHON="/home/zwu/anaconda3/envs/personalization-benchmark/bin/python"
ACCELERATE="/home/zwu/anaconda3/envs/personalization-benchmark/bin/accelerate"
ACCEL_CONFIG="${PROJECT_ROOT}/configs/accelerate_single_mig.yaml"
MODEL_DIR="${PROJECT_ROOT}/models/base/stable-diffusion-v1-5"
SUBJECT_CONFIG="${PROJECT_ROOT}/configs/main_subjects.yaml"
READER="${PROJECT_ROOT}/src/utils/read_main_subject.py"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/ablations/task_c_steps_training/${METHOD}/${SUBJECT}"
GEN_OUTPUT="${PROJECT_ROOT}/outputs/experiments/task_c_steps"
LOG_ROOT="${PROJECT_ROOT}/logs/ablations/task_c_steps/${METHOD}/${SUBJECT}"
source "${PROJECT_ROOT}/scripts/jobs/_run_with_vram_monitor.sh"

mapfile -t FIELDS < <("${PYTHON}" "${READER}" --config "${SUBJECT_CONFIG}" --subject "${SUBJECT}")
DATA_DIR="${PROJECT_ROOT}/${FIELDS[3]}"
INSTANCE_PROMPT="${FIELDS[5]}"
CLASS_PROMPT="${FIELDS[6]}"
CLASS_DIR="${PROJECT_ROOT}/${FIELDS[7]}"
PLACEHOLDER_TOKEN="${FIELDS[8]}"
INITIALIZER_TOKEN="${FIELDS[9]}"

case "${METHOD}" in
  dreambooth)
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth.py"
    interval=100; max_step=800
    ;;
  lora)
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
    interval=100; max_step=1600
    ;;
  textual_inversion)
    SCRIPT="${PROJECT_ROOT}/third_party/diffusers/examples/textual_inversion/textual_inversion.py"
    interval=250; max_step=3000
    ;;
esac
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
unlink "${OUTPUT_DIR}/FAILED" 2>/dev/null || true
unlink "${OUTPUT_DIR}/SUCCESS" 2>/dev/null || true
echo "start=$(date --iso-8601=seconds)" > "${OUTPUT_DIR}/task_runtime.txt"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >> "${OUTPUT_DIR}/task_runtime.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"

for ((target=interval; target<=max_step; target+=interval)); do
  step_output="${GEN_OUTPUT}/${METHOD}/${SUBJECT}/step_${target}"
  if [[ -d "${step_output}" ]]; then
    image_count=$(find "${step_output}" -type f -name 'seed_*.png' | wc -l)
    json_count=$(find "${step_output}" -type f -name 'seed_*.json' | wc -l)
  else
    image_count=0
    json_count=0
  fi
  if [[ "${image_count}" -eq 12 && "${json_count}" -eq 12 ]]; then
    echo "SKIP completed method=${METHOD} subject=${SUBJECT} step=${target}"
    continue
  fi
  [[ "${image_count}" -eq 0 && "${json_count}" -eq 0 ]] || {
    echo "Partial generation at step ${target}; refusing substitution" >&2; exit 4;
  }

  checkpoint="${OUTPUT_DIR}/checkpoint-${target}"
  if [[ ! -d "${checkpoint}" ]]; then
    segment="${LOG_ROOT}/step_${target}"
    mkdir -p "${segment}"
    COMMON=("${ACCELERATE}" launch --config_file "${ACCEL_CONFIG}" --num_processes 1 --num_machines 1
      --mixed_precision fp16 --dynamo_backend no "${SCRIPT}"
      --pretrained_model_name_or_path="${MODEL_DIR}" --resolution=512 --train_batch_size=1
      --lr_scheduler=constant --lr_warmup_steps=0 --mixed_precision=fp16 --seed=42
      --max_train_steps="${target}" --checkpointing_steps="${interval}" --checkpoints_total_limit=2
      --output_dir="${OUTPUT_DIR}" --report_to=tensorboard)
    case "${METHOD}" in
      dreambooth)
        COMMAND=("${COMMON[@]}" --instance_data_dir="${DATA_DIR}" --instance_prompt="${INSTANCE_PROMPT}"
          --class_data_dir="${CLASS_DIR}" --class_prompt="${CLASS_PROMPT}"
          --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
          --gradient_accumulation_steps=1 --gradient_checkpointing --learning_rate=5e-6
          --validation_prompt="${INSTANCE_PROMPT} on a beach" --num_validation_images=4 --validation_steps=200)
        ;;
      lora)
        COMMAND=("${COMMON[@]}" --instance_data_dir="${DATA_DIR}" --instance_prompt="${INSTANCE_PROMPT}"
          --class_data_dir="${CLASS_DIR}" --class_prompt="${CLASS_PROMPT}"
          --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=200
          --gradient_accumulation_steps=1 --gradient_checkpointing --rank=16 --learning_rate=1e-4
          --validation_prompt="${INSTANCE_PROMPT} on a beach" --num_validation_images=4 --validation_epochs=10)
        ;;
      textual_inversion)
        COMMAND=("${COMMON[@]}" --train_data_dir="${DATA_DIR}" --learnable_property=object
          --placeholder_token="${PLACEHOLDER_TOKEN}" --initializer_token="${INITIALIZER_TOKEN}"
          --gradient_accumulation_steps=4 --learning_rate=5e-4 --save_steps="${interval}"
          --validation_prompt="a photo of ${PLACEHOLDER_TOKEN} on a beach"
          --num_validation_images=4 --validation_steps=500)
        ;;
    esac
    if find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
      COMMAND+=(--resume_from_checkpoint=latest)
    fi
    printf '%q ' "${COMMAND[@]}" > "${segment}/training_command.txt"; echo >> "${segment}/training_command.txt"
    echo "TRAIN method=${METHOD} subject=${SUBJECT} target=${target}"
    run_with_vram_monitor "${segment}" "${segment}/training.log" "${COMMAND[@]}"
  fi
  [[ -d "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 5; }

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" \
    "${PROJECT_ROOT}/src/inference/generate_steps_checkpoint.py" \
    --project-root "${PROJECT_ROOT}" --method "${METHOD}" --subject "${SUBJECT}" \
    --step "${target}" --training-dir "${OUTPUT_DIR}" --output-root "${GEN_OUTPUT}" --resume \
    > "${LOG_ROOT}/generation_step_${target}.log" 2>&1
  echo "GENERATED method=${METHOD} subject=${SUBJECT} step=${target}"
done

expected_steps=$((max_step / interval))
expected_images=$((expected_steps * 12))
actual_images=$(find "${GEN_OUTPUT}/${METHOD}/${SUBJECT}" -type f -name 'seed_*.png' | wc -l)
actual_json=$(find "${GEN_OUTPUT}/${METHOD}/${SUBJECT}" -type f -name 'seed_*.json' | wc -l)
[[ "${actual_images}" -eq "${expected_images}" && "${actual_json}" -eq "${expected_images}" ]] || {
  echo "Count mismatch expected=${expected_images} png=${actual_images} json=${actual_json}" >&2; exit 6;
}

# Generation is complete, so training-state checkpoints are no longer needed.
while IFS= read -r -d '' checkpoint_dir; do
  case "${checkpoint_dir}" in
    "${OUTPUT_DIR}"/checkpoint-*) find "${checkpoint_dir}" -depth -delete ;;
    *) echo "Unsafe checkpoint path: ${checkpoint_dir}" >&2; exit 7 ;;
  esac
done < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print0)

touch "${OUTPUT_DIR}/SUCCESS"
echo "finish=$(date --iso-8601=seconds)" >> "${OUTPUT_DIR}/task_runtime.txt"
echo "TASK_C_LANE_COMPLETE method=${METHOD} subject=${SUBJECT} images=${actual_images}"
