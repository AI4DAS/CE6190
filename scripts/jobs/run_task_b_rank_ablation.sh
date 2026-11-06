#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"
MIGS=("${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")
SUBJECTS=(dog2 teapot bear_plushie); RANKS=(4 8 16 32 64)
TRAIN_TASKS=("teapot 4" "bear_plushie 4" "dog2 8" "teapot 8" "bear_plushie 8" "teapot 32" "bear_plushie 32" "dog2 64" "teapot 64" "bear_plushie 64")
LOG="${PROJECT_ROOT}/logs/ablations/task_b_rank"; STATUS="${PROJECT_ROOT}/outputs/ablations/task_b_rank_training"
mkdir -p "${LOG}" "${STATUS}"; unlink "${STATUS}/FAILED" 2>/dev/null || true; unlink "${STATUS}/SUCCESS" 2>/dev/null || true
echo "TASK_B_START=$(date --iso-8601=seconds) TRAIN_JOBS=10 ETA_MINUTES=20-35"
start=$(date +%s)
train_worker(){ local lane="$1" dev="$2" i subject rank out; for ((i=lane;i<${#TRAIN_TASKS[@]};i+=5)); do read -r subject rank <<<"${TRAIN_TASKS[$i]}"; out="outputs/ablations/task_b_rank_training/${subject}_r${rank}"; echo "TRAIN ${subject} r${rank}"; CUDA_VISIBLE_DEVICES="${dev}" bash "${PROJECT_ROOT}/scripts/jobs/train_lora_variant.sh" "${subject}" "${rank}" 0 on "${out}" >"${LOG}/train_${subject}_r${rank}.log" 2>&1; done; }
pids=(); for lane in "${!MIGS[@]}"; do train_worker "$lane" "${MIGS[$lane]}" & pids+=("$!"); done
code=0; for p in "${pids[@]}"; do if ! wait "$p"; then code=1; fi; done
if ((code)); then touch "${STATUS}/FAILED"; exit 1; fi

weights_for(){
  local subject="$1" rank="$2"
  if [[ "$rank" == 16 ]]; then echo "outputs/training/lora/${subject}_lora_r16_s800_seed42/pytorch_lora_weights.safetensors"
  elif [[ "$subject" == dog2 && "$rank" == 4 ]]; then echo "outputs/ablations/lora_rank/dog2_lora_r4_s800_seed42/pytorch_lora_weights.safetensors"
  elif [[ "$subject" == dog2 && "$rank" == 32 ]]; then echo "outputs/ablations/lora_rank/dog2_lora_r32_s800_seed42/pytorch_lora_weights.safetensors"
  else echo "outputs/ablations/task_b_rank_training/${subject}_r${rank}/pytorch_lora_weights.safetensors"; fi
}
GEN_TASKS=(); for subject in "${SUBJECTS[@]}"; do for rank in "${RANKS[@]}"; do GEN_TASKS+=("${subject} ${rank}"); done; done
gen_worker(){ local lane="$1" dev="$2" i subject rank weights refs; for ((i=lane;i<${#GEN_TASKS[@]};i+=5)); do read -r subject rank <<<"${GEN_TASKS[$i]}"; weights=$(weights_for "$subject" "$rank"); refs=$("${PYTHON}" -c "import yaml;print(yaml.safe_load(open('${PROJECT_ROOT}/configs/main_subjects.yaml'))['subjects']['${subject}']['expected_images'])"); CUDA_VISIBLE_DEVICES="${dev}" "${PYTHON}" "${PROJECT_ROOT}/src/inference/generate_lora_variant.py" --project-root "${PROJECT_ROOT}" --subject "$subject" --weights "$weights" --output-root "outputs/experiments/task_b_rank/r${rank}" --experiment task_b_rank_ablation --variant "lora_r${rank}" --rank "$rank" --n-refs "$refs" --prior on --resume >"${LOG}/generate_${subject}_r${rank}.log" 2>&1; done; }
pids=(); for lane in "${!MIGS[@]}"; do gen_worker "$lane" "${MIGS[$lane]}" & pids+=("$!"); done
code=0; for p in "${pids[@]}"; do if ! wait "$p"; then code=1; fi; done
if ((code)); then touch "${STATUS}/FAILED"; exit 1; fi
touch "${STATUS}/SUCCESS"; finish=$(date +%s); echo "runtime_seconds=$((finish-start))" >"${LOG}/runtime.txt"; echo "TASK_B_COMPLETE=$(date --iso-8601=seconds)"
