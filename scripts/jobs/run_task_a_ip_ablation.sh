#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"
GENERATOR="${PROJECT_ROOT}/src/inference/generate_ip_adapter_ablation.py"
LOG_ROOT="${PROJECT_ROOT}/logs/ablations/task_a_ip_adapter"
STATUS_ROOT="${PROJECT_ROOT}/outputs/experiments/task_a"
source "${PROJECT_ROOT}/scripts/mig_devices.sh"

MIGS=("${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}")
SUBSET=(dog2 teapot bear_plushie)
ALL_SUBJECTS=(dog2 cat pet_dog1 bear_plushie robot_toy teapot colorful_sneaker luggage_backpack1)
mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"
unlink "${STATUS_ROOT}/FAILED" 2>/dev/null || true
unlink "${STATUS_ROOT}/SUCCESS" 2>/dev/null || true

tasks=()
for subject in "${ALL_SUBJECTS[@]}"; do
  tasks+=("${subject}|plus|0.7|single|42,123,456,789|outputs/experiments/task_a/full_plus_s0.7|ip_adapter_plus_s0.7")
done
for adapter in plus plain; do
  if [[ "${adapter}" == "plus" ]]; then scales=(0.0 0.2 0.5 1.0); else scales=(0.0 0.2 0.3 0.5 0.7 1.0); fi
  for scale in "${scales[@]}"; do
    tag="$(printf '%s' "${scale}" | tr '.' 'p')"
    for subject in "${SUBSET[@]}"; do
      tasks+=("${subject}|${adapter}|${scale}|single|42,123|outputs/experiments/task_a/scale_sweep/${adapter}_s${tag}|ip_adapter_${adapter}_s${scale}_single")
    done
  done
done
for subject in "${SUBSET[@]}"; do
  tasks+=("${subject}|plus|0.3|average_all|42,123|outputs/experiments/task_a/reference_usage/plus_avg_all_s0p3|ip_adapter_plus_s0.3_avg_all")
done

echo "TASK_A_START=$(date --iso-8601=seconds)"
echo "TASK_A_JOBS=${#tasks[@]} EXPECTED_NEW_IMAGES=588 ETA_MINUTES=8-15"
printf '%s\n' "${tasks[@]}" > "${LOG_ROOT}/task_manifest.txt"
start_epoch=$(date +%s)

worker() {
  local lane="$1" device="$2" index subject adapter scale ref_mode seed_csv output variant log
  for ((index=lane; index<${#tasks[@]}; index+=${#MIGS[@]})); do
    IFS='|' read -r subject adapter scale ref_mode seed_csv output variant <<< "${tasks[$index]}"
    log="${LOG_ROOT}/$(printf '%03d' "${index}")_${variant}_${subject}.log"
    IFS=',' read -r -a seeds <<< "${seed_csv}"
    echo "START lane=${lane} index=${index} subject=${subject} variant=${variant} device=${device}"
    CUDA_VISIBLE_DEVICES="${device}" "${PYTHON}" "${GENERATOR}" \
      --project-root "${PROJECT_ROOT}" --subject "${subject}" \
      --adapter "${adapter}" --scale "${scale}" --reference-mode "${ref_mode}" \
      --seeds "${seeds[@]}" --output-root "${output}" \
      --experiment task_a_ip_adapter --variant "${variant}" --resume \
      > "${log}" 2>&1
    echo "DONE lane=${lane} index=${index} subject=${subject} variant=${variant}"
  done
}

pids=()
for lane in "${!MIGS[@]}"; do
  worker "${lane}" "${MIGS[$lane]}" &
  pids+=("$!")
done

code=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then code=1; fi
done
finish_epoch=$(date +%s)
echo "runtime_seconds=$((finish_epoch-start_epoch))" > "${LOG_ROOT}/runtime.txt"
if (( code != 0 )); then
  touch "${STATUS_ROOT}/FAILED"
  echo "TASK_A_FAILED=$(date --iso-8601=seconds)" >&2
  exit 1
fi
touch "${STATUS_ROOT}/SUCCESS"
echo "TASK_A_COMPLETE=$(date --iso-8601=seconds)"
