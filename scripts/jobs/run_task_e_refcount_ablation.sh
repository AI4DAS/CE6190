#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "${ROOT}/scripts/mig_devices.sh"
PYTHON="${PYTHON:-$(command -v python)}"
MIGS=("${MIG3}" "${MIG4}" "${MIG5}" "${MIG6}" "${MIG7}"); SUBJECTS=(dog2 teapot); COUNTS=(1 3 5)
LOG="${ROOT}/logs/ablations/task_e_refcount"; STATUS="${ROOT}/outputs/ablations/task_e_refcount_training"; mkdir -p "$LOG" "$STATUS"
unlink "$STATUS/FAILED" 2>/dev/null||true; unlink "$STATUS/SUCCESS" 2>/dev/null||true
# teapot has exactly five references, so k=5 is byte-for-byte the already validated
# main rank-16/prior-on training configuration. Reuse it instead of retraining.
TASKS=("dog2 1" "dog2 3" "dog2 5" "teapot 1" "teapot 3")
echo "TASK_E_START=$(date --iso-8601=seconds) TRAIN_JOBS=5 REUSED_JOBS=1 ETA_MINUTES=15-30"; start=$(date +%s)
train_worker(){ local lane=$1 dev=$2 i s k; for ((i=lane;i<${#TASKS[@]};i+=5)); do read -r s k<<<"${TASKS[$i]}"; CUDA_VISIBLE_DEVICES="$dev" bash "$ROOT/scripts/jobs/train_lora_variant.sh" "$s" 16 "$k" on "outputs/ablations/task_e_refcount_training/${s}_k${k}" >"$LOG/train_${s}_k${k}.log" 2>&1; done; }
pids=(); for l in "${!MIGS[@]}"; do train_worker "$l" "${MIGS[$l]}"& pids+=("$!"); done; code=0; for p in "${pids[@]}"; do if ! wait "$p";then code=1;fi;done; if ((code));then touch "$STATUS/FAILED";exit 1;fi
GEN=(); for s in "${SUBJECTS[@]}"; do for k in "${COUNTS[@]}"; do GEN+=("lora $s $k" "ip $s $k"); done; done
gen_worker(){ local lane=$1 dev=$2 i kind s k weights; for ((i=lane;i<${#GEN[@]};i+=5)); do read -r kind s k<<<"${GEN[$i]}"; if [[ "$kind" == lora ]];then
  if [[ "$s" == teapot && "$k" == 5 ]]; then weights="outputs/training/lora/teapot_lora_r16_s800_seed42/pytorch_lora_weights.safetensors"; else weights="outputs/ablations/task_e_refcount_training/${s}_k${k}/pytorch_lora_weights.safetensors"; fi
  CUDA_VISIBLE_DEVICES="$dev" "${PYTHON}" "$ROOT/src/inference/generate_lora_variant.py" --project-root "$ROOT" --subject "$s" --weights "$weights" --output-root "outputs/experiments/task_e_refcount/lora_k${k}" --experiment task_e_refcount --variant "lora_k${k}" --rank 16 --n-refs "$k" --prior on --resume >"$LOG/generate_lora_${s}_k${k}.log" 2>&1
else CUDA_VISIBLE_DEVICES="$dev" "${PYTHON}" "$ROOT/src/inference/generate_ip_adapter_ablation.py" --project-root "$ROOT" --subject "$s" --adapter plus --scale 0.3 --reference-mode average_all --reference-count "$k" --seeds 42 123 --output-root "outputs/experiments/task_e_refcount/ip_plus_avg_k${k}" --experiment task_e_refcount --variant "ip_plus_avg_k${k}" --resume >"$LOG/generate_ip_${s}_k${k}.log" 2>&1; fi; done; }
pids=(); for l in "${!MIGS[@]}"; do gen_worker "$l" "${MIGS[$l]}"&pids+=("$!");done;code=0;for p in "${pids[@]}";do if ! wait "$p";then code=1;fi;done;if ((code));then touch "$STATUS/FAILED";exit 1;fi
touch "$STATUS/SUCCESS"; finish=$(date +%s); echo "runtime_seconds=$((finish-start))">"$LOG/runtime.txt"; echo "TASK_E_COMPLETE=$(date --iso-8601=seconds)"
