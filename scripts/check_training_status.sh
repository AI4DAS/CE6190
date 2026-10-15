#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
show_status() {
  local label="$1" method="$2" pattern="$3" root latest log state exception
  root="${PROJECT_ROOT}/outputs/training/${method}"
  latest=$(find "${root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)
  if pgrep -af "${pattern}" | grep -v -E 'check_training_status|pgrep' >/dev/null; then state="RUNNING"
  elif [[ -n "${latest}" && -f "${latest}/SUCCESS" ]]; then state="SUCCESS"
  elif [[ -n "${latest}" && -f "${latest}/FAILED" ]]; then state="FAILED"
  elif [[ -n "${latest}" && -f "${latest}/training.log" ]] && grep -qE 'UnidentifiedImageError:|FileNotFoundError:|PermissionError:|RuntimeError:|ValueError:|TypeError:|CUDA out of memory|SIGKILL|Killed|CalledProcessError:' "${latest}/training.log"; then state="FAILED"
  else state="NOT STARTED"; fi
  echo "${label}: ${state}"
  if [[ -n "${latest}" ]]; then
    echo "  latest run: ${latest}"
    log="${latest}/training.log"
    if [[ "${state}" == "FAILED" && -f "${log}" ]]; then
      exception=$(grep -m1 -E 'UnidentifiedImageError:|FileNotFoundError:|PermissionError:|RuntimeError:|ValueError:|TypeError:|CUDA out of memory|SIGKILL|Killed' "${log}" || true)
      [[ -z "${exception}" ]] || echo "  first exception: ${exception}"
    fi
    if [[ -f "${log}" ]]; then
      tr '\r' '\n' < "${log}" | sed '/^[[:space:]]*$/d' | tail -n 3 | sed 's/^/  /'
    else
      echo "  no internal training.log"
    fi
  else echo "  no run directory"; fi
}
show_status "DreamBooth" "dreambooth" "train_dreambooth.py"
show_status "LoRA" "lora" "train_dreambooth_lora.py"
show_status "Textual Inversion" "textual_inversion" "textual_inversion.py"
