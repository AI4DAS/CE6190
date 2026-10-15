#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_status() {
  local method="$1"
  local label="$2"
  local output_dir="${PROJECT_ROOT}/outputs/modern_extension/${method}"
  local log_file="${PROJECT_ROOT}/logs/modern_extension/${method}_extension.log"
  local pattern="generate_modern_${method}.py"
  local status="NOT STARTED"

  if pgrep -af "${pattern}" >/dev/null 2>&1; then
    status="RUNNING"
  elif [[ -f "${output_dir}/SUCCESS" ]]; then
    status="SUCCESS"
  elif [[ -f "${output_dir}/FAILED" ]]; then
    status="FAILED"
  elif [[ -d "${output_dir}" ]]; then
    status="INCOMPLETE / NOT RUNNING"
  fi

  local png_count json_count
  if [[ -d "${output_dir}" ]]; then
    png_count=$(find "${output_dir}" -type f -name 'seed_*.png' | wc -l)
    json_count=$(find "${output_dir}" -type f -name 'seed_*.json' | wc -l)
  else
    png_count=0
    json_count=0
  fi
  echo "${label}: ${status} (${png_count}/72 PNG, ${json_count}/72 JSON)"
  if [[ -f "${log_file}" ]]; then
    tail -n 3 "${log_file}" | sed 's/^/  /'
  fi
}

show_status dreamo DreamO
show_status uno UNO
