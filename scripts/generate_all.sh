#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one MIG UUID." >&2
  exit 2
fi
for method in dreambooth lora textual_inversion ip_adapter_plus; do
  conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/inference/generate_pilot_benchmark.py" \
    --project-root "${PROJECT_ROOT}" --method "${method}"
done

