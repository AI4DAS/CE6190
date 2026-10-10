#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one MIG UUID." >&2
  exit 2
fi
conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/evaluation/evaluate_pilot.py" --project-root "${PROJECT_ROOT}" | tee "${PROJECT_ROOT}/results/metrics/pilot_metrics.txt"

