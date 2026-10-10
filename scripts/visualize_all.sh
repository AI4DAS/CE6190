#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/visualization/dataset_contact_sheet.py" --subject-dir "${PROJECT_ROOT}/data/processed/dreambooth/dog2" --output-dir "${PROJECT_ROOT}/results/figures/dataset"
conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/visualization/dataset_summary.py" --project-root "${PROJECT_ROOT}"
if [[ -f "${PROJECT_ROOT}/outputs/ip_adapter/dreambooth/dog2/P0/seed_42.png" ]]; then
  conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/visualization/ip_adapter_grid.py" --project-root "${PROJECT_ROOT}" --seed 42
fi

