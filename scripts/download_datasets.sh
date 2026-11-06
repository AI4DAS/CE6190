#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
DREAMBOOTH_DIR="${PROJECT_ROOT}/data/dreambooth/upstream"
CUSTOM_DIR="${PROJECT_ROOT}/data/customconcept101/upstream"
CUSTOM_ZIP="${PROJECT_ROOT}/data/customconcept101/benchmark_dataset.zip"
CUSTOM_EXTRACT="${PROJECT_ROOT}/data/customconcept101/dataset"

mkdir -p "${PROJECT_ROOT}/data/dreambooth" "${PROJECT_ROOT}/data/customconcept101"
if [[ ! -d "${DREAMBOOTH_DIR}/.git" ]]; then
  git clone https://github.com/google/dreambooth.git "${DREAMBOOTH_DIR}"
fi
if [[ ! -d "${CUSTOM_DIR}/.git" ]]; then
  git clone https://github.com/adobe-research/custom-diffusion.git "${CUSTOM_DIR}"
fi

if [[ ! -f "${CUSTOM_ZIP}" ]]; then
  curl --fail --location --retry 3 \
    https://huggingface.co/datasets/nupurkmr9/custom-diffusion/resolve/main/benchmark_dataset.zip \
    --output "${CUSTOM_ZIP}.part"
  mv "${CUSTOM_ZIP}.part" "${CUSTOM_ZIP}"
fi
if [[ ! -d "${CUSTOM_EXTRACT}/benchmark_dataset" ]]; then
  mkdir -p "${CUSTOM_EXTRACT}"
  unzip -q "${CUSTOM_ZIP}" -d "${CUSTOM_EXTRACT}"
fi

{
  echo "dreambooth_commit=$(git -C "${DREAMBOOTH_DIR}" rev-parse HEAD)"
  echo "custom_diffusion_commit=$(git -C "${CUSTOM_DIR}" rev-parse HEAD)"
  echo "customconcept101_zip_sha256=$(sha256sum "${CUSTOM_ZIP}" | awk '{print $1}')"
} > "${PROJECT_ROOT}/results/system/dataset_revisions.txt"

"${ENV_PYTHON}" "${PROJECT_ROOT}/src/data/prepare_main_subjects.py" \
  --project-root "${PROJECT_ROOT}"
echo "Datasets downloaded and all configured benchmark subjects prepared."

