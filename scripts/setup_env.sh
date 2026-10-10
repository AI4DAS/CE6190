#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
DIFFUSERS_DIR="${PROJECT_ROOT}/third_party/diffusers"

cd "${PROJECT_ROOT}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11 pip git
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

# Official examples are invoked directly from this pinned checkout.
mkdir -p "${PROJECT_ROOT}/third_party"
if [[ ! -d "${DIFFUSERS_DIR}/.git" ]]; then
  git clone https://github.com/huggingface/diffusers.git "${DIFFUSERS_DIR}"
fi
git -C "${DIFFUSERS_DIR}" fetch --tags
git -C "${DIFFUSERS_DIR}" checkout v0.35.1

# xFormers is optional. Install only when explicitly requested, then verify import.
if [[ "${INSTALL_XFORMERS:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install xformers
  conda run -n "${ENV_NAME}" python -c 'import xformers; print(xformers.__version__)'
fi

mkdir -p "${PROJECT_ROOT}/results/system"
conda run -n "${ENV_NAME}" python -m pip freeze > "${PROJECT_ROOT}/results/system/pip_freeze.txt"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
"${ENV_PYTHON}" - <<'PY' > "${PROJECT_ROOT}/results/system/environment.txt"
import importlib.metadata as md
import platform
packages = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "peft", "huggingface_hub", "safetensors", "datasets", "open_clip_torch", "timm", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow", "tensorboard"]
print(f"python={platform.python_version()}")
for name in packages:
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
try:
    import torch
    print(f"torch_cuda={torch.version.cuda}")
except ImportError:
    pass
PY

echo "Environment ready. Activate with: conda activate ${ENV_NAME}"

