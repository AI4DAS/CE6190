#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${PROJECT_ROOT}/results/system/gpu_info.txt"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  echo "=== nvidia-smi ==="
  nvidia-smi
  echo
  echo "=== nvidia-smi -L ==="
  nvidia-smi -L
  echo
  echo "=== PyTorch ==="
  python - <<'PY'
try:
    import torch
    print(f"pytorch_version={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"visible_gpu_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"cuda:{index}: name={props.name}; total_memory_mib={props.total_memory / 2**20:.0f}")
except ImportError:
    print("pytorch_version=NOT_INSTALLED")
PY
} | tee "${OUTPUT_FILE}"

echo "Wrote ${OUTPUT_FILE}"

