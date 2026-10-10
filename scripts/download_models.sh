#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"
BASE_DEST="${MODEL_DIR}/base/stable-diffusion-v1-5"
IP_DEST="${MODEL_DIR}/ip_adapter/IP-Adapter"
ENV_NAME="${ENV_NAME:-personalization-benchmark}"
ENV_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"

echo "Base model destination: ${BASE_DEST}"
echo "IP-Adapter destination: ${IP_DEST}"
df -h "${PROJECT_ROOT}"
echo "Estimated download: approximately 9-12 GiB total (repository revisions may change)."
mkdir -p "${BASE_DEST}" "${IP_DEST}" "${PROJECT_ROOT}/results/system"

PROJECT_ROOT="${PROJECT_ROOT}" "${ENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ["PROJECT_ROOT"])
items = [
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", root / "models/base/stable-diffusion-v1-5", None),
    ("h94/IP-Adapter", root / "models/ip_adapter/IP-Adapter", [
        "models/ip-adapter-plus_sd15.bin",
        "models/image_encoder/**",
        "README.md",
        ".gitattributes",
    ]),
]
revisions = {}
api = HfApi()
for repo_id, destination, patterns in items:
    info = api.model_info(repo_id)
    snapshot_download(repo_id=repo_id, revision=info.sha, local_dir=destination, allow_patterns=patterns)
    revisions[repo_id] = info.sha
(root / "results/system/model_revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
PY

test -f "${BASE_DEST}/model_index.json"
test -f "${IP_DEST}/models/ip-adapter-plus_sd15.bin"
test -f "${IP_DEST}/models/image_encoder/config.json"
echo "Verified required model files."

