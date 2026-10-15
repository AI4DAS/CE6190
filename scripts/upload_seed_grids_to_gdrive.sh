#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/results/figures/main_experiment/per_subject"
REMOTE_NAME="${1:-gdrive}"
DESTINATION="${2:-CE6190/personalization-benchmark/main_experiment/per_subject_seed_grids}"
EXPECTED_COUNT=32
PATTERN="*_four_method_grid_seed*.png"

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is not installed or is not on PATH." >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: source directory does not exist: ${SOURCE_DIR}" >&2
  exit 2
fi

if ! rclone listremotes | grep -Fxq "${REMOTE_NAME}:"; then
  echo "ERROR: rclone remote '${REMOTE_NAME}' is not configured." >&2
  echo "Configured remotes:" >&2
  rclone listremotes >&2 || true
  exit 2
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
LOCAL_LIST="${TEMP_DIR}/local.txt"
REMOTE_LIST="${TEMP_DIR}/remote.txt"

find "${SOURCE_DIR}" -maxdepth 1 -type f -name "${PATTERN}" -printf '%f\n' | sort > "${LOCAL_LIST}"
LOCAL_COUNT="$(wc -l < "${LOCAL_LIST}")"
if [[ "${LOCAL_COUNT}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_COUNT} matching PNGs, found ${LOCAL_COUNT}." >&2
  exit 3
fi

echo "Source: ${SOURCE_DIR}"
echo "Destination: ${REMOTE_NAME}:${DESTINATION}"
echo "Files: ${LOCAL_COUNT}"
echo "Total size: $(du -ch $(sed "s#^#${SOURCE_DIR}/#" "${LOCAL_LIST}") | tail -n 1 | awk '{print $1}')"

# `copy` is non-destructive: it never removes unrelated files already in Drive.
rclone copy "${SOURCE_DIR}" "${REMOTE_NAME}:${DESTINATION}" \
  --max-depth 1 \
  --include "${PATTERN}" \
  --exclude "*" \
  --transfers 4 \
  --checkers 8 \
  --progress

rclone lsf "${REMOTE_NAME}:${DESTINATION}" \
  --files-only \
  --include "${PATTERN}" | sort > "${REMOTE_LIST}"

if ! diff -u "${LOCAL_LIST}" "${REMOTE_LIST}"; then
  echo "ERROR: remote filename verification failed." >&2
  exit 4
fi

echo "UPLOAD_COMPLETED_SUCCESSFULLY"
echo "Verified ${EXPECTED_COUNT} PNG files at ${REMOTE_NAME}:${DESTINATION}"
