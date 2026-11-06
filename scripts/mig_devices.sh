#!/usr/bin/env bash
# Source this file to detect the currently configured MIG instances.  UUIDs are
# deliberately not committed because they change when MIG is reconfigured.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; cannot detect MIG devices" >&2
  return 1 2>/dev/null || exit 1
fi

mapfile -t _PB_MIG_UUIDS < <(
  nvidia-smi -L | sed -n 's/.*UUID: \(MIG-[^)]*\)).*/\1/p'
)
if ((${#_PB_MIG_UUIDS[@]} == 0)); then
  echo "ERROR: no MIG UUIDs found in nvidia-smi -L" >&2
  return 1 2>/dev/null || exit 1
fi

for _pb_index in "${!_PB_MIG_UUIDS[@]}"; do
  printf -v _pb_name 'MIG%d' "${_pb_index}"
  export "${_pb_name}=${_PB_MIG_UUIDS[$_pb_index]}"
done
export MIG_COUNT="${#_PB_MIG_UUIDS[@]}"
echo "Detected ${MIG_COUNT} MIG device(s): MIG0..MIG$((MIG_COUNT - 1))"

unset _PB_MIG_UUIDS _pb_index _pb_name
