#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <verl_global_step_actor_dir> <target_hf_dir>" >&2
  exit 2
fi

ACTOR_DIR="$1"
TARGET_DIR="$2"

if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
  echo "Missing FSDP checkpoint config: ${ACTOR_DIR}/fsdp_config.json" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}" \
  --use_cpu_initialization

echo "Merged HF model written to ${TARGET_DIR}"
