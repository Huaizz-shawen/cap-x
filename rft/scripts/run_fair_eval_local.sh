#!/usr/bin/env bash
set -euo pipefail

RFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPX_ROOT="${CAPX_ROOT:-$(cd "${RFT_ROOT}/.." && pwd)}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"
DATA_SOURCE="${DATA_SOURCE:-franka_lift_code_env}"
MODEL_PATH="${MODEL_PATH:-${CAPX_ROOT}/../rl-homework/models/Qwen2.5-Coder-7B-Instruct}"
MODEL_LABEL="${MODEL_LABEL:-base_local}"
OUT_ROOT="${OUT_ROOT:-${RFT_ROOT}/evals/local_fair_${DATA_SOURCE}_${MODEL_LABEL}}"
OUTPUT_NAME="${OUTPUT_NAME:-${MODEL_LABEL}}"
PYROKI_PORT="${PYROKI_PORT:-8116}"
PYROKI_LOG="${OUT_ROOT}/pyroki_local.log"

if [[ ! -x "${CAPX_VENV}/bin/python" ]]; then
  echo "ERROR: CAPX_VENV python is not executable: ${CAPX_VENV}/bin/python" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model config missing: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${RFT_ROOT}/.cache/robot_descriptions"

export HF_HOME="${HF_HOME:-${RFT_ROOT}/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RFT_ROOT}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RFT_ROOT}/.cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RFT_ROOT}/.cache/numba}"
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-${RFT_ROOT}/.cache/robot_descriptions}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${CAPX_ROOT}:${RFT_ROOT}/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

port_ready() {
  "${CAPX_VENV}/bin/python" - "$PYROKI_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    sock = socket.create_connection(("127.0.0.1", port), timeout=1)
    sock.close()
except OSError:
    sys.exit(1)
PY
}

if ! port_ready; then
  echo "Starting local PyRoKi IK server on port ${PYROKI_PORT}..."
  (
    cd "${CAPX_ROOT}"
    CUDA_VISIBLE_DEVICES="" "${CAPX_VENV}/bin/python" - "$PYROKI_PORT" <<'PY'
import sys
from capx.serving.launch_pyroki_server import main

main(port=int(sys.argv[1]), host="127.0.0.1")
PY
  ) > "${PYROKI_LOG}" 2>&1 &
  PYROKI_PID=$!
  trap 'kill "${PYROKI_PID}" 2>/dev/null || true' EXIT
  for _ in $(seq 1 "${PYROKI_STARTUP_TIMEOUT:-600}"); do
    if port_ready; then
      echo "PyRoKi IK server ready (PID ${PYROKI_PID})"
      break
    fi
    sleep 1
  done
fi

if ! port_ready; then
  echo "ERROR: PyRoKi IK server did not become ready on port ${PYROKI_PORT}" >&2
  tail -n 160 "${PYROKI_LOG}" >&2 || true
  exit 1
fi

cd "${CAPX_ROOT}"

"${CAPX_VENV}/bin/python" "${RFT_ROOT}/scripts/eval_capx_fair.py" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUT_ROOT}/${OUTPUT_NAME}" \
  --label "${MODEL_LABEL}" \
  --data-source "${DATA_SOURCE}" \
  --seed-base "${SEED_BASE:-60000}" \
  --num-trials "${NUM_TRIALS:-10}" \
  --samples-per-seed "${SAMPLES_PER_SEED:-1}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-1.0}" \
  --backend "${BACKEND:-transformers}" \
  --gen-batch-size "${GEN_BATCH_SIZE:-1}" \
  --generation-seed "${GENERATION_SEED:-20260621}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-1280}" \
  --reward-workers "${REWARD_WORKERS:-1}"
