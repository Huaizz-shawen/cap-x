#!/usr/bin/env bash
set -euo pipefail

GLOBAL_USER_ROOT="${GLOBAL_USER_ROOT:-/inspire/hdd/global_user/huaizezheng-p-huaizezheng}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${GLOBAL_USER_ROOT}/rl-homework}"
CAPX_ROOT="${CAPX_ROOT:-${GLOBAL_USER_ROOT}/cap-x}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"

EVAL_SCRIPT="${GLOBAL_ROOT}/scripts/eval_capx_oracle_fair.py"
OUT_ROOT="${OUT_ROOT:-${GLOBAL_ROOT}/evals/fair_lift_oracle_100seeds_1h100_globalonly_clean_0622}"
PYROKI_PORT="${PYROKI_PORT:-8116}"
PYROKI_LOG="${GLOBAL_ROOT}/logs/pyroki_eval_oracle_globalonly_0622.log"

if [[ ! -d "${CAPX_ROOT}" ]]; then
  echo "ERROR: CAPX_ROOT does not exist: ${CAPX_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${CAPX_VENV}/bin/python" ]]; then
  echo "ERROR: CAPX_VENV python is not executable: ${CAPX_VENV}/bin/python" >&2
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: eval script missing: ${EVAL_SCRIPT}" >&2
  exit 1
fi

cd "${CAPX_ROOT}"
. "${CAPX_VENV}/bin/activate"

export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export XDG_CACHE_HOME="${GLOBAL_ROOT}/cache"
export WANDB_DIR="${GLOBAL_ROOT}/logs/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONPATH="${CAPX_ROOT}:${GLOBAL_ROOT}/scripts:${PYTHONPATH:-}"
export ROBOT_DESCRIPTIONS_CACHE="${GLOBAL_ROOT}/cache/robot_descriptions"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}" "${GLOBAL_ROOT}/logs" "${ROBOT_DESCRIPTIONS_CACHE}"
echo "CAPX_ROOT=${CAPX_ROOT}"
echo "CAPX_VENV=${CAPX_VENV}"
echo "GLOBAL_ROOT=${GLOBAL_ROOT}"
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "==== GPU visibility ===="
nvidia-smi || true

port_ready() {
  python - "$PYROKI_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=1)
    s.close()
except OSError:
    sys.exit(1)
PY
}

if ! port_ready; then
  echo "Starting PyRoKi IK server on port ${PYROKI_PORT}..."
  CUDA_VISIBLE_DEVICES="" python - "$PYROKI_PORT" > "${PYROKI_LOG}" 2>&1 <<'PY' &
import sys
from capx.serving.launch_pyroki_server import main

main(port=int(sys.argv[1]), host="127.0.0.1")
PY
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

python "${EVAL_SCRIPT}" \
  --output-dir "${OUT_ROOT}" \
  --label oracle \
  --data-source franka_lift_code_env \
  --seed-base 60000 \
  --num-trials 100 \
  --reward-workers 1
