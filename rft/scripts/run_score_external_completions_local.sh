#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RAW_CANDIDATES_JSONL OUTPUT_DIR [LABEL]" >&2
  exit 2
fi

RAW_CANDIDATES="$1"
OUTPUT_DIR="$2"
LABEL="${3:-teacher_api}"
SCRIPT_ROOT="$(pwd)"

if [[ "${RAW_CANDIDATES}" != /* ]]; then
  RAW_CANDIDATES="${SCRIPT_ROOT}/${RAW_CANDIDATES}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${SCRIPT_ROOT}/${OUTPUT_DIR}"
fi

CAPX_ROOT="${CAPX_ROOT:-/media/user/B29202FA9202C2B91/cap-x}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"
DATA_SOURCE="${DATA_SOURCE:-franka_lift_code_env}"
PYROKI_PORT="${PYROKI_PORT:-8116}"
REWARD_WORKERS="${REWARD_WORKERS:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_ROOT}/.cache}"
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-${SCRIPT_ROOT}/.cache/robot_descriptions}"
export PYTHONPATH="${CAPX_ROOT}:${SCRIPT_ROOT}/scripts:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${ROBOT_DESCRIPTIONS_CACHE}"

port_ready() {
  "${CAPX_VENV}/bin/python" - "$PYROKI_PORT" <<'PY'
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
  echo "Starting local PyRoKi IK server on port ${PYROKI_PORT}..."
  (
    cd "${CAPX_ROOT}"
    "${CAPX_VENV}/bin/python" - "$PYROKI_PORT" <<'PY'
import sys
from capx.serving.launch_pyroki_server import main

main(port=int(sys.argv[1]), host="127.0.0.1")
PY
  ) > "${OUTPUT_DIR}/pyroki_local.log" 2>&1 &
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
  tail -n 120 "${OUTPUT_DIR}/pyroki_local.log" >&2 || true
  exit 1
fi

cd "${CAPX_ROOT}"

"${CAPX_VENV}/bin/python" "${SCRIPT_ROOT}/scripts/score_external_completions.py" \
  --input "${RAW_CANDIDATES}" \
  --output-dir "${OUTPUT_DIR}" \
  --label "${LABEL}" \
  --data-source "${DATA_SOURCE}" \
  --reward-workers "${REWARD_WORKERS}"
