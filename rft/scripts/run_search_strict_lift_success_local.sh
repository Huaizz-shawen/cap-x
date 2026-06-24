#!/usr/bin/env bash
set -euo pipefail

RL_ROOT="${RL_ROOT:-/media/user/B29202FA9202C2B91/rl-homework}"
CAPX_ROOT="${CAPX_ROOT:-/media/user/B29202FA9202C2B91/cap-x}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"
OUT_ROOT="${OUT_ROOT:-${RL_ROOT}/diagnostics/strict_lift_success_search_local}"
PYROKI_PORT="${PYROKI_PORT:-8116}"
PYROKI_LOG="${OUT_ROOT}/pyroki.log"
NUM_SEEDS="${NUM_SEEDS:-5}"
MAX_TEMPLATES="${MAX_TEMPLATES:-2}"
STOP_AFTER_SUCCESSES="${STOP_AFTER_SUCCESSES:-0}"

if [[ ! -x "${CAPX_VENV}/bin/python" ]]; then
  echo "ERROR: CAPX venv python missing: ${CAPX_VENV}/bin/python" >&2
  exit 1
fi
if [[ ! -f "${RL_ROOT}/scripts/search_strict_lift_success.py" ]]; then
  echo "ERROR: search script missing under RL_ROOT=${RL_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${RL_ROOT}/.cache/matplotlib" "${RL_ROOT}/.cache/robot_descriptions"

cd "${CAPX_ROOT}"
. "${CAPX_VENV}/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CAPX_ROOT}:${RL_ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${RL_ROOT}/.cache"
export MPLCONFIGDIR="${RL_ROOT}/.cache/matplotlib"
export ROBOT_DESCRIPTIONS_CACHE="${RL_ROOT}/.cache/robot_descriptions"

port_ready() {
  python - "$PYROKI_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    with socket.create_connection(("127.0.0.1", port), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
}

if ! port_ready; then
  echo "Starting local PyRoKi IK server on port ${PYROKI_PORT}..."
  python -c "from capx.serving.launch_pyroki_server import main; main(port=${PYROKI_PORT}, host='127.0.0.1')" > "${PYROKI_LOG}" 2>&1 &
  PYROKI_PID=$!
  trap 'kill "${PYROKI_PID}" 2>/dev/null || true' EXIT
  for _ in $(seq 1 "${PYROKI_STARTUP_TIMEOUT:-300}"); do
    if port_ready; then
      echo "PyRoKi IK server ready (PID ${PYROKI_PID})"
      break
    fi
    sleep 1
  done
fi

if ! port_ready; then
  echo "ERROR: PyRoKi IK server did not become ready" >&2
  tail -n 120 "${PYROKI_LOG}" >&2 || true
  exit 1
fi

python -u "${RL_ROOT}/scripts/search_strict_lift_success.py" \
  --output-dir "${OUT_ROOT}" \
  --data-source franka_lift_code_env \
  --seed-base 60000 \
  --num-seeds "${NUM_SEEDS}" \
  --max-templates "${MAX_TEMPLATES}" \
  --stop-after-successes "${STOP_AFTER_SUCCESSES}"
