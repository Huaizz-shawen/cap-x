#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="${PORT:-8114}"
HOST="${HOST:-127.0.0.1}"
DEVICE="${DEVICE:-cuda}"
VENV="${VENV:-${REPO_ROOT}/.venvs/sam3-cu128-py311}"
CHECKPOINT_PATH="${CAPX_SAM3_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/sam3.pt}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/service_logs}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "SAM3 venv not found: ${VENV}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "SAM3 checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if command -v lsof >/dev/null 2>&1; then
  old_pids="$(lsof -ti "tcp:${PORT}" || true)"
  if [[ -n "${old_pids}" ]]; then
    kill ${old_pids} || true
  fi
fi

log_path="${LOG_DIR}/sam3_${PORT}_$(date +%Y%m%d_%H%M%S).log"
pid_path="${LOG_DIR}/sam3_${PORT}.pid"

cd "${REPO_ROOT}"
nohup env \
  PYTHONPATH="${REPO_ROOT}" \
  CAPX_SAM3_CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  "${VENV}/bin/python" \
  capx/serving/launch_sam3_server.py \
  --device "${DEVICE}" \
  --port "${PORT}" \
  --host "${HOST}" \
  >"${log_path}" 2>&1 &

echo "$!" >"${pid_path}"
echo "SAM3_PID=$(cat "${pid_path}")"
echo "SAM3_LOG=${log_path}"
echo "SAM3_URL=http://${HOST}:${PORT}"
