#!/usr/bin/env bash
set -euo pipefail

# Health-check and self-heal script for SAM3 service on a notebook.
# Behavior:
# 1) If /docs on SAM3 port returns HTTP 200 -> print SAM3_ALIVE and exit 0.
# 2) Otherwise, cleanup stale listeners / old launcher processes.
# 3) Relaunch SAM3 with explicit checkpoint.
# 4) Wait until /docs is healthy -> print SAM3_RECOVERED and exit 0.
# 5) If still unhealthy after timeout -> print SAM3_FAILED and exit 1.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PORT="${PORT:-8114}"
HOST="${HOST:-127.0.0.1}"
DEVICE="${DEVICE:-cuda}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/inspire/hdd/project/exploration-topic/public/zzhuai/LIBERO/third_party/checkpoints/sam3.pt}"
LOG_PATH="${LOG_PATH:-/tmp/sam3_${PORT}.log}"

if [[ ! -x ./.venv/bin/python ]]; then
  echo "SAM3_FAILED missing_python_venv ./.venv/bin/python"
  exit 1
fi

health_code() {
  local code
  set +e
  code="$(
    curl -sS \
      --connect-timeout 2 \
      --max-time 5 \
      -o /tmp/sam3_health_${PORT}.out \
      -w "%{http_code}" \
      "http://${HOST}:${PORT}/docs" 2>/dev/null
  )"
  set -e
  # Normalize any non-standard output to a 3-digit HTTP-like code.
  if [[ ! "${code}" =~ ^[0-9]{3}$ ]]; then
    code="000"
  fi
  printf '%s' "${code}"
}

healthy_now() {
  [[ "$(health_code)" == "200" ]]
}

port_pids() {
  ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$0 ~ p {
    while (match($0, /pid=[0-9]+/)) {
      print substr($0, RSTART + 4, RLENGTH - 4)
      $0 = substr($0, RSTART + RLENGTH)
    }
  }' | sort -u
}

cleanup_stale() {
  # 1) Kill any process currently listening on the target port.
  pids="$(port_pids || true)"
  if [[ -n "${pids}" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done <<< "${pids}"
  fi

  # 2) Kill only SAM3 launcher processes that target this exact port.
  #    Important for parallel lanes: do NOT kill other SAM3 instances.
  pgrep -af "capx.serving.launch_sam3_server" | while read -r pid cmdline; do
    if [[ "${cmdline}" =~ --port([[:space:]]|=)${PORT}([^0-9]|$) ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done || true
}

launch_sam3() {
  nohup ./.venv/bin/python -m capx.serving.launch_sam3_server \
    --host "${HOST}" \
    --port "${PORT}" \
    --device "${DEVICE}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --no-load-from-hf \
    > "${LOG_PATH}" 2>&1 &
  echo "$!"
}

wait_healthy() {
  local elapsed=0
  while (( elapsed < WAIT_SECONDS )); do
    if healthy_now; then
      return 0
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$(( elapsed + POLL_INTERVAL ))
  done
  return 1
}

if healthy_now; then
  echo "SAM3_ALIVE host=${HOST} port=${PORT} code=200"
  exit 0
fi

cleanup_stale
sleep 1
pid="$(launch_sam3)"

if wait_healthy; then
  echo "SAM3_RECOVERED host=${HOST} port=${PORT} pid=${pid} code=200"
  exit 0
fi

echo "SAM3_FAILED host=${HOST} port=${PORT} pid=${pid} code=$(health_code)"
tail -n 80 "${LOG_PATH}" 2>/dev/null || true
exit 1
