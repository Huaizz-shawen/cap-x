#!/usr/bin/env bash
set -euo pipefail

# Health-check and self-heal script for Molmo service on port 8122.
# Behavior:
# 1) If /v1/models returns HTTP 200 -> print MOLMO_ALIVE and exit 0.
# 2) Otherwise, cleanup stale listeners / old vLLM processes on this port.
# 3) Relaunch vLLM OpenAI-compatible API server for Molmo2.
# 4) Wait until /v1/models is healthy -> print MOLMO_RECOVERED and exit 0.
# 5) If still unhealthy after timeout -> print MOLMO_FAILED and exit 1.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8122}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
MODEL_DIR="${MODEL_DIR:-/inspire/hdd/project/exploration-topic/public/zzhuai/LIBERO/third_party/checkpoints/molmo2-8b}"
VENV_PYTHON="${VENV_PYTHON:-${ROOT_DIR}/.venv-molmo/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-allenai/Molmo2-8B}"
LOG_PATH="${LOG_PATH:-/tmp/molmo_${PORT}.log}"
HF_HUB_OFFLINE_MODE="${HF_HUB_OFFLINE_MODE:-1}"
TRANSFORMERS_OFFLINE_MODE="${TRANSFORMERS_OFFLINE_MODE:-1}"
HF_DATASETS_OFFLINE_MODE="${HF_DATASETS_OFFLINE_MODE:-1}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "MOLMO_FAILED missing_python ${VENV_PYTHON}"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MOLMO_FAILED missing_model_dir ${MODEL_DIR}"
  exit 1
fi

health_code() {
  local code
  set +e
  code="$(
    curl -sS \
      --connect-timeout 2 \
      --max-time 8 \
      -o /tmp/molmo_health_${PORT}.out \
      -w "%{http_code}" \
      "http://${HOST}:${PORT}/v1/models" 2>/dev/null
  )"
  set -e
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
  local pids
  pids="$(port_pids || true)"
  if [[ -n "${pids}" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done <<< "${pids}"
  fi

  # Kill only vLLM api_server processes for the same port.
  pgrep -af "vllm.entrypoints.openai.api_server" | while read -r pid cmdline; do
    if [[ "${cmdline}" =~ --port([[:space:]]|=)${PORT}([^0-9]|$) ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done || true
}

launch_molmo() {
  nohup env \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE_MODE}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE_MODE}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE_MODE}" \
    "${VENV_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_DIR}" \
    --tokenizer "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
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
  echo "MOLMO_ALIVE host=${HOST} port=${PORT} code=200"
  exit 0
fi

cleanup_stale
sleep 1
pid="$(launch_molmo)"

if wait_healthy; then
  echo "MOLMO_RECOVERED host=${HOST} port=${PORT} pid=${pid} code=200"
  exit 0
fi

echo "MOLMO_FAILED host=${HOST} port=${PORT} pid=${pid} code=$(health_code)"
tail -n 120 "${LOG_PATH}" 2>/dev/null || true
exit 1
