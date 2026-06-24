#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng
GLOBAL_ROOT=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework
CAPX_ROOT="${PROJECT_ROOT}/cap-x"
EVAL_SCRIPT="${GLOBAL_ROOT}/scripts/eval_capx_fair.py"
BUILD_SCRIPT="${GLOBAL_ROOT}/scripts/build_sft_from_success_records.py"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
OUT_ROOT="${GLOBAL_ROOT}/data/warmup_lift_base_candidates_0619"
SFT_ROOT="${GLOBAL_ROOT}/data/sft_lift_success_warmup_0619"
PYROKI_PORT="${PYROKI_PORT:-8116}"

cd "${CAPX_ROOT}"
. .venv-rl/bin/activate

export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export XDG_CACHE_HOME="${GLOBAL_ROOT}/cache"
export WANDB_DIR="${GLOBAL_ROOT}/logs/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONPATH="${CAPX_ROOT}:${PYTHONPATH:-}"
export ROBOT_DESCRIPTIONS_CACHE="${PROJECT_ROOT}/.cache/robot_descriptions"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUT_ROOT}" "${SFT_ROOT}" "${GLOBAL_ROOT}/logs"
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
  CUDA_VISIBLE_DEVICES="" python -m capx.serving.launch_pyroki_server \
    --port "${PYROKI_PORT}" --host 127.0.0.1 \
    > "${GLOBAL_ROOT}/logs/pyroki_collect_warmup_0619.log" 2>&1 &
  PYROKI_PID=$!
  trap 'kill "${PYROKI_PID}" 2>/dev/null || true' EXIT
  for _ in $(seq 1 90); do
    if port_ready; then
      echo "PyRoKi IK server ready (PID ${PYROKI_PID})"
      break
    fi
    sleep 1
  done
fi

if ! port_ready; then
  echo "ERROR: PyRoKi IK server did not become ready on port ${PYROKI_PORT}" >&2
  tail -n 120 "${GLOBAL_ROOT}/logs/pyroki_collect_warmup_0619.log" >&2 || true
  exit 1
fi

if [[ ! -f "${OUT_ROOT}/base/records.jsonl" ]]; then
  python "${EVAL_SCRIPT}" \
    --model-path "${BASE_MODEL}" \
    --output-dir "${OUT_ROOT}/base" \
    --label base_train_candidates \
    --data-source franka_lift_code_env \
    --seed-base 50000 \
    --num-trials 128 \
    --samples-per-seed 4 \
    --max-tokens 256 \
    --temperature 0.7 \
    --top-p 0.95 \
    --backend transformers \
    --gen-batch-size 4 \
    --generation-seed 20260619 \
    --dtype bfloat16 \
    --max-model-len 1280 \
    --reward-workers 1
else
  echo "Candidate records already exist: ${OUT_ROOT}/base/records.jsonl"
fi

python "${BUILD_SCRIPT}" \
  --records "${OUT_ROOT}/base/records.jsonl" \
  --summary "${OUT_ROOT}/base/summary.json" \
  --output "${SFT_ROOT}/train.jsonl" \
  --summary-output "${SFT_ROOT}/dataset_summary.json"
