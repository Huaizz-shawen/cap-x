#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng
GLOBAL_ROOT=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework
CAPX_ROOT="${PROJECT_ROOT}/cap-x"
EVAL_SCRIPT="${GLOBAL_ROOT}/scripts/eval_capx_fair.py"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
WARMUP_MODEL="${GLOBAL_ROOT}/models/lora_sft_lift_success_warmup_0619_merged"
OUT_ROOT="${GLOBAL_ROOT}/evals/fair_lift_lora_sft_warmup_100seeds_2h100_0621"
PYROKI_PORT="${PYROKI_PORT:-8117}"
PYROKI_LOG="${GLOBAL_ROOT}/logs/pyroki_eval_lora_sft_warmup_0621.log"

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

mkdir -p "${OUT_ROOT}" "${GLOBAL_ROOT}/logs"
echo "==== GPU visibility ===="
nvidia-smi || true
echo "==== Eval output root: ${OUT_ROOT} ===="

if [[ ! -f "${WARMUP_MODEL}/config.json" ]]; then
  echo "ERROR: expected merged warmup model is missing: ${WARMUP_MODEL}" >&2
  exit 1
fi

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
    > "${PYROKI_LOG}" 2>&1 &
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
  tail -n 120 "${PYROKI_LOG}" >&2 || true
  exit 1
fi

COMMON_ARGS=(
  --data-source franka_lift_code_env
  --seed-base 60000
  --num-trials 100
  --samples-per-seed 1
  --max-tokens 256
  --temperature 0.2
  --top-p 1.0
  --backend transformers
  --gen-batch-size 4
  --generation-seed 20260621
  --dtype bfloat16
  --max-model-len 1280
  --reward-workers 1
)

python "${EVAL_SCRIPT}" \
  --model-path "${BASE_MODEL}" \
  --output-dir "${OUT_ROOT}/base" \
  --label base \
  "${COMMON_ARGS[@]}"

python "${EVAL_SCRIPT}" \
  --model-path "${WARMUP_MODEL}" \
  --output-dir "${OUT_ROOT}/warmup" \
  --label lora_sft_warmup \
  "${COMMON_ARGS[@]}"

python - <<'PY'
import json
from pathlib import Path

root = Path("/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_lora_sft_warmup_100seeds_2h100_0621")
base = json.loads((root / "base" / "summary.json").read_text())
warmup = json.loads((root / "warmup" / "summary.json").read_text())
comparison = {
    "base_success_rate": base["success_rate"],
    "warmup_success_rate": warmup["success_rate"],
    "delta_success_rate": warmup["success_rate"] - base["success_rate"],
    "base_success_count": base["success_count"],
    "warmup_success_count": warmup["success_count"],
    "num_records": base["num_records"],
    "seeds": base["seeds"],
    "base_valid_execution_rate": base["valid_execution_rate"],
    "warmup_valid_execution_rate": warmup["valid_execution_rate"],
    "base_mean_score": base["mean_score"],
    "warmup_mean_score": warmup["mean_score"],
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(comparison, indent=2, ensure_ascii=False))
PY
