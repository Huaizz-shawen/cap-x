#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng
GLOBAL_ROOT=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework
CAPX_ROOT="${PROJECT_ROOT}/cap-x"
EVAL_SCRIPT="${GLOBAL_ROOT}/scripts/eval_capx_fair.py"
MERGE_SCRIPT="${GLOBAL_ROOT}/scripts/merge_verl_fsdp_checkpoint.sh"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
ACTOR_DIR="${GLOBAL_ROOT}/runs/capx_grpo_lift_medium_4h100_global_0618/checkpoints/hyrl/grpo_qwen25coder7b_franka_lift_code_env_0618_medium_temperature_1.0_group_size_4_train1024_epochs2/global_step_32/actor"
RFT_MODEL="${GLOBAL_ROOT}/models/grpo_lift_medium_4h100_global_step32_hf"
OUT_ROOT="${GLOBAL_ROOT}/evals/fair_lift_medium_step32_transformers_100seeds_2h100_0619"
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

mkdir -p "${OUT_ROOT}" "${GLOBAL_ROOT}/logs"
echo "==== GPU visibility ===="
nvidia-smi || true
echo "==== Eval output root: ${OUT_ROOT} ===="

if [[ ! -f "${RFT_MODEL}/config.json" ]]; then
  echo "Merging VeRL FSDP actor checkpoint to HF: ${RFT_MODEL}"
  bash "${MERGE_SCRIPT}" "${ACTOR_DIR}" "${RFT_MODEL}"
else
  echo "Merged HF model already exists: ${RFT_MODEL}"
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
    > "${GLOBAL_ROOT}/logs/pyroki_eval_medium_0619.log" 2>&1 &
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
  tail -n 120 "${GLOBAL_ROOT}/logs/pyroki_eval_medium_0619.log" >&2 || true
  exit 1
fi

COMMON_ARGS=(
  --data-source franka_lift_code_env
  --seed-base 40000
  --num-trials 100
  --samples-per-seed 1
  --max-tokens 256
  --temperature 0.2
  --top-p 1.0
  --backend transformers
  --gen-batch-size 4
  --generation-seed 20260619
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
  --model-path "${RFT_MODEL}" \
  --output-dir "${OUT_ROOT}/rft" \
  --label rft_medium_step32 \
  "${COMMON_ARGS[@]}"

python - <<'PY'
import json
from pathlib import Path

root = Path("/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_medium_step32_transformers_100seeds_2h100_0619")
base = json.loads((root / "base" / "summary.json").read_text())
rft = json.loads((root / "rft" / "summary.json").read_text())
comparison = {
    "base_success_rate": base["success_rate"],
    "rft_success_rate": rft["success_rate"],
    "delta_success_rate": rft["success_rate"] - base["success_rate"],
    "base_success_count": base["success_count"],
    "rft_success_count": rft["success_count"],
    "num_records": base["num_records"],
    "seeds": base["seeds"],
    "base_valid_execution_rate": base["valid_execution_rate"],
    "rft_valid_execution_rate": rft["valid_execution_rate"],
    "base_mean_score": base["mean_score"],
    "rft_mean_score": rft["mean_score"],
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(comparison, indent=2, ensure_ascii=False))
PY
