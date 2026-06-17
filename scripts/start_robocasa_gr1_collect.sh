#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-robocasa_gr1_smoke}"
RUN_ID="${RUN_ID:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CONFIG_PATH="${CONFIG_PATH:-env_configs/robocasa/gr1_robocasa_smoke.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/robocasa-gr1}"
PROVIDER_SRC="${PROVIDER_SRC:-.capx_api/provider_deepseek_vdm_tunnel.sh}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"
TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS:-1200}"
USE_VISUAL_FEEDBACK="${USE_VISUAL_FEEDBACK:-False}"
USE_IMG_DIFFERENCING="${USE_IMG_DIFFERENCING:-True}"
USE_WRIST_CAMERA="${USE_WRIST_CAMERA:-True}"
ROBOCASA_VENV="${ROBOCASA_VENV:-${ROOT_DIR}/.venvs/robocasa-gr1-cpu-py310}"
ROBOCASA_GR1_SRC_ROOT="${ROBOCASA_GR1_SRC_ROOT:-${ROOT_DIR}/third_party/robocasa_gr1/robocasa-gr1-tabletop-tasks}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-osmesa}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-osmesa}"

if [[ ! -x "${ROBOCASA_VENV}/bin/python" ]]; then
  echo "[error] missing RoboCasa GR1 python: ${ROBOCASA_VENV}/bin/python" >&2
  echo "[hint] run scripts/bootstrap_robocasa_gr1_env.sh on an internet-capable notebook first" >&2
  exit 1
fi

if [[ ! -f "$PROVIDER_SRC" ]]; then
  echo "[error] provider script not found: $PROVIDER_SRC" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs/detached_logs outputs/detached_pids

OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="outputs/detached_logs/${RUN_ID}.log"
PID_PATH="outputs/detached_pids/${RUN_ID}.pid"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/capx/third_party/robosuite:${ROBOCASA_GR1_SRC_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="$MUJOCO_GL_BACKEND"
export PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM_VALUE"
export PYTHONUNBUFFERED=1
export CAPX_TRIAL_TIMEOUT_SECONDS="$TRIAL_TIMEOUT_SECONDS"

launch_cmd=(
  "${ROBOCASA_VENV}/bin/python" capx/envs/launch.py
  --api-bash-file "$PROVIDER_SRC"
  --config-path "$CONFIG_PATH"
  --output-dir "$OUT_DIR"
  --total-trials "$TOTAL_TRIALS"
  --num-workers "$NUM_WORKERS"
  --temperature "$PLANNER_TEMPERATURE"
  --record-video True
  --use-visual-feedback "$USE_VISUAL_FEEDBACK"
  --use-img-differencing "$USE_IMG_DIFFERENCING"
  --use-wrist-camera "$USE_WRIST_CAMERA"
  --enable-eap-rollback False
  --enable-eap-recovery False
)

nohup "${launch_cmd[@]}" > "$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOG=$LOG_PATH"
echo "OUT_DIR=$OUT_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON=${ROBOCASA_VENV}/bin/python"
