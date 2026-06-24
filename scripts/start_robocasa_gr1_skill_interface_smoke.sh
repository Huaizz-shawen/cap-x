#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-skill_interface_3task}"
RUN_ID="${RUN_ID:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/robocasa-gr1/diagnostics}"
TASKS="${TASKS:-pnp_pouring two_arm_lift drawer}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
FULL="${FULL:-0}"
DEPRIVILEGED_OBSERVATION="${DEPRIVILEGED_OBSERVATION:-0}"
VISUAL_ANCHOR_DIAGNOSTICS="${VISUAL_ANCHOR_DIAGNOSTICS:-0}"
ROBOCASA_VENV="${ROBOCASA_VENV:-${ROOT_DIR}/.venvs/robocasa-gr1-cpu-py310}"
ROBOCASA_GR1_SRC_ROOT="${ROBOCASA_GR1_SRC_ROOT:-${ROOT_DIR}/third_party/robocasa_gr1/robocasa-gr1-tabletop-tasks}"
SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:8114}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

DRAWER_ANCHOR_CAMERA="${DRAWER_ANCHOR_CAMERA:-egoview}"
DRAWER_ANCHOR_PROMPT="${DRAWER_ANCHOR_PROMPT:-drawer handle}"
DRAWER_ANCHOR_ROI="${DRAWER_ANCHOR_ROI:-}"
DRAWER_RENDER_CAMERA="${DRAWER_RENDER_CAMERA:-egoview}"
DRAWER_WRIST_GRASP_OFFSET="${DRAWER_WRIST_GRASP_OFFSET:-0.0,0.30,-0.12}"
DRAWER_WRIST_PREGRASP_OFFSET="${DRAWER_WRIST_PREGRASP_OFFSET:-0.0,0.18,-0.03}"
DRAWER_WRIST_AXIS_ANGLE_DELTA="${DRAWER_WRIST_AXIS_ANGLE_DELTA:-0.0,-0.45,0.0}"
DRAWER_PREGRASP_STEPS="${DRAWER_PREGRASP_STEPS:-36}"
DRAWER_APPROACH_STEPS="${DRAWER_APPROACH_STEPS:-80}"
DRAWER_CLOSE_STEPS="${DRAWER_CLOSE_STEPS:-20}"
DRAWER_PULL_STEPS="${DRAWER_PULL_STEPS:-80}"
DRAWER_PREGRASP_SCALE="${DRAWER_PREGRASP_SCALE:-0.10}"
DRAWER_APPROACH_SCALE="${DRAWER_APPROACH_SCALE:-0.12}"
DRAWER_PULL_SCALE="${DRAWER_PULL_SCALE:-0.045}"

if [[ ! -x "${ROBOCASA_VENV}/bin/python" ]]; then
  echo "[error] missing RoboCasa GR1 python: ${ROBOCASA_VENV}/bin/python" >&2
  exit 1
fi

mkdir -p outputs/detached_logs outputs/detached_pids "$OUTPUT_ROOT"

OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="outputs/detached_logs/${RUN_ID}.log"
PID_PATH="outputs/detached_pids/${RUN_ID}.pid"

export PYTHONPATH="${ROOT_DIR}:${ROBOCASA_GR1_SRC_ROOT}:${ROOT_DIR}/capx/third_party/robosuite:${PYTHONPATH:-}"
export CAPX_SAM3_SERVICE_URL="$SAM3_SERVICE_URL"
export MUJOCO_GL="$MUJOCO_GL_BACKEND"
export PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM_VALUE"
export PYTHONUNBUFFERED=1

launch_cmd=(
  "${ROBOCASA_VENV}/bin/python" scripts/run_robocasa_gr1_skill_interface_smoke.py
  --output-dir "$OUT_DIR"
  --seed "$SEED"
  --max-steps "$MAX_STEPS"
  --tasks $TASKS
  --drawer-anchor-camera "$DRAWER_ANCHOR_CAMERA"
  --drawer-anchor-prompt "$DRAWER_ANCHOR_PROMPT"
  --drawer-render-camera "$DRAWER_RENDER_CAMERA"
  --drawer-wrist-grasp-offset "$DRAWER_WRIST_GRASP_OFFSET"
  --drawer-wrist-pregrasp-offset "$DRAWER_WRIST_PREGRASP_OFFSET"
  --drawer-wrist-axis-angle-delta "$DRAWER_WRIST_AXIS_ANGLE_DELTA"
)

if [[ -n "$DRAWER_ANCHOR_ROI" ]]; then
  launch_cmd+=(--drawer-anchor-roi "$DRAWER_ANCHOR_ROI")
fi
if [[ "$SAVE_VIDEOS" == "1" || "$SAVE_VIDEOS" == "true" || "$SAVE_VIDEOS" == "True" ]]; then
  launch_cmd+=(--save-videos)
fi
if [[ "$FULL" == "1" || "$FULL" == "true" || "$FULL" == "True" ]]; then
  launch_cmd+=(--full)
fi
if [[ "$DEPRIVILEGED_OBSERVATION" == "1" || "$DEPRIVILEGED_OBSERVATION" == "true" || "$DEPRIVILEGED_OBSERVATION" == "True" ]]; then
  launch_cmd+=(--deprivileged-observation)
fi
if [[ "$VISUAL_ANCHOR_DIAGNOSTICS" == "1" || "$VISUAL_ANCHOR_DIAGNOSTICS" == "true" || "$VISUAL_ANCHOR_DIAGNOSTICS" == "True" ]]; then
  launch_cmd+=(--visual-anchor-diagnostics)
fi
if [[ -n "$DRAWER_PREGRASP_STEPS" ]]; then launch_cmd+=(--drawer-pregrasp-steps "$DRAWER_PREGRASP_STEPS"); fi
if [[ -n "$DRAWER_APPROACH_STEPS" ]]; then launch_cmd+=(--drawer-approach-steps "$DRAWER_APPROACH_STEPS"); fi
if [[ -n "$DRAWER_CLOSE_STEPS" ]]; then launch_cmd+=(--drawer-close-steps "$DRAWER_CLOSE_STEPS"); fi
if [[ -n "$DRAWER_PULL_STEPS" ]]; then launch_cmd+=(--drawer-pull-steps "$DRAWER_PULL_STEPS"); fi
if [[ -n "$DRAWER_PREGRASP_SCALE" ]]; then launch_cmd+=(--drawer-pregrasp-scale "$DRAWER_PREGRASP_SCALE"); fi
if [[ -n "$DRAWER_APPROACH_SCALE" ]]; then launch_cmd+=(--drawer-approach-scale "$DRAWER_APPROACH_SCALE"); fi
if [[ -n "$DRAWER_PULL_SCALE" ]]; then launch_cmd+=(--drawer-pull-scale "$DRAWER_PULL_SCALE"); fi

nohup "${launch_cmd[@]}" > "$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOG=$LOG_PATH"
echo "OUT_DIR=$OUT_DIR"
echo "TASKS=$TASKS"
echo "DEPRIVILEGED_OBSERVATION=$DEPRIVILEGED_OBSERVATION"
echo "VISUAL_ANCHOR_DIAGNOSTICS=$VISUAL_ANCHOR_DIAGNOSTICS"
echo "PYTHON=${ROBOCASA_VENV}/bin/python"
