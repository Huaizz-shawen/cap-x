#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

LAUNCHER="${LAUNCHER:-scripts/start_collect_notebook_18447.sh}"
if [[ ! -x "$LAUNCHER" ]]; then
  echo "[error] launcher not found or not executable: $LAUNCHER" >&2
  exit 1
fi

# Parallel orchestration knobs
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
BASE_SAM3_PORT="${BASE_SAM3_PORT:-8114}"
SAM3_PORT_STEP="${SAM3_PORT_STEP:-10}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-25}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-libero_long_parallel}"

# Per-job defaults (can still be overridden via environment)
TOTAL_TRIALS_PER_JOB="${TOTAL_TRIALS_PER_JOB:-10}"
NUM_WORKERS_PER_JOB="${NUM_WORKERS_PER_JOB:-1}"
TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS:-1200}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.7}"
SINGLE_MODEL_ENSEMBLE_TEMPS="${SINGLE_MODEL_ENSEMBLE_TEMPS:-0.1,0.5,0.9}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
LIBGL_ALWAYS_SOFTWARE_FLAG="${LIBGL_ALWAYS_SOFTWARE_FLAG:-1}"
MESA_GL_VERSION_OVERRIDE_VALUE="${MESA_GL_VERSION_OVERRIDE_VALUE:-4.1}"

# Export sidecar defaults (kept enabled)
AUTO_EXPORT_SHARDS="${AUTO_EXPORT_SHARDS:-1}"
AUTO_EXPORT_INTERVAL_S="${AUTO_EXPORT_INTERVAL_S:-60}"
DISK_GUARD_ENABLED="${DISK_GUARD_ENABLED:-1}"
DISK_GUARD_MIN_FREE_GB="${DISK_GUARD_MIN_FREE_GB:-12}"
REQUIRE_MOLMO_READY="${REQUIRE_MOLMO_READY:-0}"
PREFLIGHT_CURL_RETRIES="${PREFLIGHT_CURL_RETRIES:-10}"
PREFLIGHT_CURL_SLEEP_SECONDS="${PREFLIGHT_CURL_SLEEP_SECONDS:-3}"
PREFLIGHT_SCRIPT_TIMEOUT_SECONDS="${PREFLIGHT_SCRIPT_TIMEOUT_SECONDS:-300}"
ROBOT_DESCRIPTION_COMMIT="${ROBOT_DESCRIPTION_COMMIT:-d0d9098d752014aec3725b07766962acf06c5418}"
ROBOT_DESCRIPTIONS_CACHE_DIR="${ROBOT_DESCRIPTIONS_CACHE_DIR:-${ROOT_DIR}/.cache/robot_descriptions}"

if [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "[error] PARALLEL_JOBS must be >= 1" >&2
  exit 1
fi

prewarm_robot_descriptions_cache() {
  local cache_dir="$ROBOT_DESCRIPTIONS_CACHE_DIR"
  local commit="$ROBOT_DESCRIPTION_COMMIT"
  local lock_dir="${cache_dir}/.warmup.lock"
  local ready_flag="${cache_dir}/.ready-${commit}"
  local wait_count=0

  mkdir -p "$cache_dir"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    wait_count=$((wait_count + 1))
    if [[ "$wait_count" -eq 1 ]]; then
      echo "[parallel] waiting for robot cache warmup lock: $lock_dir"
    fi
    sleep 1
  done
  trap 'rmdir "$lock_dir" >/dev/null 2>&1 || true' EXIT

  if [[ -f "$ready_flag" ]]; then
    echo "[parallel] robot cache already warmed for commit $commit"
    rmdir "$lock_dir" >/dev/null 2>&1 || true
    trap - EXIT
    return 0
  fi

  echo "[parallel] warming robot_descriptions cache (commit=$commit, cache=$cache_dir)"
  if env ROBOT_DESCRIPTIONS_CACHE="$cache_dir" ROBOT_DESCRIPTION_COMMIT="$commit" \
      ./.venv/bin/python - <<'PY'
from robot_descriptions.loaders.yourdfpy import load_robot_description

# Triggers cache clone/checkout once in a single process.
load_robot_description("panda_description")
print("robot_descriptions cache warmup complete")
PY
  then
    touch "$ready_flag"
    echo "[parallel] robot cache warmup done"
  else
    echo "[error] robot cache warmup failed" >&2
    rmdir "$lock_dir" >/dev/null 2>&1 || true
    trap - EXIT
    return 1
  fi

  rmdir "$lock_dir" >/dev/null 2>&1 || true
  trap - EXIT
}

prewarm_robot_descriptions_cache

mkdir -p outputs/detached_logs outputs/detached_pids
BATCH_ID="${BATCH_ID:-parallel_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_PATH="outputs/detached_pids/${BATCH_ID}.tsv"

echo -e "slot\tsam3_port\trun_id\tpid\tstatus\tlog_path\tpreflight_ok\tpreflight_lines" > "$SUMMARY_PATH"
echo "[parallel] batch_id=$BATCH_ID"
echo "[parallel] summary=$SUMMARY_PATH"

fail_count=0

for ((slot=1; slot<=PARALLEL_JOBS; slot++)); do
  sam3_port=$((BASE_SAM3_PORT + (slot - 1) * SAM3_PORT_STEP))
  if [[ "$sam3_port" == "8115" || "$sam3_port" == "8116" ]]; then
    echo "[error] sam3_port=${sam3_port} conflicts with default GraspNet/PyRoKi ports (8115/8116). Adjust BASE_SAM3_PORT or SAM3_PORT_STEP." >&2
    exit 1
  fi
  run_tag="${RUN_TAG_PREFIX}_s${slot}"

  echo "[parallel] launching slot=${slot}/${PARALLEL_JOBS} sam3_port=${sam3_port} run_tag=${run_tag}"

  if out="$(
    PORT="${sam3_port}" \
    LOG_PATH="/tmp/sam3_${sam3_port}.log" \
    CAPX_SAM3_SERVICE_URL="http://127.0.0.1:${sam3_port}" \
    RUN_TAG="${run_tag}" \
    TOTAL_TRIALS="${TOTAL_TRIALS_PER_JOB}" \
    NUM_WORKERS="${NUM_WORKERS_PER_JOB}" \
    TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS}" \
    PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE}" \
    SINGLE_MODEL_ENSEMBLE_TEMPS="${SINGLE_MODEL_ENSEMBLE_TEMPS}" \
    MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND}" \
    LIBGL_ALWAYS_SOFTWARE_FLAG="${LIBGL_ALWAYS_SOFTWARE_FLAG}" \
    MESA_GL_VERSION_OVERRIDE_VALUE="${MESA_GL_VERSION_OVERRIDE_VALUE}" \
    AUTO_EXPORT_SHARDS="${AUTO_EXPORT_SHARDS}" \
    AUTO_EXPORT_INTERVAL_S="${AUTO_EXPORT_INTERVAL_S}" \
    DISK_GUARD_ENABLED="${DISK_GUARD_ENABLED}" \
    DISK_GUARD_MIN_FREE_GB="${DISK_GUARD_MIN_FREE_GB}" \
    REQUIRE_MOLMO_READY="${REQUIRE_MOLMO_READY}" \
    PREFLIGHT_CURL_RETRIES="${PREFLIGHT_CURL_RETRIES}" \
    PREFLIGHT_CURL_SLEEP_SECONDS="${PREFLIGHT_CURL_SLEEP_SECONDS}" \
    PREFLIGHT_SCRIPT_TIMEOUT_SECONDS="${PREFLIGHT_SCRIPT_TIMEOUT_SECONDS}" \
    ROBOT_DESCRIPTION_COMMIT="${ROBOT_DESCRIPTION_COMMIT}" \
    ROBOT_DESCRIPTIONS_CACHE_DIR="${ROBOT_DESCRIPTIONS_CACHE_DIR}" \
    "$LAUNCHER" 2>&1
  )"; then
    run_id="$(awk -F= '/^RUN_ID=/{print $2; exit}' <<< "$out")"
    pid="$(awk -F= '/^PID=/{print $2; exit}' <<< "$out")"
    log_path="$(awk -F= '/^LOG=/{print $2; exit}' <<< "$out")"
    preflight_lines="$(grep '^\[preflight\]' <<< "$out" | paste -sd ';' - || true)"
    if [[ -n "$preflight_lines" ]]; then
      preflight_ok="yes"
    else
      preflight_ok="unknown"
      preflight_lines="-"
    fi
    echo "$out"
    echo -e "${slot}\t${sam3_port}\t${run_id:-unknown}\t${pid:-unknown}\tOK\t${log_path:-unknown}\t${preflight_ok}\t${preflight_lines}" >> "$SUMMARY_PATH"
  else
    echo "$out" >&2
    preflight_lines="$(grep '^\[preflight\]' <<< "$out" | paste -sd ';' - || true)"
    [[ -z "$preflight_lines" ]] && preflight_lines="-"
    echo -e "${slot}\t${sam3_port}\t-\t-\tFAILED\t-\tno\t${preflight_lines}" >> "$SUMMARY_PATH"
    fail_count=$((fail_count + 1))
  fi

  if [[ "$slot" -lt "$PARALLEL_JOBS" ]]; then
    echo "[parallel] stagger sleep ${START_STAGGER_SECONDS}s before next slot..."
    sleep "$START_STAGGER_SECONDS"
  fi
done

echo
echo "[parallel] launch complete. summary: $SUMMARY_PATH"
column -ts $'\t' "$SUMMARY_PATH" || cat "$SUMMARY_PATH"

if [[ "$fail_count" -gt 0 ]]; then
  echo "[parallel] failed_slots=$fail_count"
  exit 1
fi
