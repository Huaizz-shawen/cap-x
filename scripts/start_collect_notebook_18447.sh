#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-libero_diag_1trial}"
RUN_ID="${RUN_ID:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CONFIG_PATH="${CONFIG_PATH:-env_configs/libero/franka_libero_spatial_0_persistent_sam3_lightweight.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/deepseek-v4-pro}"

# Default to the validated dual-tunnel provider:
# planner -> api.deepseek.com:18086, VDM -> open.xiaojingai.com:18447
PROVIDER_SRC="${PROVIDER_SRC:-.capx_api/provider_deepseek_vdm_tunnel.sh}"
PROVIDER_GEN="${PROVIDER_GEN:-.capx_api/provider_deepseek_vdm_tunnel.generated.sh}"
VDM_ROUTE_URL="${VDM_ROUTE_URL:-https://open.xiaojingai.com:18447/v1/chat/completions}"
PLANNER_HEALTHCHECK_URL="${PLANNER_HEALTHCHECK_URL:-https://api.deepseek.com:18086/v1/models}"

ROBOT_DESCRIPTION_COMMIT="${ROBOT_DESCRIPTION_COMMIT:-d0d9098d752014aec3725b07766962acf06c5418}"
SAM3_CKPT="${SAM3_CKPT:-/inspire/hdd/project/exploration-topic/public/zzhuai/LIBERO/third_party/checkpoints/sam3.pt}"
SAM3_HEAL_SCRIPT="${SAM3_HEAL_SCRIPT:-scripts/recover_sam3_8114.sh}"
SAM3_HEAL_WAIT_SECONDS="${SAM3_HEAL_WAIT_SECONDS:-120}"
ENABLE_MOLMO_HEAL="${ENABLE_MOLMO_HEAL:-0}"
MOLMO_HEAL_SCRIPT="${MOLMO_HEAL_SCRIPT:-scripts/recover_molmo_8122.sh}"
MOLMO_HEAL_WAIT_SECONDS="${MOLMO_HEAL_WAIT_SECONDS:-180}"
REQUIRE_MOLMO_READY="${REQUIRE_MOLMO_READY:-0}"
PREFLIGHT_CURL_RETRIES="${PREFLIGHT_CURL_RETRIES:-10}"
PREFLIGHT_CURL_SLEEP_SECONDS="${PREFLIGHT_CURL_SLEEP_SECONDS:-3}"
PREFLIGHT_SCRIPT_TIMEOUT_SECONDS="${PREFLIGHT_SCRIPT_TIMEOUT_SECONDS:-300}"
TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS:-1800}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"
ENABLE_PARALLEL_ENSEMBLE="${ENABLE_PARALLEL_ENSEMBLE:-0}"
ENABLE_MULTIMODEL_ENSEMBLE="${ENABLE_MULTIMODEL_ENSEMBLE:-0}"
SINGLE_MODEL_ENSEMBLE_TEMPS="${SINGLE_MODEL_ENSEMBLE_TEMPS:-0.1,0.5,0.9}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
LIBGL_ALWAYS_SOFTWARE_FLAG="${LIBGL_ALWAYS_SOFTWARE_FLAG:-1}"
MESA_GL_VERSION_OVERRIDE_VALUE="${MESA_GL_VERSION_OVERRIDE_VALUE:-4.1}"
AUTO_EXPORT_SHARDS="${AUTO_EXPORT_SHARDS:-1}"
AUTO_EXPORT_INTERVAL_S="${AUTO_EXPORT_INTERVAL_S:-60}"
AUTO_EXPORT_OUTPUT_ROOT="${AUTO_EXPORT_OUTPUT_ROOT:-outputs/lerobot_shards/${RUN_ID}}"
DISK_GUARD_ENABLED="${DISK_GUARD_ENABLED:-1}"
DISK_GUARD_MIN_FREE_GB="${DISK_GUARD_MIN_FREE_GB:-12}"
LIBERO_SHARED_ROOT="${LIBERO_SHARED_ROOT:-/inspire/hdd/project/exploration-topic/public/zzhuai/LIBERO}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-${LIBERO_SHARED_ROOT}/libero}"
LIBERO_DATASET_ROOT="${LIBERO_DATASET_ROOT:-${LIBERO_SHARED_ROOT}/datasets}"
LIBERO_BDDL_ROOT="${LIBERO_BDDL_ROOT:-${LIBERO_BENCHMARK_ROOT}/libero/bddl_files}"
LIBERO_INIT_ROOT="${LIBERO_INIT_ROOT:-${LIBERO_BENCHMARK_ROOT}/libero/init_files}"
LIBERO_ASSETS_ROOT="${LIBERO_ASSETS_ROOT:-${LIBERO_BENCHMARK_ROOT}/libero/assets}"
ROBOT_DESCRIPTIONS_CACHE_DIR="${ROBOT_DESCRIPTIONS_CACHE_DIR:-${ROOT_DIR}/.cache/robot_descriptions}"

wait_http_code() {
  local name="$1"
  local url="$2"
  local retries="$3"
  local sleep_s="$4"
  shift 4
  local curl_args=("$@")
  local code i

  for ((i=1; i<=retries; i++)); do
    code="$(
      curl -sS "${curl_args[@]}" \
        --connect-timeout 3 \
        --max-time 8 \
        -o /tmp/capx_preflight_${name}.out \
        -w "%{http_code}" \
        "$url" 2>/dev/null || true
    )"
    if [[ "$code" == "200" || "$code" == "401" ]]; then
      echo "[preflight] ${name}: HTTP ${code} (ok, attempt ${i}/${retries})"
      return 0
    fi
    echo "[preflight] ${name}: HTTP ${code:-000} (attempt ${i}/${retries})"
    sleep "$sleep_s"
  done

  echo "[preflight] ${name}: failed after ${retries} attempts" >&2
  return 1
}

if [[ ! -x ./.venv/bin/python ]]; then
  echo "[error] missing runtime python: $ROOT_DIR/.venv/bin/python" >&2
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

./.venv/bin/python - "$PROVIDER_SRC" "$PROVIDER_GEN" "$VDM_ROUTE_URL" <<'PY'
import subprocess
import sys

src, dst, route = sys.argv[1:4]
tokens = subprocess.check_output(["bash", src], text=True).splitlines()
for i in range(len(tokens) - 1):
    if tokens[i] == "--visual-differencing-model-server-url":
        tokens[i + 1] = route

def shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

with open(dst, "w", encoding="utf-8") as f:
    f.write("#!/usr/bin/env bash\n")
    f.write("set -euo pipefail\n")
    f.write("printf '%s\\n' \\\n")
    for idx, tok in enumerate(tokens):
        suffix = " \\\n" if idx < len(tokens) - 1 else "\n"
        f.write(f"  {shq(tok)}{suffix}")
PY
chmod +x "$PROVIDER_GEN"

mkdir -p outputs/detached_logs outputs/detached_pids
OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="outputs/detached_logs/${RUN_ID}.log"
PID_PATH="outputs/detached_pids/${RUN_ID}.pid"

# Always rewrite LIBERO config to a verified shared-storage root.
mkdir -p /root/.libero
cat > /root/.libero/config.yaml <<CFG
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
dataset_root: ${LIBERO_DATASET_ROOT}
bddl_files: ${LIBERO_BDDL_ROOT}
init_states: ${LIBERO_INIT_ROOT}
assets: ${LIBERO_ASSETS_ROOT}
CFG

# Keep hostname-based TLS/SNI while forcing local tunnel egress.
grep -q 'api.deepseek.com' /etc/hosts || echo '127.0.0.1 api.deepseek.com # capx-code-tunnel' >> /etc/hosts
grep -q 'open.xiaojingai.com' /etc/hosts || echo '127.0.0.1 open.xiaojingai.com # capx-vdm-tunnel' >> /etc/hosts

export ROBOT_DESCRIPTION_COMMIT
export CAPX_SAM3_CHECKPOINT_PATH="$SAM3_CKPT"
export CAPX_DISABLE_MOLMO="${CAPX_DISABLE_MOLMO:-1}"
export CAPX_TRIAL_TIMEOUT_SECONDS="$TRIAL_TIMEOUT_SECONDS"
export CAPX_SINGLE_MODEL_ENSEMBLE_TEMPS="$SINGLE_MODEL_ENSEMBLE_TEMPS"
export MUJOCO_GL="$MUJOCO_GL_BACKEND"
export LIBGL_ALWAYS_SOFTWARE="$LIBGL_ALWAYS_SOFTWARE_FLAG"
export MESA_GL_VERSION_OVERRIDE="$MESA_GL_VERSION_OVERRIDE_VALUE"
mkdir -p "$ROBOT_DESCRIPTIONS_CACHE_DIR"
export ROBOT_DESCRIPTIONS_CACHE="$ROBOT_DESCRIPTIONS_CACHE_DIR"

# Pre-flight: ensure SAM3 on 8114 is alive (heal if needed) before launch.
if [[ -x "$SAM3_HEAL_SCRIPT" ]]; then
  timeout "$PREFLIGHT_SCRIPT_TIMEOUT_SECONDS" \
    env CHECKPOINT_PATH="$SAM3_CKPT" WAIT_SECONDS="$SAM3_HEAL_WAIT_SECONDS" "$SAM3_HEAL_SCRIPT"
else
  echo "[warn] SAM3 heal script not found or not executable: $SAM3_HEAL_SCRIPT"
fi

# Optional pre-flight: ensure Molmo on 8122 is alive (heal if needed).
if [[ "$ENABLE_MOLMO_HEAL" == "1" ]]; then
  if [[ -x "$MOLMO_HEAL_SCRIPT" ]]; then
    timeout "$PREFLIGHT_SCRIPT_TIMEOUT_SECONDS" \
      env WAIT_SECONDS="$MOLMO_HEAL_WAIT_SECONDS" "$MOLMO_HEAL_SCRIPT"
  else
    echo "[warn] Molmo heal script not found or not executable: $MOLMO_HEAL_SCRIPT"
  fi
fi

# Hard preflight gate for tunnel/services.
wait_http_code "planner_18086" "$PLANNER_HEALTHCHECK_URL" \
  "$PREFLIGHT_CURL_RETRIES" "$PREFLIGHT_CURL_SLEEP_SECONDS" \
  -k --resolve api.deepseek.com:18086:127.0.0.1

wait_http_code "vdm_18447" "https://open.xiaojingai.com:18447/v1/models" \
  "$PREFLIGHT_CURL_RETRIES" "$PREFLIGHT_CURL_SLEEP_SECONDS" \
  -k --resolve open.xiaojingai.com:18447:127.0.0.1

wait_http_code "sam3_8114" "http://127.0.0.1:8114/docs" \
  "$PREFLIGHT_CURL_RETRIES" "$PREFLIGHT_CURL_SLEEP_SECONDS"

if [[ "$REQUIRE_MOLMO_READY" == "1" ]]; then
  wait_http_code "molmo_8122" "http://127.0.0.1:8122/v1/models" \
    "$PREFLIGHT_CURL_RETRIES" "$PREFLIGHT_CURL_SLEEP_SECONDS"
fi

launch_cmd=(
  ./.venv/bin/python capx/envs/launch.py
  --api-bash-file "$PROVIDER_GEN"
  --config-path "$CONFIG_PATH"
  --output-dir "$OUT_DIR"
  --total-trials "$TOTAL_TRIALS"
  --num-workers "$NUM_WORKERS"
  --enable-eap-rollback True
  --enable-eap-recovery True
  --use-img-differencing True
  --temperature "$PLANNER_TEMPERATURE"
)

if [[ "$ENABLE_PARALLEL_ENSEMBLE" == "1" ]]; then
  launch_cmd+=(--use-parallel-ensemble True)
  if [[ "$ENABLE_MULTIMODEL_ENSEMBLE" == "1" ]]; then
    launch_cmd+=(--use-multimodel True)
  else
    launch_cmd+=(--use-multimodel False)
  fi
fi

nohup "${launch_cmd[@]}" > "$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" > "$PID_PATH"

if [[ "$AUTO_EXPORT_SHARDS" == "1" ]]; then
  AUTO_EXPORT_LOG_PATH="outputs/detached_logs/${RUN_ID}.export_shards.log"
  AUTO_EXPORT_PID_PATH="outputs/detached_pids/${RUN_ID}.export_shards.pid"
  AUTO_EXPORT_WORKER_PATH="outputs/detached_pids/${RUN_ID}.export_shards.worker.sh"

  cat > "$AUTO_EXPORT_WORKER_PATH" <<EOF
#!/usr/bin/env bash
set -uo pipefail

cd '$ROOT_DIR'
PARENT_PID='$PID'
RUN_ID='$RUN_ID'
OUT_DIR='$OUT_DIR'
OUTPUT_ROOT='$AUTO_EXPORT_OUTPUT_ROOT'
INTERVAL='$AUTO_EXPORT_INTERVAL_S'
PYTHON_BIN='./.venv/bin/python'
LOG_PATH='$AUTO_EXPORT_LOG_PATH'
STATUS_CHECK_SCRIPT='scripts/check_collect_run_status.sh'
DISK_GUARD_ENABLED='$DISK_GUARD_ENABLED'
DISK_GUARD_MIN_FREE_GB='$DISK_GUARD_MIN_FREE_GB'

resolve_out_dir() {
  if [ -d "\$OUT_DIR" ]; then
    printf '%s\n' "\$OUT_DIR"
    return 0
  fi
  local parent base candidate
  parent="\$(dirname "\$OUT_DIR")"
  base="\$(basename "\$OUT_DIR")"
  candidate="\$(find "\$parent" -maxdepth 3 -type d -name "\$base" 2>/dev/null | head -n 1 || true)"
  if [ -n "\$candidate" ] && [ -d "\$candidate" ]; then
    printf '%s\n' "\$candidate"
    return 0
  fi
  printf '%s\n' "\$OUT_DIR"
}

run_export_once() {
  local actual_out_dir
  actual_out_dir="\$(resolve_out_dir)"
  if find "\$actual_out_dir" -path '*/transition_dataset/data.pkl.gz' -type f -print -quit | grep -q .; then
    echo "[\$(date '+%F %T')] export-shards: detected transitions under \$actual_out_dir" >> "\$LOG_PATH"
    "\$PYTHON_BIN" skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py export-shards \\
      --repo-root . \\
      --input-root "\$actual_out_dir" \\
      --output-root "\$OUTPUT_ROOT" \\
      --cleanup-transition-dataset \\
      >> "\$LOG_PATH" 2>&1 || true
  fi
}

cleanup_failed_attempt_artifacts() {
  local removed=0
  local actual_out_dir
  actual_out_dir="\$(resolve_out_dir)"
  while IFS= read -r summary; do
    local attempt_dir
    attempt_dir="\$(dirname "\$summary")"
    if ! grep -q 'Task Completed:[[:space:]]*True' "\$summary" 2>/dev/null; then
      for p in \
        "\$attempt_dir/all_responses.json" \
        "\$attempt_dir/transition_dataset" \
        "\$attempt_dir/trial_video.mp4" \
        "\$attempt_dir/rollout.mp4" \
        "\$attempt_dir/rgb_rollout.mp4"; do
        if [ -e "\$p" ]; then
          rm -rf "\$p" && removed=1 || true
        fi
      done
    fi
  done < <(find "\$actual_out_dir" -type f -name summary.txt 2>/dev/null)

  if [ "\$removed" -eq 1 ]; then
    echo "[\$(date '+%F %T')] disk_guard: cleaned failed-attempt artifacts" >> "\$LOG_PATH"
  fi
}

disk_guard_cleanup_if_needed() {
  [ "\$DISK_GUARD_ENABLED" = "1" ] || return 0

  local avail_gb
  local actual_out_dir
  actual_out_dir="\$(resolve_out_dir)"
  avail_gb="\$(df -BG "\$actual_out_dir" 2>/dev/null | awk 'NR==2 {gsub(/G/, \"\", \$4); print \$4}')"
  if [ -z "\$avail_gb" ]; then
    return 0
  fi

  if [ "\$avail_gb" -lt "\$DISK_GUARD_MIN_FREE_GB" ]; then
    echo "[\$(date '+%F %T')] disk_guard: low space \${avail_gb}G < \${DISK_GUARD_MIN_FREE_GB}G, start cleanup" >> "\$LOG_PATH"
    cleanup_failed_attempt_artifacts
    run_export_once
  fi
}

emit_final_status_summary() {
  if [ ! -x "\$STATUS_CHECK_SCRIPT" ]; then
    echo "[\$(date '+%F %T')] final-status: checker not found (\$STATUS_CHECK_SCRIPT), skip" >> "\$LOG_PATH"
    return 0
  fi

  echo "[\$(date '+%F %T')] final-status: begin (run_id=\$RUN_ID)" >> "\$LOG_PATH"
  "\$STATUS_CHECK_SCRIPT" "\$RUN_ID" "$ROOT_DIR" >> "\$LOG_PATH" 2>&1 || true
  echo "[\$(date '+%F %T')] final-status: end (run_id=\$RUN_ID)" >> "\$LOG_PATH"
}

echo "[\$(date '+%F %T')] export-shards sidecar started (parent pid=\$PARENT_PID, out_dir=\$(resolve_out_dir))" >> "\$LOG_PATH"
while kill -0 "\$PARENT_PID" 2>/dev/null; do
  run_export_once || true
  disk_guard_cleanup_if_needed || true
  sleep "\$INTERVAL"
done
run_export_once || true
disk_guard_cleanup_if_needed || true
emit_final_status_summary || true
echo "[\$(date '+%F %T')] export-shards sidecar exited (parent pid ended)" >> "\$LOG_PATH"
EOF

  chmod +x "$AUTO_EXPORT_WORKER_PATH"
  nohup bash "$AUTO_EXPORT_WORKER_PATH" >/dev/null 2>&1 &
  AUTO_EXPORT_PID=$!
  echo "$AUTO_EXPORT_PID" > "$AUTO_EXPORT_PID_PATH"
fi

echo "RUN_ID=$RUN_ID"
echo "PID=$PID"
echo "LOG=$LOG_PATH"
echo "OUTPUT=$OUT_DIR"
echo "PROVIDER=$PROVIDER_GEN"
echo "VDM_ROUTE=$VDM_ROUTE_URL"
echo "TRIAL_TIMEOUT_SECONDS=$TRIAL_TIMEOUT_SECONDS"
echo "PLANNER_TEMPERATURE=$PLANNER_TEMPERATURE"
echo "ENSEMBLE_PARALLEL=$ENABLE_PARALLEL_ENSEMBLE"
echo "ENSEMBLE_MULTIMODEL=$ENABLE_MULTIMODEL_ENSEMBLE"
echo "ENSEMBLE_TEMPS=$SINGLE_MODEL_ENSEMBLE_TEMPS"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE"
echo "MESA_GL_VERSION_OVERRIDE=$MESA_GL_VERSION_OVERRIDE"
echo "ROBOT_DESCRIPTIONS_CACHE=$ROBOT_DESCRIPTIONS_CACHE"
echo "AUTO_EXPORT_INTERVAL_S=$AUTO_EXPORT_INTERVAL_S"
echo "DISK_GUARD_ENABLED=$DISK_GUARD_ENABLED"
echo "DISK_GUARD_MIN_FREE_GB=$DISK_GUARD_MIN_FREE_GB"
if [[ "$AUTO_EXPORT_SHARDS" == "1" ]]; then
  echo "EXPORT_SHARDS_SIDE_CAR_PID=$AUTO_EXPORT_PID"
  echo "EXPORT_SHARDS_SIDE_CAR_LOG=$AUTO_EXPORT_LOG_PATH"
  echo "EXPORT_SHARDS_OUTPUT_ROOT=$AUTO_EXPORT_OUTPUT_ROOT"
fi
