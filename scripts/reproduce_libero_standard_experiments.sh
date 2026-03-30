#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/reproduce_libero_standard_experiments.sh \
    --server-url http://host:port/v1/chat/completions \
    --api-key your-key

This script reproduces the LIBERO experiments used in the repo exploration:
  1. privileged spatial with the primary model
  2. privileged goal with the primary model
  3. privileged spatial with the comparison model
  4. privileged goal with the comparison model
  5. non-privileged spatial with the primary model
  6. non-privileged goal with the primary model

Options:
  --server-url URL             OpenAI-compatible chat completions endpoint
  --api-key KEY                Bearer token for the endpoint
  --venv PATH                  LIBERO virtualenv to activate (default: .venv-libero)
  --output-root DIR            Root output directory (default: ./outputs/libero_repro)
  --primary-model NAME         Primary model for main runs (default: gpt-5.3-codex)
  --compare-model NAME         Comparison model for privileged runs (default: gpt-5.4)
  --privileged-trials N        Trials for privileged runs (default: 5)
  --non-privileged-trials N    Trials for non-privileged runs (default: 3)
  --num-workers N              Worker count for all runs (default: 1)
  --record-video BOOL          Whether to record video (default: False)
  --skip-privileged            Skip the four privileged runs
  --skip-non-privileged        Skip the two non-privileged runs
  -h, --help                   Show this help

You can also provide configuration via environment variables:
  CAPX_SERVER_URL
  CAPX_API_KEY
  CAPX_LIBERO_VENV
  CAPX_OUTPUT_ROOT
  CAPX_PRIMARY_MODEL
  CAPX_COMPARE_MODEL
  CAPX_PRIVILEGED_TRIALS
  CAPX_NON_PRIVILEGED_TRIALS
  CAPX_NUM_WORKERS
  CAPX_RECORD_VIDEO
EOF
}

SERVER_URL="${CAPX_SERVER_URL:-}"
API_KEY="${CAPX_API_KEY:-}"
VENV_PATH="${CAPX_LIBERO_VENV:-.venv-libero}"
OUTPUT_ROOT="${CAPX_OUTPUT_ROOT:-./outputs/libero_repro}"
PRIMARY_MODEL="${CAPX_PRIMARY_MODEL:-gpt-5.3-codex}"
COMPARE_MODEL="${CAPX_COMPARE_MODEL:-gpt-5.4}"
PRIVILEGED_TRIALS="${CAPX_PRIVILEGED_TRIALS:-5}"
NON_PRIVILEGED_TRIALS="${CAPX_NON_PRIVILEGED_TRIALS:-3}"
NUM_WORKERS="${CAPX_NUM_WORKERS:-1}"
RECORD_VIDEO="${CAPX_RECORD_VIDEO:-False}"
RUN_PRIVILEGED=1
RUN_NON_PRIVILEGED=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --primary-model)
            PRIMARY_MODEL="$2"
            shift 2
            ;;
        --compare-model)
            COMPARE_MODEL="$2"
            shift 2
            ;;
        --privileged-trials)
            PRIVILEGED_TRIALS="$2"
            shift 2
            ;;
        --non-privileged-trials)
            NON_PRIVILEGED_TRIALS="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --record-video)
            RECORD_VIDEO="$2"
            shift 2
            ;;
        --skip-privileged)
            RUN_PRIVILEGED=0
            shift
            ;;
        --skip-non-privileged)
            RUN_NON_PRIVILEGED=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$SERVER_URL" ]]; then
    echo "Missing --server-url or CAPX_SERVER_URL" >&2
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    echo "Missing --api-key or CAPX_API_KEY" >&2
    exit 1
fi

if [[ ! -e "$VENV_PATH/bin/activate" ]]; then
    echo "LIBERO virtualenv not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH/bin/activate"

python - <<'PY'
import importlib
for module_name in ("capx", "libero"):
    importlib.import_module(module_name)
print("Environment check passed: capx + libero import successfully.")
PY

mkdir -p "$OUTPUT_ROOT/logs"
SUMMARY_TSV="$OUTPUT_ROOT/summary.tsv"
printf "label\tmodel\tconfig\ttotal_trials\tcodegen_success\tavg_reward\ttask_completed\telapsed_seconds\tlog_path\n" > "$SUMMARY_TSV"

run_eval() {
    local label="$1"
    local config_path="$2"
    local model="$3"
    local trials="$4"

    local run_output="$OUTPUT_ROOT/$label"
    local log_path="$OUTPUT_ROOT/logs/${label}.log"

    mkdir -p "$run_output"

    echo
    echo "======================================================================"
    echo "Running $label"
    echo "  config: $config_path"
    echo "  model: $model"
    echo "  trials: $trials"
    echo "  output: $run_output"
    echo "======================================================================"

    MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        python capx/envs/launch.py \
        --config-path "$config_path" \
        --server-url "$SERVER_URL" \
        --api-key "$API_KEY" \
        --model "$model" \
        --total-trials "$trials" \
        --num-workers "$NUM_WORKERS" \
        --record-video "$RECORD_VIDEO" \
        --output-dir "$run_output" \
        2>&1 | tee "$log_path"

    python - "$label" "$model" "$config_path" "$log_path" "$SUMMARY_TSV" <<'PY'
import pathlib
import re
import sys

label, model, config_path, log_path, summary_tsv = sys.argv[1:]
text = pathlib.Path(log_path).read_text()

def extract(pattern: str, default: str = "NA") -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default

total_trials = extract(r"Total number of trials:\s*(\d+)")
triple = extract(
    r"Code generation success rate / Average reward / Task completed:\s*\n([^\n]+)"
)
elapsed = extract(r"Elapsed time:\s*([0-9.]+)\s*seconds")

codegen_success = "NA"
avg_reward = "NA"
task_completed = "NA"
if triple != "NA":
    parts = [part.strip() for part in triple.split("/")]
    if len(parts) == 3:
        codegen_success, avg_reward, task_completed = parts

with open(summary_tsv, "a", encoding="utf-8") as handle:
    handle.write(
        "\t".join(
            [
                label,
                model,
                config_path,
                total_trials,
                codegen_success,
                avg_reward,
                task_completed,
                elapsed,
                log_path,
            ]
        )
        + "\n"
    )
PY
}

if [[ "$RUN_PRIVILEGED" -eq 1 ]]; then
    run_eval \
        "privileged_spatial_${PRIMARY_MODEL}" \
        "env_configs/libero/franka_libero_spatial_0_privileged.yaml" \
        "$PRIMARY_MODEL" \
        "$PRIVILEGED_TRIALS"
    run_eval \
        "privileged_goal_${PRIMARY_MODEL}" \
        "env_configs/libero/franka_libero_goal_1_privileged.yaml" \
        "$PRIMARY_MODEL" \
        "$PRIVILEGED_TRIALS"
    run_eval \
        "privileged_spatial_${COMPARE_MODEL}" \
        "env_configs/libero/franka_libero_spatial_0_privileged.yaml" \
        "$COMPARE_MODEL" \
        "$PRIVILEGED_TRIALS"
    run_eval \
        "privileged_goal_${COMPARE_MODEL}" \
        "env_configs/libero/franka_libero_goal_1_privileged.yaml" \
        "$COMPARE_MODEL" \
        "$PRIVILEGED_TRIALS"
fi

if [[ "$RUN_NON_PRIVILEGED" -eq 1 ]]; then
    run_eval \
        "non_privileged_spatial_${PRIMARY_MODEL}" \
        "env_configs/libero/franka_libero_spatial_0.yaml" \
        "$PRIMARY_MODEL" \
        "$NON_PRIVILEGED_TRIALS"
    run_eval \
        "non_privileged_goal_${PRIMARY_MODEL}" \
        "env_configs/libero/franka_libero_goal_1.yaml" \
        "$PRIMARY_MODEL" \
        "$NON_PRIVILEGED_TRIALS"
fi

echo
echo "Summary written to $SUMMARY_TSV"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY_TSV"
else
    cat "$SUMMARY_TSV"
fi
