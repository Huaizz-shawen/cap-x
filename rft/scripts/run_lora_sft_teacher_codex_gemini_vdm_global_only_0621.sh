#!/usr/bin/env bash
set -euo pipefail

GLOBAL_USER_ROOT="${GLOBAL_USER_ROOT:-/inspire/hdd/global_user/huaizezheng-p-huaizezheng}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${GLOBAL_USER_ROOT}/rl-homework}"
CAPX_ROOT="${CAPX_ROOT:-${GLOBAL_USER_ROOT}/cap-x}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"
VENDOR_SITE="${VENDOR_SITE:-${GLOBAL_ROOT}/vendor_site}"

TRAIN_SCRIPT="${GLOBAL_ROOT}/scripts/train_lora_sft_warmup.py"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
SFT_ROOT="${GLOBAL_ROOT}/data/sft_lift_teacher_codex_gemini_vdm_pilot_0621"
OUT_ROOT="${GLOBAL_ROOT}/runs/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621"
MERGED_MODEL="${GLOBAL_ROOT}/models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged"

if [[ ! -d "${CAPX_ROOT}" ]]; then
  echo "ERROR: CAPX_ROOT does not exist: ${CAPX_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${CAPX_VENV}/bin/python" ]]; then
  echo "ERROR: CAPX_VENV python is not executable: ${CAPX_VENV}/bin/python" >&2
  echo "Set CAPX_VENV to a venv visible in this workspace." >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script missing: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SFT_ROOT}/train_all_success.jsonl" ]]; then
  echo "ERROR: SFT data missing: ${SFT_ROOT}/train_all_success.jsonl" >&2
  exit 1
fi

cd "${CAPX_ROOT}"
. "${CAPX_VENV}/bin/activate"

export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export XDG_CACHE_HOME="${GLOBAL_ROOT}/cache"
export WANDB_DIR="${GLOBAL_ROOT}/logs/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
if [[ -d "${VENDOR_SITE}" ]]; then
  export PYTHONPATH="${VENDOR_SITE}:${CAPX_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${CAPX_ROOT}:${PYTHONPATH:-}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}" "${GLOBAL_ROOT}/models" "${GLOBAL_ROOT}/logs"
echo "CAPX_ROOT=${CAPX_ROOT}"
echo "CAPX_VENV=${CAPX_VENV}"
echo "VENDOR_SITE=${VENDOR_SITE}"
echo "GLOBAL_ROOT=${GLOBAL_ROOT}"
nvidia-smi || true

python "${TRAIN_SCRIPT}" \
  --model-path "${BASE_MODEL}" \
  --train-jsonl "${SFT_ROOT}/train_all_success.jsonl" \
  --output-dir "${OUT_ROOT}" \
  --merged-output-dir "${MERGED_MODEL}" \
  --max-length 1536 \
  --epochs 4 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --save-steps 16 \
  --attn-implementation sdpa
