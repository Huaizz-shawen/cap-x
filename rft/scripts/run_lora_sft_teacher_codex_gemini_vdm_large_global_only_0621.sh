#!/usr/bin/env bash
set -euo pipefail

GLOBAL_USER_ROOT="${GLOBAL_USER_ROOT:-/inspire/hdd/global_user/huaizezheng-p-huaizezheng}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${GLOBAL_USER_ROOT}/rl-homework}"
CAPX_ROOT="${CAPX_ROOT:-${GLOBAL_USER_ROOT}/cap-x}"
CAPX_VENV="${CAPX_VENV:-${CAPX_ROOT}/.venv}"
VENDOR_SITE="${VENDOR_SITE:-${GLOBAL_ROOT}/vendor_site}"

TRAIN_SCRIPT="${GLOBAL_ROOT}/scripts/train_lora_sft_warmup.py"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
SFT_ROOT="${SFT_ROOT:-${GLOBAL_ROOT}/data/sft_lift_teacher_codex_gemini_vdm_large_0621}"
TRAIN_JSONL="${TRAIN_JSONL:-${SFT_ROOT}/train_all_raw.jsonl}"
OUT_ROOT="${OUT_ROOT:-${GLOBAL_ROOT}/runs/lora_sft_lift_teacher_codex_gemini_vdm_large_0621}"
MERGED_MODEL="${MERGED_MODEL:-${GLOBAL_ROOT}/models/lora_sft_lift_teacher_codex_gemini_vdm_large_0621_merged}"

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
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "ERROR: SFT data missing: ${TRAIN_JSONL}" >&2
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
echo "SFT_ROOT=${SFT_ROOT}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MERGED_MODEL=${MERGED_MODEL}"
nvidia-smi || true

python "${TRAIN_SCRIPT}" \
  --model-path "${BASE_MODEL}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUT_ROOT}" \
  --merged-output-dir "${MERGED_MODEL}" \
  --max-length 1536 \
  --epochs "${SFT_EPOCHS:-8}" \
  --learning-rate "${SFT_LR:-5e-5}" \
  --batch-size "${SFT_BATCH_SIZE:-2}" \
  --grad-accum "${SFT_GRAD_ACCUM:-8}" \
  --lora-r "${SFT_LORA_R:-16}" \
  --lora-alpha "${SFT_LORA_ALPHA:-32}" \
  --lora-dropout "${SFT_LORA_DROPOUT:-0.05}" \
  --save-steps "${SFT_SAVE_STEPS:-64}" \
  --attn-implementation sdpa
