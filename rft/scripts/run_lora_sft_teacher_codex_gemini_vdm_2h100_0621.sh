#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng
GLOBAL_ROOT=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework
CAPX_ROOT="${PROJECT_ROOT}/cap-x"
TRAIN_SCRIPT="${GLOBAL_ROOT}/scripts/train_lora_sft_warmup.py"
BASE_MODEL="${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
SFT_ROOT="${GLOBAL_ROOT}/data/sft_lift_teacher_codex_gemini_vdm_pilot_0621"
OUT_ROOT="${GLOBAL_ROOT}/runs/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621"
MERGED_MODEL="${GLOBAL_ROOT}/models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged"

cd "${CAPX_ROOT}"
. .venv-rl/bin/activate

export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export XDG_CACHE_HOME="${GLOBAL_ROOT}/cache"
export WANDB_DIR="${GLOBAL_ROOT}/logs/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONPATH="${CAPX_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUT_ROOT}" "${GLOBAL_ROOT}/models" "${GLOBAL_ROOT}/logs"
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
  --save-steps 16
