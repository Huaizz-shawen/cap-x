#!/usr/bin/env bash
set -euo pipefail

GLOBAL_USER_ROOT="${GLOBAL_USER_ROOT:-/inspire/hdd/global_user/huaizezheng-p-huaizezheng}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${GLOBAL_USER_ROOT}/rl-homework}"
SFT_ROOT="${SFT_ROOT:-${GLOBAL_ROOT}/data/sft_lift_teacher_codex_gemini_vdm_large_0621}"
TRAIN_JSONL="${TRAIN_JSONL:-${SFT_ROOT}/train_all_success_strict_0622.jsonl}"
OUT_ROOT="${OUT_ROOT:-${GLOBAL_ROOT}/runs/lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622}"
MERGED_MODEL="${MERGED_MODEL:-${GLOBAL_ROOT}/models/lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622_merged}"

SFT_EPOCHS="${SFT_EPOCHS:-8}" \
SFT_LR="${SFT_LR:-5e-5}" \
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}" \
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}" \
SFT_LORA_R="${SFT_LORA_R:-16}" \
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}" \
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0.05}" \
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-64}" \
GLOBAL_USER_ROOT="${GLOBAL_USER_ROOT}" \
GLOBAL_ROOT="${GLOBAL_ROOT}" \
SFT_ROOT="${SFT_ROOT}" \
TRAIN_JSONL="${TRAIN_JSONL}" \
OUT_ROOT="${OUT_ROOT}" \
MERGED_MODEL="${MERGED_MODEL}" \
bash "${GLOBAL_ROOT}/scripts/run_lora_sft_teacher_codex_gemini_vdm_large_global_only_0621.sh"
