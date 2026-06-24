#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng
GLOBAL_ROOT=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework
CAPX_ROOT="${PROJECT_ROOT}/cap-x"

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

DATE=0618
DATA_SOURCE=${DATA_SOURCE:-franka_lift_code_env}
ALGO=${ALGO:-grpo}
GROUP_SIZE=${GROUP_SIZE:-4}
TRAIN_DATASET_SIZE=${TRAIN_DATASET_SIZE:-1024}
VAL_DATASET_SIZE=${VAL_DATASET_SIZE:-64}
MODEL_PATH=${MODEL_PATH:-${GLOBAL_ROOT}/models/Qwen2.5-Coder-7B-Instruct}
DATA_ROOT=${DATA_ROOT:-${GLOBAL_ROOT}/runs/capx_grpo_lift_medium_4h100_global_0618}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
PYROKI_PORT=${PYROKI_PORT:-8116}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-16}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ALGO}_qwen25coder7b_${DATA_SOURCE}_${DATE}_medium_temperature_${TRAIN_TEMPERATURE}_group_size_${GROUP_SIZE}_train${TRAIN_DATASET_SIZE}_epochs${TOTAL_EPOCHS}}

mkdir -p "${GLOBAL_ROOT}/logs" "${HF_HOME}" "${DATA_ROOT}"

if ! python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PYROKI_PORT}), timeout=1); s.close()" 2>/dev/null; then
  echo "Starting PyRoKi IK server on port ${PYROKI_PORT}..."
  CUDA_VISIBLE_DEVICES="" python -m capx.serving.launch_pyroki_server \
    --port "${PYROKI_PORT}" --host 127.0.0.1 \
    > "${GLOBAL_ROOT}/logs/pyroki_train_medium_0618.log" 2>&1 &
  PYROKI_PID=$!
  for _ in $(seq 1 90); do
    if python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PYROKI_PORT}), timeout=1); s.close()" 2>/dev/null; then
      echo "PyRoKi IK server ready (PID ${PYROKI_PID})"
      break
    fi
    sleep 1
  done
  trap "kill ${PYROKI_PID} 2>/dev/null || true" EXIT
fi

if ! python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PYROKI_PORT}), timeout=1); s.close()" 2>/dev/null; then
  echo "ERROR: PyRoKi IK server did not become ready on port ${PYROKI_PORT}" >&2
  tail -n 120 "${GLOBAL_ROOT}/logs/pyroki_train_medium_0618.log" >&2 || true
  exit 1
fi

echo "DATA_ROOT: ${DATA_ROOT}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "N_GPUS: ${N_GPUS}"
nvidia-smi || true

if [[ ! -d "${DATA_ROOT}/train.parquet" && ! -f "${DATA_ROOT}/train.parquet" ]]; then
  echo "Preparing dataset..."
  python -m capx.cli.prepare_verl_dataset \
    --output-dir "${DATA_ROOT}" \
    --train-size "${TRAIN_DATASET_SIZE}" \
    --val-size "${VAL_DATASET_SIZE}" \
    --data-source "${DATA_SOURCE}"
fi

USE_KL_LOSS=false
if [[ "${ALGO}" == "grpo" ]]; then
  USE_KL_LOSS=true
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=${ALGO} \
  data.train_files=${DATA_ROOT}/train.parquet \
  data.val_files=${DATA_ROOT}/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=${GROUP_SIZE} \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_model_len=1280 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=verl_agent_reward/hyrl_franka_reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=[wandb] \
  trainer.project_name=hyrl \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.default_local_dir=${DATA_ROOT}/checkpoints/\${trainer.project_name}/\${trainer.experiment_name} \
  trainer.test_freq=-1 \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.val_before_train=False \
  reward_model.launch_reward_fn_async=True \
  reward_model.reward_manager=prime
