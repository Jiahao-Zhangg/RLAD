#!/bin/bash
# sft_launch.sh — in-container miles SFT launcher (pi_abs warmstart). Run by srun via
# submit_train.sbatch with LAUNCHER=this. SFT path differs from RL: train_async.py +
# miles.rollout.sft_rollout (no sglang generation), loss_type sft_loss. Modeled on
# miles/scripts/run-qwen3-4B-base-sft.sh.
#
# Required env: ARM_CONFIG = abs path to configs/sft_absgen.sh (defines MODEL_ARGS,
# CKPT_ARGS, SFT_ARGS, OPTIMIZER_ARGS, PERF_ARGS, MISC_ARGS, WANDB_ARGS + paths).

pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python; sleep 3; pkill -9 ray; pkill -9 python

set -ex
: "${ARM_CONFIG:?sft_launch.sh requires ARM_CONFIG}"
[[ -f "${ARM_CONFIG}" ]] || { echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2; exit 1; }

# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
source "${ARM_CONFIG}"

[[ -f "${SFT_ARGS[3]}" ]] || true  # prompt-data presence is checked by the config comment
[[ -f "${RUNS}/qwen3_1p7b_torch_dist/latest_checkpointed_iteration.txt" ]] || {
   echo "ERROR: base torch_dist ckpt missing — run prep_megatron_ckpt.sbatch first" >&2; exit 1; }

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd "${MILES_DIR}"
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${MILES_DIR}:${CODE}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
