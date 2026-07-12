# dapo_baseline.sh — Baseline "+DAPO": Qwen3-1.7B RL on the SAME curriculum prompts,
# NO abstractions (the paper's Table-2 baseline). Stock per-sample rollout; DAPO loss
# (asymmetric clip, token-level norm, low-var KL) from common.sh DAPO_ARGS.
# Sourced by jobs/launch_train.sh via ARM_CONFIG.
#
# Curriculum: train_curriculum.jsonl = easy(2481) then medium(773), shuffle OFF so the
# easy stage is seen first (A-curriculum-impl). Uniform 16K response budget.
#
# Batch (A-batchid, provisional — finalize from smoke step-time): rollout-batch-size 32 ×
# n-samples-per-prompt 16 = 512 gen/step = global-batch 512 => 1 optimizer step/rollout.
# 3254 prompts / 32 ≈ 102 rollouts/epoch. NUM_EPOCH (env) sets length.

ARM=dapo_baseline${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA:-${DATA}/train_curriculum.jsonl}   # POPE-hard: PROMPT_DATA=$DATA/train_pope_hard.jsonl RUN_TAG=pope
   "${ROLLOUT_COMMON[@]}"
   --rollout-max-response-len 16384
   --rollout-batch-size ${ROLLOUT_BATCH:-32}
   --n-samples-per-prompt 16
   --global-batch-size ${GLOBAL_BATCH:-512}
   --num-epoch ${NUM_EPOCH:-1}
)

# Baseline: stock generation, no abstraction injection, no reward post-process.
CUSTOM_ARGS=()
