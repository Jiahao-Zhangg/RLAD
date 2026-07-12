# rlad_joint.sh — INT-002 extension RLAD-Joint: single-model joint-trajectory (hint->solution in
# ONE generation). The model writes a <cheatsheet> then solves in the same thinking trajectory;
# the boxed answer is graded and the whole trajectory gets that reward — plain GRPO over 16
# rollouts. STOCK rollout + STOCK reward norm (no custom generate, no reward-post): the hint
# tokens naturally inherit the trajectory's advantage. Init = base Qwen3-1.7B (common.sh).
#
# PROMPT_DATA must be rendered with templates.render_joint_prompt (the combined instruction),
# NOT the plain solver prompt — build train_joint.jsonl via data_prep (build-joint).
# Sourced by jobs/launch_train.sh via ARM_CONFIG.
#
# Launch (POPE-hard): jobs/chain.sh configs/rlad_joint.sh 3 rlad-rlad_joint_pope
#   with RUN_TAG=pope PROMPT_DATA=$DATA/train_joint.jsonl NUM_EPOCH=2.

ARM=rlad_joint${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA:-${DATA}/train_joint.jsonl}
   "${ROLLOUT_COMMON[@]}"
   --rollout-max-response-len 16384
   --rollout-batch-size ${ROLLOUT_BATCH:-32}
   --n-samples-per-prompt 16
   --global-batch-size ${GLOBAL_BATCH:-512}
   --num-epoch ${NUM_EPOCH:-1}
)

# Stock GRPO: default group-norm over the 16 rollouts, no masking, no custom generate.
CUSTOM_ARGS=()
