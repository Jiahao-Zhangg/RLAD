# dapo_solgen_rlad.sh — RLAD solution-generator DAPO (the PROPOSED method's sol-gen RL).
# Identical to dapo_baseline.sh EXCEPT: (1) prompt-data is the abstraction-conditioned+bare
# mixture (train_solgen_rlad.jsonl from solgen_data.py); (2) reward_post masks the no-abs
# prompts (Eq. 3: zero advantage on bare prompts, KL retained). Stock per-sample rollout —
# the abstraction is already in the prompt (A-absinject), so no custom generate fn needed.
# Sourced by jobs/launch_train.sh via ARM_CONFIG.

ARM=dapo_solgen_rlad${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA:-${DATA}/train_solgen_rlad.jsonl}   # POPE-hard: PROMPT_DATA=$DATA/train_solgen_rlad_pope.jsonl RUN_TAG=pope
   "${ROLLOUT_COMMON[@]}"
   --rollout-max-response-len 16384
   --rollout-batch-size ${ROLLOUT_BATCH:-32}
   --n-samples-per-prompt 16
   --global-batch-size ${GLOBAL_BATCH:-512}
   --num-epoch ${NUM_EPOCH:-1}
)

# Reward masking: zero advantage on metadata.has_abstraction==False groups, GRPO-normalize
# abstraction groups. KL (--use-kl-loss in common DAPO_ARGS) still regularizes bare prompts.
CUSTOM_ARGS=(
   --custom-reward-post-process-path rlad_plugin.reward_post.rlad_reward_post
)
