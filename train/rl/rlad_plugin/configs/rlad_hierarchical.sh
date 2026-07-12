# rlad_hierarchical.sh — INT-002 extension RLAD-Hierarchical: single-model on-policy hierarchical rollout
# (the paper's Eq. 3 realized fully on-policy, one RL run, NO offline scoring).
# Each of the n=N_HINTS generate() calls emits 1 hint + m=M_SOLS solutions (custom generate
# rlad_plugin.rollout_rlad.generate_hierarchical); rlad_plugin.reward_post.hierarchical_reward_post gives each
# solution GRPO-normalized correctness (over the prompt's n*m solutions) and each hint the
# group-normalized MEAN reward of ITS m solutions. Trains on BARE problems (hints are on-policy),
# so PROMPT_DATA = train_curriculum.jsonl (rows carry the raw problem in metadata.problem).
# Init = base Qwen3-1.7B (common.sh). Sourced by jobs/launch_train.sh via ARM_CONFIG.
#
# Launch: jobs/chain.sh configs/rlad_hierarchical.sh 3 rlad-rlad_hierarchical
#   with PROMPT_DATA=$DATA/train_curriculum.jsonl NUM_EPOCH=2.

ARM=rlad_hierarchical${RUN_TAG:+_${RUN_TAG}}

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

N_HINTS=${N_HINTS:-4}                 # n = hints/prompt = n_samples_per_prompt (one hint per generate() call)
M_SOLS=${M_SOLS:-4}                   # m = solutions/hint
ROLLOUT_BATCH=${ROLLOUT_BATCH:-32}
HINT_MAX_TOKENS=${HINT_MAX_TOKENS:-1024}
# Per-prompt training samples = n*(1+m). global-batch MUST be a multiple of that block, else the
# rollout trim (data[:trim_len], rollout_data_conversion.py) can split a qid group mid-normalize.
# Default = ROLLOUT_BATCH * n * (1+m) (= 32*4*5 = 640) => whole rollout is one optimizer step.
_GBS_DEFAULT=$(( ROLLOUT_BATCH * N_HINTS * (1 + M_SOLS) ))

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA:-${DATA}/train_curriculum.jsonl}
   "${ROLLOUT_COMMON[@]}"
   --rollout-max-response-len 16384
   --rollout-batch-size ${ROLLOUT_BATCH}
   --n-samples-per-prompt ${N_HINTS}
   --global-batch-size ${GLOBAL_BATCH:-${_GBS_DEFAULT}}
   --num-epoch ${NUM_EPOCH:-1}
)

# m (M_SOLS) + hint cap reach generate_hierarchical via ENV (RLAD_SOLUTIONS_PER_HINT/RLAD_HINT_MAX_TOKENS),
# forwarded into the ray-worker RUNTIME_ENV_JSON by launch_train.sh (M_SOLS/HINT_MAX_TOKENS are read
# there). NOT via CLI: miles wires a custom generate fn's add_arguments only under the experimental
# rollout refactor (OFF here), so --solutions-per-hint would be rejected. Export so launch_train.sh sees them.
export M_SOLS HINT_MAX_TOKENS
CUSTOM_ARGS=(
   --custom-generate-function-path rlad_plugin.rollout_rlad.generate_hierarchical
   --custom-reward-post-process-path rlad_plugin.reward_post.hierarchical_reward_post
)
