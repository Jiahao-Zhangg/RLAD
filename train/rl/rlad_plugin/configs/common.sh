# common.sh — shared miles arg arrays for RLAD (sourceable, bash).
#
# NOT executable alone. Sourced by per-arm configs (dapo_baseline.sh, dapo_solgen_rlad.sh,
# sft_absgen.sh, rlad_joint.sh, rlad_hierarchical.sh), which MUST set ARM before sourcing
# (CKPT/WANDB keyed on ARM).
# Flags verified in code/miles/miles/utils/arguments.py (line refs in implementation_plan.md).

: "${ARM:?common.sh requires ARM set before sourcing (source a per-arm config instead)}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../jobs" && pwd)/cluster_env.sh"
CODE="${RLAD_HOME}"
RUNS="${RLAD_RUNS}"
DATA="${RLAD_DATA}"
BASE_HF_MODEL=Qwen/Qwen3-1.7B

mkdir -p "${RUNS}/${ARM}/ckpts" "${RUNS}/${ARM}/hf"

# MODEL_ARGS — Qwen3-1.7B (28L, hidden 2048, GQA 16/8, qk-layernorm, vocab 151936).
# Qwen3-1.7B TIES embeddings (config tie_word_embeddings=true) -> NO --untie (unlike
# an R1-Distill-style checkpoint).
source "${MILES_DIR}/scripts/models/qwen3-1.7B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF_MODEL}
   --ref-load ${RUNS}/qwen3_1p7b_torch_dist
   --load ${RUNS}/${ARM}/ckpts
   --save ${RUNS}/${ARM}/ckpts
   --save-interval 5
   --save-hf "${RUNS}/${ARM}/hf/iter_{rollout_id}"   # F001: in-training save-hf unreliable under TP+CP -> ALWAYS offline-convert + G3 before trusting evals
)

# Rollout knobs shared across arms. Per-arm config sets --prompt-data,
# --rollout-max-response-len (8K easy / 16K medium), batch sizes, CUSTOM_ARGS.
# Prompts are pre-templated offline by data_prep (NO --apply-chat-template).
# rollout-temperature 0.6 = paper train temp (Appendix Table 4).
ROLLOUT_COMMON=(
   --input-key input
   --label-key label
   --metadata-key metadata
   --rollout-seed 42
   # NOTE: --rollout-shuffle is set per-arm. Curriculum arms (baseline, RLAD) leave it
   # OFF and order the data easy->medium (train_curriculum.jsonl) so the easy stage is
   # seen first (A-curriculum-impl: single-run ordering at uniform 16K, not 2-phase 8K/16K).
   --rollout-temperature 0.6
   --rollout-max-prompt-len 4096            # paper max-prompt 3072 + cheatsheet headroom
   --custom-rm-path rlad_plugin.reward_math.custom_rm
)

# DAPO (Appendix Table 4): asymmetric clip 0.2/0.5, token-level loss norm
# (--calculate-per-token-loss), low-var KL-as-loss coeff 0.001, entropy 0.001, lr 1e-6.
DAPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.5
   --calculate-per-token-loss
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.001
   --lr 1e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.01
)

# PERF — TP2+CP2+dist-opt+chunked-logprobs (+div_ patch in clone) for 16K-resp OOM (F002).
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --use-distributed-optimizer
   --log-probs-chunk-size 2048
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 19456
   --bf16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project repro-paper003-rlad
      --wandb-group "${ARM}"
   )
fi
