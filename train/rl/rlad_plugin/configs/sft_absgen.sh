# sft_absgen.sh — SFT warmstart of the abstraction generator pi_abs (method_spec §4.2:
# 5 epochs on the seed corpus). From base Qwen3-1.7B. Sourced by jobs/sft_launch.sh.
# SFT uses miles' sft_rollout (no generation) + sft_loss; structurally unlike the RL arms.

ARM=sft_absgen

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"   # MODEL_ARGS (qwen3-1.7B), CKPT_ARGS, MISC_ARGS, WANDB_ARGS, paths

# SFT saves less often than RL (≈per-epoch); keep base init + per-arm dirs from common,
# just relax save-interval.
CKPT_ARGS+=(--save-interval 50)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data ${DATA}/train_absgen_sft.jsonl     # {messages:[user,assistant]} from warmstart_gen
   --input-key messages                              # sft_rollout reads sample.prompt as a chat-message list
   --loss-mask-type qwen3                             # mask loss to the assistant (cheatsheet) turn
   # rollout_global_dataset defaults True (flag is --disable-...); needed by sft_rollout+num-epoch, do NOT pass it
   --rollout-shuffle
   --num-epoch ${SFT_EPOCHS:-5}
   --rollout-batch-size ${SFT_BATCH:-128}
   --global-batch-size ${SFT_BATCH:-128}
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# SFT optimizer: lr 1e-5 (miles SFT default; paper does not give pi_abs SFT lr — A-sftlr).
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

# SFT prompts are short (problem + cheatsheet label); TP1 + sequence-parallel is enough
# and faster than the RL TP2+CP2. Override common's PERF_ARGS.
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --bf16
)
