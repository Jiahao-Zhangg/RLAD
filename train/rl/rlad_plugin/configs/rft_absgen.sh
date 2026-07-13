# rft_absgen.sh — offline RFT of pi_abs (method_spec §4.2: "batched offline RL via RFT").
# RFT = rejection fine-tuning = SFT on the HIGH-r_sol abstractions kept by absgen_score.py
# (Eq.1 filter). Initialized from the SFT-warmstarted pi_abs (NOT base). Same SFT machinery
# (sft_rollout + sft_loss); sourced by jobs/sft_launch.sh.
#
# Prereq: SFT pi_abs done + converted to HF at runs/sft_absgen/hf (convert_hf.sbatch), and
# train_absgen_rft.jsonl built (absgen_score.py build-rft).

ARM=sft_absgen_rft

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/common.sh"

# Init from the SFT-warmstarted pi_abs (override common's base --hf-checkpoint); fresh dirs.
# miles exports HF per-iteration under hf/iter_N — resolve the LATEST (not the parent dir).
SFT_ABSGEN_HF=$(ls -d ${RUNS}/sft_absgen/hf/iter_* 2>/dev/null | sort -V | tail -1)
[[ -f "${SFT_ABSGEN_HF}/config.json" ]] || { echo "ERROR: no SFT pi_abs HF at ${SFT_ABSGEN_HF}" >&2; exit 1; }
CKPT_ARGS=(
   --hf-checkpoint ${SFT_ABSGEN_HF}
   --ref-load ${RUNS}/qwen3_1p7b_torch_dist
   --load ${RUNS}/${ARM}/ckpts
   --save ${RUNS}/${ARM}/ckpts
   --save-interval 50
)
# NOTE: intentionally NO --save-hf (unlike common.sh). RFT is short (~38 steps) so an in-training
# save-hf may never fire, and F001 flags it as unreliable anyway — export the final Megatron ckpt
# via the offline path instead, which also yields a clean origin tokenizer (no extra_special_tokens bug):
#   jobs/sbatch.sh --export=ALL,CKPT_DIR=$PWD/runs/sft_absgen_rft/ckpts,OUT_DIR=$PWD/runs/sft_absgen_rft/hf/iter_<N>,ITER=<N> jobs/convert_hf.sbatch

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data ${DATA}/train_absgen_rft.jsonl     # high-r_sol abstractions (messages)
   --input-key messages
   --loss-mask-type qwen3
   --rollout-shuffle
   --num-epoch ${RFT_EPOCHS:-3}                       # RFT epochs (A-rft; paper unspecified)
   --rollout-batch-size 128
   --global-batch-size 128
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

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
