#!/bin/bash
#SBATCH --job-name=rlad-eval-rlad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# eval_rlad.sbatch — DUAL eval (PRIMARY metric): w/o-abs, w/abs(avg over K), w/abs(best).
# pi_abs proposes K abstractions; pi_sol solves. Three resumable stages (gen-abs / solve /
# summarize). Host vllm env. ABSGEN_HF is the SHARED (RLAD-trained) abstraction generator
# used for ALL solgen evals (base/DAPO/RLAD) to isolate solution-generator utilization
# (A-eval-absgen). Resumable: resubmit (singleton) to continue a timed-out solve.
#
# Submit (RLAD):
#   rlad_inference_sbatch --export=ALL,ABSGEN_HF=$PWD/runs/sft_absgen_rft/hf,SOLGEN_HF=$PWD/runs/dapo_solgen_rlad/hf_iter100,OUT=$PWD/runs/eval/rlad_iter100,BENCHMARK=aime25 jobs/eval_rlad.sbatch
# Submit (baseline w/abs):  SOLGEN_HF=$PWD/runs/dapo_baseline/hf_iter100, OUT=.../eval/dapo_baseline_wabs
# Submit (base w/abs):      SOLGEN_HF=Qwen/Qwen3-1.7B, OUT=.../eval/base_wabs

set -euo pipefail
source "${RLAD_HOME:-$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/..}/jobs/cluster_env.sh"
CODE=${RLAD_HOME}
: "${SOLGEN_HF:?set SOLGEN_HF (pi_sol HF dir or hub id)}"; : "${OUT:?set OUT}"
# MODE=dual (default): gen-abs(pi_abs) -> solve(pi_sol w/o & w/abs) -> summarize (needs ABSGEN_HF).
# MODE=joint (RLAD-Joint native): single model, one combined-prompt generation per sample -> summarize.
MODE=${MODE:-dual}
[[ "${MODE}" == "joint" ]] || : "${ABSGEN_HF:?set ABSGEN_HF (pi_abs HF dir) for MODE=dual}"
ABSGEN_HF=${ABSGEN_HF:-}
BENCHMARK=${BENCHMARK:-aime25}; N=${N:-32}; K=${K:-4}; MAX_TOKENS=${MAX_TOKENS:-32768}
ABS_MAX_TOKENS=${ABS_MAX_TOKENS:-1024}; TEMPERATURE=${TEMPERATURE:-0.6}; TOP_P=${TOP_P:-0.95}
SEED=${SEED:-1234}; SKIP_WOABS=${SKIP_WOABS:-0}
[[ "${SKIP_WOABS}" == 0 || "${SKIP_WOABS}" == 1 ]] || {
  echo "ERROR: SKIP_WOABS must be 0 or 1" >&2; exit 2;
}

rlad_activate_conda
# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# redirect compile caches off a possibly small $HOME (quota-exceeded kills vLLM engine init)
CACHE_ROOT=${CACHE_ROOT:-$HOME/.cache/rlad_compile}
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}" "${RLAD_LOGS}"

cd "${CODE}/eval"
NSHARDS=${NSHARDS:-$(( ${SLURM_JOB_NUM_NODES:-1} * RLAD_GPUS_PER_NODE ))}; CHUNK=${CHUNK:-8}
STAGE=solve; [[ "${MODE}" == "joint" ]] && STAGE=joint
if [[ "${MODE}" != "joint" ]]; then
  echo "=== gen-abs (pi_abs=${ABSGEN_HF}), ${NSHARDS}-way data parallel ==="
  "${CODE}/jobs/run_gpu_shards.sh" \
    "${NSHARDS}" "${RLAD_GPUS_PER_NODE}" --shard --num-shards \
    "${RLAD_LOGS}" evalrlad_genabs_shard \
    python eval_rlad.py gen-abs --absgen-hf "${ABSGEN_HF}" \
      --benchmark "${BENCHMARK}" --out "${OUT}" --k "${K}" \
      --abs-max-tokens "${ABS_MAX_TOKENS}" --seed "${SEED}" --chunk "${CHUNK}"
fi
echo "=== ${STAGE} (pi_sol=${SOLGEN_HF}), ${NSHARDS}-way data-parallel ==="
SOLVE_ARGS=(
  --solgen-hf "${SOLGEN_HF}"
  --benchmark "${BENCHMARK}"
  --out "${OUT}"
  --n "${N}"
  --k "${K}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
  --chunk "${CHUNK}"
)
[[ "${SKIP_WOABS}" == 0 ]] || SOLVE_ARGS+=(--skip-woabs)
"${CODE}/jobs/run_gpu_shards.sh" \
  "${NSHARDS}" "${RLAD_GPUS_PER_NODE}" --shard --num-shards \
  "${RLAD_LOGS}" "evalrlad_${STAGE}_shard" \
  python eval_rlad.py "${STAGE}" "${SOLVE_ARGS[@]}"
echo "=== summarize ==="
SUMMARY_ARGS=(--benchmark "${BENCHMARK}" --out "${OUT}" --mode "${MODE}" --n "${N}" --k "${K}")
[[ "${SKIP_WOABS}" == 0 ]] || SUMMARY_ARGS+=(--skip-woabs)
python eval_rlad.py summarize "${SUMMARY_ARGS[@]}"
echo "eval_rlad job ${SLURM_JOB_ID} done -> ${OUT}/summary.json"
