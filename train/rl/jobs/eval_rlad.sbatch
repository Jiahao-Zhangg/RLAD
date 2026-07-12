#!/bin/bash
#SBATCH --job-name=rlad-eval-rlad
#SBATCH --account=CHANGE_ME_slurm_account
#SBATCH --partition=CHANGE_ME_gpu_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#
# eval_rlad.sbatch — DUAL eval (PRIMARY metric): w/o-abs, w/abs(avg over K), w/abs(best).
# pi_abs proposes K abstractions; pi_sol solves. Three resumable stages (gen-abs / solve /
# summarize). Host vllm env. ABSGEN_HF is the SHARED (RLAD-trained) abstraction generator
# used for ALL solgen evals (base/DAPO/RLAD) to isolate solution-generator utilization
# (A-eval-absgen). Resumable: resubmit (singleton) to continue a timed-out solve.
#
# Submit (RLAD):
#   sbatch --export=ALL,ABSGEN_HF=$PWD/runs/sft_absgen_rft/hf,SOLGEN_HF=$PWD/runs/dapo_solgen_rlad/hf_iter100,OUT=$PWD/runs/eval/rlad_iter100,BENCHMARK=aime25 jobs/eval_rlad.sbatch
# Submit (baseline w/abs):  SOLGEN_HF=$PWD/runs/dapo_baseline/hf_iter100, OUT=.../eval/dapo_baseline_wabs
# Submit (base w/abs):      SOLGEN_HF=Qwen/Qwen3-1.7B, OUT=.../eval/base_wabs

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
: "${SOLGEN_HF:?set SOLGEN_HF (pi_sol HF dir or hub id)}"; : "${OUT:?set OUT}"
# MODE=dual (default): gen-abs(pi_abs) -> solve(pi_sol w/o & w/abs) -> summarize (needs ABSGEN_HF).
# MODE=joint (RLAD-Joint native): single model, one combined-prompt generation per sample -> summarize.
MODE=${MODE:-dual}
[[ "${MODE}" == "joint" ]] || : "${ABSGEN_HF:?set ABSGEN_HF (pi_abs HF dir) for MODE=dual}"
ABSGEN_HF=${ABSGEN_HF:-}
BENCHMARK=${BENCHMARK:-aime25}; N=${N:-32}; K=${K:-4}; MAX_TOKENS=${MAX_TOKENS:-32768}

source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" ${RLAD_CONDA_ENV}
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
# trim Megatron vocab padding on local HF dirs before vLLM (idempotent)
for M in "${ABSGEN_HF}" "${SOLGEN_HF}"; do [[ -d "$M" ]] && python "${CODE}/eval/trim_vocab_padding.py" "$M" || true; done

cd "${CODE}/eval"
NSHARDS=${NSHARDS:-8}; CHUNK=${CHUNK:-8}
STAGE=solve; [[ "${MODE}" == "joint" ]] && STAGE=joint
if [[ "${MODE}" != "joint" ]]; then
  echo "=== gen-abs (pi_abs=${ABSGEN_HF}) ==="
  CUDA_VISIBLE_DEVICES=0 python eval_rlad.py gen-abs   --absgen-hf "${ABSGEN_HF}" --benchmark "${BENCHMARK}" --out "${OUT}" --k "${K}"
fi
echo "=== ${STAGE} (pi_sol=${SOLGEN_HF}), ${NSHARDS}-way data-parallel ==="
# one process per GPU; per-shard files + skip-done => resumable across cancels/timeouts
pids=()
for s in $(seq 0 $((NSHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$s python eval_rlad.py ${STAGE} --solgen-hf "${SOLGEN_HF}" --benchmark "${BENCHMARK}" \
    --out "${OUT}" --n "${N}" --max-tokens "${MAX_TOKENS}" --chunk "${CHUNK}" --shard "$s" --num-shards "${NSHARDS}" \
    > "${RLAD_LOGS}/evalrlad_${STAGE}_shard${s}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "${STAGE} shards done (rc=$rc)"
echo "=== summarize ==="
python eval_rlad.py summarize --benchmark "${BENCHMARK}" --out "${OUT}"
echo "eval_rlad job ${SLURM_JOB_ID} done -> ${OUT}/summary.json"
