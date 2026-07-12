#!/bin/bash
#SBATCH --job-name=rlad-eval
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
# eval.sbatch — vLLM pass@1 eval of an HF checkpoint (w/o-abstraction) on math
# benchmarks. Runs in the HOST conda env (vLLM); no container.
#
# Submit (G1 base harness gate, AIME2025, n=32, 32K budget):
#   sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-1.7B,BENCHMARKS=aime25,N_SAMPLES=32,MAX_TOKENS=32768,OUT_DIR=$PWD/runs/eval/base_qwen3_1p7b eval.sbatch
# Probe:  add N_SAMPLES=4
# For RLAD w/-abstraction eval use eval_rlad.sbatch (separate; needs pi_abs).

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
RUNS=${RLAD_RUNS}

: "${MODEL_PATH:?submit with --export=ALL,MODEL_PATH=<hf model dir or HF id>}"
BENCHMARKS=${BENCHMARKS:-"aime25"}
BENCHMARKS="${BENCHMARKS//[: ]/,}"   # Slurm --export splits on commas; accept :/space too
N_SAMPLES=${N_SAMPLES:-32}
MAX_TOKENS=${MAX_TOKENS:-32768}      # paper Table-2 eval budget (max extrapolation len)
NUM_SHARDS=${NUM_SHARDS:-8}
OUT_DIR=${OUT_DIR:-${RUNS}/eval/$(basename "${MODEL_PATH}")}

source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" ${RLAD_CONDA_ENV}
# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_DIR}" "${RLAD_LOGS}"

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Model:      ${MODEL_PATH}"
echo "Benchmarks: ${BENCHMARKS} (n=${N_SAMPLES}, max_tokens=${MAX_TOKENS})"
echo "Output:     ${OUT_DIR}"
echo "Env:        $(which python)"
echo "===================================================================="

# Trim Megatron vocab padding from local HF exports (idempotent; HF-hub ids skipped).
if [[ -d "${MODEL_PATH}" ]]; then
    python "${CODE}/eval/trim_vocab_padding.py" "${MODEL_PATH}"
fi

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=${SHARD} python "${CODE}/eval/vllm_eval.py" \
        --model-path "${MODEL_PATH}" \
        --benchmarks-dir "${RLAD_DATA}/benchmarks" \
        --output-dir "${OUT_DIR}" \
        --benchmarks "${BENCHMARKS}" \
        --n-samples "${N_SAMPLES}" \
        --max-tokens "${MAX_TOKENS}" \
        --num-shards "${NUM_SHARDS}" --shard-id "${SHARD}" \
        > "${OUT_DIR}/shard${SHARD}.log" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done
[[ ${FAIL} -ne 0 ]] && echo "WARNING: a shard failed — job is resumable, resubmit to continue" >&2

python "${CODE}/eval/vllm_eval.py" \
    --model-path "${MODEL_PATH}" \
    --benchmarks-dir "${RLAD_DATA}/benchmarks" \
    --output-dir "${OUT_DIR}" \
    --benchmarks "${BENCHMARKS}" \
    --n-samples "${N_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --summarize-only

[[ ${FAIL} -eq 0 ]] || exit 1
echo "Eval job ${SLURM_JOB_ID} finished. Results in ${OUT_DIR}"
