#!/bin/bash
#SBATCH --job-name=rlad-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# eval.sbatch — vLLM pass@1 eval of an HF checkpoint (w/o-abstraction) on math
# benchmarks. Runs in the HOST conda env (vLLM); no container.
#
# Submit (G1 base harness gate, AIME2025, n=32, 32K budget):
#   rlad_inference_sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-1.7B,BENCHMARKS=aime25,N_SAMPLES=32,MAX_TOKENS=32768,OUT_DIR=$PWD/runs/eval/base_qwen3_1p7b jobs/eval.sbatch
# Probe:  add N_SAMPLES=4
# For RLAD w/-abstraction eval use eval_rlad.sbatch (separate; needs pi_abs).

set -euo pipefail

source "${RLAD_HOME:-$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/..}/jobs/cluster_env.sh"
CODE=${RLAD_HOME}
RUNS=${RLAD_RUNS}

: "${MODEL_PATH:?submit with --export=ALL,MODEL_PATH=<hf model dir or HF id>}"
BENCHMARKS=${BENCHMARKS:-"aime25"}
BENCHMARKS="${BENCHMARKS//[: ]/,}"   # Slurm --export splits on commas; accept :/space too
N_SAMPLES=${N_SAMPLES:-32}
MAX_TOKENS=${MAX_TOKENS:-32768}      # paper Table-2 eval budget (max extrapolation len)
NUM_SHARDS=${NUM_SHARDS:-$(( ${SLURM_JOB_NUM_NODES:-1} * RLAD_GPUS_PER_NODE ))}
OUT_DIR=${OUT_DIR:-${RUNS}/eval/$(basename "${MODEL_PATH}")}

rlad_activate_conda
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

if ! "${CODE}/jobs/run_gpu_shards.sh" \
    "${NUM_SHARDS}" "${RLAD_GPUS_PER_NODE}" --shard-id --num-shards \
    "${OUT_DIR}" shard \
    python "${CODE}/eval/vllm_eval.py" \
        --model-path "${MODEL_PATH}" \
        --benchmarks-dir "${RLAD_DATA}/benchmarks" \
        --output-dir "${OUT_DIR}" \
        --benchmarks "${BENCHMARKS}" \
        --n-samples "${N_SAMPLES}" \
        --max-tokens "${MAX_TOKENS}"; then
    echo "WARNING: one or more GPU shards failed; resubmit to resume" >&2
    exit 1
fi

python "${CODE}/eval/vllm_eval.py" \
    --model-path "${MODEL_PATH}" \
    --benchmarks-dir "${RLAD_DATA}/benchmarks" \
    --output-dir "${OUT_DIR}" \
    --benchmarks "${BENCHMARKS}" \
    --n-samples "${N_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --summarize-only

echo "Eval job ${SLURM_JOB_ID} finished. Results in ${OUT_DIR}"
