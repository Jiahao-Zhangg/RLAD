#!/bin/bash
#SBATCH --job-name=rlad-warmstart
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# warmstart.sbatch — build the pi_abs SFT corpus via local strong-model abstraction
# generation (rlad_plugin.warmstart_gen). Host vLLM env, sharded over every
# GPU in the allocation, followed by one deterministic merge (no container).
# Submit (smoke): rlad_inference_sbatch --export=ALL,LIMIT=16 jobs/warmstart.sbatch
#        (full):  rlad_inference_sbatch jobs/warmstart.sbatch

set -euo pipefail
source "${RLAD_HOME:-$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/..}/jobs/cluster_env.sh"
CODE=${RLAD_HOME}
rlad_activate_conda
# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
LIMIT=${LIMIT:-0}; K=${K:-2}
NSHARDS=${NSHARDS:-$(( ${SLURM_JOB_NUM_NODES:-1} * RLAD_GPUS_PER_NODE ))}
GENERATOR=${GENERATOR:-Qwen/Qwen3-4B-Instruct-2507}
GENERATOR_MAX_TOKENS=${GENERATOR_MAX_TOKENS:-4096}
GENERATOR_TEMPERATURE=${GENERATOR_TEMPERATURE:-0.7}
WARMSTART_ARGS=(
    --k "${K}"
    --limit "${LIMIT}"
    --generator "${GENERATOR}"
    --max-tokens "${GENERATOR_MAX_TOKENS}"
    --temperature "${GENERATOR_TEMPERATURE}"
)
[[ -n "${OUT:-}" ]] && WARMSTART_ARGS+=(--out "${OUT}")
cd "${CODE}"
"${CODE}/jobs/run_gpu_shards.sh" \
    "${NSHARDS}" "${RLAD_GPUS_PER_NODE}" --shard --num-shards \
    "${RLAD_LOGS}" warmstart_shard \
    python -m rlad_plugin.warmstart_gen "${WARMSTART_ARGS[@]}"
python -m rlad_plugin.warmstart_gen "${WARMSTART_ARGS[@]}" \
    --num-shards "${NSHARDS}" --merge-shards
echo "warmstart job ${SLURM_JOB_ID} done."
