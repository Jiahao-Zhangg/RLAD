#!/bin/bash
#SBATCH --job-name=rlad-warmstart
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
# warmstart.sbatch — build the pi_abs SFT corpus via local strong-model abstraction
# generation (rlad_plugin.warmstart_gen). Host vllm env (no container).
# Submit (smoke): sbatch --export=ALL,LIMIT=16 jobs/warmstart.sbatch
#        (full):  sbatch jobs/warmstart.sbatch

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" ${RLAD_CONDA_ENV}
# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
LIMIT=${LIMIT:-0}; K=${K:-2}
cd "${CODE}"
# single GPU (tp=1 in script); restrict to GPU 0 of the exclusive node
CUDA_VISIBLE_DEVICES=0 python -m rlad_plugin.warmstart_gen --k "${K}" --limit "${LIMIT}" \
    ${OUT:+--out "${OUT}"}
echo "warmstart job ${SLURM_JOB_ID} done."
