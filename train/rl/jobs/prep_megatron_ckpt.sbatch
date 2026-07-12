#!/bin/bash
#SBATCH --job-name=rlad-prep-ckpt
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
# prep_megatron_ckpt.sbatch — one-off: convert Qwen/Qwen3-1.7B (HF) to a
# Megatron torch_dist checkpoint at runs/qwen3_1p7b_torch_dist.
# Qwen3-1.7B uses TIED embeddings
# (config tie_word_embeddings=true) so NO --untie flag (unlike R1-Distill).
#
# Submit:  sbatch prep_megatron_ckpt.sbatch
# Success: runs/qwen3_1p7b_torch_dist/latest_checkpointed_iteration.txt == "release"

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
RUNS=${RLAD_RUNS}
SELF=${RLAD_HOME}/jobs/prep_megatron_ckpt.sbatch
CONTAINER=${CONTAINER:-${RLAD_CONTAINER}}
BASE_HF_MODEL=Qwen/Qwen3-1.7B
SAVE_DIR=${RUNS}/qwen3_1p7b_torch_dist

if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
    # ---------------- runs inside the container ----------------
    # (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export PYTHONPATH=/root/Megatron-LM:${MILES_DIR}:${CODE}
    export PYTHONUNBUFFERED=1

    cd "${MILES_DIR}"
    # qwen3-1.7B.sh pins --vocab-size 151936 and Qwen3 arch (qk-layernorm, GQA 16/8).
    # Qwen3-1.7B TIES embeddings -> do NOT add --untie-embeddings-and-output-weights.
    source scripts/models/qwen3-1.7B.sh

    mkdir -p "${SAVE_DIR}"

    python3 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${BASE_HF_MODEL}" \
        --save "${SAVE_DIR}"

    echo "---- tracker file ----"
    cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"
    exit 0
fi

# ---------------- sbatch prologue (host side) ----------------
if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container image not found: ${CONTAINER}" >&2
    exit 1
fi
mkdir -p "${RLAD_LOGS}"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Task:   HF -> Megatron torch_dist conversion of ${BASE_HF_MODEL}"
echo "Save:   ${SAVE_DIR}"
echo "===================================================================="

srun \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL,IN_CONTAINER=1 \
    --nodes=1 --ntasks=1 \
    bash "${SELF}"

echo "Conversion job ${SLURM_JOB_ID} finished."
