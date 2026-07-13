#!/bin/bash
#SBATCH --job-name=rlad-convert-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# convert_hf.sbatch — OFFLINE Megatron torch_dist -> HF conversion (F001: in-training
# --save-hf is unreliable under TP+CP; always use this offline path before eval).
# For Qwen3-1.7B. Converts CKPT_DIR's iter ITER (default = latest
# per tracker) via tools/convert_torch_dist_to_hf.py (also strips vocab padding).
#
# Submit (baseline iter_100):
#   jobs/sbatch.sh --export=ALL,CKPT_DIR=$PWD/runs/dapo_baseline/ckpts,OUT_DIR=$PWD/runs/dapo_baseline/hf_iter100,ITER=100 jobs/convert_hf.sbatch

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
SELF=${RLAD_HOME}/jobs/convert_hf.sbatch
CONTAINER=${CONTAINER:-${RLAD_CONTAINER}}

: "${CKPT_DIR:?submit with --export=ALL,CKPT_DIR=<megatron ckpt dir>,OUT_DIR=<hf output dir>[,ITER=N]}"
: "${OUT_DIR:?submit with --export=ALL,CKPT_DIR=<megatron ckpt dir>,OUT_DIR=<hf output dir>[,ITER=N]}"

if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
    # (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export PYTHONPATH=/root/Megatron-LM:${MILES_DIR}:${CODE}
    export PYTHONUNBUFFERED=1
    cd "${MILES_DIR}"

    ITER=${ITER:-$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")}
    ITER_DIR=$(printf "%s/iter_%07d" "${CKPT_DIR}" "${ITER}")
    [[ -d "${ITER_DIR}" ]] || { echo "ERROR: ${ITER_DIR} not found" >&2; exit 1; }
    SNAP=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-1.7B/snapshots/*/ | head -1)
    echo "converting ${ITER_DIR} (origin ${SNAP})"

    python3 tools/convert_torch_dist_to_hf.py \
        --input-dir "${ITER_DIR}" \
        --output-dir "${OUT_DIR}" \
        --origin-hf-dir "${SNAP}" \
        --vocab-size 151936 \
        --force

    # Finalize the exact immutable directory consumed by vLLM before the
    # controller fingerprints it. Downstream data jobs must not rewrite it.
    python3 "${CODE}/eval/trim_vocab_padding.py" "${OUT_DIR}"

    echo "---- converted HF dir ----"; ls -la "${OUT_DIR}"
    exit 0
fi

if [[ ! -f "${CONTAINER}" ]]; then echo "ERROR: container missing: ${CONTAINER}" >&2; exit 1; fi
mkdir -p "${RLAD_LOGS}"
echo "Convert ${CKPT_DIR} iter=${ITER:-latest} -> ${OUT_DIR}  (job ${SLURM_JOB_ID})"

CONTAINER_ARGS=(--container-image="${CONTAINER}")
[[ -n "${RLAD_CONTAINER_MOUNTS}" ]] && CONTAINER_ARGS+=(--container-mounts="${RLAD_CONTAINER_MOUNTS}")
[[ "${RLAD_CONTAINER_MOUNT_HOME}" == "1" ]] || CONTAINER_ARGS+=(--no-container-mount-home)

srun \
    "${CONTAINER_ARGS[@]}" \
    --no-container-entrypoint \
    --export=ALL,IN_CONTAINER=1,CKPT_DIR="${CKPT_DIR}",OUT_DIR="${OUT_DIR}",ITER="${ITER:-}" \
    --nodes=1 --ntasks=1 \
    bash "${SELF}"

echo "Conversion job ${SLURM_JOB_ID} finished."
