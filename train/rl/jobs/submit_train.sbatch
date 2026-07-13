#!/bin/bash
#SBATCH --job-name=rlad-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# submit_train.sbatch — one <=4h training segment for RLAD (the reference
# cluster's partition had a HARD 4h walltime; 03:55:00 leaves margin
# for teardown). Submit from train/rl/.
#
# Required (via sbatch --export, see chain.sh):
#   ARM_CONFIG  absolute path to rlad_plugin/configs/arm_*.sh
# Optional:
#   CONTAINER   override container image
#   JOB_NAME    set via `sbatch --job-name` (header default: rlad-train)
#
# Segments share the per-arm --load/--save dir, so each segment auto-resumes
# from the last checkpoint (--save-interval 2 => a TIMEOUT loses <=1 step).

set -euo pipefail

source "${RLAD_HOME:-$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/..}/jobs/cluster_env.sh"
CONTAINER=${CONTAINER:-${RLAD_CONTAINER}}
# LAUNCHER override lets SFT runs reuse this sbatch with the SFT launcher (jobs/sft_launch.sh).
LAUNCHER=${LAUNCHER:-${RLAD_HOME}/jobs/launch_train.sh}

: "${ARM_CONFIG:?submit with: jobs/sbatch.sh --export=ALL,ARM_CONFIG=<abs path to arm_*.sh> jobs/submit_train.sbatch}"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container image not found: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${ARM_CONFIG}" ]]; then
    echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2
    exit 1
fi

mkdir -p "${RLAD_LOGS}"

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "Stack:      miles (Megatron + SGLang colocated, 1 node x 8 H100)"
echo "Arm config: ${ARM_CONFIG}"
echo "RLAD-Hierarchical knobs: N_HINTS=${N_HINTS:-<config default>} M_SOLS=${M_SOLS:-<config default>}"
echo "Container:  ${CONTAINER}"
echo "===================================================================="

CONTAINER_ARGS=(--container-image="${CONTAINER}")
[[ -n "${RLAD_CONTAINER_MOUNTS}" ]] && CONTAINER_ARGS+=(--container-mounts="${RLAD_CONTAINER_MOUNTS}")
[[ "${RLAD_CONTAINER_MOUNT_HOME}" == "1" ]] || CONTAINER_ARGS+=(--no-container-mount-home)

srun \
    "${CONTAINER_ARGS[@]}" \
    --no-container-entrypoint \
    --export=ALL \
    --nodes=1 --ntasks=1 \
    bash "${LAUNCHER}"

echo "===================================================================="
echo "Slurm job ${SLURM_JOB_ID} finished."
echo "===================================================================="
