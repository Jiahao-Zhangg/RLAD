#!/bin/bash
#SBATCH --job-name=rlad-container
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Import the pinned public Miles image through Pyxis, validate it on one GPU,
# and atomically save the resulting SquashFS image on shared storage.

set -Eeuo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"

SOURCE=${RLAD_CONTAINER_SOURCE:?set RLAD_CONTAINER_SOURCE to a Pyxis/Enroot image URI}
OUT=${RLAD_CONTAINER:?set RLAD_CONTAINER to the output .sqsh path}
RECEIPT="${OUT}.source"
VERIFY="${RLAD_HOME}/scripts/verify_training_container.py"

[[ "${SOURCE}" != *','* && "${SOURCE}" != *$'\n'* ]] || {
    echo "ERROR: unsafe RLAD_CONTAINER_SOURCE value" >&2
    exit 2
}
[[ -d "${MILES_DIR}/.git" ]] || {
    echo "ERROR: pinned miles checkout is missing: ${MILES_DIR}" >&2
    exit 1
}
[[ -r "${VERIFY}" ]] || {
    echo "ERROR: container verifier is missing: ${VERIFY}" >&2
    exit 1
}

mkdir -p "$(dirname -- "${OUT}")" "${RLAD_LOGS}"
exec 9>"${OUT}.lock"
flock 9

if [[ -s "${OUT}" ]]; then
    if [[ -f "${RECEIPT}" ]] && ! grep -Fqx "source=${SOURCE}" "${RECEIPT}"; then
        echo "ERROR: existing image receipt does not match ${SOURCE}: ${RECEIPT}" >&2
        exit 1
    fi
    echo "Training container already exists: ${OUT}"
    exit 0
fi
[[ ! -e "${OUT}" ]] || {
    echo "ERROR: container target exists but is empty: ${OUT}" >&2
    exit 1
}

PARTIAL="${OUT}.partial.${SLURM_JOB_ID:-$$}"
RECEIPT_TMP="${RECEIPT}.tmp.${SLURM_JOB_ID:-$$}"
trap 'rm -f -- "${PARTIAL}" "${RECEIPT_TMP}"' EXIT
[[ ! -e "${PARTIAL}" ]] || {
    echo "ERROR: partial container path already exists: ${PARTIAL}" >&2
    exit 1
}

CONTAINER_ARGS=(
    --container-image="${SOURCE}"
    --container-save="${PARTIAL}"
)
[[ -z "${RLAD_CONTAINER_MOUNTS}" ]] ||
    CONTAINER_ARGS+=(--container-mounts="${RLAD_CONTAINER_MOUNTS}")
[[ "${RLAD_CONTAINER_MOUNT_HOME}" == 1 ]] ||
    CONTAINER_ARGS+=(--no-container-mount-home)

echo "Importing ${SOURCE}"
echo "Saving validated SquashFS image to ${OUT}"
export PYTHONPATH="/root/Megatron-LM:${MILES_DIR}:${RLAD_HOME}"
srun \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    "${CONTAINER_ARGS[@]}" \
    --no-container-entrypoint \
    --export=ALL \
    python3 "${VERIFY}"

[[ -s "${PARTIAL}" ]] || {
    echo "ERROR: Pyxis completed without creating ${PARTIAL}" >&2
    exit 1
}
{
    printf 'schema=1\n'
    printf 'source=%s\n' "${SOURCE}"
    printf 'bytes=%s\n' "$(stat -c '%s' "${PARTIAL}")"
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RECEIPT_TMP}"
# Publish the receipt first so another controller can never mistake the short
# image-before-receipt window for an unreceipted, user-provided image.
mv -- "${RECEIPT_TMP}" "${RECEIPT}"
mv -- "${PARTIAL}" "${OUT}"

echo "Training container ready: ${OUT} ($(du -h "${OUT}" | awk '{print $1}'))"
