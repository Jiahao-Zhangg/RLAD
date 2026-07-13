#!/bin/bash
# chain.sh — submit N consecutive <=4h training segments as a Slurm singleton
# chain (batch_block1's hard 4h walltime forces segmentation; checkpoints +
# miles auto-resume stitch the segments into one run).
#
# Usage:
#   ./chain.sh <ARM_CONFIG> <N_SEGMENTS> [JOB_NAME]
#
#   ARM_CONFIG  path to rlad_plugin/configs/arm_*.sh (resolved to absolute)
#   N_SEGMENTS  number of submit_train.sbatch copies to queue
#   JOB_NAME    optional; default rlad-<config basename>. All segments share
#               this name + --dependency=singleton, so Slurm runs them one
#               at a time in submission order regardless of how the previous
#               segment ended (COMPLETED / TIMEOUT / FAILED all release the
#               singleton). To abort the chain: scancel --name=<JOB_NAME>.
#
# Examples:
#   ./chain.sh ../rlad_plugin/configs/dapo_baseline.sh 8 rlad-dapo-baseline
#   N_HINTS=4 M_SOLS=4 ./chain.sh ../rlad_plugin/configs/rlad_hierarchical.sh 4 rlad-hierarchical
#
# Not an sbatch itself — run from a login/DC node.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ARM_CONFIG> <N_SEGMENTS> [JOB_NAME]" >&2
    exit 1
fi

ARM_CONFIG=$(readlink -f "$1")
N=$2
JOB_NAME=${3:-rlad-$(basename "${ARM_CONFIG}" .sh)}

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
SBATCH_SCRIPT=${RLAD_HOME}/jobs/submit_train.sbatch
SBATCH_COMMAND=${RLAD_SBATCH_COMMAND:-${RLAD_HOME}/jobs/sbatch.sh}

# Normalize the SFT launcher before forwarding it into Slurm. This accepts the
# documented basename while ensuring the in-container process receives a real path.
if [[ -n "${LAUNCHER:-}" ]]; then
    if [[ ! -f "${LAUNCHER}" && -f "${RLAD_HOME}/jobs/${LAUNCHER}" ]]; then
        LAUNCHER="${RLAD_HOME}/jobs/${LAUNCHER}"
    fi
    LAUNCHER=$(readlink -f "${LAUNCHER}")
    [[ -f "${LAUNCHER}" ]] || { echo "ERROR: launcher not found: ${LAUNCHER}" >&2; exit 1; }
    export LAUNCHER
fi

if [[ ! -f "${ARM_CONFIG}" ]]; then
    echo "ERROR: ARM_CONFIG not found: ${ARM_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

# Forward the run-selecting overrides EXPLICITLY into the job env so the arm config resolves
# the intended output dir / data / epoch count. (--export=ALL already propagates the caller's
# env; listing them makes intent explicit and survives a future non-ALL export mode.) arm_*.sh
# applies its own defaults when absent. RUN_TAG is what forks a run into a NEW dir (e.g. *_pope);
# OMITTING IT silently reuses the bare dir and miles auto-resumes it -> preflight audit w8okt63ib
# footgun (would clobber the finished DeepScaleR runs). Prefer jobs/launch_pope_retrain.sh, which
# hardcodes these and asserts a *_pope target before submitting.
EXPORTS="ALL,ARM_CONFIG=${ARM_CONFIG}"
[[ -n "${RUN_TAG:-}" ]]       && EXPORTS+=",RUN_TAG=${RUN_TAG}"
[[ -n "${PROMPT_DATA:-}" ]]   && EXPORTS+=",PROMPT_DATA=${PROMPT_DATA}"
[[ -n "${NUM_EPOCH:-}" ]]     && EXPORTS+=",NUM_EPOCH=${NUM_EPOCH}"
[[ -n "${ROLLOUT_BATCH:-}" ]] && EXPORTS+=",ROLLOUT_BATCH=${ROLLOUT_BATCH}"
[[ -n "${GLOBAL_BATCH:-}" ]]  && EXPORTS+=",GLOBAL_BATCH=${GLOBAL_BATCH}"
[[ -n "${LAUNCHER:-}" ]]      && EXPORTS+=",LAUNCHER=${LAUNCHER}"

echo "================================================================"
echo "Singleton chain: ${N} segments of ${SBATCH_SCRIPT}"
echo "  job name:   ${JOB_NAME}"
echo "  arm config: ${ARM_CONFIG}"
echo "  exports:    ${EXPORTS}"
echo "================================================================"

JOB_IDS=()
for i in $(seq 1 "${N}"); do
    JID=$("${SBATCH_COMMAND}" --parsable \
        --job-name="${JOB_NAME}" \
        --dependency=singleton \
        --export="${EXPORTS}" \
        "${SBATCH_SCRIPT}")
    JOB_IDS+=("${JID}")
    echo "  segment $(printf '%02d' "${i}"): jobid ${JID}"
done

echo "================================================================"
echo "Submitted job IDs: ${JOB_IDS[*]}"
echo "Watch:  squeue -u \$USER --name=${JOB_NAME} -o '%.12i %.24j %.8T %.10M %.10l %R'"
echo "Abort:  scancel --name=${JOB_NAME}"
echo "================================================================"
