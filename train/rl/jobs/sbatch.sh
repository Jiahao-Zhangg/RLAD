#!/bin/bash
# Submit an RLAD job with scheduler settings loaded before Slurm parses the job.
# Usage: jobs/sbatch.sh [sbatch options] jobs/<job>.sbatch

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
SBATCH_BIN=${RLAD_SBATCH_BIN:-sbatch}
command -v "${SBATCH_BIN}" >/dev/null 2>&1 || {
   echo "ERROR: sbatch executable not found: ${SBATCH_BIN}" >&2
   exit 1
}

SBATCH_SITE_ARGS=(
   --chdir="${RLAD_HOME}"
   --output="${RLAD_LOGS}/%x_%j.out"
   --error="${RLAD_LOGS}/%x_%j.err"
)
[[ -n "${RLAD_ACCOUNT}" ]] && SBATCH_SITE_ARGS+=(--account="${RLAD_ACCOUNT}")
[[ -n "${RLAD_PARTITION}" ]] && SBATCH_SITE_ARGS+=(--partition="${RLAD_PARTITION}")
[[ -n "${RLAD_SBATCH_CPUS_PER_TASK}" ]] && SBATCH_SITE_ARGS+=(--cpus-per-task="${RLAD_SBATCH_CPUS_PER_TASK}")
[[ -n "${RLAD_SBATCH_MEMORY}" ]] && SBATCH_SITE_ARGS+=(--mem="${RLAD_SBATCH_MEMORY}")
[[ -n "${RLAD_SBATCH_TIME}" ]] && SBATCH_SITE_ARGS+=(--time="${RLAD_SBATCH_TIME}")

mkdir -p "${RLAD_LOGS}"
exec "${SBATCH_BIN}" "${SBATCH_SITE_ARGS[@]}" "$@"
