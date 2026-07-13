#!/bin/bash
# Prepare the host vLLM environment and pinned external miles checkout.
# Source .env.cluster first so CONDA_BASE, HF_HOME, and site paths are correct.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RL_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${RL_DIR}/jobs/cluster_env.sh"

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || {
   echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
   exit 1
}
source "${CONDA_SH}"

if ! conda run -n "${RLAD_CONDA_ENV}" python --version >/dev/null 2>&1; then
   conda create -n "${RLAD_CONDA_ENV}" python=3.12 -y
fi
conda activate "${RLAD_CONDA_ENV}"
python -m pip install -r "${RL_DIR}/../../requirements.txt"

mkdir -p "${RLAD_DATA}" "${RLAD_RUNS}" "${RLAD_LOGS}" "${HF_HOME}"

MILES_COMMIT=9437366e0
MILES_URL=https://github.com/radixark/miles.git
PATCH="${RL_DIR}/patches/miles_div_temperature.patch"

if [[ ! -d "${MILES_DIR}/.git" ]]; then
   git clone "${MILES_URL}" "${MILES_DIR}"
fi

CURRENT=$(git -C "${MILES_DIR}" rev-parse --short=9 HEAD)
if [[ "${CURRENT}" != "${MILES_COMMIT}" ]]; then
   if [[ -n "$(git -C "${MILES_DIR}" status --porcelain)" ]]; then
      echo "ERROR: ${MILES_DIR} has local changes; refusing to switch commits" >&2
      exit 1
   fi
   git -C "${MILES_DIR}" fetch origin
   git -C "${MILES_DIR}" checkout --detach "${MILES_COMMIT}"
fi

if git -C "${MILES_DIR}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
   echo "miles patch already applied"
elif git -C "${MILES_DIR}" apply --check "${PATCH}"; then
   git -C "${MILES_DIR}" apply "${PATCH}"
else
   echo "ERROR: miles compatibility patch cannot be applied cleanly" >&2
   exit 1
fi

echo "Host setup complete"
echo "  conda env: ${RLAD_CONDA_ENV}"
echo "  miles:     ${MILES_DIR} @ ${MILES_COMMIT}"
echo "  HF_HOME:   ${HF_HOME}"
echo "Container target: ${RLAD_CONTAINER}"
echo "  (RFT_pipeline.sh setup imports ${RLAD_CONTAINER_SOURCE} when it is missing)"
