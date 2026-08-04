#!/bin/bash
#SBATCH --job-name=rlad-hinter-pool
# The dispatcher supplies partition, nodelist, and workspace-specific log paths.
#SBATCH --output=/tmp/rlad-hinter-pool-%j.out
#SBATCH --error=/tmp/rlad-hinter-pool-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${RLAD_REPO_ROOT:?set by autoresearch.hinter.pool start}"
: "${RLAD_AUTORESEARCH_WORK:?set by autoresearch.hinter.pool start}"
: "${AUTORESEARCH_PYTHON:?set by autoresearch.hinter.pool start}"

cd "${RLAD_REPO_ROOT}"
mkdir -p "${RLAD_AUTORESEARCH_WORK}/logs"

echo "[hinter-pool] allocation=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[hinter-pool] python=${AUTORESEARCH_PYTHON}"
srun --exclusive --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=12 --gpus-per-task=1 --gpu-bind=single:1 bash -c '
  visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$visible" || "$visible" == *,* ]]; then
    echo "[hinter-pool] expected one Slurm-bound CUDA device on $(hostname), got ${visible:-unset}" >&2
    exit 2
  fi
  if [[ "${SLURM_GPUS_PER_TASK:-}" != "1" ]]; then
    echo "[hinter-pool] SLURM_GPUS_PER_TASK is not 1 on $(hostname)" >&2
    exit 2
  fi
  "$AUTORESEARCH_PYTHON" -c "import torch; assert torch.cuda.device_count() == 1"
  echo "[hinter-pool] $(hostname): validated one-GPU Slurm step"
'

exec "${AUTORESEARCH_PYTHON}" -m autoresearch.hinter.pool dispatch
