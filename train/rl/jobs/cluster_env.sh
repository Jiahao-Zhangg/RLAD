#!/bin/bash
# cluster_env.sh — repo paths + cluster settings for the RLAD training/eval scripts.
# Sourced by jobs/*.sh and jobs/*.sbatch. Override any value via the environment before submitting.
# ---- repo layout (auto-detected; usually no need to edit) ----
RLAD_HOME="${RLAD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # .../train/rl
export RLAD_HOME
export MILES_DIR="${MILES_DIR:-${RLAD_HOME}/miles}"    # clone radixark/miles here (see REPRODUCTION.md)
export RLAD_RUNS="${RLAD_RUNS:-${RLAD_HOME}/runs}"     # checkpoints, HF exports, eval outputs
export RLAD_DATA="${RLAD_DATA:-${RLAD_HOME}/data}"     # prepared datasets
export RLAD_LOGS="${RLAD_LOGS:-${RLAD_HOME}/logs}"

# ---- cluster settings (export before submitting; see .env.cluster.example) ----
# Scheduler options cannot be expanded inside #SBATCH directives. jobs/sbatch.sh
# reads these values before submission and passes non-empty options to sbatch.
export RLAD_ACCOUNT="${RLAD_ACCOUNT:-}"
export RLAD_PARTITION="${RLAD_PARTITION:-}"
export RLAD_SBATCH_CPUS_PER_TASK="${RLAD_SBATCH_CPUS_PER_TASK:-}"
export RLAD_SBATCH_MEMORY="${RLAD_SBATCH_MEMORY:-}"
export RLAD_SBATCH_TIME="${RLAD_SBATCH_TIME:-}"

# Host-side vLLM/data-generation jobs can span several nodes. Training jobs do
# not consume these values and remain single-node.
export RLAD_INFERENCE_NODES="${RLAD_INFERENCE_NODES:-1}"
export RLAD_INFERENCE_NODELIST="${RLAD_INFERENCE_NODELIST:-}"
export RLAD_GPUS_PER_NODE="${RLAD_GPUS_PER_NODE:-8}"

# Container paths must be visible on compute nodes. The /lustre default preserves
# the reference-cluster behavior; FSx users should export /fsx:/fsx.
export RLAD_CONTAINER="${RLAD_CONTAINER:-/path/to/miles_container.sqsh}"
# Versioned CUDA-12 x86 image used only when RLAD_CONTAINER does not exist.
# The dated tag avoids silently changing the training stack between runs.
export RLAD_CONTAINER_SOURCE="${RLAD_CONTAINER_SOURCE:-docker.io#radixark/miles:dev-cu12-202606172131}"
export RLAD_CONTAINER_MOUNTS="${RLAD_CONTAINER_MOUNTS:-/lustre:/lustre}"
export RLAD_CONTAINER_MOUNT_HOME="${RLAD_CONTAINER_MOUNT_HOME:-0}"

# Host environment for data preparation and vLLM evaluation/scoring.
export CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
export RLAD_CONDA_ENV="${RLAD_CONDA_ENV:-rlad}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

rlad_activate_conda() {
   local conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
   if [[ ! -f "${conda_sh}" ]]; then
      echo "ERROR: conda initialization script not found: ${conda_sh}" >&2
      return 1
   fi
   source "${conda_sh}"
   conda activate "${RLAD_CONDA_ENV}"
}

rlad_inference_sbatch() {
   local args=(
      --nodes="${RLAD_INFERENCE_NODES}"
      --ntasks="${RLAD_INFERENCE_NODES}"
      --ntasks-per-node=1
      --gpus-per-node="${RLAD_GPUS_PER_NODE}"
   )
   [[ -z "${RLAD_INFERENCE_NODELIST}" ]] ||
      args+=(--nodelist="${RLAD_INFERENCE_NODELIST}")
   "${RLAD_HOME}/jobs/sbatch.sh" "${args[@]}" "$@"
}
