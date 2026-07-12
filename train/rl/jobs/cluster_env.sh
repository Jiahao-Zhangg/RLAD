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
# ---- cluster settings (EDIT these for your site, or export before submitting) ----
export RLAD_ACCOUNT="${RLAD_ACCOUNT:-CHANGE_ME_slurm_account}"
export RLAD_PARTITION="${RLAD_PARTITION:-CHANGE_ME_gpu_partition}"
export RLAD_CONTAINER="${RLAD_CONTAINER:-/path/to/miles_container.sqsh}"   # pyxis/enroot image w/ miles deps (Megatron-LM + SGLang)
export RLAD_CONDA_ENV="${RLAD_CONDA_ENV:-rlad}"       # host conda env for data-prep + eval (vllm, transformers, math-verify)
