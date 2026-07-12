#!/bin/bash
#SBATCH --job-name=rlad-solgen-data
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# solgen_data.sbatch — build train_solgen_rlad.jsonl (RLAD solver DAPO trainset): one abstraction
# per curriculum problem from the RFT'd pi_abs, mixed with ~1/8 bare prompts (reward-masked).
# Prereq for the RLAD solver DAPO. Host vllm env, 1 GPU.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
ABSGEN_HF=${ABSGEN_HF:-$(ls -d ${RLAD_RUNS}/sft_absgen_rft/hf/iter_* 2>/dev/null | sort -V | tail -1)}
[[ -f "${ABSGEN_HF}/config.json" ]] || { echo "ERROR: no RFT pi_abs HF at ABSGEN_HF=${ABSGEN_HF}" >&2; exit 1; }

source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" ${RLAD_CONDA_ENV}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1 PYTHONPATH=${CODE}
python "${CODE}/eval/trim_vocab_padding.py" "${ABSGEN_HF}" || true
cd "${CODE}"
# CURRICULUM (self-contained jsonl w/ metadata.problem, e.g. POPE-hard) + OUT are optional overrides;
# default = DeepScaleR easy+medium -> data/train_solgen_rlad.jsonl
EXTRA_ARGS=()
[[ -n "${CURRICULUM:-}" ]] && EXTRA_ARGS+=(--curriculum "${CURRICULUM}")
[[ -n "${OUT:-}" ]] && EXTRA_ARGS+=(--out "${OUT}")
echo "=== building solgen trainset from pi_abs=${ABSGEN_HF} ${CURRICULUM:+(curriculum=${CURRICULUM})} ==="
python -m rlad_plugin.solgen_data --absgen-hf "${ABSGEN_HF}" --noabs-every "${NOABS_EVERY:-8}" "${EXTRA_ARGS[@]}"
echo "=== meta ==="; cat "${RLAD_DATA}/solgen_data_meta.json"
echo "solgen_data job ${SLURM_JOB_ID} done"
