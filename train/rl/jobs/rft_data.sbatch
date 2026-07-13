#!/bin/bash
#SBATCH --job-name=rlad-rft-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# rft_data.sbatch — build the offline RFT corpus for pi_abs (absgen_score.py, method_spec §4.2).
# 8-way data-parallel over the node's GPUs, fully resumable (per-shard append + skip-completed),
# so a 4h timeout just resumes. Score stage dominates (~118 GPU-h). Submit as a singleton chain:
#   for i in $(seq 6); do jobs/sbatch.sh --job-name=rlad-rft-data --dependency=singleton jobs/rft_data.sbatch; done
# Stages per segment: gen-abs (8-way, cheap) -> score (8-way, heavy, resumes) -> build-rft (when complete).
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/cluster_env.sh"
CODE=${RLAD_HOME}
DATA=${RLAD_DATA}
# miles exports HF per-iteration under hf/iter_N; resolve the LATEST as the warm-started pi_abs
ABSGEN_HF=${ABSGEN_HF:-$(ls -d ${RLAD_RUNS}/sft_absgen/hf/iter_* 2>/dev/null | sort -V | tail -1)}
[[ -f "${ABSGEN_HF}/config.json" ]] || { echo "ERROR: no HF model_dir at ABSGEN_HF=${ABSGEN_HF}" >&2; exit 1; }
NSHARDS=${NSHARDS:-8}
NPROB=${NPROB:-1500}; K=${K:-4}; M=${M:-8}; MAXTOK=${MAXTOK:-16384}; MARGIN=${MARGIN:-0.0}

rlad_activate_conda
# (optional) source your own env file for API keys, e.g. NVIDIA_API_KEY / WANDB_API_KEY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1 PYTHONPATH=${CODE}
mkdir -p "${RLAD_LOGS}"
python "${CODE}/eval/trim_vocab_padding.py" "${ABSGEN_HF}" || true   # tokenizer/vocab fix for vLLM
cd "${CODE}"

run_shards () {   # $1 = stage name, $2... = extra args; launches one process per GPU
  local stage="$1"; shift
  local pids=()
  for s in $(seq 0 $((NSHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$s python -m rlad_plugin.absgen_score "$stage" \
        --shard "$s" --num-shards "$NSHARDS" "$@" \
        > "${RLAD_LOGS}/rftdata_${stage}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

echo "=== gen-abs (pi_abs=${ABSGEN_HF}, ${NSHARDS}-way) ==="
run_shards gen-abs --absgen-hf "${ABSGEN_HF}" --k "${K}" --n-problems "${NPROB}"
echo "=== score (base pi_sol, ${NSHARDS}-way, M=${M}) ==="
run_shards score --m "${M}" --max-tokens "${MAXTOK}"

# build-rft only when every cached abstraction has been scored (idempotent; safe to re-run)
N_ABS=$(cat ${DATA}/rft_abs_cache.shard*.jsonl ${DATA}/rft_abs_cache.jsonl 2>/dev/null | wc -l)
N_SC=$(cat ${DATA}/rft_scored.shard*.jsonl ${DATA}/rft_scored.jsonl 2>/dev/null | wc -l)
echo "scored ${N_SC} / ${N_ABS} abstractions"
if [[ "${N_ABS}" -gt 0 && "${N_SC}" -ge "${N_ABS}" ]]; then
  echo "=== build-rft (margin=${MARGIN}) ==="
  python -m rlad_plugin.absgen_score build-rft --margin "${MARGIN}"
  echo "RFT corpus complete -> ${DATA}/train_absgen_rft.jsonl"
else
  echo "score incomplete — next singleton segment will resume"
fi
echo "rft_data segment ${SLURM_JOB_ID} done"
