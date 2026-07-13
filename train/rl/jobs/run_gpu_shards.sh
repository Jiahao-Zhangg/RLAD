#!/usr/bin/env bash
# Launch one Python-style data shard per GPU across every node in this allocation.
#
# Usage:
#   run_gpu_shards.sh TOTAL_SHARDS GPUS_PER_NODE SHARD_FLAG NUM_SHARDS_FLAG \
#     LOG_DIR LOG_PREFIX COMMAND [ARGS...]
#
# The command receives "SHARD_FLAG <global-id> NUM_SHARDS_FLAG <total>".

set -Eeuo pipefail

if (( $# < 7 )); then
    echo "usage: $0 TOTAL_SHARDS GPUS_PER_NODE SHARD_FLAG NUM_SHARDS_FLAG LOG_DIR LOG_PREFIX COMMAND [ARGS...]" >&2
    exit 2
fi

TOTAL_SHARDS=$1
GPUS_PER_NODE=$2
SHARD_FLAG=$3
NUM_SHARDS_FLAG=$4
LOG_DIR=$5
LOG_PREFIX=$6
shift 6
COMMAND=("$@")

[[ "${TOTAL_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: TOTAL_SHARDS must be a positive integer" >&2; exit 2;
}
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: GPUS_PER_NODE must be a positive integer" >&2; exit 2;
}

ALLOCATED_NODES=${SLURM_JOB_NUM_NODES:-1}
[[ "${ALLOCATED_NODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: SLURM_JOB_NUM_NODES must be a positive integer" >&2; exit 2;
}
EXPECTED_SHARDS=$((ALLOCATED_NODES * GPUS_PER_NODE))
(( TOTAL_SHARDS == EXPECTED_SHARDS )) || {
    echo "ERROR: ${ALLOCATED_NODES} node(s) x ${GPUS_PER_NODE} GPUs requires " \
         "TOTAL_SHARDS=${EXPECTED_SHARDS}, got ${TOTAL_SHARDS}" >&2
    exit 2
}

mkdir -p "${LOG_DIR}"

if [[ "${RLAD_GPU_SHARD_WORKER:-0}" != 1 ]]; then
    command -v srun >/dev/null 2>&1 || {
        echo "ERROR: srun is required for the multi-node shard launch" >&2; exit 1;
    }
    exec srun \
        --nodes="${ALLOCATED_NODES}" \
        --ntasks="${ALLOCATED_NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        --gpus-per-task="${GPUS_PER_NODE}" \
        --kill-on-bad-exit=1 \
        --export=ALL,RLAD_GPU_SHARD_WORKER=1 \
        "$(readlink -f -- "$0")" "${TOTAL_SHARDS}" "${GPUS_PER_NODE}" \
        "${SHARD_FLAG}" "${NUM_SHARDS_FLAG}" "${LOG_DIR}" "${LOG_PREFIX}" \
        "${COMMAND[@]}"
fi

NODE_RANK=${SLURM_PROCID:?SLURM_PROCID is required inside the worker step}
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid SLURM_PROCID=${NODE_RANK}" >&2; exit 2;
}
(( NODE_RANK < ALLOCATED_NODES )) || {
    echo "ERROR: SLURM_PROCID=${NODE_RANK} exceeds this ${ALLOCATED_NODES}-node allocation" >&2
    exit 2
}

PIDS=()
for LOCAL_GPU in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_SHARD=$((NODE_RANK * GPUS_PER_NODE + LOCAL_GPU))
    LOG_PATH="${LOG_DIR}/${LOG_PREFIX}${GLOBAL_SHARD}_${SLURM_JOB_ID:-manual}.log"
    CACHE_BASE=${CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/rlad_compile}
    CACHE_BASE="${CACHE_BASE}/shard${GLOBAL_SHARD}"
    (
        export CUDA_VISIBLE_DEVICES="${LOCAL_GPU}"
        export RLAD_GLOBAL_SHARD_ID="${GLOBAL_SHARD}"
        export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
        export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
        export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
        export TORCH_EXTENSIONS_DIR="${CACHE_BASE}/torch_extensions"
        export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
        mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
            "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
        exec "${COMMAND[@]}" "${SHARD_FLAG}" "${GLOBAL_SHARD}" \
            "${NUM_SHARDS_FLAG}" "${TOTAL_SHARDS}"
    ) >"${LOG_PATH}" 2>&1 &
    PIDS+=("$!")
done

RC=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || RC=1
done
exit "${RC}"
