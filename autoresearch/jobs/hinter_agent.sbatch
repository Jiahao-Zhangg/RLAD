#!/bin/bash
# Submit from the RLAD root on a non-GPU coordinator partition:
#
#   sbatch --partition=<cpu-or-service-partition> \
#     --export=ALL,RLAD_REPO_ROOT="$PWD" \
#     autoresearch/jobs/hinter_agent.sbatch
#
# Do not submit this wrapper to a p4d/p5 GPU partition. The SDK coordinator launches
# autoresearch/jobs/hinter_pool.sbatch as a separate exclusive allocation for
# the configured pair of 8-GPU nodes.
#
#SBATCH --job-name=rlad-hinter-agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --output=/tmp/rlad-hinter-agent-%j.out
#SBATCH --error=/tmp/rlad-hinter-agent-%j.err

set -euo pipefail
umask 077

repo_root="${RLAD_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repo_root}" ]]; then
    echo "ERROR: submit from the RLAD root or export RLAD_REPO_ROOT" >&2
    exit 2
fi
repo_root="$(cd "${repo_root}" && pwd)"
if [[ ! -f "${repo_root}/autoresearch/run_hinter_agent.py" ]]; then
    echo "ERROR: RLAD_REPO_ROOT is not an RLAD checkout: ${repo_root}" >&2
    exit 2
fi
cd "${repo_root}"
work_root="${RLAD_AUTORESEARCH_WORK:-${repo_root}/work_zsw}"
work_root="$(cd "$(dirname "${work_root}")" && pwd)/$(basename "${work_root}")"
case "$(basename "${work_root}")" in
    work_zsw|work_zsw_lambda2|work_zsw_lambda5|work_zsw_lambda10) ;;
    *)
        echo "ERROR: unsupported autoresearch workspace: ${work_root}" >&2
        exit 2
        ;;
esac
mkdir -p "${work_root}"
export RLAD_AUTORESEARCH_WORK="${work_root}"

active_pid=""
request_pool_stop() {
    local allocation_path="${work_root}/pool/allocation.json"
    local stop_path="${work_root}/pool/control/STOP"
    local stop_tmp="${stop_path}.tmp.${SLURM_JOB_ID:-nojob}.$$"
    if [[ ! -f "${allocation_path}" ]]; then
        return
    fi
    if ! {
        mkdir -p "$(dirname "${stop_path}")"
        printf '%s\n' \
            "coordinator exited; stop after active steps finish" >"${stop_tmp}"
        mv -f -- "${stop_tmp}" "${stop_path}"
    }; then
        echo "WARNING: failed to request GPU-pool drain" >&2
    fi
}
run_interruptible() {
    local status
    "$@" &
    active_pid="$!"
    if wait "${active_pid}"; then
        status=0
    else
        status="$?"
    fi
    active_pid=""
    return "${status}"
}
cleanup_on_exit() {
    local status="$?"
    trap - EXIT
    request_pool_stop
    exit "${status}"
}
handle_signal() {
    echo "Coordinator received a termination signal; draining the GPU pool." >&2
    request_pool_stop
    if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
        kill -TERM "${active_pid}" 2>/dev/null || true
        wait "${active_pid}" 2>/dev/null || true
    fi
    exit 143
}
trap cleanup_on_exit EXIT
trap handle_signal TERM INT

coordinator_partition="${SLURM_JOB_PARTITION:-}"
coordinator_host="$(hostname -s)"
if [[ "${coordinator_partition}" == "ml.p5.48xlarge" ||
      "${coordinator_partition}" == "ml.p4d.24xlarge" ]]; then
    echo "ERROR: the SDK coordinator must not occupy a GPU pool partition" >&2
    echo "Submit this wrapper to a non-GPU CPU/service partition." >&2
    exit 2
fi
case "${coordinator_host}" in
    ip-10-1-173-179|ip-10-1-184-205|ip-10-1-196-96|ip-10-1-226-48|ip-10-1-38-11|ip-10-1-81-8)
        echo "ERROR: coordinator landed on reserved GPU host ${coordinator_host}" >&2
        echo "Use a non-GPU CPU/service partition." >&2
        exit 2
        ;;
esac

required_commands=(uv git gh codex sbatch squeue srun)
for command_name in "${required_commands[@]}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 2
    fi
done
run_interruptible gh auth status --hostname github.com >/dev/null

branch="$(git branch --show-current)"
if [[ -z "${branch}" ]]; then
    echo "ERROR: autoresearch requires a named Git branch" >&2
    exit 2
fi
git remote get-url origin >/dev/null
export HINTER_GIT_BRANCH="${HINTER_GIT_BRANCH:-${branch}}"
export RLAD_REPO_ROOT="${repo_root}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

agent_args=("$@")
resume=0
for argument in "${agent_args[@]}"; do
    if [[ "${argument}" == "--resume" ]]; then
        resume=1
    fi
done
if [[ "${HINTER_RESUME:-0}" == "1" ]]; then
    resume=1
fi
if [[ "${SLURM_RESTART_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
    resume=1
fi
if [[ "${resume}" == "1" ]]; then
    has_resume=0
    for argument in "${agent_args[@]}"; do
        if [[ "${argument}" == "--resume" ]]; then
            has_resume=1
        fi
    done
    if [[ "${has_resume}" == "0" ]]; then
        agent_args+=(--resume)
    fi
elif [[ -f "${work_root}/agent_transcript.jsonl" ]]; then
    echo "ERROR: an SDK transcript already exists; relaunch with --resume" >&2
    exit 2
fi

if [[ -f "${work_root}/research/STOPPED.json" ]]; then
    echo "Autoresearch is already sealed; nothing to launch."
    exit 0
fi
if [[ "${resume}" == "0" ]] &&
   [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: a fresh launch requires a clean committed checkout" >&2
    git status --short >&2
    exit 2
fi

if [[ "${HINTER_SKIP_SYNC:-0}" != "1" ]]; then
    run_interruptible \
        uv sync --project autoresearch --group dev --frozen
fi
if [[ "${HINTER_SKIP_TESTS:-0}" != "1" ]]; then
    run_interruptible uv run --project autoresearch --frozen \
        python -m pytest autoresearch/tests -q
fi

echo "Starting RLAD hinter SDK agent on ${coordinator_host}."
echo "Coordinator partition: ${coordinator_partition:-default}"
echo "Git branch: ${branch}"
if run_interruptible uv run --project autoresearch --frozen \
    python -m autoresearch.run_hinter_agent "${agent_args[@]}"; then
    agent_status=0
else
    agent_status="$?"
fi

if [[ ! -f "${work_root}/research/STOPPED.json" ]]; then
    echo "SDK agent exited before sealing the experiment." >&2
    if [[ "${agent_status}" == "0" ]]; then
        agent_status=3
    fi
fi
exit "${agent_status}"
