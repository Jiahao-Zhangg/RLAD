#!/usr/bin/env bash
# Stateful launcher for the original RLAD abstraction-generator SFT + offline RFT pipeline.
# Run `./RFT_pipeline.sh help` for the cross-machine workflow.

set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RL_DIR="${ROOT_DIR}/train/rl"
PROFILE="${RLAD_CLUSTER_PROFILE:-${RL_DIR}/.env.cluster}"

# Reproduction defaults. Override before invoking the script only when intentionally
# starting a new pipeline; a saved manifest rejects parameter drift during resume.
POOL_SIZE=${POOL_SIZE:-6000}
POOL_SEED=${POOL_SEED:-42}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
BASE_SAMPLES=${BASE_SAMPLES:-8}
BASE_MAX_TOKENS=${BASE_MAX_TOKENS:-8192}
NSHARDS=${NSHARDS:-8}
HARD_MAX=${HARD_MAX:-0.125}
EASY_MIN=${EASY_MIN:-0.5}
SFT_K=${SFT_K:-2}
WARMSTART_MODEL=${WARMSTART_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
WARMSTART_MAX_TOKENS=${WARMSTART_MAX_TOKENS:-4096}
WARMSTART_TEMPERATURE=${WARMSTART_TEMPERATURE:-0.7}
SFT_EPOCHS=${SFT_EPOCHS:-5}
RFT_NPROB=${RFT_NPROB:-1500}
RFT_K=${RFT_K:-4}
RFT_M=${RFT_M:-8}
RFT_MAXTOK=${RFT_MAXTOK:-16384}
RFT_MARGIN=${RFT_MARGIN:-0.0}
RFT_EPOCHS=${RFT_EPOCHS:-3}
TRAIN_BATCH=${TRAIN_BATCH:-128}
RFT_MIN_ROWS=${RFT_MIN_ROWS:-128}

POLL_SECONDS=${POLL_SECONDS:-60}
BASE_EVAL_ATTEMPTS=${BASE_EVAL_ATTEMPTS:-6}
BASE_CKPT_ATTEMPTS=${BASE_CKPT_ATTEMPTS:-2}
WARMSTART_ATTEMPTS=${WARMSTART_ATTEMPTS:-1}
SFT_SEGMENTS=${SFT_SEGMENTS:-2}
CONVERT_ATTEMPTS=${CONVERT_ATTEMPTS:-2}
RFT_GEN_ATTEMPTS=${RFT_GEN_ATTEMPTS:-2}
RFT_SCORE_SEGMENTS=${RFT_SCORE_SEGMENTS:-6}
RFT_TRAIN_SEGMENTS=${RFT_TRAIN_SEGMENTS:-2}
WARMSTART_TIME=${WARMSTART_TIME:-03:55:00}

SQUEUE_BIN=${RLAD_SQUEUE_BIN:-squeue}
SACCT_BIN=${RLAD_SACCT_BIN:-sacct}
SCONTROL_BIN=${RLAD_SCONTROL_BIN:-scontrol}
SINFO_BIN=${RLAD_SINFO_BIN:-sinfo}
SBATCH_COMMAND=${RLAD_SBATCH_COMMAND:-${RL_DIR}/jobs/sbatch.sh}

STATE_DIR=""
PIPELINE_ID=""
ACTIVE_JOB=""

log() { printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*" >&2; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./RFT_pipeline.sh <command>

Commands:
  setup         Create/load train/rl/.env.cluster and prepare Conda + pinned miles.
  doctor        Check paths, tools, environment, container, mounts, and Slurm access.
  run           Run the complete pipeline in the foreground; safe to rerun after interruption.
  resume        Alias for run.
  status        Show artifact progress and recorded Slurm jobs without submitting anything.
  archive-rft   Archive stale RFT artifacts for a same-configuration regeneration.
  help          Show this message.

Recommended on the target machine:
  cp train/rl/.env.cluster.example train/rl/.env.cluster
  $EDITOR train/rl/.env.cluster
  ./RFT_pipeline.sh setup
  tmux new -s rlad-rft
  # Then, inside tmux:
  ./RFT_pipeline.sh run

The controller submits one Slurm stage at a time and waits for it. Ctrl-C stops only the
controller; it never cancels a running Slurm job. Rerun `resume` to reconnect and continue.
After fixing a non-retryable training failure, explicitly use
`RLAD_RETRY_FAILED=1 ./RFT_pipeline.sh resume` to authorize another segment.
State and logs are kept under train/rl/runs/rft_pipeline and train/rl/logs.
EOF
}

atomic_write() {
    local path=$1 value=$2 tmp
    mkdir -p "$(dirname -- "${path}")"
    tmp="${path}.tmp.$$"
    printf '%s\n' "${value}" > "${tmp}"
    mv -f -- "${tmp}" "${path}"
}

load_profile() {
    local create=${1:-0}
    if [[ ! -f "${PROFILE}" ]]; then
        [[ "${create}" == 1 ]] || die "cluster profile missing: ${PROFILE}; run '$0 setup' first"
        cp -- "${RL_DIR}/.env.cluster.example" "${PROFILE}"
        chmod 600 "${PROFILE}"
        log "Created ${PROFILE} from the checked-in example"
    fi

    # The profile, rather than an unrelated parent shell, owns all filesystem paths.
    unset RLAD_HOME MILES_DIR RLAD_DATA RLAD_RUNS RLAD_LOGS \
        CONDA_BASE RLAD_CONDA_ENV HF_HOME RLAD_CONTAINER RLAD_CONTAINER_MOUNTS \
        RLAD_CONTAINER_MOUNT_HOME RLAD_PARTITION RLAD_ACCOUNT \
        RLAD_SBATCH_CPUS_PER_TASK RLAD_SBATCH_MEMORY RLAD_SBATCH_TIME
    # shellcheck disable=SC1090
    source "${PROFILE}"

    SQUEUE_BIN=${RLAD_SQUEUE_BIN:-squeue}
    SACCT_BIN=${RLAD_SACCT_BIN:-sacct}
    SCONTROL_BIN=${RLAD_SCONTROL_BIN:-scontrol}
    SINFO_BIN=${RLAD_SINFO_BIN:-sinfo}
    SBATCH_COMMAND=${RLAD_SBATCH_COMMAND:-${RL_DIR}/jobs/sbatch.sh}

    [[ "$(readlink -f -- "${RLAD_HOME}")" == "$(readlink -f -- "${RL_DIR}")" ]] ||
        die "RLAD_HOME=${RLAD_HOME} does not point to this checkout (${RL_DIR})"
    # data_prep.py and absgen_score.py currently use RLAD_HOME/data and RLAD_HOME/runs.
    [[ "$(readlink -m -- "${RLAD_DATA}")" == "$(readlink -m -- "${RL_DIR}/data")" ]] ||
        die "RLAD_DATA must be ${RL_DIR}/data for the original pipeline"
    [[ "$(readlink -m -- "${RLAD_RUNS}")" == "$(readlink -m -- "${RL_DIR}/runs")" ]] ||
        die "RLAD_RUNS must be ${RL_DIR}/runs for the original pipeline"

    export PYTHONPATH="${RLAD_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
    STATE_DIR="${RLAD_RUNS}/rft_pipeline"
    local path_hash
    path_hash=$(printf '%s' "${RLAD_HOME}" | sha256sum | awk '{print substr($1, 1, 6)}')
    PIPELINE_ID="absrft-$(git -C "${ROOT_DIR}" rev-parse --short=6 HEAD)-${path_hash}"
}

mounted_path() {
    local path=$1 entry host
    local -a mounts=()
    IFS=',' read -ra mounts <<< "${RLAD_CONTAINER_MOUNTS:-}"
    for entry in "${mounts[@]}"; do
        host=${entry%%:*}
        [[ -n "${host}" && ( "${path}" == "${host}" || "${path}" == "${host}/"* ) ]] && return 0
    done
    [[ "${RLAD_CONTAINER_MOUNT_HOME:-0}" == 1 &&
       ( "${path}" == "${HOME}" || "${path}" == "${HOME}/"* ) ]]
}

doctor() {
    local failures=0 path value
    for path in "${CONDA_BASE}/etc/profile.d/conda.sh" "${RLAD_CONTAINER}"; do
        if [[ ! -s "${path}" || ! -r "${path}" ]]; then
            warn "missing or empty: ${path}"
            failures=1
        fi
    done
    for path in "${RLAD_DATA}" "${RLAD_RUNS}" "${RLAD_LOGS}" "${HF_HOME}"; do
        if [[ ! -d "${path}" || ! -w "${path}" ]]; then
            warn "directory is missing or not writable: ${path}"
            failures=1
        fi
    done
    for path in "${RLAD_HOME}" "${MILES_DIR}" "${HF_HOME}"; do
        if ! mounted_path "${path}"; then
            warn "container-visible path is not covered by RLAD_CONTAINER_MOUNTS: ${path}"
            failures=1
        fi
    done
    for path in "${RLAD_HOME}" "${RLAD_DATA}" "${RLAD_RUNS}" "${RLAD_CONTAINER}" \
        "${BASE_MODEL}" "${WARMSTART_MODEL}"; do
        if [[ "${path}" == *','* || "${path}" == *$'\n'* ]]; then
            warn "value cannot be passed safely through Slurm --export: ${path}"
            failures=1
        fi
    done
    if [[ "${BASE_MODEL}" != "Qwen/Qwen3-1.7B" ]]; then
        warn "BASE_MODEL is fixed to Qwen/Qwen3-1.7B by checkpoint and RFT code"
        failures=1
    fi
    for path in POOL_SIZE POOL_SEED BASE_SAMPLES BASE_MAX_TOKENS NSHARDS SFT_K SFT_EPOCHS \
        RFT_NPROB RFT_K RFT_M RFT_MAXTOK RFT_EPOCHS TRAIN_BATCH RFT_MIN_ROWS; do
        value=${!path}
        if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
            warn "${path} must be a nonnegative integer (got ${value})"
            failures=1
        fi
    done
    for path in POOL_SIZE BASE_SAMPLES BASE_MAX_TOKENS NSHARDS SFT_K SFT_EPOCHS \
        RFT_NPROB RFT_K RFT_M RFT_MAXTOK RFT_EPOCHS TRAIN_BATCH RFT_MIN_ROWS; do
        value=${!path}
        if [[ "${value}" =~ ^[0-9]+$ ]] && (( value < 1 )); then
            warn "${path} must be greater than zero"
            failures=1
        fi
    done
    if [[ "${NSHARDS}" =~ ^[0-9]+$ ]] && (( NSHARDS < 1 || NSHARDS > 8 )); then
        warn "NSHARDS must be between 1 and the allocated 8 GPUs"
        failures=1
    fi

    for path in git flock "${RLAD_SBATCH_BIN:-sbatch}" "${SBATCH_COMMAND}" "${SQUEUE_BIN}" \
        "${SACCT_BIN}" "${SCONTROL_BIN}" "${SINFO_BIN}" srun python3; do
        if ! command -v "${path}" >/dev/null 2>&1; then
            warn "required command not found: ${path}"
            failures=1
        fi
    done

    if [[ -z "${RLAD_PARTITION:-}" ]]; then
        warn "RLAD_PARTITION is empty"
        failures=1
    elif command -v "${SINFO_BIN}" >/dev/null 2>&1 &&
         ! "${SINFO_BIN}" -h -p "${RLAD_PARTITION}" -o '%P' 2>/dev/null | grep -q .; then
        warn "partition was not visible through sinfo: ${RLAD_PARTITION}"
        failures=1
    fi

    if [[ -d "${MILES_DIR}/.git" ]]; then
        [[ "$(git -C "${MILES_DIR}" rev-parse --short=9 HEAD)" == 9437366e0 ]] || {
            warn "miles is not pinned at 9437366e0"
            failures=1
        }
        miles_patch_exact || {
            warn "miles must contain exactly the checked-in compatibility patch and no other local changes"
            failures=1
        }
    else
        warn "pinned miles checkout missing: ${MILES_DIR}"
        failures=1
    fi

    if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        if ! conda run -n "${RLAD_CONDA_ENV}" python -c \
            'import datasets, math_verify, transformers, vllm' >/dev/null 2>&1; then
            warn "Conda env ${RLAD_CONDA_ENV} is missing one or more required Python packages"
            failures=1
        fi
    fi

    [[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN is unset (public downloads may still work)"
    [[ -n "${WANDB_API_KEY:-}" ]] || warn "WANDB_API_KEY is unset (W&B logging will be disabled)"
    if command -v srun >/dev/null 2>&1 &&
       ! srun --help 2>&1 | grep -q -- '--container-image'; then
        warn "srun help does not advertise Pyxis --container-image; verify it on a compute node"
    fi

    [[ ${failures} -eq 0 ]] || die "preflight failed; fix the warnings above"
    log "Preflight passed for ${RLAD_PARTITION} using ${RLAD_CONTAINER}"
}

miles_patch_exact() {
    local target patch status expected actual
    target="miles/backends/training_utils/loss_hub/logit_processors.py"
    patch="${RL_DIR}/patches/miles_div_temperature.patch"
    status=$(git -C "${MILES_DIR}" status --porcelain --untracked-files=all 2>/dev/null) || return 1
    [[ "${status}" == " M ${target}" ]] || return 1
    expected=$(sed '/^index /d' "${patch}" | sha256sum | awk '{print $1}')
    actual=$(git -C "${MILES_DIR}" --no-pager diff --no-ext-diff --no-color -- "${target}" |
        sed '/^index /d' | sha256sum | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]]
}

miles_ready() {
    [[ -d "${MILES_DIR}/.git" ]] || return 1
    [[ "$(git -C "${MILES_DIR}" rev-parse --short=9 HEAD 2>/dev/null)" == 9437366e0 ]] || return 1
    miles_patch_exact
}

bootstrap_host() {
    local marker key
    [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] ||
        die "Conda initialization script missing: ${CONDA_BASE}/etc/profile.d/conda.sh"
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    mkdir -p "${RLAD_DATA}" "${RLAD_RUNS}" "${RLAD_LOGS}" "${HF_HOME}" "${STATE_DIR}"
    marker="${STATE_DIR}/bootstrap.key"
    key=$(
        sha256sum "${ROOT_DIR}/requirements.txt" \
            "${RL_DIR}/scripts/bootstrap_host.sh" \
            "${RL_DIR}/patches/miles_div_temperature.patch" | sha256sum | awk '{print $1}'
    )
    if [[ -f "${marker}" && "$(<"${marker}")" == "${key}" ]] && miles_ready &&
       conda run -n "${RLAD_CONDA_ENV}" python -c \
           'import datasets, math_verify, transformers, vllm' >/dev/null 2>&1; then
        log "Host bootstrap already matches this checkout"
        return
    fi
    "${RL_DIR}/scripts/bootstrap_host.sh"
    atomic_write "${marker}" "${key}"
}

activate_host_env() {
    rlad_activate_conda
    export PYTHONPATH="${RLAD_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
}

manifest_text() {
    cat <<EOF
schema=1
repo_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD)
rlad_home=${RLAD_HOME}
profile=$(readlink -f -- "${PROFILE}")
base_model=${BASE_MODEL}
pool_size=${POOL_SIZE}
pool_seed=${POOL_SEED}
base_samples=${BASE_SAMPLES}
base_max_tokens=${BASE_MAX_TOKENS}
nshards=${NSHARDS}
hard_max=${HARD_MAX}
easy_min=${EASY_MIN}
sft_k=${SFT_K}
warmstart_model=${WARMSTART_MODEL}
warmstart_max_tokens=${WARMSTART_MAX_TOKENS}
warmstart_temperature=${WARMSTART_TEMPERATURE}
sft_epochs=${SFT_EPOCHS}
rft_nprob=${RFT_NPROB}
rft_k=${RFT_K}
rft_m=${RFT_M}
rft_maxtok=${RFT_MAXTOK}
rft_margin=${RFT_MARGIN}
rft_epochs=${RFT_EPOCHS}
rft_min_rows=${RFT_MIN_ROWS}
train_batch=${TRAIN_BATCH}
EOF
}

unsafe_artifacts_exist() {
    local path
    for path in \
        "${RLAD_DATA}/benchmarks/dsr_pool.jsonl" \
        "${RLAD_DATA}/dsr_pool_meta.json" \
        "${RLAD_RUNS}/eval/dsr_pool_score" \
        "${RLAD_RUNS}/qwen3_1p7b_torch_dist" \
        "${RLAD_DATA}/train_easy.jsonl" \
        "${RLAD_DATA}/train_medium.jsonl" \
        "${RLAD_DATA}/curriculum_meta.json" \
        "${RLAD_DATA}/train_absgen_sft.jsonl" \
        "${RLAD_DATA}/absgen_sft_meta.json" \
        "${RLAD_RUNS}/sft_absgen" \
        "${RLAD_RUNS}/sft_absgen_rft" \
        "${RLAD_DATA}/train_absgen_rft.jsonl"; do
        [[ -e "${path}" ]] && return 0
    done
    compgen -G "${RLAD_DATA}/rft_abs_cache*.jsonl" >/dev/null ||
        compgen -G "${RLAD_DATA}/rft_scored*.jsonl" >/dev/null
}

ensure_manifest() {
    local manifest="${STATE_DIR}/manifest.env" current
    current=$(manifest_text)
    if [[ -f "${manifest}" ]]; then
        [[ "$(<"${manifest}")" == "${current}" ]] ||
            die "pipeline parameters or repository commit changed; resume with the original settings, or move train/rl/data and train/rl/runs aside for a new full run"
        return
    fi
    if unsafe_artifacts_exist; then
        die "pipeline artifacts exist without a matching manifest; move train/rl/data and train/rl/runs aside before starting"
    fi
    atomic_write "${manifest}" "${current}"
}

acquire_lock() {
    mkdir -p "${STATE_DIR}"
    exec 9>"${STATE_DIR}/controller.lock"
    flock -n 9 || die "another RFT_pipeline controller is active"
}

on_signal() {
    warn "controller interrupted${ACTIVE_JOB:+; Slurm job ${ACTIVE_JOB} was left running}"
    exit 130
}

job_state() {
    local jid=$1 state raw
    state=""
    if raw=$("${SACCT_BIN}" -X -n -P -j "${jid}" --format=JobIDRaw,State 2>/dev/null); then
        state=$(awk -F'|' -v id="${jid}" \
            '$1 == id {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' <<< "${raw}")
    fi
    if [[ -z "${state}" ]] && command -v "${SCONTROL_BIN}" >/dev/null 2>&1; then
        if raw=$("${SCONTROL_BIN}" show job -o "${jid}" 2>/dev/null); then
            state=$(sed -n 's/.*JobState=\([^ ]*\).*/\1/p' <<< "${raw}")
            state=${state%%$'\n'*}
        fi
    fi
    printf '%s\n' "${state}"
}

job_active() {
    local jid=$1 output line
    # `squeue -j <finished-id>` returns rc=1 on common Slurm versions. Querying
    # the user's queue makes normal job completion distinguishable from a real
    # scheduler transport failure.
    output=$("${SQUEUE_BIN}" -h -u "${USER}" -o '%A' 2>/dev/null) ||
        die "squeue failed while checking job ${jid}"
    while IFS= read -r line; do
        [[ "${line}" == "${jid}" ]] && return 0
    done <<< "${output}"
    return 1
}

normalized_state() {
    local state=$1
    state=${state%% *}
    state=${state%%+*}
    printf '%s\n' "${state}"
}

state_is_terminal() {
    case "$1" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|\
        DEADLINE|REVOKED|SPECIAL_EXIT) return 0 ;;
        *) return 1 ;;
    esac
}

submit_stage() {
    local stage=$1 job_name=$2 job_file raw jid existing prior_state
    shift 2
    job_file="${STATE_DIR}/jobs/${stage}.job"
    mkdir -p "${STATE_DIR}/jobs"

    if [[ -f "${job_file}" ]]; then
        jid=$(<"${job_file}")
        if [[ "${jid}" =~ ^[0-9]+$ ]] && job_active "${jid}"; then
            log "Reusing active ${stage} job ${jid}" >&2
            printf '%s\n' "${jid}"
            return
        fi
        prior_state=$(normalized_state "$(job_state "${jid}")")
        [[ -n "${prior_state}" ]] ||
            die "job ${jid} disappeared but Slurm accounting has no terminal state; not resubmitting ${stage}"
        state_is_terminal "${prior_state}" ||
            die "job ${jid} is ${prior_state} in accounting but absent from squeue; retry when Slurm reconciles it"
    fi

    existing=$("${SQUEUE_BIN}" -h -u "${USER}" -n "${job_name}" -o '%A' 2>/dev/null) ||
        die "squeue failed while recovering ${stage}"
    existing=${existing%%$'\n'*}
    if [[ "${existing}" =~ ^[0-9]+$ ]]; then
        atomic_write "${job_file}" "${existing}"
        log "Recovered active ${stage} job ${existing}" >&2
        printf '%s\n' "${existing}"
        return
    fi

    if ! raw=$("${SBATCH_COMMAND}" --parsable --job-name="${job_name}" "$@"); then
        die "failed to submit ${stage}"
    fi
    jid=${raw%%;*}
    [[ "${jid}" =~ ^[0-9]+$ ]] || die "unexpected sbatch response for ${stage}: ${raw}"
    atomic_write "${job_file}" "${jid}"
    printf '%(%Y-%m-%dT%H:%M:%SZ)T\t%s\t%s\t%s\n' -1 "${stage}" "${jid}" "${job_name}" \
        >> "${STATE_DIR}/jobs.tsv"
    log "Submitted ${stage} as job ${jid} (${job_name})" >&2
    printf '%s\n' "${jid}"
}

wait_for_job() {
    local jid=$1 stage=$2 polls=0 state="" i
    ACTIVE_JOB=${jid}
    log "Waiting for ${stage} job ${jid}"
    while job_active "${jid}"; do
        sleep "${POLL_SECONDS}"
        polls=$((polls + 1))
        (( polls % 10 == 0 )) && log "${stage} job ${jid} is still active"
    done
    for i in $(seq 1 12); do
        state=$(normalized_state "$(job_state "${jid}")")
        state_is_terminal "${state}" && break
        sleep 5
    done
    ACTIVE_JOB=""
    state_is_terminal "${state}" || die "no terminal accounting state for ${stage} job ${jid}"
    atomic_write "${STATE_DIR}/jobs/${stage}.state" "${state}"
    log "${stage} job ${jid} ended in ${state}"
    [[ "${state}" == COMPLETED ]]
}

recorded_job_completed() {
    local stage=$1 file jid state
    file="${STATE_DIR}/jobs/${stage}.job"
    [[ -f "${file}" ]] || return 1
    jid=$(<"${file}")
    [[ "${jid}" =~ ^[0-9]+$ ]] || return 1
    job_active "${jid}" && return 1
    state=$(normalized_state "$(job_state "${jid}")")
    [[ "${state}" == COMPLETED ]]
}

pool_complete() {
    python3 - "${RLAD_DATA}/benchmarks/dsr_pool.jsonl" "${RLAD_DATA}/dsr_pool_meta.json" \
        "${POOL_SIZE}" "${POOL_SEED}" <<'PY' >/dev/null 2>&1
import json, sys
rows_path, meta_path, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    rows = [json.loads(x) for x in open(rows_path, encoding="utf-8") if x.strip()]
    meta = json.load(open(meta_path, encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
ids = [r.get("id") for r in rows]
assert len(rows) == n and len(set(ids)) == n
assert all(i and i.startswith("dsr-") and r.get("problem") and str(r.get("answer", "")) for i, r in zip(ids, rows))
assert meta.get("n_pool") == n and meta.get("seed") == seed
PY
}

base_score_progress() {
    python3 - "${RLAD_RUNS}/eval/dsr_pool_score/dsr_pool" <<'PY'
import glob, json, os, sys
counts = {}
for path in glob.glob(os.path.join(sys.argv[1], "samples*.jsonl")):
    try:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line); counts[row["id"]] = counts.get(row["id"], 0) + 1
    except (OSError, ValueError):
        pass
print(sum(counts.values()))
PY
}

base_scores_complete() {
    python3 - "${RLAD_DATA}/benchmarks/dsr_pool.jsonl" \
        "${RLAD_RUNS}/eval/dsr_pool_score/dsr_pool" "${BASE_SAMPLES}" <<'PY' >/dev/null 2>&1
import collections, glob, json, os, sys
pool_path, out_dir, n_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    pool = [json.loads(x) for x in open(pool_path, encoding="utf-8") if x.strip()]
    valid = {r["id"] for r in pool}
    rows = []
    for path in glob.glob(os.path.join(out_dir, "samples*.jsonl")):
        rows.extend(json.loads(x) for x in open(path, encoding="utf-8") if x.strip())
except (OSError, ValueError, KeyError):
    raise SystemExit(1)
seen = collections.defaultdict(set)
for row in rows:
    qid, idx = row.get("id"), row.get("sample_idx")
    assert qid in valid and isinstance(idx, int) and 0 <= idx < n_samples
    assert row.get("correct") in (0, 1)
    assert idx not in seen[qid]
    seen[qid].add(idx)
assert set(seen) == valid
assert all(v == set(range(n_samples)) for v in seen.values())
assert len(rows) == len(valid) * n_samples
PY
}

base_checkpoint_complete() {
    local dir="${RLAD_RUNS}/qwen3_1p7b_torch_dist"
    [[ -f "${dir}/latest_checkpointed_iteration.txt" &&
       "$(<"${dir}/latest_checkpointed_iteration.txt")" == release ]] || return 1
    find "${dir}" -type f ! -name latest_checkpointed_iteration.txt -size +0c \
        -print -quit 2>/dev/null | grep -q .
}

curriculum_complete() {
    python3 - "${RLAD_DATA}" "${POOL_SIZE}" "${HARD_MAX}" "${EASY_MIN}" "${RFT_NPROB}" <<'PY' >/dev/null 2>&1
import json, os, sys
d, n, hard, easy, nprob = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
def read(name):
    return [json.loads(x) for x in open(os.path.join(d, name), encoding="utf-8") if x.strip()]
try:
    er, mr = read("train_easy.jsonl"), read("train_medium.jsonl")
    hr = read("benchmarks/dsr_hard.jsonl")
    pool = read("benchmarks/dsr_pool.jsonl")
    meta = json.load(open(os.path.join(d, "curriculum_meta.json"), encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
eids = {r["metadata"]["qid"] for r in er}; mids = {r["metadata"]["qid"] for r in mr}
hids = {r["id"] for r in hr}
pool_ids = {r["id"] for r in pool}
assert er and mr and not (eids & mids or eids & hids or mids & hids)
assert len(er) == len(eids) and len(mr) == len(mids) and len(hr) == len(hids)
assert len(pool) == len(pool_ids) == n and (eids | mids | hids) == pool_ids
assert len(eids) + len(mids) >= nprob
assert len(eids) == meta["counts"]["easy"] and len(mids) == meta["counts"]["medium"]
assert len(hids) == meta["counts"]["hard"] and meta.get("n_scored") == n
assert abs(float(meta.get("hard_max")) - hard) < 1e-12
assert abs(float(meta.get("easy_min")) - easy) < 1e-12
PY
}

warmstart_complete() {
    python3 - "${RLAD_DATA}" "${SFT_K}" "${WARMSTART_MODEL}" <<'PY' >/dev/null 2>&1
import json, os, sys
d, k, generator = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def read(name):
    return [json.loads(x) for x in open(os.path.join(d, name), encoding="utf-8") if x.strip()]
try:
    curriculum = read("train_easy.jsonl") + read("train_medium.jsonl")
    rows = read("train_absgen_sft.jsonl")
    meta = json.load(open(os.path.join(d, "absgen_sft_meta.json"), encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
qids = {r["metadata"]["qid"] for r in curriculum}
assert rows and meta.get("n_problems") == len(qids) and meta.get("k") == k
assert meta.get("kept") == len(rows) and meta.get("generator") == generator
for row in rows:
    assert row.get("metadata", {}).get("qid") in qids
    msgs = row.get("messages"); assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant"
    assert msgs[1].get("content")
PY
}

jsonl_rows() {
    python3 - "$1" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip(): json.loads(line); n += 1
print(n)
PY
}

files_digest() {
    local file
    for file in "$@"; do sha256sum "${file}"; done | sha256sum | awk '{print $1}'
}

expected_iteration() {
    local data=$1 epochs=$2 rows rollouts
    rows=$(jsonl_rows "${data}")
    rollouts=$((rows / TRAIN_BATCH))
    (( rollouts > 0 )) || die "${data} has ${rows} rows; at least ${TRAIN_BATCH} are required"
    printf '%s\n' "$((rollouts * epochs - 1))"
}

checkpoint_at_iteration() {
    local dir=$1 expected=$2 tracker iter_dir
    tracker="${dir}/latest_checkpointed_iteration.txt"
    [[ -f "${tracker}" ]] || return 1
    [[ "$(<"${tracker}")" == "${expected}" ]] || return 1
    printf -v iter_dir '%s/iter_%07d' "${dir}" "${expected}"
    [[ -d "${iter_dir}" ]] && find "${iter_dir}" -type f -size +0c -print -quit | grep -q .
}

hf_artifacts_valid() {
    python3 - "$1" <<'PY' >/dev/null 2>&1
import json, os, sys
d = sys.argv[1]
config_path = os.path.join(d, "config.json")
assert os.path.isfile(config_path) and os.path.getsize(config_path) > 0
config = json.load(open(config_path, encoding="utf-8"))
assert isinstance(config, dict) and config
tokenizer_ok = False
for name in ("tokenizer_config.json", "tokenizer.json"):
    path = os.path.join(d, name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        value = json.load(open(path, encoding="utf-8"))
        tokenizer_ok |= isinstance(value, dict) and bool(value)
assert tokenizer_ok
valid_weights = False
for name in ("model.safetensors", "pytorch_model.bin"):
    path = os.path.join(d, name)
    valid_weights |= os.path.isfile(path) and os.path.getsize(path) > 0
for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    path = os.path.join(d, name)
    if os.path.isfile(path):
        index = json.load(open(path, encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        assert shards
        for shard in shards:
            shard_path = os.path.join(d, shard)
            assert os.path.isfile(shard_path) and os.path.getsize(shard_path) > 0
        valid_weights = True
assert valid_weights
PY
}

hf_fingerprint() {
    local dir=$1 files=()
    mapfile -t files < <(find "${dir}" -maxdepth 1 -type f -print | sort)
    ((${#files[@]} > 0)) || return 1
    files_digest "${files[@]}"
}

rft_cache_files() {
    find "${RLAD_DATA}" -maxdepth 1 -type f \
        \( -name 'rft_abs_cache*.jsonl' -o -name 'rft_scored*.jsonl' \) -print | sort
}

rft_abs_files() {
    find "${RLAD_DATA}" -maxdepth 1 -type f -name 'rft_abs_cache*.jsonl' -print | sort
}

rft_abs_digest() {
    local files=()
    mapfile -t files < <(rft_abs_files)
    ((${#files[@]} > 0)) || return 1
    files_digest "${files[@]}"
}

rft_generation_complete() {
    local marker="${STATE_DIR}/rft_gen.complete"
    [[ -f "${marker}" ]] && validate_rft gen &&
        [[ "$(<"${marker}")" == "$(rft_abs_digest 2>/dev/null || true)" ]]
}

rft_scoring_complete() {
    rft_generation_complete && validate_rft score
}

rft_corpus_complete() {
    [[ -f "${STATE_DIR}/rft_data.input.key" ]] && rft_scoring_complete && validate_rft corpus
}

validate_rft() {
    local mode=$1
    python3 - "${mode}" "${RLAD_DATA}" "${RLAD_RUNS}" "${RFT_K}" "${RFT_M}" \
        "${RFT_MARGIN}" "${RFT_MIN_ROWS}" "${RFT_NPROB}" "${RLAD_HOME}" <<'PY' >/dev/null 2>&1
import collections, glob, json, os, re, sys
mode, d, runs = sys.argv[1:4]
k, m, margin = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
min_rows, nprob = int(sys.argv[7]), int(sys.argv[8])
rlad_home = sys.argv[9]
def paths(stem):
    base = os.path.join(d, stem + ".jsonl")
    shards = sorted(glob.glob(os.path.join(d, stem + ".shard*.jsonl")))
    assert not (os.path.exists(base) and shards), "mixed sharded/unsharded caches"
    return ([base] if os.path.exists(base) else []) + shards
def rows(stem):
    out = []
    for path in paths(stem):
        for line in open(path, encoding="utf-8"):
            if line.strip(): out.append(json.loads(line))
    return out
abs_rows = rows("rft_abs_cache")
assert abs_rows
abs_keys = [(r["qid"], int(r["abs_idx"])) for r in abs_rows]
assert len(abs_keys) == len(set(abs_keys))
assert all(0 <= idx < k and qid and r.get("abstraction") and "answer" in r
           for (qid, idx), r in zip(abs_keys, abs_rows))
curriculum_rows = []
for name in ("train_easy.jsonl", "train_medium.jsonl"):
    curriculum_rows += [json.loads(x) for x in open(os.path.join(d, name), encoding="utf-8") if x.strip()]
curriculum = [row["metadata"]["qid"] for row in curriculum_rows]
expected_qids = set(curriculum[:nprob])
assert expected_qids and all(qid in expected_qids for qid, idx in abs_keys)
if mode == "gen": raise SystemExit(0)
scored = rows("rft_scored")
score_keys = [(r["qid"], int(r["abs_idx"])) for r in scored]
assert len(score_keys) == len(set(score_keys))
assert set(score_keys).issubset(set(abs_keys))
abs_by_key = dict(zip(abs_keys, abs_rows))
for r in scored:
    original = abs_by_key[(r["qid"], int(r["abs_idx"]))]
    assert r.get("answer") == original.get("answer")
    assert r.get("abstraction") == original.get("abstraction")
    value = float(r["r_sol"])
    assert 0.0 <= value <= 1.0 and abs(value * m - round(value * m)) < 1e-8
if mode == "partial-score": raise SystemExit(0)
assert set(score_keys) == set(abs_keys)
if mode == "score": raise SystemExit(0)
try:
    corpus = [json.loads(x) for x in open(os.path.join(d, "train_absgen_rft.jsonl"), encoding="utf-8") if x.strip()]
    meta = json.load(open(os.path.join(d, "absgen_rft_meta.json"), encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
assert len(corpus) >= min_rows and meta.get("kept") == len(corpus)
assert abs(float(meta.get("margin")) - margin) < 1e-12
assert meta.get("kept", 0) + meta.get("dropped_ineffective", 0) + meta.get("dropped_leak", 0) == len(scored)
problem_by_qid = {row["metadata"]["qid"]: row["metadata"].get("problem") for row in curriculum_rows}
assert all(problem_by_qid.get(qid) for qid in expected_qids)
sys.path.insert(0, rlad_home)
from rlad_plugin.templates import ABSGEN_INSTRUCTION
base_values = collections.defaultdict(list)
for path in glob.glob(os.path.join(runs, "eval", "dsr_pool_score", "dsr_pool", "samples*.jsonl")):
    for line in open(path, encoding="utf-8"):
        if line.strip():
            value = json.loads(line)
            base_values[value["id"]].append(int(value["correct"]))
base = {qid: sum(values) / len(values) for qid, values in base_values.items() if values}
def leaks(abstraction, answer):
    normalized = re.sub(r"\s+", "", str(answer))
    if len(normalized) < 3:
        return False
    if re.fullmatch(r"[A-Za-z0-9]+", normalized):
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(normalized) +
                         r"(?![A-Za-z0-9])", re.sub(r"\s+", " ", abstraction)) is not None
    return normalized in re.sub(r"\s+", "", abstraction)
expected = collections.Counter()
dropped_ineffective = dropped_leak = 0
for row in scored:
    base_score = base.get(row["qid"], 0.0)
    score = float(row["r_sol"])
    if score <= base_score + margin:
        dropped_ineffective += 1
        continue
    if leaks(row["abstraction"], row["answer"]):
        dropped_leak += 1
        continue
    user = ABSGEN_INSTRUCTION + "\n\nProblem:\n" + problem_by_qid[row["qid"]]
    expected[(row["qid"], user, row["abstraction"], round(score, 3), round(base_score, 3))] += 1
assert meta.get("dropped_ineffective") == dropped_ineffective
assert meta.get("dropped_leak") == dropped_leak
actual = collections.Counter()
for row in corpus:
    msgs = row.get("messages"); assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant"
    assert msgs[0].get("content") and msgs[1].get("content")
    metadata = row.get("metadata"); assert isinstance(metadata, dict)
    actual[(metadata.get("qid"), msgs[0]["content"], msgs[1]["content"],
            round(float(metadata["r_sol"]), 3), round(float(metadata["base"]), 3))] += 1
assert actual == expected and len(corpus) == sum(expected.values())
PY
}

ensure_stage_key() {
    local name=$1 key=$2 unsafe_path=$3 file
    file="${STATE_DIR}/${name}.input.key"
    if [[ -f "${file}" ]]; then
        [[ "$(<"${file}")" == "${key}" ]] ||
            die "${name} inputs changed while its artifacts exist; archive the old run"
    else
        [[ ! -e "${unsafe_path}" ]] ||
            die "${name} artifacts exist without pipeline provenance: ${unsafe_path}"
        atomic_write "${file}" "${key}"
    fi
}

run_base_stages() {
    local prep_jid="" jid state progress previous=-1 stagnant=0 attempt
    if ! base_checkpoint_complete; then
        prep_jid=$(submit_stage base_ckpt "${PIPELINE_ID}-baseckpt" \
            "${RL_DIR}/jobs/prep_megatron_ckpt.sbatch")
    fi

    if ! pool_complete; then
        log "Building deterministic ${POOL_SIZE}-problem curriculum pool"
        python -m rlad_plugin.data_prep build-pool --n-pool "${POOL_SIZE}" --seed "${POOL_SEED}"
        pool_complete || die "pool validation failed"
    fi

    for attempt in $(seq 1 "${BASE_EVAL_ATTEMPTS}"); do
        base_scores_complete && break
        progress=$(base_score_progress)
        log "Base-score progress: ${progress}/$((POOL_SIZE * BASE_SAMPLES)) rows"
        jid=$(submit_stage base_eval "${PIPELINE_ID}-basescore" \
            --export="ALL,MODEL_PATH=${BASE_MODEL},BENCHMARKS=dsr_pool,N_SAMPLES=${BASE_SAMPLES},MAX_TOKENS=${BASE_MAX_TOKENS},NUM_SHARDS=${NSHARDS},OUT_DIR=${RLAD_RUNS}/eval/dsr_pool_score" \
            "${RL_DIR}/jobs/eval.sbatch")
        if ! wait_for_job "${jid}" base_eval; then
            state=$(normalized_state "$(job_state "${jid}")")
            warn "base evaluation segment ended in ${state}; artifacts will decide whether to retry"
        fi
        progress=$(base_score_progress)
        if [[ "${progress}" == "${previous}" ]]; then stagnant=$((stagnant + 1)); else stagnant=0; fi
        previous=${progress}
        (( stagnant < 2 )) || die "base evaluation made no progress across two segments"
    done
    base_scores_complete || die "base scoring is incomplete after ${BASE_EVAL_ATTEMPTS} segments; rerun resume after inspecting logs"

    if ! base_checkpoint_complete; then
        for attempt in $(seq 1 "${BASE_CKPT_ATTEMPTS}"); do
            [[ -n "${prep_jid}" ]] || prep_jid=$(submit_stage base_ckpt "${PIPELINE_ID}-baseckpt" \
                "${RL_DIR}/jobs/prep_megatron_ckpt.sbatch")
            if wait_for_job "${prep_jid}" base_ckpt && base_checkpoint_complete; then
                break
            fi
            warn "base checkpoint attempt ${attempt} failed validation"
            prep_jid=""
        done
    fi
    base_checkpoint_complete || die "base checkpoint validation failed"

    if ! curriculum_complete; then
        log "Partitioning the complete base scores"
        python -m rlad_plugin.data_prep partition --hard-max "${HARD_MAX}" --easy-min "${EASY_MIN}"
        curriculum_complete || die "curriculum validation failed"
    fi
}

run_warmstart() {
    local jid attempt
    warmstart_complete && return
    for attempt in $(seq 1 "${WARMSTART_ATTEMPTS}"); do
        jid=$(submit_stage warmstart "${PIPELINE_ID}-warmstart" \
            --time="${WARMSTART_TIME}" \
            --export="ALL,LIMIT=0,K=${SFT_K},GENERATOR=${WARMSTART_MODEL},GENERATOR_MAX_TOKENS=${WARMSTART_MAX_TOKENS},GENERATOR_TEMPERATURE=${WARMSTART_TEMPERATURE},OUT=${RLAD_DATA}/train_absgen_sft.jsonl" \
            "${RL_DIR}/jobs/warmstart.sbatch")
        if wait_for_job "${jid}" warmstart && warmstart_complete; then return; fi
    done
    die "warm-start corpus is incomplete; this stage regenerates from scratch, so inspect its log before retrying"
}

run_training() {
    local name=$1 data=$2 epochs=$3 segments=$4 arm_config=$5 job_suffix=$6 extra_export=${7:-}
    local ckpt_dir expected input_key marker blocked jid attempt state exports
    ckpt_dir="${RLAD_RUNS}/${name}/ckpts"
    expected=$(expected_iteration "${data}" "${epochs}")
    input_key=$(printf '%s|%s' \
        "$(files_digest "${data}" "${arm_config}" "${RL_DIR}/jobs/sft_launch.sh")" \
        "${extra_export}" | sha256sum | awk '{print $1}')
    ensure_stage_key "${name}" "${input_key}" "${ckpt_dir}/latest_checkpointed_iteration.txt"
    marker="${STATE_DIR}/${name}.complete"
    blocked="${STATE_DIR}/${name}.blocked"
    if [[ -f "${blocked}" ]]; then
        [[ "${RLAD_RETRY_FAILED:-0}" == 1 ]] ||
            die "${name} is blocked after $(<"${blocked}"); inspect logs, then set RLAD_RETRY_FAILED=1 to retry"
        mv -- "${blocked}" "${blocked}.retried.$(date -u +%Y%m%dT%H%M%SZ).$$"
    fi
    if [[ -f "${marker}" ]]; then
        if [[ "$(<"${marker}")" == "${input_key}:${expected}" ]] &&
           checkpoint_at_iteration "${ckpt_dir}" "${expected}"; then
            return
        fi
        die "${name} completion marker no longer matches its checkpoint; move ${RLAD_RUNS}/${name} and ${marker} aside before retrying"
    fi
    if checkpoint_at_iteration "${ckpt_dir}" "${expected}" &&
       recorded_job_completed "${name}_train"; then
        atomic_write "${marker}" "${input_key}:${expected}"
        log "Reconciled completed ${name} training job from Slurm accounting"
        return
    fi

    exports="ALL,ARM_CONFIG=${arm_config},LAUNCHER=${RL_DIR}/jobs/sft_launch.sh"
    [[ -n "${extra_export}" ]] && exports+=",${extra_export}"
    for attempt in $(seq 1 "${segments}"); do
        jid=$(submit_stage "${name}_train" "${PIPELINE_ID}-${job_suffix}" \
            --export="${exports}" "${RL_DIR}/jobs/submit_train.sbatch")
        if wait_for_job "${jid}" "${name}_train"; then
            checkpoint_at_iteration "${ckpt_dir}" "${expected}" ||
                die "${name} job completed but tracker did not reach expected iteration ${expected}"
            atomic_write "${marker}" "${input_key}:${expected}"
            return
        fi
        state=$(normalized_state "$(job_state "${jid}")")
        case "${state}" in
            TIMEOUT|PREEMPTED|NODE_FAIL|BOOT_FAIL)
                warn "${name} segment ${attempt} is resumable after ${state}"
                ;;
            *)
                atomic_write "${blocked}" "${state}"
                die "${name} training stopped in non-retryable state ${state}"
                ;;
        esac
    done
    die "${name} did not finish after ${segments} segments; rerun resume to append more"
}

run_conversion() {
    local name=$1 ckpt_dir=$2 expected=$3 job_suffix=$4
    local out_dir marker marker_prefix marker_value fingerprint jid attempt
    out_dir="${RLAD_RUNS}/${name}/hf/iter_${expected}"
    marker="${STATE_DIR}/${name}.hf.complete"
    marker_prefix="$(files_digest "${ckpt_dir}/latest_checkpointed_iteration.txt" \
        "${STATE_DIR}/${name}.complete"):${expected}:"
    if [[ -f "${marker}" ]]; then
        hf_artifacts_valid "${out_dir}" ||
            die "trusted HF export for ${name} is incomplete; move ${out_dir} and ${marker} aside before retrying"
        fingerprint=$(hf_fingerprint "${out_dir}")
        marker_value="${marker_prefix}${fingerprint}"
        [[ "$(<"${marker}")" == "${marker_value}" ]] ||
            die "trusted HF export for ${name} changed after conversion; move ${out_dir} and ${marker} aside before retrying"
        printf '%s\n' "${out_dir}"
        return
    fi
    if hf_artifacts_valid "${out_dir}" && recorded_job_completed "${name}_convert"; then
        fingerprint=$(hf_fingerprint "${out_dir}")
        marker_value="${marker_prefix}${fingerprint}"
        atomic_write "${marker}" "${marker_value}"
        log "Reconciled completed offline conversion for ${name}"
        printf '%s\n' "${out_dir}"
        return
    fi
    for attempt in $(seq 1 "${CONVERT_ATTEMPTS}"); do
        jid=$(submit_stage "${name}_convert" "${PIPELINE_ID}-${job_suffix}" \
            --export="ALL,CKPT_DIR=${ckpt_dir},OUT_DIR=${out_dir},ITER=${expected}" \
            "${RL_DIR}/jobs/convert_hf.sbatch")
        if wait_for_job "${jid}" "${name}_convert" && hf_artifacts_valid "${out_dir}"; then
            fingerprint=$(hf_fingerprint "${out_dir}")
            marker_value="${marker_prefix}${fingerprint}"
            atomic_write "${marker}" "${marker_value}"
            printf '%s\n' "${out_dir}"
            return
        fi
    done
    die "offline conversion for ${name} failed validation"
}

ensure_rft_cache_key() {
    local sft_hf=$1 file="${STATE_DIR}/rft_data.input.key" key existing=0
    [[ -f "${STATE_DIR}/sft_absgen.hf.complete" ]] || die "trusted SFT conversion marker is missing"
    key="sft_hf=$(readlink -f -- "${sft_hf}")|digest=$(files_digest "${sft_hf}/config.json" "${STATE_DIR}/sft_absgen.hf.complete")|nprob=${RFT_NPROB}|k=${RFT_K}|m=${RFT_M}|maxtok=${RFT_MAXTOK}|margin=${RFT_MARGIN}|shards=${NSHARDS}"
    rft_cache_files | grep -q . && existing=1
    [[ -e "${RLAD_DATA}/train_absgen_rft.jsonl" ]] && existing=1
    if [[ -f "${file}" ]]; then
        [[ "$(<"${file}")" == "${key}" ]] ||
            die "RFT cache provenance no longer matches the trusted SFT export; run '$0 archive-rft' before regenerating"
    else
        (( existing == 0 )) || die "RFT caches exist without provenance; run '$0 archive-rft'"
        atomic_write "${file}" "${key}"
    fi
}

run_rft_data() {
    local sft_hf=$1 jid attempt state progress previous=-1 stagnant=0 gen_marker cache_hash
    ensure_rft_cache_key "${sft_hf}"
    gen_marker="${STATE_DIR}/rft_gen.complete"

    if rft_abs_files | grep -q . && ! validate_rft gen; then
        die "existing RFT abstraction cache is malformed or has duplicate/stale keys; run '$0 archive-rft'"
    fi
    if [[ -f "${gen_marker}" ]] && ! rft_generation_complete; then
        die "frozen RFT abstraction cache changed after generation; run '$0 archive-rft'"
    fi
    if [[ ! -f "${gen_marker}" ]] && validate_rft gen && recorded_job_completed rft_gen; then
        atomic_write "${gen_marker}" "$(rft_abs_digest)"
        log "Reconciled completed RFT abstraction-generation job"
    fi
    if ! rft_generation_complete; then
        for attempt in $(seq 1 "${RFT_GEN_ATTEMPTS}"); do
            jid=$(submit_stage rft_gen "${PIPELINE_ID}-rftgen" \
                --export="ALL,RFT_STAGE=gen-abs,ABSGEN_HF=${sft_hf},NSHARDS=${NSHARDS},NPROB=${RFT_NPROB},K=${RFT_K},M=${RFT_M},MAXTOK=${RFT_MAXTOK},MARGIN=${RFT_MARGIN}" \
                "${RL_DIR}/jobs/rft_data.sbatch")
            if wait_for_job "${jid}" rft_gen && validate_rft gen; then
                cache_hash=$(rft_abs_digest)
                atomic_write "${gen_marker}" "${cache_hash}"
                break
            fi
        done
    fi
    validate_rft gen || die "RFT abstraction generation cache is invalid; archive it before retrying"
    cache_hash=$(rft_abs_digest)
    [[ -f "${gen_marker}" && "$(<"${gen_marker}")" == "${cache_hash}" ]] ||
        die "RFT abstraction cache changed after generation was frozen"

    for attempt in $(seq 1 "${RFT_SCORE_SEGMENTS}"); do
        rft_scoring_complete && break
        if compgen -G "${RLAD_DATA}/rft_scored*.jsonl" >/dev/null &&
           ! validate_rft partial-score; then
            die "partial RFT score cache is malformed or contains duplicate/stale keys; run '$0 archive-rft'"
        fi
        progress=$(python3 - "${RLAD_DATA}" <<'PY'
import glob, json, os, sys
keys=set()
for p in glob.glob(os.path.join(sys.argv[1], "rft_scored*.jsonl")):
    try:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r=json.loads(line); keys.add((r["qid"], int(r["abs_idx"])))
    except (OSError, ValueError, KeyError): pass
print(len(keys))
PY
)
        log "RFT-score progress: ${progress} unique abstractions"
        jid=$(submit_stage rft_score "${PIPELINE_ID}-rftscore" \
            --export="ALL,RFT_STAGE=score,ABSGEN_HF=${sft_hf},NSHARDS=${NSHARDS},NPROB=${RFT_NPROB},K=${RFT_K},M=${RFT_M},MAXTOK=${RFT_MAXTOK},MARGIN=${RFT_MARGIN}" \
            "${RL_DIR}/jobs/rft_data.sbatch")
        if ! wait_for_job "${jid}" rft_score; then
            state=$(normalized_state "$(job_state "${jid}")")
            warn "RFT scoring segment ended in ${state}; validating partial progress"
        fi
        [[ "$(rft_abs_digest)" == "${cache_hash}" ]] ||
            die "frozen abstraction cache changed during RFT scoring"
        if [[ "${progress}" == "${previous}" ]]; then stagnant=$((stagnant + 1)); else stagnant=0; fi
        previous=${progress}
        (( stagnant < 2 )) || die "RFT scoring made no progress across two segments"
    done
    rft_scoring_complete || die "RFT scoring is incomplete after ${RFT_SCORE_SEGMENTS} segments; inspect logs and resume"

    if ! rft_corpus_complete; then
        log "Applying rejection rule and building the RFT corpus"
        python -m rlad_plugin.absgen_score build-rft --margin "${RFT_MARGIN}"
    fi
    rft_corpus_complete ||
        die "RFT corpus validation failed or produced fewer than ${RFT_MIN_ROWS} accepted rows"
}

run_pipeline() {
    local sft_expected rft_expected sft_hf rft_hf
    trap on_signal INT TERM HUP
    acquire_lock
    ensure_manifest

    run_base_stages
    run_warmstart

    run_training sft_absgen "${RLAD_DATA}/train_absgen_sft.jsonl" "${SFT_EPOCHS}" \
        "${SFT_SEGMENTS}" "${RL_DIR}/rlad_plugin/configs/sft_absgen.sh" sft \
        "SFT_EPOCHS=${SFT_EPOCHS},SFT_BATCH=${TRAIN_BATCH}"
    sft_expected=$(expected_iteration "${RLAD_DATA}/train_absgen_sft.jsonl" "${SFT_EPOCHS}")
    sft_hf=$(run_conversion sft_absgen "${RLAD_RUNS}/sft_absgen/ckpts" "${sft_expected}" sft-hf)
    hf_artifacts_valid "${sft_hf}" || die "trusted SFT HF export is invalid"

    run_rft_data "${sft_hf}"
    run_training sft_absgen_rft "${RLAD_DATA}/train_absgen_rft.jsonl" "${RFT_EPOCHS}" \
        "${RFT_TRAIN_SEGMENTS}" "${RL_DIR}/rlad_plugin/configs/rft_absgen.sh" rft \
        "RFT_EPOCHS=${RFT_EPOCHS},SFT_BATCH=${TRAIN_BATCH},SFT_ABSGEN_HF=${sft_hf}"
    rft_expected=$(expected_iteration "${RLAD_DATA}/train_absgen_rft.jsonl" "${RFT_EPOCHS}")
    rft_hf=$(run_conversion sft_absgen_rft "${RLAD_RUNS}/sft_absgen_rft/ckpts" \
        "${rft_expected}" rft-hf)

    atomic_write "${STATE_DIR}/final_model" "${rft_hf}"
    log "RFT pipeline complete"
    log "Final abstraction generator: ${rft_hf}"
}

stage_status() {
    local label=$1
    shift
    if "$@"; then printf '  %-22s COMPLETE\n' "${label}"; else printf '  %-22s INCOMPLETE\n' "${label}"; fi
}

show_status() {
    printf 'Pipeline: %s\n' "${PIPELINE_ID}"
    printf 'Profile:  %s\n' "${PROFILE}"
    printf 'State:    %s\n' "${STATE_DIR}"
    stage_status 'curriculum pool' pool_complete
    stage_status 'base scores' base_scores_complete
    stage_status 'base checkpoint' base_checkpoint_complete
    stage_status 'curriculum split' curriculum_complete
    stage_status 'SFT corpus' warmstart_complete
    if [[ -f "${RLAD_RUNS}/sft_absgen/ckpts/latest_checkpointed_iteration.txt" ]]; then
        printf '  %-22s %s\n' 'SFT tracker' "$(<"${RLAD_RUNS}/sft_absgen/ckpts/latest_checkpointed_iteration.txt")"
    else
        printf '  %-22s INCOMPLETE\n' 'SFT tracker'
    fi
    stage_status 'RFT generated cache' rft_generation_complete
    stage_status 'RFT scored cache' rft_scoring_complete
    stage_status 'RFT corpus' rft_corpus_complete
    if [[ -f "${RLAD_RUNS}/sft_absgen_rft/ckpts/latest_checkpointed_iteration.txt" ]]; then
        printf '  %-22s %s\n' 'RFT tracker' "$(<"${RLAD_RUNS}/sft_absgen_rft/ckpts/latest_checkpointed_iteration.txt")"
    else
        printf '  %-22s INCOMPLETE\n' 'RFT tracker'
    fi
    if [[ -f "${STATE_DIR}/final_model" ]]; then
        local final_model
        final_model=$(<"${STATE_DIR}/final_model")
        if hf_artifacts_valid "${final_model}"; then
            printf '  %-22s %s\n' 'final model' "${final_model}"
        else
            printf '  %-22s INCOMPLETE (stale path: %s)\n' 'final model' "${final_model}"
        fi
    fi
    if [[ -d "${STATE_DIR}/jobs" ]]; then
        printf '\nRecorded jobs:\n'
        local file jid state
        for file in "${STATE_DIR}"/jobs/*.job; do
            [[ -e "${file}" ]] || continue
            jid=$(<"${file}"); state=$(normalized_state "$(job_state "${jid}")")
            printf '  %-22s %-12s %s\n' "$(basename "${file}" .job)" "${jid}" "${state:-UNKNOWN}"
        done
    fi
}

assert_no_active_rft_jobs() {
    local file jid output active_id active_name raw workdir

    # Recorded IDs catch jobs submitted by an older commit whose job-name prefix
    # differs from the controller currently on disk.
    for file in "${STATE_DIR}"/jobs/rft_*.job "${STATE_DIR}"/jobs/sft_absgen_rft*.job; do
        [[ -e "${file}" ]] || continue
        jid=$(<"${file}")
        if [[ "${jid}" =~ ^[0-9]+$ ]] && job_active "${jid}"; then
            die "refusing to archive while recorded RFT job ${jid} is active"
        fi
    done

    # Also catch a submitted job if the controller was interrupted before it
    # could persist the ID locally.
    output=$("${SQUEUE_BIN}" -h -u "${USER}" -o '%A|%j' 2>/dev/null) ||
        die "squeue failed while checking for active RFT jobs"
    while IFS='|' read -r active_id active_name; do
        case "${active_name}" in
            "${PIPELINE_ID}-rftgen"|"${PIPELINE_ID}-rftscore"|\
            "${PIPELINE_ID}-rft"|"${PIPELINE_ID}-rft-hf")
                die "refusing to archive while RFT job ${active_id} (${active_name}) is active"
                ;;
            rlad-rft-data|rlad-rft-absgen|rlad-convert-hf)
                raw=$("${SCONTROL_BIN}" show job -o "${active_id}" 2>/dev/null) ||
                    die "cannot verify the checkout used by active ${active_name} job ${active_id}; refusing to archive"
                workdir=$(sed -n 's/.* WorkDir=\([^ ]*\).*/\1/p' <<< "${raw}")
                [[ -n "${workdir}" ]] ||
                    die "cannot read WorkDir for active ${active_name} job ${active_id}; refusing to archive"
                if [[ "$(readlink -m -- "${workdir}")" == "$(readlink -m -- "${RLAD_HOME}")" ]]; then
                    die "refusing to archive while manual RFT job ${active_id} (${active_name}) is active for this checkout"
                fi
                ;;
        esac
    done <<< "${output}"
}

archive_rft() {
    local stamp dest files=() path
    acquire_lock
    assert_no_active_rft_jobs
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dest="${RLAD_RUNS}/archive/rft_${stamp}_$$"
    mkdir -p "${dest}/data" "${dest}/runs" "${dest}/state"
    [[ ! -f "${STATE_DIR}/manifest.env" ]] ||
        cp -- "${STATE_DIR}/manifest.env" "${dest}/state/"
    [[ ! -f "${STATE_DIR}/jobs.tsv" ]] ||
        cp -- "${STATE_DIR}/jobs.tsv" "${dest}/state/"
    mapfile -t files < <(find "${RLAD_DATA}" -maxdepth 1 -type f \
        \( -name 'rft_abs_cache*.jsonl' -o -name 'rft_scored*.jsonl' -o \
           -name 'train_absgen_rft.jsonl' -o -name 'absgen_rft_meta.json' \) -print)
    for path in "${files[@]}"; do mv -- "${path}" "${dest}/data/"; done
    [[ ! -e "${RLAD_RUNS}/sft_absgen_rft" ]] ||
        mv -- "${RLAD_RUNS}/sft_absgen_rft" "${dest}/runs/"
    for path in "${STATE_DIR}"/rft* "${STATE_DIR}"/sft_absgen_rft* "${STATE_DIR}/final_model"; do
        [[ -e "${path}" ]] && mv -- "${path}" "${dest}/state/"
    done
    if [[ -d "${STATE_DIR}/jobs" ]]; then
        mkdir -p "${dest}/state/jobs"
        for path in "${STATE_DIR}/jobs"/rft_* "${STATE_DIR}/jobs"/sft_absgen_rft*; do
            [[ -e "${path}" ]] && mv -- "${path}" "${dest}/state/jobs/"
        done
    fi
    log "Archived RFT data and training artifacts to ${dest}"
}

main() {
    local command=${1:-help}
    case "${command}" in
        help|-h|--help) usage ;;
        setup)
            load_profile 1
            bootstrap_host
            doctor
            log "Setup complete; run '$0 run' inside tmux"
            ;;
        doctor)
            load_profile 0
            doctor
            ;;
        run|resume)
            load_profile 1
            bootstrap_host
            doctor
            activate_host_env
            [[ -z "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no)" ]] ||
                die "tracked repository files are modified; commit or restore them before starting a reproducible run"
            run_pipeline
            ;;
        status)
            load_profile 0
            show_status
            ;;
        archive-rft)
            load_profile 0
            archive_rft
            ;;
        *) die "unknown command: ${command}; run '$0 help'" ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
