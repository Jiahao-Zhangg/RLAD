#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLAD_REPO_ROOT="${repo_root}"
export RLAD_AUTORESEARCH_WORK="${repo_root}/work_zsw_lambda2"
export RLAD_AUTORESEARCH_LAMBDA=2
export RLAD_AUTORESEARCH_PARTITION="ml.p4d.24xlarge"
export RLAD_AUTORESEARCH_NODES="ip-10-1-173-179,ip-10-1-184-205"

mkdir -p "${RLAD_AUTORESEARCH_WORK}"
cd "${repo_root}"
exec uv run --project autoresearch --frozen \
    python -m autoresearch.run_hinter_agent "$@"
