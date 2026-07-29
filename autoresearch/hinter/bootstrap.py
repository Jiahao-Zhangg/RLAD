"""Resume-safe setup for pinned data, model, grader, and smoke packets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .core import (
    LOCK_PATH,
    RESEARCH_ROOT,
    SETUP_SMOKE_HINT,
    WORK_ROOT,
    atomic_write_json,
    canonical_json_bytes,
    hint_hash,
    load_config,
    load_json,
    reject_answer_leak,
    sha256_bytes,
    sha256_file,
    source_bundle_hash,
    source_hash_manifest,
    verify_runtime,
)
from .data import public_training_rows
from .grader import grader_checkout, verify_grader
from .job import validate_private_input, validate_train_packet


def _write_or_verify(path: Path, value: Any) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"setup artifact drift: {path}")
        return
    atomic_write_json(path, value)


def _prepare_grader(config: dict[str, Any]) -> dict[str, Any]:
    checkout = grader_checkout()
    grader = config["grader"]
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        if checkout.exists() and any(checkout.iterdir()):
            raise RuntimeError(f"non-git grader directory exists: {checkout}")
        subprocess.run(
            ["git", "clone", "--no-checkout", grader["repo"], str(checkout)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "checkout",
                "--detach",
                grader["revision"],
            ],
            check=True,
        )
    verify_grader(config)
    return {
        "schema_version": 1,
        "repo": grader["repo"],
        "revision": grader["revision"],
        "checkout": str(checkout),
        "deepscaler_sha256": sha256_file(
            checkout / "miles/rollout/rm_hub/deepscaler.py"
        ),
        "math_utils_sha256": sha256_file(
            checkout / "miles/rollout/rm_hub/math_utils.py"
        ),
        "clean": True,
    }


def bootstrap() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("refusing setup changes after experiment freeze")
    config, config_hash = load_config(require_frozen=False)
    setup = RESEARCH_ROOT / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    runtime = verify_runtime(config)
    rows = public_training_rows(config, config_hash)
    grader_manifest = _prepare_grader(config)
    snapshot = Path(
        snapshot_download(
            config["student"]["repo_id"],
            revision=config["student"]["revision"],
        )
    ).resolve()
    _write_or_verify(
        setup / "runtime_manifest.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "observed": runtime,
        },
    )
    _write_or_verify(
        setup / "model_manifest.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "repo_id": config["student"]["repo_id"],
            "revision": config["student"]["revision"],
            "snapshot": str(snapshot),
            "max_model_len": config["student"]["max_model_len"],
        },
    )
    _write_or_verify(setup / "grader_manifest.json", grader_manifest)

    row = rows[0]
    reject_answer_leak(SETUP_SMOKE_HINT, row["answer"])
    smoke_input = {
        "schema_version": 1,
        "round": 0,
        "hint_id": row["hint_id"],
        "train_position": row["train_position"],
        "train_qid": row["train_qid"],
        "problem": row["problem"],
        "answer": row["answer"],
        "hint": SETUP_SMOKE_HINT,
        "hint_hash": hint_hash(SETUP_SMOKE_HINT),
        "config_hash": config_hash,
        "previous_train_i": 0.0,
        "previous_heldout_i": 0.0,
        "previous_J_i": 0.0,
        "hard_tokens_per_hint":
            config["hint_limits"]["hard_tokens_per_hint"],
        "worker_tokens_per_hint":
            config["hint_limits"]["worker_tokens_per_hint"],
        "hard_total_tokens": config["hint_limits"]["hard_total_tokens"],
    }
    validate_train_packet(smoke_input, config_hash)
    private_input = {
        "schema_version": 1,
        "hint_id": row["hint_id"],
        "hint": SETUP_SMOKE_HINT,
        "hint_hash": hint_hash(SETUP_SMOKE_HINT),
        "config_hash": config_hash,
    }
    validate_private_input(private_input, config_hash)
    _write_or_verify(setup / "smoke_input.json", smoke_input)
    _write_or_verify(setup / "private_smoke_input.json", private_input)
    return {
        "config_hash": config_hash,
        "work_root": str(WORK_ROOT),
        "public_training_rows": len(rows),
        "heldout_materialized": False,
        "model_snapshot": str(snapshot),
        "grader_revision": grader_manifest["revision"],
        "smoke_ready": True,
    }


def review_material() -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=False)
    del config
    manifest = source_hash_manifest()
    return {
        "schema_version": 1,
        "config_hash": config_hash,
        "source_sha256": manifest,
        "source_bundle_hash": sha256_bytes(canonical_json_bytes(manifest)),
        "required_receipt": str(
            WORK_ROOT / "review" / "independent_review.json"
        ),
        "required_reviewer": "independent Codex process",
        "heldout_details_must_be_inspected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("run", "review-material"),
        nargs="?",
        default="run",
    )
    args = parser.parse_args()
    value = bootstrap() if args.command == "run" else review_material()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
