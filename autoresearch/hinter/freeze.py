"""Freeze the reviewed experiment after one-GPU smoke gates pass."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .core import (
    LOCK_PATH,
    PRIVATE_RESULT_KEYS,
    REPO_ROOT,
    RESEARCH_ROOT,
    WORK_ROOT,
    atomic_write_json,
    load_config,
    load_json,
    refuse_existing,
    require_exact_keys,
    sha256_file,
    source_bundle_hash,
    source_hash_manifest,
    validate_private_metrics,
    validate_receipt,
)
from .data import validate_split
from .grader import verify_grader


REVIEW_KEYS = {
    "schema_version",
    "verdict",
    "reviewer",
    "reviewed_epoch",
    "config_hash",
    "source_sha256",
    "source_bundle_hash",
    "heldout_details_inspected",
    "findings",
}

WORKSPACE_PREFIX = str(WORK_ROOT.relative_to(REPO_ROOT))
SETUP_FILES = tuple(
    f"{WORKSPACE_PREFIX}/research/setup/{name}"
    for name in (
        "data_manifest.json",
        "train_public.json",
        "runtime_manifest.json",
        "model_manifest.json",
        "grader_manifest.json",
        "smoke_input.json",
        "smoke_output.json",
        "smoke_receipt.json",
        "private_smoke_input.json",
        "private_smoke_output.json",
        "private_smoke_receipt.json",
    )
)


def _validate_smoke(config: dict[str, Any], config_hash: str) -> None:
    setup = RESEARCH_ROOT / "setup"
    smoke_request = load_json(setup / "smoke_input.json")
    smoke_output = load_json(setup / "smoke_output.json")
    require_exact_keys(
        smoke_output,
        {
            "schema_version",
            "mode",
            "hint_id",
            "train_position",
            "hint_hash",
            "config_hash",
            "conditions",
        },
        "smoke output",
    )
    if (
        smoke_output["mode"] != "smoke"
        or smoke_output["config_hash"] != config_hash
        or smoke_output["hint_id"] != smoke_request["hint_id"]
        or smoke_output["train_position"] != smoke_request["train_position"]
        or smoke_output["hint_hash"] != smoke_request["hint_hash"]
    ):
        raise RuntimeError("normal smoke output identity is stale")
    if set(smoke_output["conditions"]) != {"without_hint", "with_hint"}:
        raise RuntimeError("smoke did not exercise both prompt conditions")
    for name, value in smoke_output["conditions"].items():
        require_exact_keys(
            value,
            {
                "correct",
                "total",
                "sample_indices",
                "completion_tokens",
                "finish_reasons",
            },
            f"smoke condition {name}",
        )
        if (
            value["total"] != 8
            or value["sample_indices"] != list(range(8))
            or len(value["completion_tokens"]) != 8
            or len(value["finish_reasons"]) != 8
            or isinstance(value["correct"], bool)
            or not isinstance(value["correct"], int)
            or not 0 <= value["correct"] <= 8
        ):
            raise RuntimeError(f"smoke condition {name} is incomplete")
    smoke_input = setup / "smoke_input.json"
    smoke_result = setup / "smoke_output.json"
    validate_receipt(
        load_json(setup / "smoke_receipt.json"),
        task_id="setup-smoke",
        mode="smoke",
        input_path=smoke_input,
        output_path=smoke_result,
        config_hash=config_hash,
    )

    private_output = load_json(setup / "private_smoke_output.json")
    require_exact_keys(private_output, PRIVATE_RESULT_KEYS, "private smoke output")
    validate_private_metrics(
        private_output,
        objective_lambda=int(config["objective"]["lambda"]),
    )
    private_request = load_json(setup / "private_smoke_input.json")
    if (
        private_output["hint_id"] != private_request["hint_id"]
        or private_output["hint_hash"] != private_request["hint_hash"]
        or private_output["config_hash"] != config_hash
    ):
        raise RuntimeError("private smoke output identity is stale")
    validate_receipt(
        load_json(setup / "private_smoke_receipt.json"),
        task_id="setup-private-smoke",
        mode="private-smoke",
        input_path=setup / "private_smoke_input.json",
        output_path=setup / "private_smoke_output.json",
        config_hash=config_hash,
    )


def freeze() -> dict[str, Any]:
    refuse_existing(LOCK_PATH)
    config, config_hash = load_config(require_frozen=False)
    validate_split(config)
    verify_grader(config)
    for relative in SETUP_FILES:
        if not (REPO_ROOT / relative).is_file():
            raise RuntimeError(f"required setup artifact missing: {relative}")
    _validate_smoke(config, config_hash)

    review_path = WORK_ROOT / "review" / "independent_review.json"
    review = load_json(review_path)
    require_exact_keys(review, REVIEW_KEYS, "independent review")
    current_sources = source_hash_manifest()
    if (
        review["schema_version"] != 1
        or review["verdict"] != "pass"
        or review["reviewer"] != "codex"
        or review["config_hash"] != config_hash
        or review["heldout_details_inspected"] is not False
        or review["source_sha256"] != current_sources
        or review["source_bundle_hash"] != source_bundle_hash()
    ):
        raise RuntimeError("independent Codex review receipt is invalid or stale")

    setup_manifest = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in SETUP_FILES
    }
    lock = {
        "schema_version": 1,
        "status": "frozen",
        "config_hash": config_hash,
        "source_sha256": current_sources,
        "setup_sha256": setup_manifest,
        "review_sha256": sha256_file(review_path),
        "created_epoch": time.time(),
    }
    atomic_write_json(LOCK_PATH, lock)
    load_config(require_frozen=True)
    return lock


def main() -> None:
    print(json.dumps(freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
