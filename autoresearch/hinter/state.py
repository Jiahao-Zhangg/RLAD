"""Hint-book lifecycle, private decisions, round logs, and stop conditions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from .core import (
    PRIVATE_INPUT_KEYS,
    PRIVATE_RESULT_KEYS,
    RESEARCH_ROOT,
    TRAIN_PACKET_KEYS,
    TRAIN_POSITIONS,
    TRAIN_RESULT_KEYS,
    TRAIN_ROLLOUT_KEYS,
    WORK_ROOT,
    atomic_write_bytes,
    atomic_write_json,
    hint_hash,
    load_config,
    load_json,
    proposal_wins,
    refuse_existing,
    reject_answer_leak,
    require_exact_keys,
    sha256_file,
    tokenizer_factory,
    validate_book,
    validate_hint,
    validate_private_metrics,
    validate_receipt,
    validate_registered_training_packet,
)
from .pool import enqueue_task


PROPOSAL_KEYS = {
    "hint_id",
    "hint",
    "mutation",
    "subagent_summary",
    "sampling_slurm_job_id",
}

ROUND_SUMMARY_COLUMNS = [
    "round",
    "book_hash_before",
    "book_hash_after",
    "mean_train_before",
    "mean_train_after",
    "mean_heldout_before",
    "mean_heldout_after",
    "mean_J_before",
    "mean_J_after",
    "num_kept",
    "num_discarded",
    "total_tokens_before",
    "total_tokens_after",
    "elapsed_seconds",
    "notes",
]

HINT_HISTORY_COLUMNS = [
    "round",
    "hint_id",
    "train_qid",
    "sampling_slurm_job_id",
    "proposal_eval_slurm_job_id",
    "incumbent_hash",
    "proposal_hash",
    "final_hash",
    "old_train",
    "new_train",
    "delta_train",
    "old_heldout",
    "new_heldout",
    "delta_heldout",
    "old_J",
    "new_J",
    "delta_J",
    "old_tokens",
    "new_tokens",
    "decision",
    "mutation",
    "subagent_summary",
]


def _public_rows() -> list[dict[str, Any]]:
    rows = load_json(RESEARCH_ROOT / "setup" / "train_public.json")
    if not isinstance(rows, list) or len(rows) != 10:
        raise RuntimeError("public training manifest must have ten rows")
    expected = {
        "hint_id",
        "train_position",
        "train_qid",
        "problem",
        "answer",
    }
    for hint_id, row in enumerate(rows, start=1):
        require_exact_keys(row, expected, "public training row")
        if row["hint_id"] != hint_id:
            raise RuntimeError("public training rows are not ordered")
    return rows


def _book(path: Path) -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    return validate_book(
        load_json(path),
        tokenizer_factory(config),
        config,
        config_hash,
    )


def initialize_book(hints: list[str]) -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    if len(hints) != 10:
        raise ValueError("initial book requires exactly ten hints")
    tokenizer = tokenizer_factory(config)
    rows = _public_rows()
    normalized = []
    limit = int(config["hint_limits"]["worker_tokens_per_hint"])
    for row, raw_hint in zip(rows, hints, strict=True):
        reject_answer_leak(raw_hint, row["answer"])
        hint, tokens, digest = validate_hint(raw_hint, tokenizer, limit)
        normalized.append(
            {
                "hint_id": row["hint_id"],
                "train_position": row["train_position"],
                "train_qid": row["train_qid"],
                "hint": hint,
                "tokens": tokens,
                "hint_hash": digest,
            }
        )
    book = validate_book(
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "generation": 0,
            "hints": normalized,
        },
        tokenizer,
        config,
        config_hash,
    )
    initial = RESEARCH_ROOT / "initial_book.json"
    current = RESEARCH_ROOT / "current_book.json"
    if initial.exists():
        existing = validate_book(
            load_json(initial),
            tokenizer,
            config,
            config_hash,
        )
        if existing != book:
            raise RuntimeError("initial book artifact drift")
        if not current.exists():
            atomic_write_json(current, existing)
        return existing
    if current.exists():
        raise RuntimeError("current book exists without an initial book")
    atomic_write_json(initial, book)
    atomic_write_json(current, book)
    return book


def make_private_inputs(book_path: Path, output_dir: Path) -> list[Path]:
    _, config_hash = load_config(require_frozen=True)
    book = _book(book_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in book["hints"]:
        path = output_dir / f"hint_{row['hint_id']:02d}.json"
        value = {
            "schema_version": 1,
            "hint_id": row["hint_id"],
            "hint": row["hint"],
            "hint_hash": row["hint_hash"],
            "config_hash": config_hash,
        }
        if path.exists():
            if load_json(path) != value:
                raise RuntimeError(f"private input drift: {path}")
        else:
            atomic_write_json(path, value)
        paths.append(path)
    return paths


def enqueue_private_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    receipt_dir: Path,
    task_prefix: str,
) -> list[dict[str, Any]]:
    results = []
    for hint_id in range(1, 11):
        stem = f"hint_{hint_id:02d}"
        results.append(
            enqueue_task(
                task_id=f"{task_prefix}-h{hint_id:02d}",
                mode="private",
                input_path=input_dir / f"{stem}.json",
                output_path=output_dir / f"{stem}.json",
                receipt_path=receipt_dir / f"{stem}.json",
            )
        )
    return results


def _private_result(
    *,
    hint_row: dict[str, Any],
    input_path: Path,
    output_path: Path,
    receipt_path: Path,
    task_id: str,
    config_hash: str,
    objective_lambda: int,
) -> tuple[dict[str, Any], str]:
    request = load_json(input_path)
    require_exact_keys(request, PRIVATE_INPUT_KEYS, "private input")
    expected = {
        "schema_version": 1,
        "hint_id": hint_row["hint_id"],
        "hint": hint_row["hint"],
        "hint_hash": hint_row["hint_hash"],
        "config_hash": config_hash,
    }
    if request != expected:
        raise RuntimeError(f"private input changed for hint {hint_row['hint_id']}")
    result = load_json(output_path)
    require_exact_keys(result, PRIVATE_RESULT_KEYS, "private result")
    validate_private_metrics(
        result,
        objective_lambda=objective_lambda,
    )
    if (
        result["hint_id"] != hint_row["hint_id"]
        or result["hint_hash"] != hint_row["hint_hash"]
        or result["config_hash"] != config_hash
    ):
        raise RuntimeError("private result identity is stale")
    receipt = load_json(receipt_path)
    validate_receipt(
        receipt,
        task_id=task_id,
        mode="private",
        input_path=input_path,
        output_path=output_path,
        config_hash=config_hash,
    )
    return result, receipt["execution_id"]


def initialize_metrics(
    *,
    input_dir: Path,
    output_dir: Path,
    receipt_dir: Path,
    task_prefix: str,
) -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    book = _book(RESEARCH_ROOT / "current_book.json")
    target = RESEARCH_ROOT / "best_per_hint.json"
    if target.exists():
        value = _best()
        _rebuild_csv_logs()
        return value
    rows = []
    execution_ids = set()
    for hint in book["hints"]:
        hint_id = hint["hint_id"]
        result, execution_id = _private_result(
            hint_row=hint,
            input_path=input_dir / f"hint_{hint_id:02d}.json",
            output_path=output_dir / f"hint_{hint_id:02d}.json",
            receipt_path=receipt_dir / f"hint_{hint_id:02d}.json",
            task_id=f"{task_prefix}-h{hint_id:02d}",
            config_hash=config_hash,
            objective_lambda=int(config["objective"]["lambda"]),
        )
        if execution_id in execution_ids:
            raise RuntimeError("baseline evaluator reused a Slurm step")
        execution_ids.add(execution_id)
        rows.append(
            {
                **hint,
                "metrics": result,
                "history": [
                    {
                        "round": 0,
                        "decision": "initial",
                        "metrics": result,
                        "proposal_eval_slurm_job_id": execution_id,
                    }
                ],
            }
        )
    value = {
        "schema_version": 1,
        "config_hash": config_hash,
        "hints": rows,
    }
    atomic_write_json(target, value)
    _rebuild_csv_logs()
    return value


def _best() -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    value = load_json(RESEARCH_ROOT / "best_per_hint.json")
    require_exact_keys(
        value,
        {"schema_version", "config_hash", "hints"},
        "best-per-hint ledger",
    )
    if (
        value.get("schema_version") != 1
        or value.get("config_hash") != config_hash
        or [row.get("hint_id") for row in value.get("hints", [])]
        != list(range(1, 11))
    ):
        raise RuntimeError("best-per-hint ledger is stale or incomplete")
    public = _public_rows()
    tokenizer = tokenizer_factory(config)
    total_tokens = 0
    for hint_id, (row, public_row) in enumerate(
        zip(value["hints"], public, strict=True),
        start=1,
    ):
        require_exact_keys(
            row,
            {
                "hint_id",
                "train_position",
                "train_qid",
                "hint",
                "tokens",
                "hint_hash",
                "metrics",
                "history",
            },
            "best-per-hint row",
        )
        if (
            row["hint_id"] != hint_id
            or row["train_position"] != TRAIN_POSITIONS[hint_id - 1]
            or row["train_qid"] != public_row["train_qid"]
            or not isinstance(row["history"], list)
            or not row["history"]
        ):
            raise RuntimeError("best-per-hint assignment/history is invalid")
        reject_answer_leak(row["hint"], public_row["answer"])
        hint, tokens, digest = validate_hint(
            row["hint"],
            tokenizer,
            int(config["hint_limits"]["hard_tokens_per_hint"]),
        )
        if (
            row["hint"] != hint
            or row["tokens"] != tokens
            or row["hint_hash"] != digest
        ):
            raise RuntimeError("best-per-hint text identity is stale")
        total_tokens += tokens
        validate_private_metrics(
            row["metrics"],
            objective_lambda=int(config["objective"]["lambda"]),
        )
        if row["metrics"]["hint_hash"] != row["hint_hash"]:
            raise RuntimeError("incumbent metrics do not match hint")
    if total_tokens > int(config["hint_limits"]["hard_total_tokens"]):
        raise RuntimeError("best-per-hint ledger exceeds the token budget")
    return value


def _require_previous_publication(
    round_number: int,
    config: dict[str, Any],
    config_hash: str,
) -> None:
    if round_number == 1:
        return
    previous = round_number - 1
    path = WORK_ROOT / "publication" / f"round_{previous:03d}.json"
    if not path.is_file():
        raise RuntimeError(
            f"round {previous} has not been successfully pushed to GitHub"
        )
    receipt = load_json(path)
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "round",
            "config_hash",
            "commit",
            "remote",
            "branch",
            "paths",
            "pushed_epoch",
        },
        "publication receipt",
    )
    commit = receipt["commit"]
    workspace_prefix = str(WORK_ROOT.relative_to(WORK_ROOT.parent))
    expected_metrics = (
        f"{workspace_prefix}/research/rounds/{previous:03d}/metrics.json"
    )
    if (
        receipt["schema_version"] != 1
        or receipt["round"] != previous
        or receipt["config_hash"] != config_hash
        or receipt["remote"] != config["publication"]["remote"]
        or not isinstance(receipt["branch"], str)
        or not receipt["branch"]
        or not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(receipt["paths"], list)
        or expected_metrics not in receipt["paths"]
        or isinstance(receipt["pushed_epoch"], bool)
        or not isinstance(receipt["pushed_epoch"], (int, float))
        or receipt["pushed_epoch"] <= 0
    ):
        raise RuntimeError("previous round publication receipt is invalid")


def _history_artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(WORK_ROOT):
        raise RuntimeError(f"worker-history artifact escapes workspace: {path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"worker-history artifact is missing or unsafe: {path}")
    return {
        "path": str(resolved.relative_to(WORK_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _optional_history_artifact(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return _history_artifact(path)


def _same_question_history(
    *,
    round_number: int,
    hint: dict[str, Any],
    incumbent: dict[str, Any],
    packet_path: Path,
    config_hash: str,
    objective_lambda: int,
) -> dict[str, Any]:
    """Build the held-out-safe, same-question context for one worker."""
    hint_id = int(hint["hint_id"])
    stem = f"hint_{hint_id:02d}.json"
    baseline = RESEARCH_ROOT / "baseline"
    baseline_artifacts = [
        _history_artifact(baseline / directory / stem)
        for directory in ("inputs", "outputs", "receipts")
    ]
    prior_rounds = []
    for previous in range(1, round_number):
        prior_dir = RESEARCH_ROOT / "rounds" / f"{previous:03d}"
        metrics = load_json(prior_dir / "metrics.json")
        records = [
            row
            for row in metrics["records"]
            if row.get("hint_id") == hint_id
        ]
        history_rows = [
            row
            for row in metrics["history_rows"]
            if row.get("hint_id") == hint_id
        ]
        if len(records) != 1 or len(history_rows) != 1:
            raise RuntimeError(
                f"round {previous} lacks unique history for hint {hint_id}"
            )
        artifacts = [
            _history_artifact(prior_dir / directory / stem)
            for directory in (
                "training_inputs",
                "training_outputs",
                "training_receipts",
                "worker_proposals",
                "proposal_private_inputs",
                "proposal_private_outputs",
                "proposal_private_receipts",
                "worker_history",
            )
        ]
        task_id = f"r{previous:02d}-train-h{hint_id:02d}"
        for suffix in ("out", "err"):
            optional = _optional_history_artifact(
                WORK_ROOT / "logs" / "tasks" / f"{task_id}.{suffix}"
            )
            if optional is not None:
                artifacts.append(optional)
        book_snapshots = {}
        for name in ("book_before", "book_proposals", "book_after"):
            book = load_json(prior_dir / f"{name}.json")
            rows = [
                row
                for row in book.get("hints", [])
                if row.get("hint_id") == hint_id
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    f"round {previous} {name} lacks hint {hint_id}"
                )
            book_snapshots[name] = rows[0]
        prior_rounds.append(
            {
                "round": previous,
                "artifacts": artifacts,
                "book_snapshots": book_snapshots,
                "decision_record": records[0],
                "history_row": history_rows[0],
            }
        )
    return {
        "schema_version": 1,
        "config_hash": config_hash,
        "round": round_number,
        "hint_id": hint_id,
        "train_position": hint["train_position"],
        "train_qid": hint["train_qid"],
        "objective_lambda": objective_lambda,
        "current_training_input": _history_artifact(packet_path),
        "baseline_artifacts": baseline_artifacts,
        "incumbent_history": incumbent["history"],
        "prior_rounds": prior_rounds,
    }


def prepare_round(round_number: int) -> Path:
    config, config_hash = load_config(require_frozen=True)
    if round_number < 1 or round_number > int(config["budget"]["max_rounds"]):
        raise ValueError("round is outside the declared budget")
    status = stopping_status()
    if status["should_stop"]:
        raise RuntimeError(f"search already satisfies stop condition: {status}")
    expected = status["completed_rounds"] + 1
    if round_number != expected:
        raise RuntimeError(f"next round must be {expected}")
    _require_previous_publication(round_number, config, config_hash)
    current = _book(RESEARCH_ROOT / "current_book.json")
    best = _best()
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "training_inputs",
        "training_outputs",
        "training_receipts",
        "worker_history",
        "worker_proposals",
        "proposal_private_inputs",
        "proposal_private_outputs",
        "proposal_private_receipts",
    ):
        (round_dir / name).mkdir(exist_ok=True)
    before_path = round_dir / "book_before.json"
    if before_path.exists():
        if load_json(before_path) != current:
            raise RuntimeError("partially prepared round has a stale incumbent book")
    else:
        atomic_write_json(before_path, current)
    state_path = round_dir / "round_state.json"
    if state_path.exists():
        round_state = load_json(state_path)
        if (
            round_state.get("schema_version") != 1
            or round_state.get("round") != round_number
            or round_state.get("config_hash") != config_hash
            or round_state.get("status") != "sampling_incumbents"
        ):
            raise RuntimeError("partially prepared round has stale state")
    else:
        atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "round": round_number,
                "config_hash": config_hash,
                "started_epoch": time.time(),
                "status": "sampling_incumbents",
            },
        )
    public = _public_rows()
    for hint, incumbent, public_row in zip(
        current["hints"],
        best["hints"],
        public,
        strict=True,
    ):
        metrics = incumbent["metrics"]
        packet = {
            "schema_version": 1,
            "round": round_number,
            "hint_id": hint["hint_id"],
            "train_position": hint["train_position"],
            "train_qid": hint["train_qid"],
            "problem": public_row["problem"],
            "answer": public_row["answer"],
            "hint": hint["hint"],
            "hint_hash": hint["hint_hash"],
            "config_hash": config_hash,
            "previous_train_i": metrics["train_i"],
            "previous_heldout_i": metrics["heldout_i"],
            "previous_J_i": metrics["J_i"],
            "hard_tokens_per_hint":
                config["hint_limits"]["hard_tokens_per_hint"],
            "worker_tokens_per_hint":
                config["hint_limits"]["worker_tokens_per_hint"],
            "hard_total_tokens": config["hint_limits"]["hard_total_tokens"],
        }
        require_exact_keys(packet, TRAIN_PACKET_KEYS, "training packet")
        packet_path = (
            round_dir / "training_inputs"
            / f"hint_{hint['hint_id']:02d}.json"
        )
        if packet_path.exists():
            if load_json(packet_path) != packet:
                raise RuntimeError(f"training packet drift: {packet_path}")
        else:
            atomic_write_json(packet_path, packet)
        history_path = (
            round_dir / "worker_history"
            / f"hint_{hint['hint_id']:02d}.json"
        )
        history = _same_question_history(
            round_number=round_number,
            hint=hint,
            incumbent=incumbent,
            packet_path=packet_path,
            config_hash=config_hash,
            objective_lambda=int(config["objective"]["lambda"]),
        )
        if history_path.exists():
            if load_json(history_path) != history:
                raise RuntimeError(f"worker history drift: {history_path}")
        else:
            atomic_write_json(history_path, history)
    return round_dir


def _training_evidence(
    *,
    round_number: int,
    hint_id: int,
    config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    config, current_config_hash = load_config(require_frozen=True)
    if current_config_hash != config_hash:
        raise RuntimeError("training evidence config hash is stale")
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    packet_path = round_dir / "training_inputs" / f"hint_{hint_id:02d}.json"
    output_path = round_dir / "training_outputs" / f"hint_{hint_id:02d}.json"
    receipt_path = round_dir / "training_receipts" / f"hint_{hint_id:02d}.json"
    packet = validate_registered_training_packet(
        packet_path,
        task_id=f"r{round_number:02d}-train-h{hint_id:02d}",
        config=config,
        config_hash=config_hash,
        research_root=RESEARCH_ROOT,
    )
    result = load_json(output_path)
    require_exact_keys(result, TRAIN_RESULT_KEYS, "training result")
    if (
        result["round"] != round_number
        or result["hint_id"] != hint_id
        or result["train_position"] != packet["train_position"]
        or result["train_qid"] != packet["train_qid"]
        or result["hint_hash"] != packet["hint_hash"]
        or result["config_hash"] != config_hash
        or isinstance(result["total"], bool)
        or not isinstance(result["total"], int)
        or result["total"] != 8
        or not isinstance(result["rollouts"], list)
        or len(result["rollouts"]) != 8
    ):
        raise RuntimeError(f"training result identity is stale for hint {hint_id}")
    rewards = []
    for sample_idx, rollout in enumerate(result["rollouts"]):
        require_exact_keys(rollout, TRAIN_ROLLOUT_KEYS, "training rollout")
        reward = rollout["reward"]
        if (
            rollout["sample_idx"] != sample_idx
            or isinstance(reward, bool)
            or not isinstance(reward, int)
            or reward not in (0, 1)
            or not isinstance(rollout["response"], str)
            or (
                rollout["finish_reason"] is not None
                and not isinstance(rollout["finish_reason"], str)
            )
            or isinstance(rollout["completion_tokens"], bool)
            or not isinstance(rollout["completion_tokens"], int)
            or rollout["completion_tokens"] < 0
        ):
            raise RuntimeError("training rollout index/reward is invalid")
        rewards.append(reward)
    if (
        isinstance(result["correct"], bool)
        or not isinstance(result["correct"], int)
        or result["correct"] != sum(rewards)
        or result["train_i"] != sum(rewards) / 8
    ):
        raise RuntimeError("training aggregate was not recomputed from rollouts")
    receipt = load_json(receipt_path)
    validate_receipt(
        receipt,
        task_id=f"r{round_number:02d}-train-h{hint_id:02d}",
        mode="train",
        input_path=packet_path,
        output_path=output_path,
        config_hash=config_hash,
    )
    return packet, result, receipt["execution_id"]


def _validate_proposal_book(
    value: dict[str, Any],
    *,
    round_number: int,
    before: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema_version",
            "config_hash",
            "round",
            "book_hash_before",
            "hints",
        },
        "proposal book",
    )
    rows = value["hints"]
    if (
        value["schema_version"] != 1
        or value["config_hash"] != config_hash
        or value["round"] != round_number
        or value["book_hash_before"] != before["book_hash"]
        or not isinstance(rows, list)
        or len(rows) != 10
    ):
        raise RuntimeError("proposal book identity is stale")
    tokenizer = tokenizer_factory(config)
    public = _public_rows()
    execution_ids = set()
    expected_keys = PROPOSAL_KEYS | {
        "tokens",
        "hint_hash",
        "train_position",
        "train_qid",
        "incumbent_hash",
    }
    for hint_id, (row, incumbent, public_row) in enumerate(
        zip(rows, before["hints"], public, strict=True),
        start=1,
    ):
        require_exact_keys(row, expected_keys, "proposal row")
        if (
            row["hint_id"] != hint_id
            or row["train_position"] != incumbent["train_position"]
            or row["train_qid"] != incumbent["train_qid"]
            or row["incumbent_hash"] != incumbent["hint_hash"]
        ):
            raise RuntimeError("proposal changed its permanent assignment")
        for label, maximum in (
            ("mutation", 500),
            ("subagent_summary", 1000),
        ):
            text = row[label]
            if not isinstance(text, str) or not text.strip() or len(text) > maximum:
                raise RuntimeError(f"proposal {label} is invalid")
        execution_id = row["sampling_slurm_job_id"]
        if (
            not isinstance(execution_id, str)
            or re.fullmatch(r"[0-9]+\.[0-9]+", execution_id) is None
            or execution_id in execution_ids
        ):
            raise RuntimeError("proposal has an invalid/reused Slurm execution ID")
        execution_ids.add(execution_id)
        reject_answer_leak(row["hint"], public_row["answer"])
        hint, tokens, digest = validate_hint(
            row["hint"],
            tokenizer,
            int(config["hint_limits"]["worker_tokens_per_hint"]),
        )
        if (
            row["hint"] != hint
            or row["tokens"] != tokens
            or row["hint_hash"] != digest
            or digest == incumbent["hint_hash"]
        ):
            raise RuntimeError("proposal hint identity is invalid")
    return value


def collect_proposals(round_number: int) -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    output = round_dir / "book_proposals.json"
    before = _book(round_dir / "book_before.json")
    if output.exists():
        return _validate_proposal_book(
            load_json(output),
            round_number=round_number,
            before=before,
            config=config,
            config_hash=config_hash,
        )
    tokenizer = tokenizer_factory(config)
    rows = []
    execution_ids = set()
    public = _public_rows()
    for hint_id in range(1, 11):
        packet, _, execution_id = _training_evidence(
            round_number=round_number,
            hint_id=hint_id,
            config_hash=config_hash,
        )
        raw = load_json(
            round_dir / "worker_proposals" / f"hint_{hint_id:02d}.json"
        )
        require_exact_keys(raw, PROPOSAL_KEYS, "worker proposal")
        if raw["hint_id"] != hint_id:
            raise RuntimeError("worker proposal changed its assignment")
        if raw["sampling_slurm_job_id"] != execution_id:
            raise RuntimeError("worker proposal cites the wrong Slurm step")
        if execution_id in execution_ids:
            raise RuntimeError("training tasks reused a Slurm step")
        execution_ids.add(execution_id)
        reject_answer_leak(raw["hint"], public[hint_id - 1]["answer"])
        hint, tokens, digest = validate_hint(
            raw["hint"],
            tokenizer,
            int(config["hint_limits"]["worker_tokens_per_hint"]),
        )
        if digest == packet["hint_hash"]:
            raise RuntimeError("worker did not revise its hint")
        rows.append(
            {
                **raw,
                "hint": hint,
                "tokens": tokens,
                "hint_hash": digest,
                "train_position": packet["train_position"],
                "train_qid": packet["train_qid"],
                "incumbent_hash": before["hints"][hint_id - 1]["hint_hash"],
            }
        )
    value = {
        "schema_version": 1,
        "config_hash": config_hash,
        "round": round_number,
        "book_hash_before": before["book_hash"],
        "hints": rows,
    }
    _validate_proposal_book(
        value,
        round_number=round_number,
        before=before,
        config=config,
        config_hash=config_hash,
    )
    atomic_write_json(output, value)
    return value


def proposal_private_inputs(round_number: int) -> list[Path]:
    _, config_hash = load_config(require_frozen=True)
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    proposals = collect_proposals(round_number)
    output_dir = round_dir / "proposal_private_inputs"
    paths = []
    for row in proposals["hints"]:
        path = output_dir / f"hint_{row['hint_id']:02d}.json"
        value = {
            "schema_version": 1,
            "hint_id": row["hint_id"],
            "hint": row["hint"],
            "hint_hash": row["hint_hash"],
            "config_hash": config_hash,
        }
        if path.exists() and load_json(path) != value:
            raise RuntimeError("proposal private input drift")
        if not path.exists():
            atomic_write_json(path, value)
        paths.append(path)
    return paths


def enqueue_proposal_private(round_number: int) -> list[dict[str, Any]]:
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    proposal_private_inputs(round_number)
    return enqueue_private_directory(
        input_dir=round_dir / "proposal_private_inputs",
        output_dir=round_dir / "proposal_private_outputs",
        receipt_dir=round_dir / "proposal_private_receipts",
        task_prefix=f"r{round_number:02d}-private",
    )


def _averages(metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "train": mean(row["train_i"] for row in metrics),
        "heldout": mean(row["heldout_i"] for row in metrics),
        "J": mean(row["J_i"] for row in metrics),
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _rebuild_csv_logs() -> None:
    summaries = []
    histories = []
    rounds_root = RESEARCH_ROOT / "rounds"
    if rounds_root.exists():
        for round_dir in sorted(rounds_root.iterdir()):
            metrics_path = round_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            summaries.append(metrics["aggregates"])
            histories.extend(metrics["history_rows"])
    _write_csv(
        RESEARCH_ROOT / "round_summary.csv",
        ROUND_SUMMARY_COLUMNS,
        summaries,
    )
    _write_csv(
        RESEARCH_ROOT / "hint_history.csv",
        HINT_HISTORY_COLUMNS,
        histories,
    )


def _validate_finalization_transaction(
    transaction: dict[str, Any],
    *,
    round_number: int,
    config: dict[str, Any],
    config_hash: str,
) -> None:
    require_exact_keys(
        transaction,
        {
            "schema_version",
            "config_hash",
            "round",
            "book_after",
            "best_after",
            "metrics",
            "started_epoch",
        },
        "round finalization transaction",
    )
    if (
        transaction["schema_version"] != 1
        or transaction["config_hash"] != config_hash
        or transaction["round"] != round_number
        or isinstance(transaction["started_epoch"], bool)
        or not isinstance(transaction["started_epoch"], (int, float))
        or transaction["started_epoch"] <= 0
    ):
        raise RuntimeError("round finalization transaction identity is stale")
    tokenizer = tokenizer_factory(config)
    after = validate_book(
        transaction["book_after"],
        tokenizer,
        config,
        config_hash,
    )
    if (
        after != transaction["book_after"]
        or after.get("generation") != round_number
    ):
        raise RuntimeError("finalization book is invalid")
    best_after = transaction["best_after"]
    require_exact_keys(
        best_after,
        {"schema_version", "config_hash", "hints"},
        "finalization best ledger",
    )
    if (
        best_after["schema_version"] != 1
        or best_after["config_hash"] != config_hash
        or not isinstance(best_after["hints"], list)
        or len(best_after["hints"]) != 10
    ):
        raise RuntimeError("finalization best ledger is invalid")
    for final_hint, best_row in zip(
        after["hints"],
        best_after["hints"],
        strict=True,
    ):
        require_exact_keys(
            best_row,
            {
                "hint_id",
                "train_position",
                "train_qid",
                "hint",
                "tokens",
                "hint_hash",
                "metrics",
                "history",
            },
            "finalization best row",
        )
        for key in (
            "hint_id",
            "train_position",
            "train_qid",
            "hint",
            "tokens",
            "hint_hash",
        ):
            if best_row[key] != final_hint[key]:
                raise RuntimeError("finalization book/best ledger disagree")
        validate_private_metrics(
            best_row["metrics"],
            objective_lambda=int(config["objective"]["lambda"]),
        )
        if (
            best_row["metrics"]["hint_hash"] != best_row["hint_hash"]
            or not isinstance(best_row["history"], list)
            or not best_row["history"]
        ):
            raise RuntimeError("finalization best metrics/history are invalid")
    metrics = transaction["metrics"]
    require_exact_keys(
        metrics,
        {
            "schema_version",
            "config_hash",
            "round",
            "records",
            "aggregates",
            "history_rows",
        },
        "finalization metrics",
    )
    if (
        metrics["schema_version"] != 1
        or metrics["config_hash"] != config_hash
        or metrics["round"] != round_number
        or not isinstance(metrics["records"], list)
        or len(metrics["records"]) != 10
        or not isinstance(metrics["history_rows"], list)
        or len(metrics["history_rows"]) != 10
        or not isinstance(metrics["aggregates"], dict)
        or set(metrics["aggregates"]) != set(ROUND_SUMMARY_COLUMNS)
        or metrics["aggregates"]["round"] != round_number
    ):
        raise RuntimeError("finalization metrics are invalid")


def finalize_round(round_number: int, notes: str = "") -> dict[str, Any]:
    if (RESEARCH_ROOT / "STOPPED.json").exists():
        raise RuntimeError("cannot finalize after the search has been sealed")
    config, config_hash = load_config(require_frozen=True)
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    metrics_path = round_dir / "metrics.json"
    transaction_path = round_dir / "finalization.json"
    if transaction_path.exists():
        transaction = load_json(transaction_path)
        _validate_finalization_transaction(
            transaction,
            round_number=round_number,
            config=config,
            config_hash=config_hash,
        )
        _apply_finalization(round_dir, transaction)
        return transaction["metrics"]
    if metrics_path.exists():
        raise RuntimeError(
            "metrics exist without the resume-safe finalization transaction"
        )
    before = _book(round_dir / "book_before.json")
    current = _book(RESEARCH_ROOT / "current_book.json")
    if current["book_hash"] != before["book_hash"]:
        raise RuntimeError("current book changed during the round")
    proposals = collect_proposals(round_number)
    best = _best()
    final_hints = []
    next_best = []
    records = []
    history_rows = []
    execution_ids = set()
    for incumbent, proposal, best_row in zip(
        before["hints"],
        proposals["hints"],
        best["hints"],
        strict=True,
    ):
        hint_id = incumbent["hint_id"]
        proposed_metrics, execution_id = _private_result(
            hint_row=proposal,
            input_path=round_dir / "proposal_private_inputs"
            / f"hint_{hint_id:02d}.json",
            output_path=round_dir / "proposal_private_outputs"
            / f"hint_{hint_id:02d}.json",
            receipt_path=round_dir / "proposal_private_receipts"
            / f"hint_{hint_id:02d}.json",
            task_id=f"r{round_number:02d}-private-h{hint_id:02d}",
            config_hash=config_hash,
            objective_lambda=int(config["objective"]["lambda"]),
        )
        if execution_id in execution_ids:
            raise RuntimeError("proposal evaluators reused a Slurm step")
        execution_ids.add(execution_id)
        old_metrics = best_row["metrics"]
        keep = proposal_wins(
            old_metrics,
            proposed_metrics,
            incumbent["tokens"],
            proposal["tokens"],
        )
        decision = "kept" if keep else "discarded"
        if keep:
            final_hint = {
                key: proposal[key]
                for key in (
                    "hint_id",
                    "train_position",
                    "train_qid",
                    "hint",
                    "tokens",
                    "hint_hash",
                )
            }
            final_metrics = proposed_metrics
        else:
            final_hint = incumbent
            final_metrics = old_metrics
        final_hints.append(final_hint)
        next_best.append(
            {
                **final_hint,
                "metrics": final_metrics,
                "history": [
                    *best_row["history"],
                    {
                        "round": round_number,
                        "decision": decision,
                        "proposal_hash": proposal["hint_hash"],
                        "proposal_metrics": proposed_metrics,
                        "proposal_eval_slurm_job_id": execution_id,
                    },
                ],
            }
        )
        record = {
            "hint_id": hint_id,
            "incumbent": {
                "hint_hash": incumbent["hint_hash"],
                "tokens": incumbent["tokens"],
                "metrics": old_metrics,
            },
            "proposal": {
                "hint_hash": proposal["hint_hash"],
                "tokens": proposal["tokens"],
                "metrics": proposed_metrics,
            },
            "final": {
                "hint_hash": final_hint["hint_hash"],
                "tokens": final_hint["tokens"],
                "metrics": final_metrics,
            },
            "decision": decision,
            "sampling_slurm_job_id": proposal["sampling_slurm_job_id"],
            "proposal_eval_slurm_job_id": execution_id,
            "mutation": proposal["mutation"],
            "subagent_summary": proposal["subagent_summary"],
        }
        records.append(record)
        history_rows.append(
            {
                "round": round_number,
                "hint_id": hint_id,
                "train_qid": incumbent["train_qid"],
                "sampling_slurm_job_id":
                    proposal["sampling_slurm_job_id"],
                "proposal_eval_slurm_job_id": execution_id,
                "incumbent_hash": incumbent["hint_hash"],
                "proposal_hash": proposal["hint_hash"],
                "final_hash": final_hint["hint_hash"],
                "old_train": old_metrics["train_i"],
                "new_train": proposed_metrics["train_i"],
                "delta_train":
                    proposed_metrics["train_i"] - old_metrics["train_i"],
                "old_heldout": old_metrics["heldout_i"],
                "new_heldout": proposed_metrics["heldout_i"],
                "delta_heldout":
                    proposed_metrics["heldout_i"]
                    - old_metrics["heldout_i"],
                "old_J": old_metrics["J_i"],
                "new_J": proposed_metrics["J_i"],
                "delta_J": proposed_metrics["J_i"] - old_metrics["J_i"],
                "old_tokens": incumbent["tokens"],
                "new_tokens": proposal["tokens"],
                "decision": decision,
                "mutation": proposal["mutation"],
                "subagent_summary": proposal["subagent_summary"],
            }
        )
    tokenizer = tokenizer_factory(config)
    after = validate_book(
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "generation": round_number,
            "hints": final_hints,
        },
        tokenizer,
        config,
        config_hash,
    )
    old_values = [row["metrics"] for row in best["hints"]]
    final_values = [row["metrics"] for row in next_best]
    old_avg = _averages(old_values)
    new_avg = _averages(final_values)
    started = load_json(round_dir / "round_state.json")["started_epoch"]
    kept = sum(row["decision"] == "kept" for row in records)
    aggregates = {
        "round": round_number,
        "book_hash_before": before["book_hash"],
        "book_hash_after": after["book_hash"],
        "mean_train_before": old_avg["train"],
        "mean_train_after": new_avg["train"],
        "mean_heldout_before": old_avg["heldout"],
        "mean_heldout_after": new_avg["heldout"],
        "mean_J_before": old_avg["J"],
        "mean_J_after": new_avg["J"],
        "num_kept": kept,
        "num_discarded": 10 - kept,
        "total_tokens_before": before["total_tokens"],
        "total_tokens_after": after["total_tokens"],
        "elapsed_seconds": time.time() - float(started),
        "notes": notes,
    }
    metrics = {
        "schema_version": 1,
        "config_hash": config_hash,
        "round": round_number,
        "records": records,
        "aggregates": aggregates,
        "history_rows": history_rows,
    }
    transaction = {
        "schema_version": 1,
        "config_hash": config_hash,
        "round": round_number,
        "book_after": after,
        "best_after": {
            "schema_version": 1,
            "config_hash": config_hash,
            "hints": next_best,
        },
        "metrics": metrics,
        "started_epoch": started,
    }
    _validate_finalization_transaction(
        transaction,
        round_number=round_number,
        config=config,
        config_hash=config_hash,
    )
    atomic_write_json(transaction_path, transaction)
    _apply_finalization(round_dir, transaction)
    return metrics


def _write_same_or_new(path: Path, value: Any) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"finalization artifact drift: {path}")
        return
    atomic_write_json(path, value)


def _apply_finalization(
    round_dir: Path,
    transaction: dict[str, Any],
) -> None:
    round_number = int(transaction["round"])
    config_hash = transaction["config_hash"]
    after = transaction["book_after"]
    best_after = transaction["best_after"]
    metrics = transaction["metrics"]
    _write_same_or_new(round_dir / "book_after.json", after)
    current_path = RESEARCH_ROOT / "current_book.json"
    current_generation = -1
    if current_path.exists():
        current_generation = int(load_json(current_path).get("generation", -1))
    if current_generation < round_number:
        atomic_write_json(RESEARCH_ROOT / "best_per_hint.json", best_after)
        atomic_write_json(current_path, after)
    elif current_generation == round_number:
        if load_json(current_path) != after:
            raise RuntimeError("current book conflicts with finalized round")
        existing_best = load_json(RESEARCH_ROOT / "best_per_hint.json")
        if existing_best != best_after:
            raise RuntimeError("best-per-hint conflicts with finalized round")
    else:
        raise RuntimeError("cannot replay a finalization behind the current book")
    _write_same_or_new(round_dir / "metrics.json", metrics)
    atomic_write_json(
        round_dir / "round_state.json",
        {
            "schema_version": 1,
            "round": round_number,
            "config_hash": config_hash,
            "started_epoch": transaction["started_epoch"],
            "status": "finalized",
        },
    )
    _rebuild_csv_logs()


def _completed_round_metrics(config_hash: str) -> list[dict[str, Any]]:
    values = []
    root = RESEARCH_ROOT / "rounds"
    if not root.exists():
        return values
    numeric_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and re.fullmatch(r"[0-9]{3}", path.name)
    ]
    for round_dir in numeric_dirs:
        metrics_path = round_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        value = load_json(metrics_path)
        require_exact_keys(
            value,
            {
                "schema_version",
                "config_hash",
                "round",
                "records",
                "aggregates",
                "history_rows",
            },
            "round metrics",
        )
        round_number = int(round_dir.name)
        aggregates = value["aggregates"]
        if (
            value["schema_version"] != 1
            or value["config_hash"] != config_hash
            or value["round"] != round_number
            or not isinstance(value["records"], list)
            or len(value["records"]) != 10
            or not isinstance(value["history_rows"], list)
            or len(value["history_rows"]) != 10
            or not isinstance(aggregates, dict)
            or set(aggregates) != set(ROUND_SUMMARY_COLUMNS)
            or aggregates["round"] != round_number
            or isinstance(aggregates["num_kept"], bool)
            or not isinstance(aggregates["num_kept"], int)
            or isinstance(aggregates["num_discarded"], bool)
            or not isinstance(aggregates["num_discarded"], int)
            or not 0 <= aggregates["num_kept"] <= 10
            or aggregates["num_discarded"] != 10 - aggregates["num_kept"]
        ):
            raise RuntimeError(f"round {round_number} metrics are invalid")
        values.append(value)
    numbers = [value["round"] for value in values]
    if numbers != list(range(1, len(values) + 1)):
        raise RuntimeError(f"completed rounds are not contiguous: {numbers}")
    return values


def stopping_status() -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    metrics_values = _completed_round_metrics(config_hash)
    completed = [value["round"] for value in metrics_values]
    zero_kept = 0
    for value in reversed(metrics_values):
        if value["aggregates"]["num_kept"] == 0:
            zero_kept += 1
        else:
            break
    max_rounds = int(config["budget"]["max_rounds"])
    stopped_path = RESEARCH_ROOT / "STOPPED.json"
    if stopped_path.exists():
        stopped = load_json(stopped_path)
        require_exact_keys(
            stopped,
            {
                "schema_version",
                "completed_rounds",
                "last_round",
                "consecutive_zero_keep_rounds",
                "max_rounds",
                "should_stop",
                "reason",
            },
            "persisted stop receipt",
        )
        if (
            stopped["schema_version"] != 1
            or stopped["completed_rounds"] != len(completed)
            or stopped["last_round"] != (completed[-1] if completed else 0)
            or stopped["consecutive_zero_keep_rounds"] != zero_kept
            or stopped["max_rounds"] != max_rounds
            or stopped["should_stop"] is not True
            or stopped["reason"] not in {
                "round_budget_exhausted",
                "human_stop",
            }
            or not (RESEARCH_ROOT / "final_book.json").is_file()
        ):
            raise RuntimeError("persisted stop receipt is invalid")
        return {
            key: value
            for key, value in stopped.items()
            if key != "schema_version"
        }
    reason = None
    if len(completed) >= max_rounds:
        reason = "round_budget_exhausted"
    return {
        "completed_rounds": len(completed),
        "last_round": completed[-1] if completed else 0,
        "consecutive_zero_keep_rounds": zero_kept,
        "max_rounds": max_rounds,
        "should_stop": reason is not None,
        "reason": reason,
    }


def seal_final_book(*, human_stop: bool = False) -> dict[str, Any]:
    status = stopping_status()
    if human_stop and not status["should_stop"]:
        status = {
            **status,
            "should_stop": True,
            "reason": "human_stop",
        }
    if not status["should_stop"]:
        raise RuntimeError("cannot seal before a declared stop condition")
    current = _book(RESEARCH_ROOT / "current_book.json")
    final = {**current, "stopping_status": status}
    path = RESEARCH_ROOT / "final_book.json"
    if path.exists() and load_json(path) != final:
        raise RuntimeError("final book artifact drift")
    if not path.exists():
        atomic_write_json(path, final)
    atomic_write_json(
        RESEARCH_ROOT / "STOPPED.json",
        {"schema_version": 1, **status},
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-book")
    init.add_argument("--hints", type=Path, required=True)
    private = commands.add_parser("private-inputs")
    private.add_argument("--book", type=Path, required=True)
    private.add_argument("--output-dir", type=Path, required=True)
    enqueue = commands.add_parser("enqueue-private")
    enqueue.add_argument("--input-dir", type=Path, required=True)
    enqueue.add_argument("--output-dir", type=Path, required=True)
    enqueue.add_argument("--receipt-dir", type=Path, required=True)
    enqueue.add_argument("--task-prefix", required=True)
    metrics = commands.add_parser("init-metrics")
    metrics.add_argument("--input-dir", type=Path, required=True)
    metrics.add_argument("--output-dir", type=Path, required=True)
    metrics.add_argument("--receipt-dir", type=Path, required=True)
    metrics.add_argument("--task-prefix", required=True)
    prepare = commands.add_parser("prepare-round")
    prepare.add_argument("--round", type=int, required=True)
    collect = commands.add_parser("collect-proposals")
    collect.add_argument("--round", type=int, required=True)
    proposal_private = commands.add_parser("proposal-private")
    proposal_private.add_argument("--round", type=int, required=True)
    finalize = commands.add_parser("finalize-round")
    finalize.add_argument("--round", type=int, required=True)
    finalize.add_argument("--notes", default="")
    commands.add_parser("stopping-status")
    seal = commands.add_parser("seal-final")
    seal.add_argument("--human-stop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init-book":
        value = initialize_book(load_json(args.hints))
    elif args.command == "private-inputs":
        value = [
            str(path)
            for path in make_private_inputs(args.book, args.output_dir)
        ]
    elif args.command == "enqueue-private":
        value = enqueue_private_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            receipt_dir=args.receipt_dir,
            task_prefix=args.task_prefix,
        )
    elif args.command == "init-metrics":
        value = initialize_metrics(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            receipt_dir=args.receipt_dir,
            task_prefix=args.task_prefix,
        )
    elif args.command == "prepare-round":
        value = {"round_dir": str(prepare_round(args.round))}
    elif args.command == "collect-proposals":
        value = collect_proposals(args.round)
    elif args.command == "proposal-private":
        value = enqueue_proposal_private(args.round)
    elif args.command == "finalize-round":
        value = finalize_round(args.round, args.notes)
    elif args.command == "stopping-status":
        value = stopping_status()
    else:
        value = seal_final_book(human_stop=args.human_stop)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
