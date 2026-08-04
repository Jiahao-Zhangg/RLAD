"""One-GPU Slurm-step entry point for train, smoke, and private evaluation."""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path
from typing import Any

from .core import (
    HELDOUT_POSITIONS,
    PRIVATE_INPUT_KEYS,
    PRIVATE_RESULT_KEYS,
    RECEIPT_KEYS,
    RUNTIME_SOURCE_FILES,
    TASK_KEYS,
    TRAIN_PACKET_KEYS,
    TRAIN_POSITIONS,
    TRAIN_RESULT_KEYS,
    WORK_ROOT,
    atomic_write_json,
    exact_metrics,
    hint_hash,
    load_config,
    load_json,
    load_pool_allocation,
    require_exact_keys,
    sha256_file,
    source_bundle_hash,
    validate_registered_private_input,
    validate_task_identity,
)
from .data import load_pinned_dataset
from .sampling import StudentSampler


def _inside_work(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(WORK_ROOT):
        raise ValueError(f"task path escapes autoresearch workspace: {resolved}")
    return resolved


def validate_train_packet(
    packet: dict[str, Any],
    config_hash: str,
) -> None:
    require_exact_keys(packet, TRAIN_PACKET_KEYS, "training packet")
    if packet["schema_version"] != 1 or packet["config_hash"] != config_hash:
        raise ValueError("training packet config hash is stale")
    hint_id = packet["hint_id"]
    if isinstance(hint_id, bool) or not isinstance(hint_id, int):
        raise ValueError("hint_id must be an integer")
    if hint_id not in range(1, 11):
        raise ValueError("hint_id must be between 1 and 10")
    if packet["train_position"] != TRAIN_POSITIONS[hint_id - 1]:
        raise ValueError("training packet assignment changed")
    if packet["hint_hash"] != hint_hash(packet["hint"]):
        raise ValueError("training packet hint hash mismatch")
    round_number = packet["round"]
    if (
        isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or not 0 <= round_number <= 20
    ):
        raise ValueError("training packet round is invalid")


def validate_private_input(
    value: dict[str, Any],
    config_hash: str,
) -> None:
    require_exact_keys(value, PRIVATE_INPUT_KEYS, "private input")
    if value["schema_version"] != 1 or value["config_hash"] != config_hash:
        raise ValueError("private input config hash is stale")
    hint_id = value["hint_id"]
    if isinstance(hint_id, bool) or not isinstance(hint_id, int):
        raise ValueError("private hint_id must be an integer")
    if hint_id not in range(1, 11):
        raise ValueError("private hint_id must be between 1 and 10")
    if value["hint_hash"] != hint_hash(value["hint"]):
        raise ValueError("private input hint hash mismatch")


def _run_train(
    sampler: StudentSampler,
    input_path: Path,
    output_path: Path,
    config_hash: str,
) -> None:
    packet = load_json(input_path)
    validate_train_packet(packet, config_hash)
    responses = sampler.sample(
        problem=packet["problem"],
        answer=packet["answer"],
        hint=packet["hint"],
    )
    correct = sum(row.reward for row in responses)
    result = {
        "schema_version": 1,
        "round": packet["round"],
        "hint_id": packet["hint_id"],
        "train_position": packet["train_position"],
        "train_qid": packet["train_qid"],
        "hint_hash": packet["hint_hash"],
        "config_hash": config_hash,
        "correct": correct,
        "total": len(responses),
        "train_i": correct / len(responses),
        "rollouts": [
            {
                "sample_idx": row.sample_idx,
                "response": row.response,
                "reward": row.reward,
                "finish_reason": row.finish_reason,
                "completion_tokens": row.completion_tokens,
            }
            for row in responses
        ],
    }
    require_exact_keys(result, TRAIN_RESULT_KEYS, "training result")
    if len(responses) != 8:
        raise RuntimeError("training task did not return eight rollouts")
    atomic_write_json(output_path, result)


def _run_smoke(
    sampler: StudentSampler,
    input_path: Path,
    output_path: Path,
    config_hash: str,
) -> None:
    packet = load_json(input_path)
    validate_train_packet(packet, config_hash)
    groups = sampler.sample_batch(
        [
            {
                "problem": packet["problem"],
                "answer": packet["answer"],
                "hint": None,
            },
            {
                "problem": packet["problem"],
                "answer": packet["answer"],
                "hint": packet["hint"],
            },
        ]
    )
    conditions = {}
    for name, responses in zip(
        ("without_hint", "with_hint"),
        groups,
        strict=True,
    ):
        conditions[name] = {
            "correct": sum(row.reward for row in responses),
            "total": len(responses),
            "sample_indices": [row.sample_idx for row in responses],
            "completion_tokens": [
                row.completion_tokens for row in responses
            ],
            "finish_reasons": [row.finish_reason for row in responses],
        }
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "mode": "smoke",
            "hint_id": packet["hint_id"],
            "train_position": packet["train_position"],
            "hint_hash": packet["hint_hash"],
            "config_hash": config_hash,
            "conditions": conditions,
        },
    )


def _run_private(
    sampler: StudentSampler,
    config: dict[str, Any],
    input_path: Path,
    output_path: Path,
    config_hash: str,
) -> None:
    request = load_json(input_path)
    validate_private_input(request, config_hash)
    hint_id = int(request["hint_id"])
    dataset = load_pinned_dataset(config)
    positions = [TRAIN_POSITIONS[hint_id - 1], *HELDOUT_POSITIONS]
    requests = [
        {
            "problem": str(dataset[position]["problem"]),
            "answer": str(dataset[position]["answer"]),
            "hint": request["hint"],
        }
        for position in positions
    ]
    groups = sampler.sample_batch(requests)
    train_correct = sum(row.reward for row in groups[0])
    heldout_correct = sum(
        row.reward
        for group in groups[1:]
        for row in group
    )
    result = {
        "schema_version": 1,
        "hint_id": hint_id,
        "hint_hash": request["hint_hash"],
        "config_hash": config_hash,
        **exact_metrics(
            train_correct,
            heldout_correct,
            objective_lambda=int(config["objective"]["lambda"]),
        ),
    }
    require_exact_keys(result, PRIVATE_RESULT_KEYS, "private result")
    atomic_write_json(output_path, result)


def _validate_task_input_identity(
    task_id: str,
    mode: str,
    input_path: Path,
    config: dict[str, Any],
    config_hash: str,
) -> None:
    value = load_json(input_path)
    if mode in {"train", "smoke"}:
        validate_train_packet(value, config_hash)
        if mode == "smoke":
            if value["round"] != 0 or task_id != "setup-smoke":
                raise ValueError("smoke task/input identity mismatch")
        else:
            if not 1 <= int(value["round"]) <= int(
                config["budget"]["max_rounds"]
            ):
                raise ValueError("training packet round is outside the budget")
            expected = (
                f"r{int(value['round']):02d}-train-"
                f"h{int(value['hint_id']):02d}"
            )
            if task_id != expected:
                raise ValueError("training task/input identity mismatch")
    else:
        validate_private_input(value, config_hash)
        validate_registered_private_input(
            input_path,
            task_id=task_id,
            mode=mode,
            config=config,
            config_hash=config_hash,
        )


def run_task(task_path: Path) -> None:
    task = load_json(task_path)
    require_exact_keys(task, TASK_KEYS, "pool task")
    if task["schema_version"] != 1:
        raise ValueError("pool task schema is not 1")
    if task["gpu_count"] != 1:
        raise ValueError("every student task must request exactly one GPU")
    if (
        isinstance(task["created_epoch"], bool)
        or not isinstance(task["created_epoch"], (int, float))
        or task["created_epoch"] <= 0
    ):
        raise ValueError("pool task creation time is invalid")
    mode = task["mode"]
    if mode not in {"train", "private", "smoke", "private-smoke"}:
        raise ValueError(f"unsupported task mode: {mode}")
    validate_task_identity(task["task_id"], mode)
    config, config_hash = load_config(
        require_frozen=mode in {"train", "private"}
    )
    if task["config_hash"] != config_hash:
        raise ValueError("pool task config hash is stale")
    expected_source = source_bundle_hash(RUNTIME_SOURCE_FILES)
    if task["source_hash"] != expected_source:
        raise ValueError("pool task runtime source hash is stale")
    input_path = _inside_work(Path(task["input_path"]))
    output_path = _inside_work(Path(task["output_path"]))
    receipt_path = _inside_work(Path(task["receipt_path"]))
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite task artifacts")
    if task["input_sha256"] != sha256_file(input_path):
        raise ValueError("pool task input changed after enqueue")
    _validate_task_input_identity(
        task["task_id"],
        mode,
        input_path,
        config,
        config_hash,
    )

    allocation = os.environ.get("SLURM_JOB_ID")
    step = os.environ.get("SLURM_STEP_ID")
    if (
        not allocation
        or not allocation.isdigit()
        or not step
        or not step.isdigit()
    ):
        raise RuntimeError("task is not running in a Slurm job step")
    pool_slot_raw = os.environ.get("AUTORESEARCH_POOL_SLOT")
    if not pool_slot_raw or not pool_slot_raw.isdigit():
        raise RuntimeError("task has no audited dispatcher pool slot")
    pool_slot = int(pool_slot_raw)
    if not 0 <= pool_slot < int(config["slurm"]["gpus_per_node"]):
        raise RuntimeError("dispatcher pool slot is outside the node capacity")
    slurm_gpus_raw = os.environ.get("SLURM_GPUS_PER_TASK")
    if slurm_gpus_raw != "1":
        raise RuntimeError("Slurm did not grant exactly one GPU to the task")
    visible_cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_cuda_device or "," in visible_cuda_device:
        raise RuntimeError("task does not have exactly one CUDA device binding")
    node = socket.gethostname().split(".", 1)[0]
    pool_allocation = load_pool_allocation(config, config_hash)
    if allocation != pool_allocation["job_id"]:
        raise RuntimeError("task job does not match the active pool allocation")
    if node not in pool_allocation["nodes"]:
        raise RuntimeError("task is running outside the two-node pool")

    start = time.time()
    sampler = StudentSampler(config)
    if mode == "train":
        _run_train(sampler, input_path, output_path, config_hash)
    elif mode == "smoke":
        _run_smoke(sampler, input_path, output_path, config_hash)
    else:
        _run_private(
            sampler,
            config,
            input_path,
            output_path,
            config_hash,
        )
    end = time.time()
    receipt = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "mode": mode,
        "allocation_job_id": allocation,
        "slurm_step_id": step,
        "execution_id": f"{allocation}.{step}",
        "node": node,
        "gpu_count": 1,
        "pool_slot": pool_slot,
        "slurm_gpus_per_task": 1,
        "visible_cuda_device": visible_cuda_device,
        "config_hash": config_hash,
        "source_hash": expected_source,
        "input_path": str(input_path),
        "input_sha256": task["input_sha256"],
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "start_epoch": start,
        "end_epoch": end,
        "elapsed_seconds": end - start,
        "exit_code": 0,
    }
    require_exact_keys(receipt, RECEIPT_KEYS, "task receipt")
    atomic_write_json(receipt_path, receipt)
    print(f"completed task {task['task_id']} in Slurm step {allocation}.{step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    run_task(args.task.resolve())


if __name__ == "__main__":
    main()
