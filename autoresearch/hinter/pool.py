"""Two-node exclusive Slurm pool with exact one-GPU task steps."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .core import (
    LOCK_PATH,
    POOL_ALLOCATION_KEYS,
    REPO_ROOT,
    RUNTIME_SOURCE_FILES,
    TASK_KEYS,
    WORK_ROOT,
    atomic_write_json,
    load_config,
    load_json,
    load_pool_allocation,
    refuse_existing,
    require_exact_keys,
    sha256_file,
    source_bundle_hash,
    validate_pool_allocation,
    validate_receipt,
    validate_registered_private_input,
    validate_registered_training_packet,
    validate_task_identity,
)


POOL_ROOT = WORK_ROOT / "pool"
PENDING = POOL_ROOT / "pending"
RUNNING = POOL_ROOT / "running"
DONE = POOL_ROOT / "done"
FAILED = POOL_ROOT / "failed"
CONTROL = POOL_ROOT / "control"
ALLOCATION_PATH = POOL_ROOT / "allocation.json"
STOP_PATH = CONTROL / "STOP"
SEALED_PATH = WORK_ROOT / "research" / "STOPPED.json"
TASK_LOGS = WORK_ROOT / "logs" / "tasks"
SBATCH_SCRIPT = REPO_ROOT / "autoresearch" / "jobs" / "hinter_pool.sbatch"
TASK_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")


def _make_dirs() -> None:
    for path in (
        PENDING,
        RUNNING,
        DONE,
        FAILED,
        CONTROL,
        TASK_LOGS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _queue_state(job_id: str) -> str | None:
    completed = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return states[0] if states else None


def _validate_allocation(
    value: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    *,
    require_current_source: bool = True,
) -> None:
    validate_pool_allocation(
        value,
        config=config,
        config_hash=config_hash,
        require_current_source=require_current_source,
    )


def start_pool(*, restart: bool = False) -> dict[str, Any]:
    if SEALED_PATH.exists():
        raise RuntimeError("cannot start a pool for a sealed experiment")
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is unavailable")
    config, config_hash = load_config(require_frozen=False)
    _make_dirs()
    if ALLOCATION_PATH.exists():
        previous = load_json(ALLOCATION_PATH)
        _validate_allocation(
            previous,
            config,
            config_hash,
            require_current_source=False,
        )
        state = _queue_state(str(previous["job_id"]))
        if state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}:
            if previous["source_hash"] != source_bundle_hash(
                RUNTIME_SOURCE_FILES
            ):
                raise RuntimeError(
                    "active pool allocation uses stale runtime source; "
                    "stop it before restarting"
                )
            return {**previous, "state": state, "reused": True}
        if not restart:
            raise RuntimeError(
                "previous pool allocation is terminal; pass --restart after "
                "checking its logs"
            )
        archived = POOL_ROOT / f"allocation.{previous['job_id']}.json"
        refuse_existing(archived)
        os.replace(ALLOCATION_PATH, archived)
        if STOP_PATH.exists():
            STOP_PATH.unlink()

    slurm = config["slurm"]
    nodes = ",".join(slurm["nodes"])
    pool_slots = int(slurm["pool_slots"])
    gpus_per_node = int(slurm["gpus_per_node"])
    command = [
        "sbatch",
        "--parsable",
        f"--partition={slurm['partition']}",
        "--nodes=2",
        f"--ntasks={pool_slots}",
        f"--ntasks-per-node={gpus_per_node}",
        f"--gpus-per-node={gpus_per_node}",
        f"--cpus-per-task={slurm['cpus_per_step']}",
        f"--mem={slurm['memory']}",
        "--exclusive",
        f"--nodelist={nodes}",
        f"--time={slurm['time']}",
        f"--output={WORK_ROOT / 'logs' / 'pool_%j.out'}",
        f"--error={WORK_ROOT / 'logs' / 'pool_%j.err'}",
        "--export="
        + ",".join(
            [
                "ALL",
                f"RLAD_REPO_ROOT={REPO_ROOT}",
                f"RLAD_AUTORESEARCH_WORK={WORK_ROOT}",
                f"AUTORESEARCH_PYTHON={sys.executable}",
            ]
        ),
        str(SBATCH_SCRIPT),
    ]
    environment = os.environ.copy()
    environment["RLAD_AUTORESEARCH_LAMBDA"] = str(
        config["objective"]["lambda"]
    )
    environment["RLAD_AUTORESEARCH_PARTITION"] = slurm["partition"]
    environment["RLAD_AUTORESEARCH_NODES"] = nodes
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
    )
    raw = completed.stdout.strip()
    job_id = raw.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch response: {raw!r}")
    value = {
        "schema_version": 1,
        "job_id": job_id,
        "partition": slurm["partition"],
        "nodes": slurm["nodes"],
        "pool_slots": slurm["pool_slots"],
        "exclusive": True,
        "gpus_per_node": slurm["gpus_per_node"],
        "gpus_per_task": slurm["gpus_per_task"],
        "gpu_binding": slurm["gpu_binding"],
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
        "work_root": str(WORK_ROOT),
        "config_hash": config_hash,
        "source_hash": source_bundle_hash(RUNTIME_SOURCE_FILES),
        "submitted_epoch": time.time(),
    }
    require_exact_keys(value, POOL_ALLOCATION_KEYS, "pool allocation")
    atomic_write_json(ALLOCATION_PATH, value)
    return {**value, "state": _queue_state(job_id), "reused": False}


def _inside_work(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(WORK_ROOT):
        raise ValueError(f"pool path escapes autoresearch workspace: {resolved}")
    return resolved


def _task_locations(task_id: str) -> dict[str, Path]:
    return {
        "pending": PENDING / f"{task_id}.json",
        "running": RUNNING / f"{task_id}.json",
        "done": DONE / f"{task_id}.json",
        "failed": FAILED / f"{task_id}.json",
    }


def enqueue_task(
    *,
    task_id: str,
    mode: str,
    input_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if SEALED_PATH.exists():
        raise RuntimeError("cannot enqueue work for a sealed experiment")
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("task ID must be lowercase alphanumeric, '_' or '-'")
    if mode not in {"train", "private", "smoke", "private-smoke"}:
        raise ValueError(f"unsupported mode: {mode}")
    validate_task_identity(task_id, mode)
    config, config_hash = load_config(
        require_frozen=mode in {"train", "private"}
    )
    _make_dirs()
    input_path = _inside_work(input_path)
    output_path = _inside_work(output_path)
    receipt_path = _inside_work(receipt_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if mode == "train":
        validate_registered_training_packet(
            input_path,
            task_id=task_id,
            config=config,
            config_hash=config_hash,
        )
    elif mode in {"private", "private-smoke"}:
        validate_registered_private_input(
            input_path,
            task_id=task_id,
            mode=mode,
            config=config,
            config_hash=config_hash,
        )
    locations = _task_locations(task_id)
    present = [path for path in locations.values() if path.exists()]
    if present:
        if locations["done"] in present:
            existing = load_json(locations["done"])
            require_exact_keys(existing, TASK_KEYS, "completed task")
            expected_existing = {
                "schema_version": 1,
                "task_id": task_id,
                "mode": mode,
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "output_path": str(output_path),
                "receipt_path": str(receipt_path),
                "config_hash": config_hash,
                "source_hash": source_bundle_hash(RUNTIME_SOURCE_FILES),
                "gpu_count": 1,
            }
            for key, expected in expected_existing.items():
                if existing[key] != expected:
                    raise RuntimeError(
                        f"completed task has stale {key}: {task_id}"
                    )
            if not output_path.is_file() or not receipt_path.is_file():
                raise RuntimeError("completed task artifacts are missing")
            validate_receipt(
                load_json(receipt_path),
                task_id=task_id,
                mode=mode,
                input_path=input_path,
                output_path=output_path,
                config_hash=config_hash,
            )
            return {
                "task_id": task_id,
                "state": "done",
                "path": str(locations["done"]),
                "reused": True,
            }
        raise RuntimeError(f"task ID already exists: {present}")
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("refusing task with pre-existing artifacts")
    if SEALED_PATH.exists():
        raise RuntimeError("experiment was sealed while enqueueing work")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": mode,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "receipt_path": str(receipt_path),
        "config_hash": config_hash,
        "source_hash": source_bundle_hash(RUNTIME_SOURCE_FILES),
        "gpu_count": 1,
        "created_epoch": time.time(),
    }
    require_exact_keys(task, TASK_KEYS, "pool task")
    atomic_write_json(locations["pending"], task)
    return {
        "task_id": task_id,
        "state": "pending",
        "path": str(locations["pending"]),
        "reused": False,
    }


def task_status(task_id: str) -> dict[str, Any]:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("invalid task ID")
    for state, path in _task_locations(task_id).items():
        if path.exists():
            task = load_json(path)
            result: dict[str, Any] = {
                "task_id": task_id,
                "state": state,
                "mode": task["mode"],
                "output_present": Path(task["output_path"]).is_file(),
                "receipt_present": Path(task["receipt_path"]).is_file(),
            }
            if result["receipt_present"]:
                receipt = load_json(Path(task["receipt_path"]))
                result["execution_id"] = receipt.get("execution_id")
                result["node"] = receipt.get("node")
                result["elapsed_seconds"] = receipt.get("elapsed_seconds")
            error_path = FAILED / f"{task_id}.error.json"
            if state == "failed" and error_path.exists():
                result["error"] = load_json(error_path)
            return result
    return {"task_id": task_id, "state": "missing"}


def wait_task(task_id: str, poll_seconds: float = 10.0) -> dict[str, Any]:
    while True:
        status = task_status(task_id)
        if status["state"] in {"done", "failed", "missing"}:
            return status
        if ALLOCATION_PATH.exists():
            allocation = load_json(ALLOCATION_PATH)
            config, config_hash = load_config(require_frozen=False)
            _validate_allocation(allocation, config, config_hash)
            allocation_state = _queue_state(str(allocation["job_id"]))
            if allocation_state is None:
                return {
                    **status,
                    "state": "pool-terminal",
                    "allocation_job_id": allocation["job_id"],
                }
        time.sleep(poll_seconds)


def pool_status() -> dict[str, Any]:
    allocation = None
    if ALLOCATION_PATH.exists():
        allocation = load_json(ALLOCATION_PATH)
        config, config_hash = load_config(require_frozen=False)
        _validate_allocation(allocation, config, config_hash)
        allocation = {
            **allocation,
            "state": _queue_state(str(allocation["job_id"])),
        }
    counts = {}
    for state, directory in (
        ("pending", PENDING),
        ("running", RUNNING),
        ("done", DONE),
        ("failed", FAILED),
    ):
        counts[state] = (
            len(
                [
                    path
                    for path in directory.glob("*.json")
                    if not path.name.endswith(".error.json")
                ]
            )
            if directory.exists()
            else 0
        )
    return {"allocation": allocation, "task_counts": counts}


def request_stop() -> dict[str, Any]:
    _make_dirs()
    STOP_PATH.write_text(
        "stop after active steps finish; do not claim pending tasks\n",
        encoding="utf-8",
    )
    return {"stop_requested": True, "path": str(STOP_PATH)}


def _recover_running() -> None:
    for path in sorted(RUNNING.glob("*.json")):
        task = load_json(path)
        output = Path(task["output_path"])
        receipt = Path(task["receipt_path"])
        if output.is_file() and receipt.is_file():
            try:
                validate_receipt(
                    load_json(receipt),
                    task_id=task["task_id"],
                    mode=task["mode"],
                    input_path=Path(task["input_path"]),
                    output_path=output,
                    config_hash=task["config_hash"],
                )
                os.replace(path, DONE / path.name)
                continue
            except Exception:
                pass
        if output.exists() or receipt.exists():
            failure = FAILED / f"{task['task_id']}.error.json"
            atomic_write_json(
                failure,
                {
                    "error": "stale running task has partial artifacts; "
                    "manual inspection is required"
                },
            )
            os.replace(path, FAILED / path.name)
        else:
            os.replace(path, PENDING / path.name)


def retry_failed(
    task_id: str,
    *,
    refresh_source: bool = False,
) -> dict[str, Any]:
    """Archive a failed attempt and requeue it, optionally refreshing setup code."""
    if SEALED_PATH.exists():
        raise RuntimeError("cannot retry work for a sealed experiment")
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("invalid task ID")
    _make_dirs()
    failed_path = FAILED / f"{task_id}.json"
    if not failed_path.is_file():
        raise RuntimeError(f"failed task does not exist: {task_id}")
    task = load_json(failed_path)
    require_exact_keys(task, TASK_KEYS, "failed task")
    config, config_hash = load_config(
        require_frozen=task["mode"] in {"train", "private"}
    )
    del config
    if refresh_source and task["mode"] not in {"smoke", "private-smoke"}:
        raise RuntimeError(
            "source refresh is allowed only for pre-freeze setup smoke tasks"
        )
    if refresh_source and LOCK_PATH.exists():
        raise RuntimeError("cannot refresh setup tasks after experiment freeze")
    current_source = source_bundle_hash(RUNTIME_SOURCE_FILES)
    if task["source_hash"] != current_source and not refresh_source:
        raise RuntimeError(
            "failed task uses stale source; setup smoke retries require "
            "--refresh-source"
        )
    archive = POOL_ROOT / "retries" / task_id / str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(failed_path, archive / "task.json")
    for label, path in (
        ("output", Path(task["output_path"])),
        ("receipt", Path(task["receipt_path"])),
        ("error", FAILED / f"{task_id}.error.json"),
    ):
        if path.exists():
            os.replace(path, archive / f"{label}{path.suffix}")
    pending_path = PENDING / failed_path.name
    if pending_path.exists():
        raise RuntimeError("pending retry path already exists")
    if refresh_source:
        input_path = Path(task["input_path"])
        refreshed = {
            **task,
            "input_sha256": sha256_file(input_path),
            "config_hash": config_hash,
            "source_hash": current_source,
            "created_epoch": time.time(),
        }
        require_exact_keys(refreshed, TASK_KEYS, "refreshed setup task")
        failed_path.unlink()
        atomic_write_json(pending_path, refreshed)
    else:
        os.replace(failed_path, pending_path)
    return {
        "task_id": task_id,
        "state": "pending",
        "archived_attempt": str(archive),
        "source_refreshed": refresh_source,
    }


def refresh_setup_task(task_id: str) -> dict[str, Any]:
    """Archive and recreate a completed/failed smoke after reviewed source edits."""
    if SEALED_PATH.exists():
        raise RuntimeError("cannot refresh work for a sealed experiment")
    if task_id not in {"setup-smoke", "setup-private-smoke"}:
        raise ValueError("only the two setup smoke gates can be refreshed")
    if LOCK_PATH.exists():
        raise RuntimeError("cannot refresh setup tasks after experiment freeze")
    _make_dirs()
    locations = _task_locations(task_id)
    if locations["pending"].exists() or locations["running"].exists():
        raise RuntimeError("cannot refresh an active setup task")
    source_path = (
        locations["done"]
        if locations["done"].is_file()
        else locations["failed"]
    )
    if not source_path.is_file():
        raise RuntimeError(f"setup task has no completed attempt: {task_id}")
    task = load_json(source_path)
    require_exact_keys(task, TASK_KEYS, "setup task")
    if task["mode"] not in {"smoke", "private-smoke"}:
        raise RuntimeError("setup task has the wrong mode")
    config, config_hash = load_config(require_frozen=False)
    del config
    archive = POOL_ROOT / "retries" / task_id / str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_path, archive / "task.json")
    for label, path in (
        ("output", Path(task["output_path"])),
        ("receipt", Path(task["receipt_path"])),
        ("error", FAILED / f"{task_id}.error.json"),
    ):
        if path.exists():
            os.replace(path, archive / f"{label}{path.suffix}")
    source_path.unlink()
    input_path = Path(task["input_path"])
    refreshed = {
        **task,
        "input_sha256": sha256_file(input_path),
        "config_hash": config_hash,
        "source_hash": source_bundle_hash(RUNTIME_SOURCE_FILES),
        "created_epoch": time.time(),
    }
    require_exact_keys(refreshed, TASK_KEYS, "refreshed setup task")
    atomic_write_json(locations["pending"], refreshed)
    return {
        "task_id": task_id,
        "state": "pending",
        "archived_attempt": str(archive),
        "source_refreshed": True,
    }


def _claim_one() -> tuple[Path, dict[str, Any]] | None:
    if SEALED_PATH.exists():
        return None
    for pending in sorted(PENDING.glob("*.json")):
        running = RUNNING / pending.name
        try:
            os.replace(pending, running)
        except FileNotFoundError:
            continue
        task = load_json(running)
        require_exact_keys(task, TASK_KEYS, "claimed task")
        return running, task
    return None


def _launch(
    running_path: Path,
    task: dict[str, Any],
    config: dict[str, Any],
    slot: tuple[str, int],
) -> subprocess.Popen[bytes]:
    task_id = task["task_id"]
    node, gpu_index = slot
    stdout = TASK_LOGS / f"{task_id}.out"
    stderr = TASK_LOGS / f"{task_id}.err"
    command = [
        "srun",
        "--exclusive",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        f"--cpus-per-task={config['slurm']['cpus_per_step']}",
        "--gpus-per-task=1",
        "--gpu-bind=single:1",
        "--kill-on-bad-exit=1",
        f"--job-name={task_id}",
        f"--output={stdout}",
        f"--error={stderr}",
        sys.executable,
        "-m",
        "autoresearch.hinter.job",
        "--task",
        str(running_path),
    ]
    environment = os.environ.copy()
    environment["RLAD_REPO_ROOT"] = str(REPO_ROOT)
    environment["RLAD_AUTORESEARCH_WORK"] = str(WORK_ROOT)
    environment["AUTORESEARCH_POOL_SLOT"] = str(gpu_index)
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def dispatch() -> int:
    """Run inside the two-node parent allocation and launch one-GPU steps."""
    allocation = os.environ.get("SLURM_JOB_ID")
    if not allocation or not allocation.isdigit():
        raise RuntimeError("pool dispatcher must run inside sbatch")
    config, config_hash = load_config(require_frozen=False)
    deadline = time.monotonic() + 30
    while not ALLOCATION_PATH.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    active = load_pool_allocation(config, config_hash)
    if active["job_id"] != allocation:
        raise RuntimeError("dispatcher job does not match the active allocation")
    _make_dirs()
    _recover_running()
    processes: dict[
        str,
        tuple[
            subprocess.Popen[bytes],
            Path,
            dict[str, Any],
            tuple[str, int],
        ],
    ] = {}
    max_active = int(config["slurm"]["pool_slots"])
    slots = [
        (node, gpu_index)
        for node in config["slurm"]["nodes"]
        for gpu_index in range(int(config["slurm"]["gpus_per_node"]))
    ]
    if len(slots) != max_active:
        raise RuntimeError("configured GPU slot count is inconsistent")
    while True:
        for task_id, (
            process,
            running_path,
            task,
            slot,
        ) in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            output = Path(task["output_path"])
            receipt_path = Path(task["receipt_path"])
            try:
                if return_code != 0:
                    raise RuntimeError(f"srun exited {return_code}")
                if not output.is_file() or not receipt_path.is_file():
                    raise RuntimeError("task ended without output and receipt")
                receipt = load_json(receipt_path)
                validate_receipt(
                    receipt,
                    task_id=task_id,
                    mode=task["mode"],
                    input_path=Path(task["input_path"]),
                    output_path=output,
                    config_hash=task["config_hash"],
                    expected_node=slot[0],
                    expected_pool_slot=slot[1],
                )
                os.replace(running_path, DONE / running_path.name)
            except Exception as error:
                atomic_write_json(
                    FAILED / f"{task_id}.error.json",
                    {"error": f"{type(error).__name__}: {error}"},
                )
                os.replace(running_path, FAILED / running_path.name)
            del processes[task_id]

        stopping = STOP_PATH.exists() or SEALED_PATH.exists()
        if not stopping:
            while len(processes) < max_active:
                used_slots = {value[3] for value in processes.values()}
                available_slots = [
                    slot for slot in slots if slot not in used_slots
                ]
                if not available_slots:
                    break
                claimed = _claim_one()
                if claimed is None:
                    break
                running_path, task = claimed
                slot = available_slots[0]
                processes[task["task_id"]] = (
                    _launch(running_path, task, config, slot),
                    running_path,
                    task,
                    slot,
                )

        if (STOP_PATH.exists() or SEALED_PATH.exists()) and not processes:
            return 0
        time.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--restart", action="store_true")
    commands.add_parser("status")
    stop = commands.add_parser("stop")
    del stop
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--task-id", required=True)
    enqueue.add_argument(
        "--mode",
        choices=("train", "private", "smoke", "private-smoke"),
        required=True,
    )
    enqueue.add_argument("--input", type=Path, required=True)
    enqueue.add_argument("--output", type=Path, required=True)
    enqueue.add_argument("--receipt", type=Path, required=True)
    inspect = commands.add_parser("task-status")
    inspect.add_argument("--task-id", required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--task-id", required=True)
    wait.add_argument("--poll-seconds", type=float, default=10.0)
    retry = commands.add_parser("retry")
    retry.add_argument("--task-id", required=True)
    retry.add_argument("--refresh-source", action="store_true")
    refresh = commands.add_parser("refresh-setup")
    refresh.add_argument("--task-id", required=True)
    commands.add_parser("dispatch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "start":
        value = start_pool(restart=args.restart)
    elif args.command == "status":
        value = pool_status()
    elif args.command == "stop":
        value = request_stop()
    elif args.command == "enqueue":
        value = enqueue_task(
            task_id=args.task_id,
            mode=args.mode,
            input_path=args.input,
            output_path=args.output,
            receipt_path=args.receipt,
        )
    elif args.command == "task-status":
        value = task_status(args.task_id)
    elif args.command == "wait":
        value = wait_task(args.task_id, args.poll_seconds)
    elif args.command == "retry":
        value = retry_failed(
            args.task_id,
            refresh_source=args.refresh_source,
        )
    elif args.command == "refresh-setup":
        value = refresh_setup_task(args.task_id)
    else:
        raise SystemExit(dispatch())
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
