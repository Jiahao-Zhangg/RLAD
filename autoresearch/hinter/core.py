"""Experiment invariants, hashing, schemas, and held-out-safe validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from packaging.version import Version


PACKAGE_ROOT = Path(__file__).resolve().parent
AUTORESEARCH_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = AUTORESEARCH_ROOT.parent
WORK_ROOT = Path(
    os.environ.get("RLAD_AUTORESEARCH_WORK", REPO_ROOT / "work_zsw")
).resolve()
WORKSPACE_LAMBDAS = {
    "work_zsw": 1,
    "work_zsw_lambda2": 2,
    "work_zsw_lambda5": 5,
    "work_zsw_lambda10": 10,
}
if (
    not WORK_ROOT.is_relative_to(REPO_ROOT)
    or WORK_ROOT.parent != REPO_ROOT
    or WORK_ROOT.name not in WORKSPACE_LAMBDAS
):
    raise RuntimeError(
        "RLAD_AUTORESEARCH_WORK must be one of this repository's isolated "
        f"workspaces: {sorted(WORKSPACE_LAMBDAS)}"
    )
RESEARCH_ROOT = WORK_ROOT / "research"
CONFIG_PATH = AUTORESEARCH_ROOT / "config.json"
LOCK_PATH = WORK_ROOT / "experiment.lock.json"

ROUND_ID_PATTERN = r"(?:0[1-9]|1[0-9]|20)"
ALLOWED_ALLOCATIONS = {
    1: (
        "ml.p5.48xlarge",
        ("ip-10-1-38-11", "ip-10-1-81-8"),
    ),
    2: (
        "ml.p4d.24xlarge",
        ("ip-10-1-173-179", "ip-10-1-184-205"),
    ),
    5: (
        "ml.p4d.24xlarge",
        ("ip-10-1-196-96", "ip-10-1-226-48"),
    ),
    10: (
        "ml.p5.48xlarge",
        ("ip-10-1-38-11", "ip-10-1-81-8"),
    ),
}

TRAIN_POSITIONS = (163, 28, 6, 189, 70, 62, 57, 35, 188, 26)
HELDOUT_POSITIONS = (173, 139, 22, 151, 108, 8, 7, 23, 55, 59)

SETUP_SMOKE_HINT = (
    "Identify the governing mathematical structure and translate every "
    "condition into an explicit equation, invariant, or finite case split. "
    "Check exceptional and boundary cases before accepting the result."
)

RUNTIME_SOURCE_FILES = (
    "autoresearch/hinter/core.py",
    "autoresearch/hinter/data.py",
    "autoresearch/hinter/grader.py",
    "autoresearch/hinter/sampling.py",
    "autoresearch/hinter/job.py",
    "autoresearch/hinter/pool.py",
    "autoresearch/jobs/hinter_pool.sbatch",
    "train/rl/eval/vllm_eval.py",
    "train/rl/rlad_plugin/templates.py",
)

TRAIN_PACKET_KEYS = {
    "schema_version",
    "round",
    "hint_id",
    "train_position",
    "train_qid",
    "problem",
    "answer",
    "hint",
    "hint_hash",
    "config_hash",
    "previous_train_i",
    "previous_heldout_i",
    "previous_J_i",
    "hard_tokens_per_hint",
    "worker_tokens_per_hint",
    "hard_total_tokens",
}

TRAIN_RESULT_KEYS = {
    "schema_version",
    "round",
    "hint_id",
    "train_position",
    "train_qid",
    "hint_hash",
    "config_hash",
    "correct",
    "total",
    "train_i",
    "rollouts",
}

TRAIN_ROLLOUT_KEYS = {
    "sample_idx",
    "response",
    "reward",
    "finish_reason",
    "completion_tokens",
}

PRIVATE_INPUT_KEYS = {
    "schema_version",
    "hint_id",
    "hint",
    "hint_hash",
    "config_hash",
}

PRIVATE_RESULT_KEYS = {
    "schema_version",
    "hint_id",
    "hint_hash",
    "config_hash",
    "train_correct",
    "train_total",
    "train_i",
    "heldout_correct",
    "heldout_total",
    "heldout_i",
    "J_numerator",
    "J_denominator",
    "J_i",
}

TASK_KEYS = {
    "schema_version",
    "task_id",
    "mode",
    "input_path",
    "input_sha256",
    "output_path",
    "receipt_path",
    "config_hash",
    "source_hash",
    "gpu_count",
    "created_epoch",
}

RECEIPT_KEYS = {
    "schema_version",
    "task_id",
    "mode",
    "allocation_job_id",
    "slurm_step_id",
    "execution_id",
    "node",
    "gpu_count",
    "pool_slot",
    "slurm_gpus_per_task",
    "visible_cuda_device",
    "config_hash",
    "source_hash",
    "input_path",
    "input_sha256",
    "output_path",
    "output_sha256",
    "start_epoch",
    "end_epoch",
    "elapsed_seconds",
    "exit_code",
}

POOL_ALLOCATION_KEYS = {
    "schema_version",
    "job_id",
    "partition",
    "nodes",
    "pool_slots",
    "exclusive",
    "gpus_per_node",
    "gpus_per_task",
    "gpu_binding",
    "python",
    "repo_root",
    "work_root",
    "config_hash",
    "source_hash",
    "submitted_epoch",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        + b"\n",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")


def require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def validate_task_identity(task_id: str, mode: str) -> None:
    patterns = {
        "smoke": r"setup-smoke",
        "private-smoke": r"setup-private-smoke",
        "train": rf"r{ROUND_ID_PATTERN}-train-h(?:0[1-9]|10)",
        "private": (
            rf"(?:baseline|r{ROUND_ID_PATTERN}-private)-h(?:0[1-9]|10)"
        ),
    }
    pattern = patterns.get(mode)
    if pattern is None or re.fullmatch(pattern, task_id) is None:
        raise ValueError(f"task ID {task_id!r} is invalid for mode {mode!r}")


def source_hash_manifest() -> dict[str, str]:
    """Hash all experiment source, plus the shared sampler and prompt templates."""
    allowed_suffixes = {
        ".json",
        ".lock",
        ".md",
        ".py",
        ".sbatch",
        ".sh",
        ".toml",
    }
    paths = []
    for path in AUTORESEARCH_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(AUTORESEARCH_ROOT).parts
        if any(
            part in {".venv", "__pycache__", ".pytest_cache"}
            for part in relative_parts
        ):
            continue
        if path.suffix in allowed_suffixes:
            paths.append(path)
    paths.extend(
        [
            REPO_ROOT / ".gitignore",
            REPO_ROOT / "docs/plan/hinter.md",
            REPO_ROOT / "train/rl/eval/vllm_eval.py",
            REPO_ROOT / "train/rl/rlad_plugin/templates.py",
        ]
    )
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in sorted(set(paths))
    }


def source_bundle_hash(paths: tuple[str, ...] | None = None) -> str:
    if paths is None:
        manifest = source_hash_manifest()
    else:
        manifest = {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in paths
        }
    return sha256_bytes(canonical_json_bytes(manifest))


def _apply_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return the workspace-specific config selected by a launch wrapper."""
    effective = json.loads(json.dumps(config))
    raw_lambda = os.environ.get("RLAD_AUTORESEARCH_LAMBDA")
    raw_partition = os.environ.get("RLAD_AUTORESEARCH_PARTITION")
    raw_nodes = os.environ.get("RLAD_AUTORESEARCH_NODES")
    if raw_lambda is not None:
        try:
            objective_lambda = int(raw_lambda)
        except ValueError as error:
            raise RuntimeError(
                "RLAD_AUTORESEARCH_LAMBDA must be an integer"
            ) from error
        effective["objective"]["lambda"] = objective_lambda
    if raw_partition is not None:
        effective["slurm"]["partition"] = raw_partition
    if raw_nodes is not None:
        nodes = [node.strip() for node in raw_nodes.split(",")]
        if any(not node for node in nodes):
            raise RuntimeError("RLAD_AUTORESEARCH_NODES contains an empty node")
        effective["slurm"]["nodes"] = nodes
    objective_lambda = effective["objective"]["lambda"]
    effective["publication"]["commit_prefix"] = (
        f"autoresearch: hinter lambda-{objective_lambda} round"
    )
    return effective


def _validate_config(config: dict[str, Any]) -> None:
    require_exact_keys(
        config,
        {
            "schema_version",
            "name",
            "dataset",
            "student",
            "grader",
            "runtime",
            "sampling",
            "objective",
            "hint_limits",
            "budget",
            "slurm",
            "publication",
        },
        "experiment config",
    )
    if (
        config["schema_version"] != 1
        or config["name"] != "parallel-10-hint-book-autoresearch"
    ):
        raise RuntimeError("experiment config schema must be 1")
    dataset = config["dataset"]
    expected_dataset = {
        "repo_id":
            "zjhhhh/DeepScaleR-Qwen3-1.7B-2k-strategy-error-200",
        "revision": "1764f5c2ed03ddd7a9b7b9d9252e62f1e8fab3a6",
        "fingerprint": "ff45e4dc2b5c512a",
        "parquet_sha256":
            "a322a82aa1c8e6cb81dbcee11be982ad8d1a9f35e5c3d050b3de8fda3b146271",
        "row_count": 200,
        "train_positions": list(TRAIN_POSITIONS),
        "heldout_positions": list(HELDOUT_POSITIONS),
    }
    if dataset != expected_dataset:
        raise RuntimeError("pinned dataset identity or split changed")
    if tuple(dataset["train_positions"]) != TRAIN_POSITIONS:
        raise RuntimeError("fixed training split changed")
    if tuple(dataset["heldout_positions"]) != HELDOUT_POSITIONS:
        raise RuntimeError("fixed held-out split changed")
    if set(TRAIN_POSITIONS) & set(HELDOUT_POSITIONS):
        raise RuntimeError("training and held-out splits overlap")
    if config["student"] != {
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "max_model_len": 40960,
        "gpu_memory_utilization": 0.9,
    }:
        raise RuntimeError("student checkpoint settings changed")
    if config["grader"] != {
        "repo": "https://github.com/radixark/miles.git",
        "revision": "9437366e0aa3a25294720f70d18b081067595f85",
        "deepscaler_sha256":
            "7be863a12650d5054f5245bee0a9f01e1efe4b314db2e272c9d4261e18feb0a0",
        "math_utils_sha256":
            "db83eabd2941fb21253c9f733752d775d80488797582e093f2b94bb29d3d39cc",
    }:
        raise RuntimeError("pinned grader identity changed")
    if config["runtime"] != {
        "python_min": "3.10",
        "python_max_exclusive": "3.13",
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "datasets": "3.6.0",
        "huggingface_hub": "0.36.2",
        "vllm": "0.8.5.post1",
        "math_verify": "0.6.0",
        "pylatexenc": "2.10",
        "sympy": "1.13.1",
        "packaging": "25.0",
    }:
        raise RuntimeError("runtime version pins changed")
    limits = config["hint_limits"]
    expected_limits = {
        "hints": 10,
        "hard_tokens_per_hint": 256,
        "worker_tokens_per_hint": 200,
        "hard_total_tokens": 2048,
    }
    if limits != expected_limits:
        raise RuntimeError(f"hint limits changed: {limits}")
    sampling = config["sampling"]
    if sampling != {
        "rollouts": 8,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": False,
        "seed": 1234,
        "tensor_parallel_size": 1,
        "max_tokens": 16384,
        "thinking_enabled": True,
    }:
        raise RuntimeError("student sampling invariants changed")
    objective = config["objective"]
    objective_lambda = objective.get("lambda")
    if (
        isinstance(objective_lambda, bool)
        or not isinstance(objective_lambda, int)
        or objective_lambda not in ALLOWED_ALLOCATIONS
    ):
        raise RuntimeError("objective lambda must be one of 1, 2, 5, or 10")
    if objective != {
        "lambda": objective_lambda,
        "train_denominator": 8,
        "heldout_questions": 10,
        "heldout_denominator": 80,
        "comparison": ["higher_J", "higher_heldout", "shorter_hint"],
    }:
        raise RuntimeError("objective invariants changed")
    if config["budget"] != {"max_rounds": 20}:
        raise RuntimeError("experiment budget changed")
    slurm = config["slurm"]
    partition, nodes = ALLOWED_ALLOCATIONS[objective_lambda]
    if slurm != {
        "partition": partition,
        "nodes": list(nodes),
        "exclusive": True,
        "gpus_per_node": 8,
        "pool_slots": 16,
        "gpus_per_task": 1,
        "gpu_binding": "slurm_gpus_per_task_1_gpu_bind_single_1",
        "cpus_per_step": 12,
        "memory": "0",
        "time": "1-00:00:00",
    }:
        raise RuntimeError(
            f"two-node allocation changed for lambda={objective_lambda}"
        )
    if config["publication"] != {
        "remote": "origin",
        "commit_prefix": (
            f"autoresearch: hinter lambda-{objective_lambda} round"
        ),
    }:
        raise RuntimeError("publication settings changed")


def load_config(*, require_frozen: bool = False) -> tuple[dict[str, Any], str]:
    config = _apply_environment_overrides(load_json(CONFIG_PATH))
    _validate_config(config)
    expected_lambda = WORKSPACE_LAMBDAS.get(WORK_ROOT.name)
    if (
        expected_lambda is not None
        and config["objective"]["lambda"] != expected_lambda
    ):
        raise RuntimeError(
            f"workspace {WORK_ROOT.name} is reserved for lambda="
            f"{expected_lambda}, not lambda={config['objective']['lambda']}"
        )
    config_hash = sha256_bytes(canonical_json_bytes(config))
    if require_frozen:
        if not LOCK_PATH.exists():
            raise RuntimeError(f"experiment is not frozen: {LOCK_PATH}")
        lock = load_json(LOCK_PATH)
        require_exact_keys(
            lock,
            {
                "schema_version",
                "status",
                "config_hash",
                "source_sha256",
                "setup_sha256",
                "review_sha256",
                "created_epoch",
            },
            "experiment lock",
        )
        if lock["schema_version"] != 1 or lock["status"] != "frozen":
            raise RuntimeError("experiment lock is not frozen")
        if lock["config_hash"] != config_hash:
            raise RuntimeError("experiment configuration drifted after freeze")
        current = source_hash_manifest()
        if lock["source_sha256"] != current:
            changed = sorted(
                key
                for key in set(lock["source_sha256"]) | set(current)
                if lock["source_sha256"].get(key) != current.get(key)
            )
            raise RuntimeError(f"frozen experiment source drift: {changed}")
        for relative, digest in lock["setup_sha256"].items():
            path = REPO_ROOT / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"frozen setup artifact drift: {relative}")
        review_path = WORK_ROOT / "review" / "independent_review.json"
        if (
            not review_path.is_file()
            or sha256_file(review_path) != lock["review_sha256"]
        ):
            raise RuntimeError("frozen independent review receipt drift")
    return config, config_hash


def verify_runtime(config: dict[str, Any]) -> dict[str, str]:
    import torch

    names = {
        "transformers": "transformers",
        "datasets": "datasets",
        "huggingface_hub": "huggingface-hub",
        "vllm": "vllm",
        "math_verify": "math-verify",
        "pylatexenc": "pylatexenc",
        "sympy": "sympy",
        "packaging": "packaging",
    }
    observed = {
        "python": platform.python_version(),
        "torch": Version(torch.__version__).base_version,
        **{
            key: importlib.metadata.version(distribution)
            for key, distribution in names.items()
        },
    }
    expected = config["runtime"]
    python_version = Version(observed["python"])
    if not (
        python_version >= Version(expected["python_min"])
        and python_version < Version(expected["python_max_exclusive"])
    ):
        raise RuntimeError(f"unsupported Python runtime: {observed['python']}")
    for key in ("torch", *names):
        if observed[key] != str(expected[key]):
            raise RuntimeError(
                f"runtime drift for {key}: {observed[key]} != {expected[key]}"
            )
    return observed


def canonicalize_hint(hint: str) -> str:
    if not isinstance(hint, str):
        raise TypeError("hint must be a string")
    result = unicodedata.normalize(
        "NFC",
        hint.replace("\r\n", "\n").replace("\r", "\n"),
    ).strip()
    if not result:
        raise ValueError("hint must not be empty")
    for character in result:
        if unicodedata.category(character) == "Cc" and character not in "\n\t":
            raise ValueError("hint contains a disallowed control character")
    return result


def hint_hash(hint: str) -> str:
    return sha256_bytes(canonicalize_hint(hint).encode("utf-8"))


def reject_answer_leak(hint: str, answer: str) -> None:
    canonical = canonicalize_hint(hint)
    lowered = canonical.casefold()
    if "\\boxed" in lowered or "final answer" in lowered:
        raise ValueError("hint contains explicit answer language")
    compact_hint = "".join(lowered.split()).replace("$", "")
    compact_answer = "".join(str(answer).casefold().split()).replace("$", "")
    if not compact_answer:
        raise ValueError("assigned answer is empty")
    if compact_answer in compact_hint:
        raise ValueError("hint contains the assigned answer")


def tokenizer_factory(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    student = config["student"]
    return AutoTokenizer.from_pretrained(
        student["repo_id"],
        revision=student["revision"],
        trust_remote_code=True,
    )


def validate_hint(
    hint: str,
    tokenizer: Any,
    maximum: int,
) -> tuple[str, int, str]:
    canonical = canonicalize_hint(hint)
    tokens = len(tokenizer.encode(canonical, add_special_tokens=False))
    if tokens > maximum:
        raise ValueError(f"hint has {tokens} tokens; maximum is {maximum}")
    return canonical, tokens, hint_hash(canonical)


def validate_book(
    book: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    rows = book.get("hints")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("hint book must contain exactly ten hints")
    if [row.get("hint_id") for row in rows] != list(range(1, 11)):
        raise ValueError("hint IDs must be ordered 1 through 10")
    if [row.get("train_position") for row in rows] != list(TRAIN_POSITIONS):
        raise ValueError("hint assignments changed")
    if book.get("config_hash", config_hash) != config_hash:
        raise ValueError("book config hash is stale")
    normalized = []
    hard_limit = int(config["hint_limits"]["hard_tokens_per_hint"])
    for row in rows:
        hint, tokens, digest = validate_hint(row["hint"], tokenizer, hard_limit)
        if "tokens" in row and row["tokens"] != tokens:
            raise ValueError(f"stale token count for hint {row['hint_id']}")
        if "hint_hash" in row and row["hint_hash"] != digest:
            raise ValueError(f"stale hash for hint {row['hint_id']}")
        normalized.append(
            {**row, "hint": hint, "tokens": tokens, "hint_hash": digest}
        )
    total_tokens = sum(row["tokens"] for row in normalized)
    if total_tokens > int(config["hint_limits"]["hard_total_tokens"]):
        raise ValueError("hint book exceeds the total token limit")
    identity = [
        {
            "hint_id": row["hint_id"],
            "train_position": row["train_position"],
            "hint_hash": row["hint_hash"],
        }
        for row in normalized
    ]
    digest = sha256_bytes(
        canonical_json_bytes({"config_hash": config_hash, "hints": identity})
    )
    if "book_hash" in book and book["book_hash"] != digest:
        raise ValueError("stored book hash is stale")
    return {
        **book,
        "schema_version": 1,
        "config_hash": config_hash,
        "hints": normalized,
        "total_tokens": total_tokens,
        "book_hash": digest,
    }


def exact_metrics(
    train_correct: int,
    heldout_correct: int,
    *,
    objective_lambda: int = 1,
) -> dict[str, Any]:
    if (
        isinstance(train_correct, bool)
        or not isinstance(train_correct, int)
        or not 0 <= train_correct <= 8
    ):
        raise ValueError("train_correct must be an integer in [0, 8]")
    if (
        isinstance(heldout_correct, bool)
        or not isinstance(heldout_correct, int)
        or not 0 <= heldout_correct <= 80
    ):
        raise ValueError("heldout_correct must be an integer in [0, 80]")
    if (
        isinstance(objective_lambda, bool)
        or not isinstance(objective_lambda, int)
        or objective_lambda not in ALLOWED_ALLOCATIONS
    ):
        raise ValueError("objective_lambda must be one of 1, 2, 5, or 10")
    numerator = train_correct * 10 + objective_lambda * heldout_correct
    return {
        "train_correct": train_correct,
        "train_total": 8,
        "train_i": train_correct / 8,
        "heldout_correct": heldout_correct,
        "heldout_total": 80,
        "heldout_i": heldout_correct / 80,
        "J_numerator": numerator,
        "J_denominator": 80,
        "J_i": numerator / 80,
    }


def validate_private_metrics(
    value: dict[str, Any],
    *,
    objective_lambda: int = 1,
) -> None:
    require_exact_keys(value, PRIVATE_RESULT_KEYS, "private result")
    train_correct = value["train_correct"]
    heldout_correct = value["heldout_correct"]
    if (
        isinstance(train_correct, bool)
        or not isinstance(train_correct, int)
        or isinstance(heldout_correct, bool)
        or not isinstance(heldout_correct, int)
    ):
        raise ValueError("private correct counts must be integers")
    recomputed = exact_metrics(
        train_correct,
        heldout_correct,
        objective_lambda=objective_lambda,
    )
    for key, expected in recomputed.items():
        if value[key] != expected:
            raise ValueError(f"private metric {key} was not count-derived")


def validate_registered_training_packet(
    input_path: Path,
    *,
    task_id: str,
    config: dict[str, Any],
    config_hash: str,
    research_root: Path | None = None,
) -> dict[str, Any]:
    root = RESEARCH_ROOT if research_root is None else research_root.resolve()
    validate_task_identity(task_id, "train")
    match = re.fullmatch(
        r"r([0-9]{2})-train-h(0[1-9]|10)",
        task_id,
    )
    assert match is not None
    round_number = int(match.group(1))
    hint_id = int(match.group(2))
    if round_number > int(config["budget"]["max_rounds"]):
        raise ValueError("training task exceeds the round budget")
    expected_path = (
        root / "rounds" / f"{round_number:03d}"
        / "training_inputs" / f"hint_{hint_id:02d}.json"
    ).resolve()
    if input_path.resolve() != expected_path:
        raise ValueError("training task does not use its registered packet path")

    packet = load_json(input_path)
    require_exact_keys(packet, TRAIN_PACKET_KEYS, "training packet")
    public_rows = load_json(root / "setup" / "train_public.json")
    if not isinstance(public_rows, list) or len(public_rows) != 10:
        raise ValueError("public training manifest is incomplete")
    public = public_rows[hint_id - 1]
    book = load_json(
        root / "rounds" / f"{round_number:03d}"
        / "book_before.json"
    )
    if (
        book.get("config_hash") != config_hash
        or book.get("generation") != round_number - 1
        or len(book.get("hints", [])) != 10
    ):
        raise ValueError("round incumbent book is stale")
    incumbent = book["hints"][hint_id - 1]
    best = load_json(root / "best_per_hint.json")
    if (
        best.get("config_hash") != config_hash
        or len(best.get("hints", [])) != 10
    ):
        raise ValueError("best-per-hint ledger is stale")
    metrics = best["hints"][hint_id - 1]["metrics"]
    validate_private_metrics(
        metrics,
        objective_lambda=int(config["objective"]["lambda"]),
    )
    expected = {
        "schema_version": 1,
        "round": round_number,
        "hint_id": hint_id,
        "train_position": TRAIN_POSITIONS[hint_id - 1],
        "train_qid": public["train_qid"],
        "problem": public["problem"],
        "answer": public["answer"],
        "hint": incumbent["hint"],
        "hint_hash": incumbent["hint_hash"],
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
    if packet != expected:
        differing = sorted(
            key
            for key in expected
            if packet.get(key) != expected[key]
        )
        raise ValueError(f"registered training packet drift: {differing}")
    return packet


def validate_registered_private_input(
    input_path: Path,
    *,
    task_id: str,
    mode: str,
    config: dict[str, Any],
    config_hash: str,
    research_root: Path | None = None,
) -> dict[str, Any]:
    """Bind every held-out query to one setup, baseline, or round artifact."""
    if mode not in {"private", "private-smoke"}:
        raise ValueError(f"mode {mode!r} is not a private evaluator mode")
    validate_task_identity(task_id, mode)
    root = RESEARCH_ROOT if research_root is None else research_root.resolve()
    request = load_json(input_path)
    require_exact_keys(request, PRIVATE_INPUT_KEYS, "private input")
    hint_id = request["hint_id"]
    if (
        request["schema_version"] != 1
        or request["config_hash"] != config_hash
        or isinstance(hint_id, bool)
        or not isinstance(hint_id, int)
        or hint_id not in range(1, 11)
        or request["hint_hash"] != hint_hash(request["hint"])
    ):
        raise ValueError("private input identity is invalid")

    if mode == "private-smoke":
        expected_path = (
            root / "setup" / "private_smoke_input.json"
        ).resolve()
        if input_path.resolve() != expected_path:
            raise ValueError(
                "private smoke does not use its registered setup path"
            )
        smoke = load_json(root / "setup" / "smoke_input.json")
        require_exact_keys(smoke, TRAIN_PACKET_KEYS, "smoke input")
        public_rows = load_json(root / "setup" / "train_public.json")
        if not isinstance(public_rows, list) or len(public_rows) != 10:
            raise ValueError("public training manifest is incomplete")
        public = public_rows[0]
        expected_smoke = {
            "schema_version": 1,
            "round": 0,
            "hint_id": 1,
            "train_position": TRAIN_POSITIONS[0],
            "train_qid": public["train_qid"],
            "problem": public["problem"],
            "answer": public["answer"],
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
            "hard_total_tokens":
                config["hint_limits"]["hard_total_tokens"],
        }
        if smoke != expected_smoke:
            raise ValueError("registered setup smoke packet is stale")
        expected = {
            "schema_version": 1,
            "hint_id": smoke["hint_id"],
            "hint": smoke["hint"],
            "hint_hash": smoke["hint_hash"],
            "config_hash": config_hash,
        }
    elif task_id.startswith("baseline-"):
        match = re.fullmatch(r"baseline-h(0[1-9]|10)", task_id)
        assert match is not None
        registered_hint_id = int(match.group(1))
        expected_path = (
            root / "baseline" / "inputs"
            / f"hint_{registered_hint_id:02d}.json"
        ).resolve()
        if input_path.resolve() != expected_path:
            raise ValueError(
                "baseline task does not use its registered input path"
            )
        book = load_json(root / "initial_book.json")
        if (
            book.get("config_hash") != config_hash
            or book.get("generation") != 0
            or len(book.get("hints", [])) != 10
        ):
            raise ValueError("initial hint book is stale")
        row = book["hints"][registered_hint_id - 1]
        expected = {
            "schema_version": 1,
            "hint_id": registered_hint_id,
            "hint": row["hint"],
            "hint_hash": row["hint_hash"],
            "config_hash": config_hash,
        }
    else:
        match = re.fullmatch(
            rf"r({ROUND_ID_PATTERN})-private-h(0[1-9]|10)",
            task_id,
        )
        assert match is not None
        round_number = int(match.group(1))
        registered_hint_id = int(match.group(2))
        if round_number > int(config["budget"]["max_rounds"]):
            raise ValueError("private task exceeds the round budget")
        expected_path = (
            root / "rounds" / f"{round_number:03d}"
            / "proposal_private_inputs"
            / f"hint_{registered_hint_id:02d}.json"
        ).resolve()
        if input_path.resolve() != expected_path:
            raise ValueError(
                "proposal evaluator does not use its registered input path"
            )
        proposals = load_json(
            root / "rounds" / f"{round_number:03d}"
            / "book_proposals.json"
        )
        rows = proposals.get("hints", [])
        if (
            proposals.get("schema_version") != 1
            or proposals.get("config_hash") != config_hash
            or proposals.get("round") != round_number
            or len(rows) != 10
            or [row.get("hint_id") for row in rows]
            != list(range(1, 11))
        ):
            raise ValueError("registered proposal book is stale")
        row = rows[registered_hint_id - 1]
        expected = {
            "schema_version": 1,
            "hint_id": registered_hint_id,
            "hint": row["hint"],
            "hint_hash": row["hint_hash"],
            "config_hash": config_hash,
        }

    if request != expected:
        raise ValueError("private input differs from its registered hint")
    return request


def proposal_wins(
    incumbent: dict[str, Any],
    proposal: dict[str, Any],
    incumbent_tokens: int,
    proposal_tokens: int,
) -> bool:
    old_j = int(incumbent["J_numerator"])
    new_j = int(proposal["J_numerator"])
    if new_j != old_j:
        return new_j > old_j
    old_heldout = int(incumbent["heldout_correct"])
    new_heldout = int(proposal["heldout_correct"])
    if new_heldout != old_heldout:
        return new_heldout > old_heldout
    return proposal_tokens < incumbent_tokens


def validate_pool_allocation(
    value: dict[str, Any],
    *,
    config: dict[str, Any],
    config_hash: str,
    require_current_source: bool = True,
) -> None:
    require_exact_keys(value, POOL_ALLOCATION_KEYS, "pool allocation")
    slurm = config["slurm"]
    expected = {
        "schema_version": 1,
        "partition": slurm["partition"],
        "nodes": slurm["nodes"],
        "pool_slots": slurm["pool_slots"],
        "exclusive": True,
        "gpus_per_node": slurm["gpus_per_node"],
        "gpus_per_task": 1,
        "gpu_binding": slurm["gpu_binding"],
        "repo_root": str(REPO_ROOT),
        "work_root": str(WORK_ROOT),
        "config_hash": config_hash,
    }
    if require_current_source:
        expected["source_hash"] = source_bundle_hash(RUNTIME_SOURCE_FILES)
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(f"pool allocation has stale {key}")
    source_hash = value["source_hash"]
    if (
        not isinstance(source_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
    ):
        raise ValueError("pool allocation has an invalid source hash")
    if not str(value["job_id"]).isdigit():
        raise ValueError("pool allocation has an invalid job ID")
    python = Path(value["python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("pool allocation has an invalid Python executable")
    submitted = value["submitted_epoch"]
    if (
        isinstance(submitted, bool)
        or not isinstance(submitted, (int, float))
        or submitted <= 0
    ):
        raise ValueError("pool allocation has an invalid submission time")


def load_pool_allocation(
    config: dict[str, Any],
    config_hash: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    active_path = WORK_ROOT / "pool" / "allocation.json"
    path = active_path
    if job_id is not None:
        if not job_id.isdigit():
            raise ValueError("pool allocation job ID must be numeric")
        if active_path.is_file():
            active = load_json(active_path)
            if str(active.get("job_id")) != job_id:
                path = WORK_ROOT / "pool" / f"allocation.{job_id}.json"
        else:
            path = WORK_ROOT / "pool" / f"allocation.{job_id}.json"
    if not path.is_file():
        raise RuntimeError(f"pool allocation receipt is missing: {path}")
    value = load_json(path)
    validate_pool_allocation(
        value,
        config=config,
        config_hash=config_hash,
    )
    return value


def validate_receipt(
    receipt: dict[str, Any],
    *,
    task_id: str,
    mode: str,
    input_path: Path,
    output_path: Path,
    config_hash: str,
    expected_node: str | None = None,
    expected_pool_slot: int | None = None,
) -> None:
    require_exact_keys(receipt, RECEIPT_KEYS, "task receipt")
    config, current_config_hash = load_config(require_frozen=False)
    if current_config_hash != config_hash:
        raise ValueError("receipt validation used a stale config hash")
    allocation_job_id = str(receipt["allocation_job_id"])
    if not allocation_job_id.isdigit():
        raise ValueError("receipt has an invalid allocation job ID")
    allocation = load_pool_allocation(
        config,
        config_hash,
        allocation_job_id,
    )
    expected = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": mode,
        "config_hash": config_hash,
        "source_hash": source_bundle_hash(RUNTIME_SOURCE_FILES),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path.resolve()),
        "gpu_count": 1,
        "slurm_gpus_per_task": 1,
        "exit_code": 0,
    }
    for key, value in expected.items():
        if receipt[key] != value:
            raise ValueError(f"task receipt has stale {key}")
    if receipt["allocation_job_id"] != allocation["job_id"]:
        raise ValueError("receipt does not belong to the active pool allocation")
    if not str(receipt["slurm_step_id"]).isdigit():
        raise ValueError("receipt has an invalid Slurm step ID")
    if receipt["node"] not in allocation["nodes"]:
        raise ValueError("receipt ran outside the allocated nodes")
    if (
        isinstance(receipt["pool_slot"], bool)
        or not isinstance(receipt["pool_slot"], int)
        or not 0 <= receipt["pool_slot"] < int(allocation["gpus_per_node"])
    ):
        raise ValueError("receipt has an invalid dispatcher pool slot")
    visible = receipt["visible_cuda_device"]
    if (
        not isinstance(visible, str)
        or not visible
        or visible != visible.strip()
        or "," in visible
    ):
        raise ValueError("receipt does not prove exactly one visible CUDA device")
    if expected_node is not None and receipt["node"] != expected_node:
        raise ValueError("receipt ran on the wrong dispatched node")
    if (
        expected_pool_slot is not None
        and receipt["pool_slot"] != expected_pool_slot
    ):
        raise ValueError("receipt used the wrong dispatcher pool slot")
    if receipt["execution_id"] != (
        f"{receipt['allocation_job_id']}.{receipt['slurm_step_id']}"
    ):
        raise ValueError("receipt execution ID does not match its Slurm step")
    if receipt["output_sha256"] != sha256_file(output_path):
        raise ValueError("receipt output hash mismatch")
    timing = [
        receipt["start_epoch"],
        receipt["end_epoch"],
        receipt["elapsed_seconds"],
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in timing
    ):
        raise ValueError("receipt timing is not finite numeric data")
    start, end, elapsed = timing
    if (
        end < start
        or elapsed < 0
        or not math.isclose(elapsed, end - start, rel_tol=1e-9, abs_tol=1e-6)
    ):
        raise ValueError("receipt timing is invalid")


def epoch() -> float:
    return time.time()
