"""Commit and push one completed round's held-out-safe public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .core import (
    REPO_ROOT,
    RESEARCH_ROOT,
    WORK_ROOT,
    atomic_write_bytes,
    atomic_write_json,
    load_config,
    load_json,
    require_exact_keys,
    sha256_file,
)


FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "answers",
    "heldout_answer",
    "heldout_answers",
    "heldout_problem",
    "heldout_problems",
    "per_question",
    "problem",
    "problems",
    "response",
    "responses",
    "rollout",
    "rollouts",
}

PUBLICATION_RECEIPT_KEYS = {
    "schema_version",
    "round",
    "config_hash",
    "commit",
    "remote",
    "branch",
    "paths",
    "pushed_epoch",
}

PUBLICATION_TRANSACTION_KEYS = {
    "schema_version",
    "message",
    "remote",
    "branch",
    "head_before",
    "paths",
    "artifact_sha256",
    "created_epoch",
}


def _assert_public_shape(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = set(value) & FORBIDDEN_PUBLIC_KEYS
        if forbidden:
            raise RuntimeError(
                f"public artifact {path} contains forbidden keys: "
                f"{sorted(forbidden)}"
            )
        for key, child in value.items():
            _assert_public_shape(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_shape(child, f"{path}[{index}]")


def _git(*args: str, capture: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=capture,
        text=True,
    )
    return completed.stdout.strip() if capture else ""


def _validate_public_paths(paths: list[Path]) -> list[Path]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"publication artifacts missing: {missing}")
    for path in paths:
        if not path.resolve().is_relative_to(REPO_ROOT):
            raise RuntimeError(f"publication path escapes repository: {path}")
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink publication path: {path}")
        if path.suffix == ".json":
            _assert_public_shape(
                json.loads(path.read_text(encoding="utf-8")),
                str(path.relative_to(REPO_ROOT)),
            )
    return paths


def public_round_paths(round_number: int) -> list[Path]:
    round_dir = RESEARCH_ROOT / "rounds" / f"{round_number:03d}"
    paths = [
        RESEARCH_ROOT / "initial_book.json",
        RESEARCH_ROOT / "current_book.json",
        RESEARCH_ROOT / "best_per_hint.json",
        RESEARCH_ROOT / "round_summary.csv",
        RESEARCH_ROOT / "hint_history.csv",
        round_dir / "book_before.json",
        round_dir / "book_proposals.json",
        round_dir / "book_after.json",
        round_dir / "metrics.json",
    ]
    for final_name in ("final_book.json", "STOPPED.json"):
        final_path = RESEARCH_ROOT / final_name
        if final_path.exists():
            paths.append(final_path)
    current = load_json(RESEARCH_ROOT / "current_book.json")
    metrics = load_json(round_dir / "metrics.json")
    if (
        current.get("generation") != round_number
        or metrics.get("round") != round_number
    ):
        raise RuntimeError("round publication does not match current generation")
    return _validate_public_paths(paths)


def public_terminal_paths() -> list[Path]:
    paths = [
        RESEARCH_ROOT / "initial_book.json",
        RESEARCH_ROOT / "current_book.json",
        RESEARCH_ROOT / "final_book.json",
        RESEARCH_ROOT / "STOPPED.json",
    ]
    for optional in (
        "best_per_hint.json",
        "round_summary.csv",
        "hint_history.csv",
    ):
        path = RESEARCH_ROOT / optional
        if path.is_file():
            paths.append(path)
    return _validate_public_paths(paths)


def _changed_paths() -> set[str]:
    tracked = set(
        line
        for line in _git("diff", "--name-only").splitlines()
        if line
    )
    untracked = set(
        line
        for line in _git(
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if line
    )
    return tracked | untracked


def _staged_paths() -> set[str]:
    return {
        line
        for line in _git("diff", "--cached", "--name-only").splitlines()
        if line
    }


def _transaction_path() -> Path:
    return WORK_ROOT / "publication" / "git_transaction.json"


def _artifact_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in paths
    }


def _validate_publication_transaction(
    value: dict[str, Any],
    *,
    message: str,
    remote: str,
    branch: str,
    relative_paths: list[str],
    artifact_sha256: dict[str, str],
) -> None:
    require_exact_keys(
        value,
        PUBLICATION_TRANSACTION_KEYS,
        "publication transaction",
    )
    head_before = value["head_before"]
    if (
        value["schema_version"] != 1
        or value["message"] != message
        or value["remote"] != remote
        or value["branch"] != branch
        or value["paths"] != relative_paths
        or value["artifact_sha256"] != artifact_sha256
        or not isinstance(head_before, str)
        or len(head_before) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in head_before)
        or isinstance(value["created_epoch"], bool)
        or not isinstance(value["created_epoch"], (int, float))
        or value["created_epoch"] <= 0
    ):
        raise RuntimeError("publication transaction is stale or invalid")


def _commit_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _validate_checkpoint_commit(
    commit: str,
    transaction: dict[str, Any],
) -> None:
    if _git("log", "-1", "--format=%s", commit) != transaction["message"]:
        raise RuntimeError("checkpoint commit subject does not match transaction")
    ancestry = _git("rev-list", "--parents", "-n", "1", commit).split()
    if (
        len(ancestry) != 2
        or ancestry[0] != commit
        or ancestry[1] != transaction["head_before"]
    ):
        raise RuntimeError("checkpoint commit does not directly extend transaction")
    changed = {
        line
        for line in _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).splitlines()
        if line
    }
    allowed = set(transaction["paths"])
    if not changed or not changed.issubset(allowed):
        raise RuntimeError(
            f"checkpoint commit changed disallowed paths: {changed - allowed}"
        )
    for relative_path, expected in transaction["artifact_sha256"].items():
        if _commit_blob_sha256(commit, relative_path) != expected:
            raise RuntimeError(
                f"checkpoint commit has stale artifact: {relative_path}"
            )


def _finish_publication_transaction(
    commit: str,
    *,
    allow_mismatch: bool = False,
) -> None:
    path = _transaction_path()
    if not path.exists():
        return
    transaction = load_json(path)
    require_exact_keys(
        transaction,
        PUBLICATION_TRANSACTION_KEYS,
        "publication transaction",
    )
    if (
        _git("branch", "--show-current") != transaction["branch"]
        or _git("rev-parse", "HEAD") != commit
    ):
        if allow_mismatch:
            return
        raise RuntimeError(
            "cannot finish a publication transaction from another HEAD"
        )
    _validate_checkpoint_commit(commit, transaction)
    path.unlink()


def _remote_host(remote_url: str) -> str | None:
    if "://" in remote_url:
        return urlsplit(remote_url).hostname
    if ":" not in remote_url or remote_url.startswith(("/", "./", "../")):
        return None
    authority = remote_url.split(":", 1)[0]
    return authority.rsplit("@", 1)[-1] or None


def _push(commit: str, remote: str, branch: str) -> None:
    push_urls = [
        line.strip()
        for line in _git(
            "remote",
            "get-url",
            "--push",
            "--all",
            remote,
        ).splitlines()
        if line.strip()
    ]
    invalid = [
        url
        for url in push_urls
        if (_remote_host(url) or "").casefold() != "github.com"
    ]
    if not push_urls or invalid:
        raise RuntimeError(
            f"{remote} has non-GitHub effective push URLs: {invalid}"
        )
    if shutil.which("gh") is None:
        raise RuntimeError("gh is required for authenticated round publication")
    subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "push", "-u", remote, f"{commit}:refs/heads/{branch}"],
        cwd=REPO_ROOT,
        check=True,
    )


def _validate_existing_receipt(
    receipt: dict[str, Any],
    *,
    round_number: int,
    config_hash: str,
    remote: str,
) -> None:
    require_exact_keys(
        receipt,
        PUBLICATION_RECEIPT_KEYS,
        "publication receipt",
    )
    commit = receipt["commit"]
    if (
        receipt["schema_version"] != 1
        or receipt["round"] != round_number
        or receipt["config_hash"] != config_hash
        or receipt["remote"] != remote
        or not isinstance(receipt["branch"], str)
        or not receipt["branch"]
        or not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(receipt["paths"], list)
        or any(not isinstance(path, str) for path in receipt["paths"])
        or isinstance(receipt["pushed_epoch"], bool)
        or not isinstance(receipt["pushed_epoch"], (int, float))
        or receipt["pushed_epoch"] <= 0
    ):
        raise RuntimeError("stored publication receipt is invalid")
    required_path = (
        f"work_zsw/research/rounds/{round_number:03d}/metrics.json"
        if round_number > 0
        else "work_zsw/research/final_book.json"
    )
    if required_path not in receipt["paths"]:
        raise RuntimeError("publication receipt omits its required result")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
    )
    current_branch = _git("branch", "--show-current")
    if current_branch != receipt["branch"]:
        raise RuntimeError("publication receipt belongs to another branch")
    subprocess.run(
        [
            "git",
            "fetch",
            "--quiet",
            remote,
            f"refs/heads/{receipt['branch']}",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "FETCH_HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )


def _commit_and_push(
    paths: list[Path],
    *,
    message: str,
    remote: str,
) -> tuple[str, str, list[str]]:
    allowed = {str(path.relative_to(REPO_ROOT)) for path in paths}
    relative_paths = sorted(allowed)
    branch = _git("branch", "--show-current")
    if not branch:
        raise RuntimeError("round publication requires a named git branch")
    expected_branch = os.environ.get("HINTER_GIT_BRANCH")
    if expected_branch and branch != expected_branch:
        raise RuntimeError(
            f"current branch {branch!r} != HINTER_GIT_BRANCH "
            f"{expected_branch!r}"
        )
    artifact_sha256 = _artifact_hashes(paths)
    transaction_path = _transaction_path()
    if transaction_path.exists():
        transaction = load_json(transaction_path)
        _validate_publication_transaction(
            transaction,
            message=message,
            remote=remote,
            branch=branch,
            relative_paths=relative_paths,
            artifact_sha256=artifact_sha256,
        )
    else:
        if _staged_paths():
            raise RuntimeError(
                "refusing publication with an unregistered staged index"
            )
        unexpected = sorted(_changed_paths() - allowed)
        if unexpected:
            raise RuntimeError(
                "refusing to publish with unrelated visible changes: "
                f"{unexpected}"
            )
        transaction = {
            "schema_version": 1,
            "message": message,
            "remote": remote,
            "branch": branch,
            "head_before": _git("rev-parse", "HEAD"),
            "paths": relative_paths,
            "artifact_sha256": artifact_sha256,
            "created_epoch": time.time(),
        }
        atomic_write_json(transaction_path, transaction)

    staged_before = _staged_paths()
    unexpected_staged = staged_before - allowed
    if unexpected_staged:
        raise RuntimeError(
            "publication transaction found disallowed staged paths: "
            f"{sorted(unexpected_staged)}"
        )
    unexpected = sorted(_changed_paths() - allowed)
    if unexpected:
        raise RuntimeError(
            "refusing to publish with unrelated visible changes: "
            f"{unexpected}"
        )
    if _artifact_hashes(paths) != transaction["artifact_sha256"]:
        raise RuntimeError("publication artifacts changed during transaction")

    head = _git("rev-parse", "HEAD")
    try:
        if head == transaction["head_before"]:
            subprocess.run(
                ["git", "add", "-f", "--", *relative_paths],
                cwd=REPO_ROOT,
                check=True,
            )
            staged = _staged_paths()
            if not staged.issubset(allowed):
                raise RuntimeError(
                    f"publication staged disallowed paths: {staged - allowed}"
                )
            if not staged:
                raise RuntimeError(
                    "publication transaction has no artifact changes to commit"
                )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=REPO_ROOT,
                check=True,
            )
            commit = _git("rev-parse", "HEAD")
        else:
            if staged_before:
                raise RuntimeError(
                    "checkpoint commit exists but the index is still staged"
                )
            commit = head
    except Exception:
        subprocess.run(
            ["git", "restore", "--staged", "--", *relative_paths],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise

    _validate_checkpoint_commit(commit, transaction)
    _push(commit, remote, branch)
    return commit, branch, relative_paths


def _request_pool_drain() -> None:
    path = WORK_ROOT / "pool" / "control" / "STOP"
    if not path.exists():
        atomic_write_bytes(
            path,
            b"terminal checkpoint pushed; stop after active steps finish\n",
        )


def _publish_round_zero_terminal(
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    remote = config["publication"]["remote"]
    terminal_path = WORK_ROOT / "publication" / "terminal.json"
    if terminal_path.exists():
        terminal = load_json(terminal_path)
        _validate_existing_receipt(
            terminal,
            round_number=0,
            config_hash=config_hash,
            remote=remote,
        )
        _finish_publication_transaction(terminal["commit"])
        _request_pool_drain()
        return {**terminal, "reused": True}
    stopped = load_json(RESEARCH_ROOT / "STOPPED.json")
    current = load_json(RESEARCH_ROOT / "current_book.json")
    if (
        stopped.get("should_stop") is not True
        or stopped.get("last_round") != 0
        or current.get("generation") != 0
    ):
        raise RuntimeError("round-zero terminal state is inconsistent")
    paths = public_terminal_paths()
    message = f"{config['publication']['commit_prefix']} 000 terminal"
    commit, branch, relative_paths = _commit_and_push(
        paths,
        message=message,
        remote=remote,
    )
    terminal = {
        "schema_version": 1,
        "round": 0,
        "config_hash": config_hash,
        "commit": commit,
        "remote": remote,
        "branch": branch,
        "paths": relative_paths,
        "pushed_epoch": time.time(),
    }
    atomic_write_json(terminal_path, terminal)
    _finish_publication_transaction(commit)
    _request_pool_drain()
    return {**terminal, "reused": False}


def publish_round(round_number: int) -> dict[str, Any]:
    config, config_hash = load_config(require_frozen=True)
    if round_number < 0:
        raise ValueError("round number must not be negative")
    if round_number == 0:
        return _publish_round_zero_terminal(config, config_hash)
    receipt_path = (
        WORK_ROOT / "publication" / f"round_{round_number:03d}.json"
    )
    remote = config["publication"]["remote"]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_existing_receipt(
            receipt,
            round_number=round_number,
            config_hash=config_hash,
            remote=remote,
        )
        _finish_publication_transaction(
            receipt["commit"],
            allow_mismatch=True,
        )
        final_paths = [
            RESEARCH_ROOT / "final_book.json",
            RESEARCH_ROOT / "STOPPED.json",
        ]
        present = [path.is_file() for path in final_paths]
        if any(present) and not all(present):
            raise RuntimeError("terminal checkpoint artifacts are incomplete")
        final_relatives = {
            str(path.relative_to(REPO_ROOT))
            for path in final_paths
        }
        if all(present) and not final_relatives.issubset(receipt["paths"]):
            terminal_path = WORK_ROOT / "publication" / "terminal.json"
            if terminal_path.exists():
                terminal = load_json(terminal_path)
                _validate_existing_receipt(
                    terminal,
                    round_number=round_number,
                    config_hash=config_hash,
                    remote=remote,
                )
                _finish_publication_transaction(terminal["commit"])
                _request_pool_drain()
                return {**terminal, "reused": True}
            paths = public_round_paths(round_number)
            message = (
                f"{config['publication']['commit_prefix']} "
                f"{round_number:03d} terminal"
            )
            commit, branch, relative_paths = _commit_and_push(
                paths,
                message=message,
                remote=remote,
            )
            terminal = {
                "schema_version": 1,
                "round": round_number,
                "config_hash": config_hash,
                "commit": commit,
                "remote": remote,
                "branch": branch,
                "paths": relative_paths,
                "pushed_epoch": time.time(),
            }
            atomic_write_json(terminal_path, terminal)
            _finish_publication_transaction(commit)
            _request_pool_drain()
            return {**terminal, "reused": False}
        if all(present):
            _request_pool_drain()
        return {**receipt, "reused": True}

    paths = public_round_paths(round_number)
    message = (
        f"{config['publication']['commit_prefix']} {round_number:03d}"
    )
    commit, branch, relative_paths = _commit_and_push(
        paths,
        message=message,
        remote=remote,
    )
    receipt = {
        "schema_version": 1,
        "round": round_number,
        "config_hash": config_hash,
        "commit": commit,
        "remote": remote,
        "branch": branch,
        "paths": relative_paths,
        "pushed_epoch": time.time(),
    }
    atomic_write_json(receipt_path, receipt)
    _finish_publication_transaction(commit)
    if all(
        (RESEARCH_ROOT / name).is_file()
        for name in ("final_book.json", "STOPPED.json")
    ):
        _request_pool_drain()
    return {**receipt, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish_round(args.round),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
