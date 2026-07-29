"""Pinned `radixark/miles` DeepScaleR grader verification and scoring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import WORK_ROOT, sha256_file


def grader_checkout() -> Path:
    return (WORK_ROOT / "deps" / "miles").resolve()


def install_grader_import_path() -> Path:
    checkout = grader_checkout()
    value = str(checkout)
    if value not in sys.path:
        sys.path.insert(0, value)
    return checkout


def verify_grader(config: dict[str, Any]) -> None:
    checkout = install_grader_import_path()
    grader = config["grader"]
    if not (checkout / ".git").exists():
        raise RuntimeError(f"pinned miles checkout is missing: {checkout}")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != grader["revision"]:
        raise RuntimeError(f"miles revision drift: {head}")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("miles grader checkout is not clean")
    expected = {
        checkout / "miles/rollout/rm_hub/deepscaler.py":
            grader["deepscaler_sha256"],
        checkout / "miles/rollout/rm_hub/math_utils.py":
            grader["math_utils_sha256"],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"grader source drift: {path}")

    import miles.rollout.rm_hub.deepscaler as module

    expected_module = (
        checkout / "miles/rollout/rm_hub/deepscaler.py"
    ).resolve()
    if Path(module.__file__).resolve() != expected_module:
        raise RuntimeError("DeepScaleR grader imported from the wrong checkout")


def grade_response(response: str, answer: str) -> int:
    from miles.rollout.rm_hub.deepscaler import (
        get_deepscaler_rule_based_reward,
    )

    try:
        reward = int(get_deepscaler_rule_based_reward(response, answer))
    except Exception:
        reward = 0
    if reward not in (0, 1):
        raise RuntimeError(f"DeepScaleR grader returned {reward}, not binary")
    return reward

