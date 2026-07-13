#!/usr/bin/env python3
"""Publish the original RLAD abstraction-generator corpora and checkpoints.

Authentication is intentionally delegated to ``huggingface_hub`` so either
``HF_TOKEN`` or a prior ``hf auth login`` is used without exposing a token on
the command line. Re-running the command is safe: unchanged, complete remote
repositories are accepted from the local receipt, while interrupted folder
uploads are resumed by ``HfApi.upload_folder``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path


BASE_MODEL = "Qwen/Qwen3-1.7B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _model_files(path: Path) -> list[str]:
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.relative_to(path).parts:
            continue
        files.append(item.relative_to(path).as_posix())
    return files


def _validate_model(path: Path) -> list[str]:
    if not (path / "config.json").is_file():
        raise SystemExit(f"model config is missing: {path / 'config.json'}")
    files = _model_files(path)
    weights = [name for name in files if name.endswith((".safetensors", ".bin"))]
    if not weights:
        raise SystemExit(f"model weights are missing under {path}")
    return files


def _repo_id(explicit: str | None, namespace: str, default_name: str) -> str:
    repo_id = explicit or f"{namespace}/{default_name}"
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
        raise SystemExit(f"invalid Hugging Face repository id: {repo_id!r}")
    return repo_id


def resolve_targets(args: argparse.Namespace, namespace: str) -> dict[str, dict[str, str]]:
    prefix = args.repo_prefix.strip("-")
    if not prefix:
        raise SystemExit("--repo-prefix cannot be empty")
    return {
        "sft_dataset": {
            "id": _repo_id(args.sft_dataset_repo, namespace, f"{prefix}-absgen-sft-data"),
            "type": "dataset",
        },
        "rft_dataset": {
            "id": _repo_id(args.rft_dataset_repo, namespace, f"{prefix}-absgen-rft-data"),
            "type": "dataset",
        },
        "sft_model": {
            "id": _repo_id(args.sft_model_repo, namespace, f"{prefix}-absgen-sft-model"),
            "type": "model",
        },
        "rft_model": {
            "id": _repo_id(args.rft_model_repo, namespace, f"{prefix}-absgen-rft-model"),
            "type": "model",
        },
    }


def source_key(args: argparse.Namespace, targets: dict[str, dict[str, str]], private: bool) -> str:
    inputs = {
        "schema": 1,
        "private": private,
        "targets": targets,
        "source_commit": args.source_commit,
        "sft_dataset_sha256": _sha256(args.sft_dataset),
        "sft_metadata_sha256": _sha256(args.sft_metadata),
        "rft_dataset_sha256": _sha256(args.rft_dataset),
        "rft_metadata_sha256": _sha256(args.rft_metadata),
        "sft_model_marker": args.sft_model_marker.read_text(encoding="utf-8").strip(),
        "rft_model_marker": args.rft_model_marker.read_text(encoding="utf-8").strip(),
    }
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _dataset_card(stage: str, source_commit: str, rows_file: str) -> str:
    title = f"RLAD abstraction-generator {stage.upper()} corpus"
    return f"""---
pretty_name: {title}
task_categories:
- text-generation
language:
- en
---

# {title}

This repository contains the `{stage}` training corpus produced by the original
RLAD abstraction-generator pipeline. The examples are stored in `{rows_file}`
with user/assistant messages; `metadata.json` records generation or rejection
statistics.

- Base model: `{BASE_MODEL}`
- RLAD source commit: `{source_commit}`
- Source data: `agentica-org/DeepScaleR-Preview-Dataset`

Review the source dataset and base-model licenses before redistribution or use.
"""


def _model_card(stage: str, source_commit: str, dataset_repo: str) -> str:
    title = f"RLAD {stage.upper()} abstraction generator"
    return f"""---
base_model: {BASE_MODEL}
library_name: transformers
pipeline_tag: text-generation
datasets:
- {dataset_repo}
tags:
- rlad
- abstraction-generator
- math
---

# {title}

This is the offline-converted `{stage}` checkpoint from the original RLAD
abstraction-generator reproduction.

- Base model: `{BASE_MODEL}`
- Training corpus: `{dataset_repo}`
- RLAD source commit: `{source_commit}`

The model proposes concise non-answer-revealing math hints/cheatsheets. Review
the base-model and training-data licenses before redistribution or deployment.
"""


def _remote_files(api, target: dict[str, str]) -> tuple[object, set[str]]:
    info = api.repo_info(repo_id=target["id"], repo_type=target["type"], files_metadata=False)
    return info, {item.rfilename for item in (info.siblings or [])}


def _ensure_repo(api, target: dict[str, str], private: bool) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info, _ = _remote_files(api, target)
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=target["id"], repo_type=target["type"], private=private, exist_ok=False
        )
        info, _ = _remote_files(api, target)
    if bool(info.private) != private:
        visibility = "private" if info.private else "public"
        wanted = "private" if private else "public"
        raise SystemExit(
            f"{target['type']} repo {target['id']} already exists as {visibility}; "
            f"refusing to change it to {wanted}. Choose a different repo id or change visibility explicitly."
        )


def _receipt_is_current(api, receipt: Path, key: str, required: dict[str, set[str]]) -> bool:
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if saved.get("source_key") != key:
        return False
    repositories = saved.get("repositories", {})
    for name, needed in required.items():
        target = repositories.get(name)
        if not isinstance(target, dict):
            return False
        try:
            info, remote = _remote_files(api, target)
        except Exception:
            return False
        if (
            bool(info.private) != bool(saved.get("private"))
            or info.sha != target.get("revision")
            or not needed.issubset(remote)
        ):
            return False
    print(f"Hugging Face publication already complete: {receipt}")
    return True


def _upload_bytes(api, target: dict[str, str], path: str, content: str, message: str) -> None:
    api.upload_file(
        path_or_fileobj=io.BytesIO(content.encode("utf-8")),
        path_in_repo=path,
        repo_id=target["id"],
        repo_type=target["type"],
        commit_message=message,
    )


def _upload_dataset(api, target: dict[str, str], data: Path, metadata: Path, card: str) -> None:
    api.upload_file(
        path_or_fileobj=str(data), path_in_repo="train.jsonl", repo_id=target["id"],
        repo_type="dataset", commit_message="Upload RLAD training corpus"
    )
    api.upload_file(
        path_or_fileobj=str(metadata), path_in_repo="metadata.json", repo_id=target["id"],
        repo_type="dataset", commit_message="Upload RLAD corpus metadata"
    )
    _upload_bytes(api, target, "README.md", card, "Add dataset card")


def _upload_model(api, target: dict[str, str], model_dir: Path, card: str) -> None:
    api.upload_folder(
        folder_path=str(model_dir), repo_id=target["id"], repo_type="model",
        ignore_patterns=[".cache/**", "**/*.lock"], commit_message="Upload RLAD HF checkpoint"
    )
    _upload_bytes(api, target, "README.md", card, "Add model card")


def publish(args: argparse.Namespace) -> dict:
    from huggingface_hub import HfApi

    for path in (
        args.sft_dataset, args.sft_metadata, args.rft_dataset, args.rft_metadata,
        args.sft_model_marker, args.rft_model_marker,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"required publication input is missing or empty: {path}")
    sft_model_files = _validate_model(args.sft_model)
    rft_model_files = _validate_model(args.rft_model)

    api = HfApi()
    identity = api.whoami()
    namespace = args.namespace or identity.get("name")
    if not namespace:
        raise SystemExit("could not infer a Hugging Face namespace; set HF_NAMESPACE")
    private = args.private == "1"
    targets = resolve_targets(args, namespace)
    key = source_key(args, targets, private)
    required = {
        "sft_dataset": {"README.md", "train.jsonl", "metadata.json"},
        "rft_dataset": {"README.md", "train.jsonl", "metadata.json"},
        "sft_model": {"README.md", *sft_model_files},
        "rft_model": {"README.md", *rft_model_files},
    }
    if _receipt_is_current(api, args.receipt, key, required):
        return json.loads(args.receipt.read_text(encoding="utf-8"))

    for target in targets.values():
        _ensure_repo(api, target, private)

    _upload_dataset(
        api, targets["sft_dataset"], args.sft_dataset, args.sft_metadata,
        _dataset_card("sft", args.source_commit, "train.jsonl"),
    )
    _upload_dataset(
        api, targets["rft_dataset"], args.rft_dataset, args.rft_metadata,
        _dataset_card("rft", args.source_commit, "train.jsonl"),
    )
    _upload_model(
        api, targets["sft_model"], args.sft_model,
        _model_card("sft", args.source_commit, targets["sft_dataset"]["id"]),
    )
    _upload_model(
        api, targets["rft_model"], args.rft_model,
        _model_card("rft", args.source_commit, targets["rft_dataset"]["id"]),
    )

    repositories = {}
    for name, target in targets.items():
        info, remote = _remote_files(api, target)
        if not required[name].issubset(remote):
            missing = sorted(required[name] - remote)
            raise SystemExit(f"upload verification failed for {target['id']}; missing: {missing}")
        prefix = "datasets/" if target["type"] == "dataset" else ""
        repositories[name] = {
            **target,
            "revision": info.sha,
            "url": f"https://huggingface.co/{prefix}{target['id']}",
        }
    receipt = {
        "schema": 1,
        "source_key": key,
        "source_commit": args.source_commit,
        "private": private,
        "repositories": repositories,
    }
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-dataset", type=Path, required=True)
    ap.add_argument("--sft-metadata", type=Path, required=True)
    ap.add_argument("--rft-dataset", type=Path, required=True)
    ap.add_argument("--rft-metadata", type=Path, required=True)
    ap.add_argument("--sft-model", type=Path, required=True)
    ap.add_argument("--rft-model", type=Path, required=True)
    ap.add_argument("--sft-model-marker", type=Path, required=True)
    ap.add_argument("--rft-model-marker", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE"))
    ap.add_argument("--repo-prefix", default=os.environ.get("HF_REPO_PREFIX", "rlad-original"))
    ap.add_argument("--private", choices=("0", "1"), default=os.environ.get("HF_REPO_PRIVATE", "1"))
    ap.add_argument("--sft-dataset-repo", default=os.environ.get("HF_SFT_DATASET_REPO"))
    ap.add_argument("--rft-dataset-repo", default=os.environ.get("HF_RFT_DATASET_REPO"))
    ap.add_argument("--sft-model-repo", default=os.environ.get("HF_SFT_MODEL_REPO"))
    ap.add_argument("--rft-model-repo", default=os.environ.get("HF_RFT_MODEL_REPO"))
    return ap


if __name__ == "__main__":
    publish(parser().parse_args())
