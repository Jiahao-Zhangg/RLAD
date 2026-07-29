"""Pinned dataset access with strict public/private materialization boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import (
    HELDOUT_POSITIONS,
    RESEARCH_ROOT,
    TRAIN_POSITIONS,
    atomic_write_json,
    load_json,
    sha256_file,
)


def validate_split(config: dict[str, Any]) -> None:
    train = tuple(config["dataset"]["train_positions"])
    heldout = tuple(config["dataset"]["heldout_positions"])
    if train != TRAIN_POSITIONS or heldout != HELDOUT_POSITIONS:
        raise RuntimeError("fixed train/held-out split changed")
    if len(train) != 10 or len(heldout) != 10:
        raise RuntimeError("split must have ten train and ten held-out rows")
    if set(train) & set(heldout):
        raise RuntimeError("train and held-out rows overlap")
    row_count = int(config["dataset"]["row_count"])
    if min(train + heldout) < 0 or max(train + heldout) >= row_count:
        raise RuntimeError("split position is outside the pinned dataset")


def load_pinned_dataset(config: dict[str, Any]) -> Any:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    validate_split(config)
    dataset_spec = config["dataset"]
    dataset = load_dataset(
        dataset_spec["repo_id"],
        revision=dataset_spec["revision"],
        split="train",
    )
    if len(dataset) != int(dataset_spec["row_count"]):
        raise RuntimeError("dataset row-count drift")
    if dataset._fingerprint != dataset_spec["fingerprint"]:
        raise RuntimeError(
            f"dataset fingerprint drift: {dataset._fingerprint} != "
            f"{dataset_spec['fingerprint']}"
        )
    parquet = Path(
        hf_hub_download(
            dataset_spec["repo_id"],
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=dataset_spec["revision"],
        )
    )
    digest = sha256_file(parquet)
    if digest != dataset_spec["parquet_sha256"]:
        raise RuntimeError(
            f"dataset parquet drift: {digest} != "
            f"{dataset_spec['parquet_sha256']}"
        )
    return dataset


def public_training_rows(
    config: dict[str, Any],
    config_hash: str,
) -> list[dict[str, Any]]:
    dataset = load_pinned_dataset(config)
    rows = []
    for hint_id, position in enumerate(TRAIN_POSITIONS, start=1):
        row = dataset[position]
        rows.append(
            {
                "hint_id": hint_id,
                "train_position": position,
                "train_qid": str(row["idx"]),
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
            }
        )
    setup = RESEARCH_ROOT / "setup"
    output = setup / "train_public.json"
    manifest_path = setup / "data_manifest.json"
    manifest = {
        "schema_version": 1,
        "config_hash": config_hash,
        "dataset_repo_id": config["dataset"]["repo_id"],
        "dataset_revision": config["dataset"]["revision"],
        "dataset_fingerprint": config["dataset"]["fingerprint"],
        "dataset_parquet_sha256": config["dataset"]["parquet_sha256"],
        "row_count": len(dataset),
        "train_positions": list(TRAIN_POSITIONS),
        "heldout_positions": list(HELDOUT_POSITIONS),
        "heldout_materialized": False,
    }
    if output.exists() and load_json(output) != rows:
        raise RuntimeError("public training manifest drift")
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise RuntimeError("dataset setup manifest drift")
    if not output.exists():
        atomic_write_json(output, rows)
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    return rows
