"""Pure helper tests for Hugging Face publication planning and provenance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publish_hf.py"
SPEC = importlib.util.spec_from_file_location("publish_hf_test", SCRIPT)
publish_hf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(publish_hf)


class PublishPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.paths = {}
        for name in ("sft_dataset", "sft_metadata", "rft_dataset", "rft_metadata"):
            path = root / name
            path.write_text(name + "\n", encoding="utf-8")
            self.paths[name] = path
        for stage in ("sft", "rft"):
            model = root / f"{stage}_model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            marker = root / f"{stage}.marker"
            marker.write_text(f"{stage}-digest\n", encoding="utf-8")
            self.paths[f"{stage}_model"] = model
            self.paths[f"{stage}_model_marker"] = marker
        self.args = argparse.Namespace(
            **self.paths,
            source_commit="1" * 40,
            receipt=root / "receipt.json",
            namespace=None,
            private="1",
            repo_prefix="rlad-original",
            sft_dataset_repo=None,
            rft_dataset_repo=None,
            sft_model_repo=None,
            rft_model_repo=None,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_default_targets_and_stable_source_key(self) -> None:
        targets = publish_hf.resolve_targets(self.args, "alice")
        self.assertEqual(targets["rft_model"]["id"], "alice/rlad-original-absgen-rft-model")
        first = publish_hf.source_key(self.args, targets, True)
        second = publish_hf.source_key(self.args, targets, True)
        self.assertEqual(first, second)
        self.paths["rft_dataset"].write_text("changed\n", encoding="utf-8")
        self.assertNotEqual(first, publish_hf.source_key(self.args, targets, True))

    def test_model_validation_and_cards(self) -> None:
        files = publish_hf._validate_model(self.paths["sft_model"])
        self.assertIn("model.safetensors", files)
        card = publish_hf._model_card("sft", "abc", "alice/data")
        self.assertIn("alice/data", card)
        self.assertNotIn("HF_TOKEN", card)

    def test_end_to_end_publication_receipt_is_idempotent(self) -> None:
        class RepositoryNotFoundError(Exception):
            pass

        class FakeApi:
            def __init__(self):
                self.repos = {}
                self.uploads = 0

            def whoami(self):
                return {"name": "alice"}

            def repo_info(self, repo_id, repo_type, files_metadata=False):
                key = (repo_type, repo_id)
                if key not in self.repos:
                    raise RepositoryNotFoundError(repo_id)
                repo = self.repos[key]
                siblings = [types.SimpleNamespace(rfilename=name) for name in sorted(repo["files"])]
                return types.SimpleNamespace(private=repo["private"], sha="deadbeef", siblings=siblings)

            def create_repo(self, repo_id, repo_type, private, exist_ok):
                self.repos[(repo_type, repo_id)] = {"private": private, "files": set()}

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
                self.uploads += 1
                self.repos[(repo_type, repo_id)]["files"].add(path_in_repo)

            def upload_folder(self, folder_path, repo_id, repo_type, ignore_patterns, commit_message):
                self.uploads += 1
                root = Path(folder_path)
                self.repos[(repo_type, repo_id)]["files"].update(
                    item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
                )

        api = FakeApi()
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: api
        utils = types.ModuleType("huggingface_hub.utils")
        utils.RepositoryNotFoundError = RepositoryNotFoundError
        previous_hub = sys.modules.get("huggingface_hub")
        previous_utils = sys.modules.get("huggingface_hub.utils")
        sys.modules["huggingface_hub"] = hub
        sys.modules["huggingface_hub.utils"] = utils
        try:
            first = publish_hf.publish(self.args)
            uploads = api.uploads
            second = publish_hf.publish(self.args)
        finally:
            if previous_hub is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = previous_hub
            if previous_utils is None:
                sys.modules.pop("huggingface_hub.utils", None)
            else:
                sys.modules["huggingface_hub.utils"] = previous_utils
        self.assertEqual(first, second)
        self.assertEqual(api.uploads, uploads)
        self.assertEqual(len(api.repos), 4)
        saved = json.loads(self.args.receipt.read_text(encoding="utf-8"))
        self.assertTrue(saved["private"])
        self.assertIn("rft_model", saved["repositories"])


if __name__ == "__main__":
    unittest.main()
