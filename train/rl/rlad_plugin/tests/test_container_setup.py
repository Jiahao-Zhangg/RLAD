"""CPU-only tests for automatic Pyxis training-container preparation."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RL_ROOT.parents[1]
PREP_JOB = RL_ROOT / "jobs" / "prepare_container.sbatch"
ROOT_PIPELINE = REPO_ROOT / "RFT_pipeline.sh"
SOURCE = "docker.io#radixark/miles:dev-cu12-202606172131"


class ContainerSetupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.miles = self.root / "miles"
        (self.miles / ".git").mkdir(parents=True)
        self.out = self.root / "images" / "miles.sqsh"
        self.args_file = self.root / "srun.args"
        self.fake_srun = self.bin_dir / "srun"
        self.fake_srun.write_text(
            """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$@" >> "$FAKE_SRUN_ARGS"
save=''
for arg in "$@"; do
  case "$arg" in --container-save=*) save=${arg#--container-save=} ;; esac
done
test -n "$save"
mkdir -p "$(dirname "$save")"
printf 'partial-squashfs' > "$save"
test "${FAKE_SRUN_FAIL:-0}" = 0
""",
            encoding="utf-8",
        )
        self.fake_srun.chmod(0o755)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "RLAD_HOME": str(RL_ROOT),
            "MILES_DIR": str(self.miles),
            "RLAD_CONTAINER": str(self.out),
            "RLAD_CONTAINER_SOURCE": SOURCE,
            "RLAD_CONTAINER_MOUNTS": "",
            "RLAD_CONTAINER_MOUNT_HOME": "0",
            "RLAD_LOGS": str(self.root / "logs"),
            "SLURM_JOB_ID": "123",
            "FAKE_SRUN_ARGS": str(self.args_file),
        }

    def test_prepare_job_saves_receipted_image_and_is_idempotent(self) -> None:
        subprocess.run(
            ["bash", str(PREP_JOB)], env=self.environment(), check=True,
            text=True, capture_output=True,
        )
        self.assertEqual(self.out.read_text(encoding="utf-8"), "partial-squashfs")
        receipt = Path(f"{self.out}.source").read_text(encoding="utf-8")
        self.assertIn("schema=1\n", receipt)
        self.assertIn(f"source={SOURCE}\n", receipt)
        self.assertIn(f"bytes={self.out.stat().st_size}\n", receipt)
        arguments = self.args_file.read_text(encoding="utf-8").splitlines()
        self.assertIn(f"--container-image={SOURCE}", arguments)
        self.assertIn(f"--container-save={self.out}.partial.123", arguments)
        self.assertIn("--gpus-per-task=1", arguments)
        self.assertIn(str(RL_ROOT / "scripts" / "verify_training_container.py"), arguments)

        first_arguments = self.args_file.read_text(encoding="utf-8")
        subprocess.run(
            ["bash", str(PREP_JOB)], env=self.environment(), check=True,
            text=True, capture_output=True,
        )
        self.assertEqual(self.args_file.read_text(encoding="utf-8"), first_arguments)

    def test_failed_import_never_publishes_partial_image(self) -> None:
        env = {**self.environment(), "FAKE_SRUN_FAIL": "1"}
        completed = subprocess.run(
            ["bash", str(PREP_JOB)], env=env, check=False,
            text=True, capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(self.out.exists())
        self.assertFalse(Path(f"{self.out}.source").exists())
        self.assertEqual(list(self.out.parent.glob("miles.sqsh.partial.*")), [])

    def test_controller_validates_pipeline_receipt(self) -> None:
        self.out.parent.mkdir(parents=True)
        self.out.write_bytes(b"image")
        receipt = Path(f"{self.out}.source")
        receipt.write_text(
            f"schema=1\nsource={SOURCE}\nbytes={self.out.stat().st_size}\n",
            encoding="utf-8",
        )
        command = (
            'source "$1"; RLAD_CONTAINER="$2"; RLAD_CONTAINER_SOURCE="$3"; '
            "training_container_ready"
        )
        valid = subprocess.run(
            ["bash", "-c", command, "receipt-test", str(ROOT_PIPELINE),
             str(self.out), SOURCE],
            check=False, text=True, capture_output=True,
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        invalid = subprocess.run(
            ["bash", "-c", command, "receipt-test", str(ROOT_PIPELINE),
             str(self.out), "docker.io#example/other:tag"],
            check=False, text=True, capture_output=True,
        )
        self.assertNotEqual(invalid.returncode, 0)

    def test_setup_and_run_prepare_container_before_doctor(self) -> None:
        text = ROOT_PIPELINE.read_text(encoding="utf-8")
        for command in ("setup)", "run|resume)"):
            block = text.split(command, 1)[1].split(";;", 1)[0]
            self.assertLess(block.index("bootstrap_host"), block.index("prepare_training_container"))
            self.assertLess(block.index("prepare_training_container"), block.index("doctor"))


if __name__ == "__main__":
    unittest.main()
