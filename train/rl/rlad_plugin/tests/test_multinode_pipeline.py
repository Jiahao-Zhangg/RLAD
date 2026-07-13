"""CPU-only tests for two-node shard launching and warm-start merging."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[2]
WARMSTART_PATH = RL_ROOT / "rlad_plugin" / "warmstart_gen.py"
SHARD_LAUNCHER = RL_ROOT / "jobs" / "run_gpu_shards.sh"
ROOT_PIPELINE = RL_ROOT.parents[1] / "RFT_pipeline.sh"
SPEC = importlib.util.spec_from_file_location("warmstart_gen_test", WARMSTART_PATH)
warmstart_gen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(warmstart_gen)


class MultiNodePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def row(qid: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"problem-{qid}"},
                {"role": "assistant", "content": f"abstraction-{qid}"},
            ],
            "metadata": {"qid": qid, "generator": "teacher"},
        }

    def test_warmstart_shards_merge_in_curriculum_order(self) -> None:
        qids = [f"dsr-{index}" for index in range(6)]
        out = self.root / "train_absgen_sft.jsonl"
        meta = self.root / "absgen_sft_meta.json"
        args = argparse.Namespace(
            out=str(out), num_shards=2, k=2, generator="teacher"
        )
        for shard in range(2):
            shard_qids = qids[shard::2]
            rows = [self.row(qid) for qid in reversed(shard_qids)]
            warmstart_gen._atomic_jsonl(warmstart_gen._shard_path(out, shard, 2), rows)
            warmstart_gen._atomic_json(
                warmstart_gen._shard_path(meta, shard, 2),
                {
                    "shard": shard,
                    "num_shards": 2,
                    "n_problems": len(shard_qids),
                    "total_problems": len(qids),
                    "k": 2,
                    "generator": "teacher",
                    "kept": len(rows),
                    "leaked_dropped": shard,
                },
            )

        warmstart_gen._merge_shards(args, qids)
        merged = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["metadata"]["qid"] for row in merged], qids)
        merged_meta = json.loads(meta.read_text(encoding="utf-8"))
        self.assertEqual(merged_meta["kept"], len(qids))
        self.assertEqual(merged_meta["leaked_dropped"], 1)
        self.assertEqual(merged_meta["num_shards"], 2)

    def test_controller_allocates_both_named_inference_nodes(self) -> None:
        env = {
            **os.environ,
            "RLAD_CLUSTER_PROFILE": str(RL_ROOT / ".env.cluster.example"),
        }
        completed = subprocess.run(
            [
                "bash", "-c",
                'source "$1"; load_profile 0; printf "%s\\n" "${INFERENCE_SBATCH_ARGS[@]}"',
                "controller-test", str(ROOT_PIPELINE),
            ],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = completed.stdout.splitlines()
        self.assertIn("--nodes=2", arguments)
        self.assertIn("--ntasks=2", arguments)
        self.assertIn("--ntasks-per-node=1", arguments)
        self.assertIn("--gpus-per-node=8", arguments)
        self.assertIn("--nodelist=ip-10-1-81-8,ip-10-1-38-11", arguments)

    def test_worker_rank_uses_disjoint_global_shards_and_local_gpus(self) -> None:
        recorder = self.root / "record.py"
        results = self.root / "results"
        results.mkdir()
        recorder.write_text(
            """import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
shard = os.environ["RLAD_GLOBAL_SHARD_ID"]
(out / f"shard{shard}.json").write_text(json.dumps({
    "argv": sys.argv[2:],
    "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
    "cache": os.environ["VLLM_CACHE_ROOT"],
}))
""",
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "RLAD_GPU_SHARD_WORKER": "1",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_PROCID": "1",
            "SLURM_JOB_ID": "99",
            "HF_HOME": str(self.root / "hf"),
        }
        subprocess.run(
            [
                str(SHARD_LAUNCHER), "4", "2", "--shard", "--num-shards",
                str(self.root / "logs"), "worker", sys.executable, str(recorder),
                str(results),
            ],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        shard2 = json.loads((results / "shard2.json").read_text(encoding="utf-8"))
        shard3 = json.loads((results / "shard3.json").read_text(encoding="utf-8"))
        self.assertEqual(shard2["argv"], ["--shard", "2", "--num-shards", "4"])
        self.assertEqual(shard3["argv"], ["--shard", "3", "--num-shards", "4"])
        self.assertEqual((shard2["cuda"], shard3["cuda"]), ("0", "1"))
        self.assertIn("shard2/vllm", shard2["cache"])
        self.assertIn("shard3/vllm", shard3["cache"])

    def test_coordinator_requests_one_eight_gpu_task_per_node(self) -> None:
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        args_file = self.root / "srun-args"
        fake_srun = bin_dir / "srun"
        fake_srun.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$SRUN_ARGS_FILE"\n',
            encoding="utf-8",
        )
        fake_srun.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_CPUS_PER_TASK": "32",
            "SRUN_ARGS_FILE": str(args_file),
        }
        subprocess.run(
            [
                str(SHARD_LAUNCHER), "16", "8", "--shard", "--num-shards",
                str(self.root / "logs"), "worker", "true",
            ],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = args_file.read_text(encoding="utf-8").splitlines()
        self.assertIn("--nodes=2", arguments)
        self.assertIn("--ntasks=2", arguments)
        self.assertIn("--ntasks-per-node=1", arguments)
        self.assertIn("--cpus-per-task=32", arguments)
        self.assertIn("--gpus-per-task=8", arguments)
        self.assertIn("--export=ALL,RLAD_GPU_SHARD_WORKER=1", arguments)

    def test_worker_propagates_a_gpu_shard_failure(self) -> None:
        fail_one = self.root / "fail_one.py"
        fail_one.write_text(
            """import os, sys
raise SystemExit(7 if os.environ["RLAD_GLOBAL_SHARD_ID"] == "1" else 0)
""",
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "RLAD_GPU_SHARD_WORKER": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_PROCID": "0",
            "SLURM_JOB_ID": "101",
            "HF_HOME": str(self.root / "hf"),
        }
        completed = subprocess.run(
            [
                str(SHARD_LAUNCHER), "2", "2", "--shard", "--num-shards",
                str(self.root / "logs"), "failure", sys.executable, str(fail_one),
            ],
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertTrue((self.root / "logs" / "failure1_101.log").exists())


if __name__ == "__main__":
    unittest.main()
