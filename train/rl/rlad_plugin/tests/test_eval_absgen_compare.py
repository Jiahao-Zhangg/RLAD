"""CPU-only integrity tests for the resumable five-metric abstraction evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


EVAL_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_rlad.py"
SPEC = importlib.util.spec_from_file_location("eval_rlad_test", EVAL_PATH)
eval_rlad = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eval_rlad)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvalIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.bench = self.root / "benchmarks"
        write_jsonl(
            self.bench / "dsr_hard.jsonl",
            [
                {"id": "p1", "problem": "one", "answer": "1"},
                {"id": "p2", "problem": "two", "answer": "2"},
            ],
        )
        eval_rlad.BENCH_DIR = self.bench

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def abstractions(self, out: Path) -> None:
        write_jsonl(
            out / "abstractions.jsonl",
            [
                {"id": pid, "abs_idx": idx, "abstraction": f"hint-{pid}-{idx}"}
                for pid in ("p1", "p2")
                for idx in range(2)
            ],
        )

    @staticmethod
    def solve_rows(skip_woabs: bool = False) -> list[dict]:
        rewards = {
            ("p1", "woabs"): [1, 0],
            ("p2", "woabs"): [0, 0],
            ("p1", "abs0"): [1, 1],
            ("p1", "abs1"): [0, 0],
            ("p2", "abs0"): [1, 0],
            ("p2", "abs1"): [1, 1],
        }
        rows = []
        for (pid, condition), values in rewards.items():
            if skip_woabs and condition == "woabs":
                continue
            for idx, correct in enumerate(values):
                rows.append({"id": pid, "cond": condition, "sample_idx": idx, "correct": correct})
        return rows

    def args(self, out: Path, skip_woabs: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            out=str(out), benchmark="dsr_hard", mode="dual", n=2, k=2,
            skip_woabs=skip_woabs,
        )

    def test_exact_summary_and_five_metric_comparison(self) -> None:
        untrained = self.root / "untrained"
        rft = self.root / "rft"
        for out, skip in ((untrained, False), (rft, True)):
            self.abstractions(out)
            write_jsonl(out / "solve_samples.shard0.jsonl", self.solve_rows(skip))

        untrained_summary = eval_rlad._build_summary(self.args(untrained))
        rft_summary = eval_rlad._build_summary(self.args(rft, True))
        self.assertEqual(untrained_summary["woabs_pass1"], 25.0)
        self.assertEqual(untrained_summary["wabs_avg_pass1"], 62.5)
        self.assertEqual(untrained_summary["wabs_best_pass1"], 100.0)
        combined = eval_rlad.build_comparison(untrained_summary, rft_summary, "dsr_hard", "a" * 64)
        self.assertEqual(combined["base_without_hint_pass1"], 25.0)
        self.assertEqual(combined["untrained_hint_avg_pass1"], 62.5)
        self.assertEqual(combined["rft_hint_best_pass1"], 100.0)

    def test_compare_logs_all_five_metrics_and_writes_wandb_receipt(self) -> None:
        untrained = self.root / "untrained-wandb"
        rft = self.root / "rft-wandb"
        out = self.root / "comparison"
        for directory, skip in ((untrained, False), (rft, True)):
            self.abstractions(directory)
            write_jsonl(directory / "solve_samples.shard0.jsonl", self.solve_rows(skip))
            eval_rlad._atomic_json(
                directory / "summary.json", eval_rlad._build_summary(self.args(directory, skip))
            )

        class FakeRun:
            entity = "alice"
            project = "proj"
            id = "run-1"
            url = "https://wandb.ai/alice/proj/runs/run-1"

            def __init__(self):
                self.logged = None
                self.summary = {}

            def log(self, values, step):
                self.logged = (values, step)

            def finish(self):
                return None

        fake_run = FakeRun()
        init_kwargs = {}

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                init_kwargs.update(kwargs)
                return fake_run

        previous = sys.modules.get("wandb")
        sys.modules["wandb"] = FakeWandb
        try:
            eval_rlad.stage_compare(
                SimpleNamespace(
                    untrained_out=str(untrained), rft_out=str(rft), out=str(out),
                    benchmark="dsr_hard", input_key="b" * 64,
                    wandb_project="proj", wandb_entity="alice", wandb_group="group",
                    wandb_run_name="comparison", wandb_run_id="run-1",
                    untrained_absgen="base", rft_absgen="rft", solver_model="base",
                    source_commit="c" * 40,
                )
            )
        finally:
            if previous is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = previous
        saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(fake_run.logged[0]), 5)
        self.assertEqual(init_kwargs["mode"], "online")
        self.assertEqual(init_kwargs["resume"], "allow")
        self.assertEqual(saved["wandb"]["group"], "group")
        self.assertFalse((out / "summary.pending.json").exists())

    def test_summary_rejects_missing_or_duplicate_samples(self) -> None:
        out = self.root / "incomplete"
        self.abstractions(out)
        rows = self.solve_rows()
        write_jsonl(out / "solve_samples.shard0.jsonl", rows[:-1])
        with self.assertRaises(SystemExit):
            eval_rlad._build_summary(self.args(out))
        write_jsonl(out / "solve_samples.shard0.jsonl", rows + [rows[0]])
        with self.assertRaises(SystemExit):
            eval_rlad._build_summary(self.args(out))

    def test_resume_removes_only_incomplete_condition(self) -> None:
        path = self.root / "solve_samples.shard0.jsonl"
        rows = [
            {"id": "p1", "cond": "abs0", "sample_idx": 0, "correct": 1},
            {"id": "p1", "cond": "abs0", "sample_idx": 1, "correct": 0},
            {"id": "p1", "cond": "abs1", "sample_idx": 0, "correct": 1},
        ]
        write_jsonl(path, rows)
        complete = eval_rlad._repair_solve_file(path, {("p1", "abs0"), ("p1", "abs1")}, 2)
        self.assertEqual(complete, {("p1", "abs0")})
        self.assertEqual(len(eval_rlad._read(path)), 2)

    def test_resume_removes_truncated_final_jsonl_record(self) -> None:
        path = self.root / "solve_samples.shard0.jsonl"
        complete = [
            {"id": "p1", "cond": "abs0", "sample_idx": 0, "correct": 1},
            {"id": "p1", "cond": "abs0", "sample_idx": 1, "correct": 0},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in complete) + '{"id":"p1"',
            encoding="utf-8",
        )
        kept = eval_rlad._repair_solve_file(path, {("p1", "abs0")}, 2)
        self.assertEqual(kept, {("p1", "abs0")})
        self.assertEqual(eval_rlad._read(path), complete)

    def test_resume_repairs_partial_abstraction_group(self) -> None:
        out = self.root / "partial-abs"
        write_jsonl(
            out / "abstractions.jsonl",
            [
                {"id": "p1", "abs_idx": 0, "abstraction": "h0"},
                {"id": "p1", "abs_idx": 1, "abstraction": "h1"},
                {"id": "p2", "abs_idx": 0, "abstraction": "partial"},
            ],
        )
        complete = eval_rlad._abstractions(out, "dsr_hard", 2, repair=True)
        self.assertEqual(set(complete), {"p1"})
        self.assertEqual({row["id"] for row in eval_rlad._read(out / "abstractions.jsonl")}, {"p1"})


if __name__ == "__main__":
    unittest.main()
