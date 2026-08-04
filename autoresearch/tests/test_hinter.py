from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autoresearch.hinter import core, pool, publish, state


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


@pytest.fixture
def config() -> dict:
    return json.loads(
        (core.REPO_ROOT / "autoresearch/config.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_matches_contract(config: dict) -> None:
    core._validate_config(config)
    assert config["dataset"]["train_positions"] == list(core.TRAIN_POSITIONS)
    assert config["dataset"]["heldout_positions"] == list(
        core.HELDOUT_POSITIONS
    )
    assert config["slurm"]["pool_slots"] == 16
    assert config["sampling"]["rollouts"] == 8
    assert config["sampling"]["max_tokens"] == 16384
    assert config["budget"] == {"max_rounds": 20}
    assert config["objective"]["lambda"] == 1


@pytest.mark.parametrize(
    ("objective_lambda", "partition", "nodes"),
    [
        (
            2,
            "ml.p4d.24xlarge",
            ["ip-10-1-173-179", "ip-10-1-184-205"],
        ),
        (
            5,
            "ml.p4d.24xlarge",
            ["ip-10-1-196-96", "ip-10-1-226-48"],
        ),
        (
            10,
            "ml.p5.48xlarge",
            ["ip-10-1-38-11", "ip-10-1-81-8"],
        ),
    ],
)
def test_lambda_launch_config_overrides(
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
    objective_lambda: int,
    partition: str,
    nodes: list[str],
) -> None:
    monkeypatch.setenv("RLAD_AUTORESEARCH_LAMBDA", str(objective_lambda))
    monkeypatch.setenv("RLAD_AUTORESEARCH_PARTITION", partition)
    monkeypatch.setenv("RLAD_AUTORESEARCH_NODES", ",".join(nodes))
    effective = core._apply_environment_overrides(config)
    core._validate_config(effective)
    assert effective["objective"]["lambda"] == objective_lambda
    assert effective["slurm"]["partition"] == partition
    assert effective["slurm"]["nodes"] == nodes
    assert f"lambda-{objective_lambda}" in effective["publication"][
        "commit_prefix"
    ]


def test_lambda_launchers_are_isolated_and_executable() -> None:
    expected = {
        2: (
            "work_zsw_lambda2",
            "ip-10-1-173-179,ip-10-1-184-205",
        ),
        5: (
            "work_zsw_lambda5",
            "ip-10-1-196-96,ip-10-1-226-48",
        ),
        10: (
            "work_zsw_lambda10",
            "ip-10-1-38-11,ip-10-1-81-8",
        ),
    }
    for objective_lambda, (workspace, nodes) in expected.items():
        path = (
            core.REPO_ROOT
            / "autoresearch"
            / f"run_lambda_{objective_lambda}.sh"
        )
        script = path.read_text(encoding="utf-8")
        assert os.access(path, os.X_OK)
        assert f"RLAD_AUTORESEARCH_LAMBDA={objective_lambda}" in script
        assert workspace in script
        assert nodes in script
        assert "python -m autoresearch.run_hinter_agent" in script


def test_agent_sbatch_does_not_consume_gpu_pool() -> None:
    script = (
        core.REPO_ROOT / "autoresearch/jobs/hinter_agent.sbatch"
    ).read_text(encoding="utf-8")
    directives = [
        line.strip()
        for line in script.splitlines()
        if line.startswith("#SBATCH ")
    ]
    assert "#SBATCH --nodes=1" in directives
    assert "#SBATCH --cpus-per-task=4" in directives
    assert not any("--partition=" in line for line in directives)
    assert not any("--gpus" in line or "--gres" in line for line in directives)
    assert not any("--nodelist" in line for line in directives)
    assert "ml.p5.48xlarge" in script
    assert "ml.p4d.24xlarge" in script
    assert "ip-10-1-38-11|ip-10-1-81-8" in script
    assert script.index("trap cleanup_on_exit EXIT") < script.index(
        "gh auth status"
    )
    signal_handler = script.split("handle_signal() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert signal_handler.index("request_pool_stop") < signal_handler.index(
        'wait "${active_pid}"'
    )
    assert "run_interruptible \\\n        uv sync" in script
    assert (
        "run_interruptible uv run --project autoresearch --frozen"
        in script
    )


def test_exact_metrics_and_tie_break() -> None:
    old = core.exact_metrics(2, 20)
    higher_j = core.exact_metrics(3, 20)
    higher_heldout_same_j = core.exact_metrics(1, 30)
    assert core.proposal_wins(old, higher_j, 40, 100)
    assert core.proposal_wins(old, higher_heldout_same_j, 40, 100)
    assert core.proposal_wins(old, old, 40, 39)
    assert not core.proposal_wins(old, old, 40, 40)
    value = {
        "schema_version": 1,
        "hint_id": 1,
        "hint_hash": "x",
        "config_hash": "c",
        **old,
    }
    core.validate_private_metrics(value)
    value["J_i"] += 0.01
    with pytest.raises(ValueError, match="count-derived"):
        core.validate_private_metrics(value)
    assert core.exact_metrics(
        2,
        20,
        objective_lambda=10,
    )["J_numerator"] == 220
    lambda_ten = {
        "schema_version": 1,
        "hint_id": 1,
        "hint_hash": "x",
        "config_hash": "c",
        **core.exact_metrics(2, 20, objective_lambda=10),
    }
    core.validate_private_metrics(lambda_ten, objective_lambda=10)
    with pytest.raises(ValueError, match="count-derived"):
        core.validate_private_metrics(lambda_ten, objective_lambda=1)


def test_round_twenty_task_ids_are_valid() -> None:
    core.validate_task_identity("r20-train-h10", "train")
    core.validate_task_identity("r20-private-h01", "private")
    with pytest.raises(ValueError, match="invalid"):
        core.validate_task_identity("r21-train-h01", "train")


def test_zero_keep_streak_never_stops_before_round_twenty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    metrics = [
        {"round": round_number, "aggregates": {"num_kept": 0}}
        for round_number in range(1, 4)
    ]
    monkeypatch.setattr(state, "RESEARCH_ROOT", tmp_path)
    monkeypatch.setattr(
        state,
        "load_config",
        lambda require_frozen=True: (config, "config-hash"),
    )
    monkeypatch.setattr(
        state,
        "_completed_round_metrics",
        lambda config_hash: metrics,
    )
    status = state.stopping_status()
    assert status["consecutive_zero_keep_rounds"] == 3
    assert status["should_stop"] is False
    assert status["reason"] is None

    metrics[:] = [
        {"round": round_number, "aggregates": {"num_kept": 0}}
        for round_number in range(1, 21)
    ]
    status = state.stopping_status()
    assert status["should_stop"] is True
    assert status["reason"] == "round_budget_exhausted"


def test_publication_ignores_only_other_lambda_workspace_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path("/tmp/rlad-publication-scope-test")
    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "WORK_ROOT", repo / "work_zsw_lambda2")
    monkeypatch.setattr(
        publish,
        "_changed_paths",
        lambda: {
            "work_zsw_lambda2/research/current_book.json",
            "work_zsw_lambda5/research/current_book.json",
            "docs/plan/hinter.md",
        },
    )
    assert publish._unexpected_changes(set()) == [
        "docs/plan/hinter.md",
        "work_zsw_lambda2/research/current_book.json",
    ]


def test_publication_waits_for_other_workspace_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    transaction = (
        tmp_path
        / "work_zsw_lambda5"
        / "publication"
        / "git_transaction.json"
    )
    transaction.parent.mkdir(parents=True)
    transaction.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(publish, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(publish, "WORK_ROOT", tmp_path / "work_zsw_lambda2")
    monkeypatch.setattr(
        publish,
        "_git",
        lambda *args, **kwargs: ".git",
    )
    with pytest.raises(RuntimeError, match="recover"):
        with publish._repo_publication_lock():
            pytest.fail("lock should reject an outstanding transaction")


def test_hint_limits_and_leak_gate() -> None:
    tokenizer = FakeTokenizer()
    hint, tokens, digest = core.validate_hint(
        "Use a reusable invariant.",
        tokenizer,
        10,
    )
    assert hint == "Use a reusable invariant."
    assert tokens == 4
    assert digest == core.hint_hash(hint)
    with pytest.raises(ValueError, match="maximum"):
        core.validate_hint("one two three", tokenizer, 2)
    with pytest.raises(ValueError, match="assigned answer"):
        core.reject_answer_leak("The value is 173.", "173")
    with pytest.raises(ValueError, match="answer language"):
        core.reject_answer_leak("Do not reveal the final answer.", "173")


def _patch_pool(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(pool, "LOCK_PATH", root / "experiment.lock.json")
    monkeypatch.setattr(pool, "WORK_ROOT", root)
    monkeypatch.setattr(pool, "POOL_ROOT", root / "pool")
    monkeypatch.setattr(pool, "PENDING", root / "pool/pending")
    monkeypatch.setattr(pool, "RUNNING", root / "pool/running")
    monkeypatch.setattr(pool, "DONE", root / "pool/done")
    monkeypatch.setattr(pool, "FAILED", root / "pool/failed")
    monkeypatch.setattr(pool, "CONTROL", root / "pool/control")
    monkeypatch.setattr(pool, "ALLOCATION_PATH", root / "pool/allocation.json")
    monkeypatch.setattr(pool, "STOP_PATH", root / "pool/control/STOP")
    monkeypatch.setattr(pool, "SEALED_PATH", root / "research/STOPPED.json")
    monkeypatch.setattr(pool, "TASK_LOGS", root / "logs/tasks")
    monkeypatch.setattr(
        pool,
        "load_config",
        lambda require_frozen=False: ({}, "config-hash"),
    )
    monkeypatch.setattr(pool, "source_bundle_hash", lambda paths: "source-hash")
    monkeypatch.setattr(
        pool,
        "validate_registered_training_packet",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        pool,
        "validate_registered_private_input",
        lambda *args, **kwargs: {},
    )


def test_pool_enqueue_is_immutable_and_work_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pool(monkeypatch, tmp_path)
    input_path = tmp_path / "inputs/hint.json"
    input_path.parent.mkdir()
    input_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "outputs/hint.json"
    receipt = tmp_path / "receipts/hint.json"
    value = pool.enqueue_task(
        task_id="r01-train-h01",
        mode="train",
        input_path=input_path,
        output_path=output,
        receipt_path=receipt,
    )
    assert value["state"] == "pending"
    task = json.loads(
        (tmp_path / "pool/pending/r01-train-h01.json").read_text()
    )
    assert set(task) == core.TASK_KEYS
    assert task["input_sha256"] == core.sha256_file(input_path)
    with pytest.raises(RuntimeError, match="already exists"):
        pool.enqueue_task(
            task_id="r01-train-h01",
            mode="train",
            input_path=input_path,
            output_path=output,
            receipt_path=receipt,
        )
    outside = tmp_path.parent / "escape.json"
    with pytest.raises(ValueError, match="escapes"):
        pool.enqueue_task(
            task_id="r02-train-h01",
            mode="train",
            input_path=input_path,
            output_path=outside,
            receipt_path=receipt,
        )


def test_sealed_experiment_refuses_pool_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pool(monkeypatch, tmp_path)
    input_path = tmp_path / "inputs/hint.json"
    input_path.parent.mkdir()
    input_path.write_text("{}\n", encoding="utf-8")
    pool.enqueue_task(
        task_id="r01-train-h01",
        mode="train",
        input_path=input_path,
        output_path=tmp_path / "outputs/hint.json",
        receipt_path=tmp_path / "receipts/hint.json",
    )
    core.atomic_write_json(
        tmp_path / "research/STOPPED.json",
        {"should_stop": True},
    )
    assert pool._claim_one() is None
    with pytest.raises(RuntimeError, match="sealed"):
        pool.enqueue_task(
            task_id="r02-train-h01",
            mode="train",
            input_path=input_path,
            output_path=tmp_path / "outputs/hint_2.json",
            receipt_path=tmp_path / "receipts/hint_2.json",
        )
    with pytest.raises(RuntimeError, match="sealed"):
        pool.start_pool()


def test_private_scoring_is_bound_to_registered_round_artifacts(
    tmp_path: Path,
    config: dict,
) -> None:
    config_hash = "config-hash"
    research = tmp_path / "research"
    public_rows = [
        {
            "hint_id": hint_id,
            "train_position": core.TRAIN_POSITIONS[hint_id - 1],
            "train_qid": f"qid-{hint_id}",
            "problem": f"problem {hint_id}",
            "answer": f"answer {hint_id}",
        }
        for hint_id in range(1, 11)
    ]
    core.atomic_write_json(
        research / "setup/train_public.json",
        public_rows,
    )
    smoke = {
        "schema_version": 1,
        "round": 0,
        "hint_id": 1,
        "train_position": core.TRAIN_POSITIONS[0],
        "train_qid": public_rows[0]["train_qid"],
        "problem": public_rows[0]["problem"],
        "answer": public_rows[0]["answer"],
        "hint": core.SETUP_SMOKE_HINT,
        "hint_hash": core.hint_hash(core.SETUP_SMOKE_HINT),
        "config_hash": config_hash,
        "previous_train_i": 0.0,
        "previous_heldout_i": 0.0,
        "previous_J_i": 0.0,
        "hard_tokens_per_hint":
            config["hint_limits"]["hard_tokens_per_hint"],
        "worker_tokens_per_hint":
            config["hint_limits"]["worker_tokens_per_hint"],
        "hard_total_tokens": config["hint_limits"]["hard_total_tokens"],
    }
    core.atomic_write_json(research / "setup/smoke_input.json", smoke)
    private_smoke = {
        "schema_version": 1,
        "hint_id": 1,
        "hint": core.SETUP_SMOKE_HINT,
        "hint_hash": core.hint_hash(core.SETUP_SMOKE_HINT),
        "config_hash": config_hash,
    }
    private_smoke_path = research / "setup/private_smoke_input.json"
    core.atomic_write_json(private_smoke_path, private_smoke)
    assert core.validate_registered_private_input(
        private_smoke_path,
        task_id="setup-private-smoke",
        mode="private-smoke",
        config=config,
        config_hash=config_hash,
        research_root=research,
    ) == private_smoke

    hints = [
        {
            "hint_id": hint_id,
            "hint": f"registered strategy {hint_id}",
            "hint_hash": core.hint_hash(f"registered strategy {hint_id}"),
        }
        for hint_id in range(1, 11)
    ]
    core.atomic_write_json(
        research / "initial_book.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "generation": 0,
            "hints": hints,
        },
    )
    baseline = research / "baseline/inputs/hint_01.json"
    baseline_request = {
        "schema_version": 1,
        "hint_id": 1,
        "hint": hints[0]["hint"],
        "hint_hash": hints[0]["hint_hash"],
        "config_hash": config_hash,
    }
    core.atomic_write_json(baseline, baseline_request)
    assert core.validate_registered_private_input(
        baseline,
        task_id="baseline-h01",
        mode="private",
        config=config,
        config_hash=config_hash,
        research_root=research,
    ) == baseline_request

    proposals = [
        {
            "hint_id": hint_id,
            "hint": f"proposal strategy {hint_id}",
            "hint_hash": core.hint_hash(f"proposal strategy {hint_id}"),
        }
        for hint_id in range(1, 11)
    ]
    core.atomic_write_json(
        research / "rounds/001/book_proposals.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "round": 1,
            "book_hash_before": "before",
            "hints": proposals,
        },
    )
    proposal_input = (
        research / "rounds/001/proposal_private_inputs/hint_01.json"
    )
    proposal_request = {
        "schema_version": 1,
        "hint_id": 1,
        "hint": proposals[0]["hint"],
        "hint_hash": proposals[0]["hint_hash"],
        "config_hash": config_hash,
    }
    core.atomic_write_json(proposal_input, proposal_request)
    assert core.validate_registered_private_input(
        proposal_input,
        task_id="r01-private-h01",
        mode="private",
        config=config,
        config_hash=config_hash,
        research_root=research,
    ) == proposal_request

    arbitrary = {
        **proposal_request,
        "hint": "unregistered held-out query",
        "hint_hash": core.hint_hash("unregistered held-out query"),
    }
    core.atomic_write_json(proposal_input, arbitrary)
    with pytest.raises(ValueError, match="registered hint"):
        core.validate_registered_private_input(
            proposal_input,
            task_id="r01-private-h01",
            mode="private",
            config=config,
            config_hash=config_hash,
            research_root=research,
        )
    with pytest.raises(ValueError, match="invalid"):
        core.validate_task_identity("r99-private-h01", "private")


def test_failed_task_retry_archives_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pool(monkeypatch, tmp_path)
    pool._make_dirs()
    task_id = "r01-train-h01"
    output = tmp_path / "outputs/hint.json"
    receipt = tmp_path / "receipts/hint.json"
    output.parent.mkdir()
    receipt.parent.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")
    output.write_text("partial\n", encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": "train",
        "input_path": str(input_path),
        "input_sha256": core.sha256_file(input_path),
        "output_path": str(output),
        "receipt_path": str(receipt),
        "config_hash": "config-hash",
        "source_hash": "source-hash",
        "gpu_count": 1,
        "created_epoch": 1.0,
    }
    core.atomic_write_json(tmp_path / f"pool/failed/{task_id}.json", task)
    core.atomic_write_json(
        tmp_path / f"pool/failed/{task_id}.error.json",
        {"error": "test"},
    )
    value = pool.retry_failed(task_id)
    assert value["state"] == "pending"
    assert (tmp_path / f"pool/pending/{task_id}.json").is_file()
    archive = Path(value["archived_attempt"])
    assert (archive / "output.json").is_file()
    assert (archive / "receipt.json").is_file()
    assert (archive / "error.json").is_file()
    assert (archive / "task.json").is_file()
    assert not output.exists()
    assert not receipt.exists()


def test_setup_retry_can_refresh_stale_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pool(monkeypatch, tmp_path)
    pool._make_dirs()
    input_path = tmp_path / "setup/smoke_input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}\n", encoding="utf-8")
    task_id = "setup-smoke"
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": "smoke",
        "input_path": str(input_path),
        "input_sha256": "old-input-hash",
        "output_path": str(tmp_path / "setup/smoke_output.json"),
        "receipt_path": str(tmp_path / "setup/smoke_receipt.json"),
        "config_hash": "old-config-hash",
        "source_hash": "old-source-hash",
        "gpu_count": 1,
        "created_epoch": 1.0,
    }
    core.atomic_write_json(tmp_path / f"pool/failed/{task_id}.json", task)
    core.atomic_write_json(
        tmp_path / f"pool/failed/{task_id}.error.json",
        {"error": "implementation failure"},
    )
    result = pool.retry_failed(task_id, refresh_source=True)
    refreshed = core.load_json(tmp_path / f"pool/pending/{task_id}.json")
    assert result["source_refreshed"] is True
    assert refreshed["source_hash"] == "source-hash"
    assert refreshed["config_hash"] == "config-hash"
    assert refreshed["input_sha256"] == core.sha256_file(input_path)
    assert Path(result["archived_attempt"], "task.json").is_file()


def test_completed_setup_gate_can_be_refreshed_after_review_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pool(monkeypatch, tmp_path)
    pool._make_dirs()
    task_id = "setup-private-smoke"
    input_path = tmp_path / "setup/private_smoke_input.json"
    output_path = tmp_path / "setup/private_smoke_output.json"
    receipt_path = tmp_path / "setup/private_smoke_receipt.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}\n", encoding="utf-8")
    output_path.write_text("{}\n", encoding="utf-8")
    receipt_path.write_text("{}\n", encoding="utf-8")
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": "private-smoke",
        "input_path": str(input_path),
        "input_sha256": core.sha256_file(input_path),
        "output_path": str(output_path),
        "receipt_path": str(receipt_path),
        "config_hash": "config-hash",
        "source_hash": "old-source-hash",
        "gpu_count": 1,
        "created_epoch": 1.0,
    }
    core.atomic_write_json(tmp_path / f"pool/done/{task_id}.json", task)
    result = pool.refresh_setup_task(task_id)
    refreshed = core.load_json(tmp_path / f"pool/pending/{task_id}.json")
    assert result["state"] == "pending"
    assert refreshed["source_hash"] == "source-hash"
    archive = Path(result["archived_attempt"])
    assert (archive / "task.json").is_file()
    assert (archive / "output.json").is_file()
    assert (archive / "receipt.json").is_file()


def test_receipt_binds_archived_allocation_and_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    _, config_hash = core.load_config(require_frozen=False)
    monkeypatch.setattr(core, "WORK_ROOT", tmp_path)
    pool_root = tmp_path / "pool"
    pool_root.mkdir()
    source_hash = core.source_bundle_hash(core.RUNTIME_SOURCE_FILES)

    def allocation(job_id: str) -> dict:
        return {
            "schema_version": 1,
            "job_id": job_id,
            "partition": config["slurm"]["partition"],
            "nodes": config["slurm"]["nodes"],
            "pool_slots": 16,
            "exclusive": True,
            "gpus_per_node": 8,
            "gpus_per_task": 1,
            "gpu_binding": config["slurm"]["gpu_binding"],
            "python": sys.executable,
            "repo_root": str(core.REPO_ROOT),
            "work_root": str(tmp_path),
            "config_hash": config_hash,
            "source_hash": source_hash,
            "submitted_epoch": time.time(),
        }

    core.atomic_write_json(pool_root / "allocation.json", allocation("222"))
    core.atomic_write_json(
        pool_root / "allocation.111.json",
        allocation("111"),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text('{"packet": 1}\n', encoding="utf-8")
    output_path.write_text('{"result": 1}\n', encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "task_id": "setup-smoke",
        "mode": "smoke",
        "allocation_job_id": "111",
        "slurm_step_id": "7",
        "execution_id": "111.7",
        "node": config["slurm"]["nodes"][0],
        "gpu_count": 1,
        "pool_slot": 3,
        "slurm_gpus_per_task": 1,
        "visible_cuda_device": "0",
        "config_hash": config_hash,
        "source_hash": source_hash,
        "input_path": str(input_path),
        "input_sha256": core.sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": core.sha256_file(output_path),
        "start_epoch": 1.0,
        "end_epoch": 2.0,
        "elapsed_seconds": 1.0,
        "exit_code": 0,
    }
    core.validate_receipt(
        receipt,
        task_id="setup-smoke",
        mode="smoke",
        input_path=input_path,
        output_path=output_path,
        config_hash=config_hash,
    )
    receipt["node"] = "outside-node"
    with pytest.raises(ValueError, match="allocated nodes"):
        core.validate_receipt(
            receipt,
            task_id="setup-smoke",
            mode="smoke",
            input_path=input_path,
            output_path=output_path,
            config_hash=config_hash,
        )


def _configure_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> tuple[Path, str]:
    research = tmp_path / "research"
    (research / "setup").mkdir(parents=True)
    rows = [
        {
            "hint_id": hint_id,
            "train_position": core.TRAIN_POSITIONS[hint_id - 1],
            "train_qid": f"qid-{hint_id}",
            "problem": f"public problem {hint_id}",
            "answer": f"gold-{hint_id}",
        }
        for hint_id in range(1, 11)
    ]
    core.atomic_write_json(research / "setup/train_public.json", rows)
    config_hash = "frozen-config"
    monkeypatch.setattr(state, "RESEARCH_ROOT", research)
    monkeypatch.setattr(state, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(
        state,
        "load_config",
        lambda require_frozen=True: (config, config_hash),
    )
    monkeypatch.setattr(state, "tokenizer_factory", lambda _: FakeTokenizer())
    monkeypatch.setattr(state, "validate_receipt", lambda *args, **kwargs: None)
    return research, config_hash


def _private_output(
    path: Path,
    *,
    hint_id: int,
    digest: str,
    config_hash: str,
    train_correct: int,
    heldout_correct: int,
) -> None:
    core.atomic_write_json(
        path,
        {
            "schema_version": 1,
            "hint_id": hint_id,
            "hint_hash": digest,
            "config_hash": config_hash,
            **core.exact_metrics(train_correct, heldout_correct),
        },
    )


def test_full_round_state_machine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    research, config_hash = _configure_state(tmp_path, monkeypatch, config)
    hints = [
        f"Initial reusable strategy for case {hint_id} with extra detail."
        for hint_id in range(1, 11)
    ]
    book = state.initialize_book(hints)
    assert book["total_tokens"] < 2048

    baseline = research / "baseline"
    inputs = state.make_private_inputs(
        research / "initial_book.json",
        baseline / "inputs",
    )
    for hint_id, input_path in enumerate(inputs, start=1):
        request = core.load_json(input_path)
        _private_output(
            baseline / "outputs" / f"hint_{hint_id:02d}.json",
            hint_id=hint_id,
            digest=request["hint_hash"],
            config_hash=config_hash,
            train_correct=2,
            heldout_correct=20,
        )
        core.atomic_write_json(
            baseline / "receipts" / f"hint_{hint_id:02d}.json",
            {"execution_id": f"100.{hint_id}"},
        )
    state.initialize_metrics(
        input_dir=baseline / "inputs",
        output_dir=baseline / "outputs",
        receipt_dir=baseline / "receipts",
        task_prefix="baseline",
    )

    partial = research / "rounds/001/training_inputs"
    partial.mkdir(parents=True)
    round_dir = state.prepare_round(1)
    assert len(list(partial.glob("hint_*.json"))) == 10
    first_history = core.load_json(
        round_dir / "worker_history" / "hint_01.json"
    )
    assert first_history["hint_id"] == 1
    assert first_history["objective_lambda"] == 1
    assert first_history["config_hash"] == config_hash
    assert first_history["prior_rounds"] == []
    assert len(first_history["baseline_artifacts"]) == 3
    assert all(
        artifact["path"].endswith("hint_01.json")
        for artifact in first_history["baseline_artifacts"]
    )
    packet_path = partial / "hint_01.json"
    original_packet = core.load_json(packet_path)
    changed_packet = {**original_packet, "problem": "mutated question"}
    core.atomic_write_json(packet_path, changed_packet)
    with pytest.raises(ValueError, match="packet drift"):
        state.validate_registered_training_packet(
            packet_path,
            task_id="r01-train-h01",
            config=config,
            config_hash=config_hash,
            research_root=research,
        )
    core.atomic_write_json(packet_path, original_packet)
    for hint_id in range(1, 11):
        packet = core.load_json(
            round_dir / "training_inputs" / f"hint_{hint_id:02d}.json"
        )
        rewards = [1, 0, 0, 0, 0, 0, 0, 0]
        core.atomic_write_json(
            round_dir / "training_outputs" / f"hint_{hint_id:02d}.json",
            {
                "schema_version": 1,
                "round": 1,
                "hint_id": hint_id,
                "train_position": packet["train_position"],
                "train_qid": packet["train_qid"],
                "hint_hash": packet["hint_hash"],
                "config_hash": config_hash,
                "correct": 1,
                "total": 8,
                "train_i": 0.125,
                "rollouts": [
                    {
                        "sample_idx": sample_idx,
                        "response": f"response {sample_idx}",
                        "reward": reward,
                        "finish_reason": "stop",
                        "completion_tokens": 10,
                    }
                    for sample_idx, reward in enumerate(rewards)
                ],
            },
        )
        core.atomic_write_json(
            round_dir / "training_receipts" / f"hint_{hint_id:02d}.json",
            {"execution_id": f"200.{hint_id}"},
        )
        core.atomic_write_json(
            round_dir / "worker_proposals" / f"hint_{hint_id:02d}.json",
            {
                "hint_id": hint_id,
                "hint": f"Revised strategy {hint_id}.",
                "mutation": "shorten and clarify",
                "subagent_summary": "Training errors favored a clearer invariant.",
                "sampling_slurm_job_id": f"200.{hint_id}",
            },
        )
    proposals = state.collect_proposals(1)
    assert len(proposals["hints"]) == 10
    state.proposal_private_inputs(1)
    for hint_id, proposal in enumerate(proposals["hints"], start=1):
        if hint_id == 1:
            train_correct, heldout_correct = 3, 25
        elif hint_id == 2:
            train_correct, heldout_correct = 2, 20
        else:
            train_correct, heldout_correct = 1, 10
        _private_output(
            round_dir
            / "proposal_private_outputs"
            / f"hint_{hint_id:02d}.json",
            hint_id=hint_id,
            digest=proposal["hint_hash"],
            config_hash=config_hash,
            train_correct=train_correct,
            heldout_correct=heldout_correct,
        )
        core.atomic_write_json(
            round_dir
            / "proposal_private_receipts"
            / f"hint_{hint_id:02d}.json",
            {"execution_id": f"300.{hint_id}"},
        )
    metrics = state.finalize_round(1)
    assert metrics["aggregates"]["num_kept"] == 2
    assert metrics["aggregates"]["num_discarded"] == 8
    assert len(metrics["history_rows"]) == 10
    assert (round_dir / "book_after.json").is_file()
    assert len((research / "hint_history.csv").read_text().splitlines()) == 11
    assert state.stopping_status()["completed_rounds"] == 1
    assert state.finalize_round(1) == metrics
    with pytest.raises(RuntimeError, match="not been successfully pushed"):
        state.prepare_round(2)
    publication = tmp_path / "publication"
    publication.mkdir()
    core.atomic_write_json(
        publication / "round_001.json",
        {
            "schema_version": 1,
            "round": 1,
            "config_hash": config_hash,
            "commit": "a" * 40,
            "remote": config["publication"]["remote"],
                "branch": "research",
                "paths": [
                    f"{tmp_path.name}/research/rounds/001/metrics.json",
                ],
            "pushed_epoch": time.time(),
        },
    )
    second_round = state.prepare_round(2)
    assert second_round.name == "002"
    second_history = core.load_json(
        second_round / "worker_history" / "hint_01.json"
    )
    assert len(second_history["incumbent_history"]) == 2
    assert [row["round"] for row in second_history["prior_rounds"]] == [1]
    prior = second_history["prior_rounds"][0]
    assert prior["decision_record"]["hint_id"] == 1
    assert prior["history_row"]["hint_id"] == 1
    prior_paths = [artifact["path"] for artifact in prior["artifacts"]]
    assert len(prior_paths) == 8
    assert all("hint_01.json" in path for path in prior_paths)
    assert not any("hint_02" in path for path in prior_paths)
    assert not any("private-h01" in path for path in prior_paths)
    final = state.seal_final_book(human_stop=True)
    assert final["stopping_status"]["reason"] == "human_stop"
    assert state.stopping_status()["reason"] == "human_stop"
    with pytest.raises(RuntimeError, match="stop condition"):
        state.prepare_round(2)
    with pytest.raises(RuntimeError, match="sealed"):
        state.finalize_round(2)


def test_publication_allowlist_excludes_private_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    research = repo / "work_zsw/research"
    round_dir = research / "rounds/001"
    round_dir.mkdir(parents=True)
    for path in (
        research / "initial_book.json",
        research / "current_book.json",
        research / "best_per_hint.json",
        research / "round_summary.csv",
        research / "hint_history.csv",
        round_dir / "book_before.json",
        round_dir / "book_proposals.json",
        round_dir / "book_after.json",
        round_dir / "metrics.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    core.atomic_write_json(research / "current_book.json", {"generation": 1})
    core.atomic_write_json(round_dir / "metrics.json", {"round": 1})
    private = round_dir / "proposal_private_outputs/hint_01.json"
    private.parent.mkdir()
    private.write_text('{"private": true}\n', encoding="utf-8")
    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "RESEARCH_ROOT", research)
    paths = publish.public_round_paths(1)
    assert private not in paths
    assert round_dir / "metrics.json" in paths
    (round_dir / "metrics.json").write_text(
        '{"round": 1, "heldout_problem": "must not publish"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="forbidden keys"):
        publish.public_round_paths(1)


def _publication_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "research"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
    )
    (repo / ".gitignore").write_text("work_zsw/*\n", encoding="utf-8")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "base.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)
    artifact = repo / "work_zsw/research/rounds/001/metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"round": 1}\n', encoding="utf-8")
    return repo, artifact


def test_publication_recovers_exact_staged_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, artifact = _publication_repo(tmp_path)
    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "WORK_ROOT", repo / "work_zsw")
    monkeypatch.setattr(publish, "_push", lambda *args: None)
    real_run = publish.subprocess.run

    def interrupt_commit(command: list[str], *args, **kwargs):
        if command[:2] == ["git", "commit"]:
            raise KeyboardInterrupt
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(publish.subprocess, "run", interrupt_commit)
    message = "autoresearch: hinter round 001"
    with pytest.raises(KeyboardInterrupt):
        publish._commit_and_push(
            [artifact],
            message=message,
            remote="origin",
        )
    staged = real_run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert staged == "work_zsw/research/rounds/001/metrics.json"
    assert (
        repo / "work_zsw/publication/git_transaction.json"
    ).is_file()

    monkeypatch.setattr(publish.subprocess, "run", real_run)
    commit, branch, _ = publish._commit_and_push(
        [artifact],
        message=message,
        remote="origin",
    )
    assert branch == "research"
    assert commit == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    publish._finish_publication_transaction(commit)
    assert not (
        repo / "work_zsw/publication/git_transaction.json"
    ).exists()


def test_publication_never_recovers_same_subject_from_another_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, artifact = _publication_repo(tmp_path)
    subprocess.run(
        ["git", "add", "-f", str(artifact.relative_to(repo))],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "tracked artifact"],
        cwd=repo,
        check=True,
    )
    research_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "switch", "-c", "other"], cwd=repo, check=True)
    artifact.write_text('{"round": 999}\n', encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(artifact.relative_to(repo))],
        cwd=repo,
        check=True,
    )
    message = "autoresearch: hinter round 001"
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True)
    subprocess.run(["git", "switch", "research"], cwd=repo, check=True)

    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "WORK_ROOT", repo / "work_zsw")
    monkeypatch.setattr(publish, "_push", lambda *args: None)
    with pytest.raises(RuntimeError, match="no artifact changes"):
        publish._commit_and_push(
            [artifact],
            message=message,
            remote="origin",
        )
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == research_head


def test_github_remote_host_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert publish._remote_host(
        "https://github.com/example/repository.git"
    ) == "github.com"
    assert publish._remote_host(
        "git@github.com:example/repository.git"
    ) == "github.com"
    assert publish._remote_host(
        "https://github.com.evil.example/repository.git"
    ) == "github.com.evil.example"
    monkeypatch.setattr(
        publish,
        "_git",
        lambda *args: (
            "https://github.com/example/repository.git\n"
            "https://github.com.evil.example/repository.git"
        ),
    )
    with pytest.raises(RuntimeError, match="effective push URLs"):
        publish._push("a" * 40, "origin", "research")


def test_round_publisher_commits_only_allowlisted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "research"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)

    research = repo / "work_zsw/research"
    round_dir = research / "rounds/001"
    round_dir.mkdir(parents=True)
    for path in (
        research / "initial_book.json",
        research / "current_book.json",
        research / "best_per_hint.json",
        round_dir / "book_before.json",
        round_dir / "book_proposals.json",
        round_dir / "book_after.json",
        round_dir / "metrics.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    core.atomic_write_json(research / "current_book.json", {"generation": 1})
    core.atomic_write_json(round_dir / "metrics.json", {"round": 1})
    (research / "round_summary.csv").write_text("round\n1\n", encoding="utf-8")
    (research / "hint_history.csv").write_text("round\n1\n", encoding="utf-8")
    private = round_dir / "training_outputs/hint_01.json"
    private.parent.mkdir()
    private.write_text('{"response": "private"}\n', encoding="utf-8")
    (repo / ".gitignore").write_text(
        "work_zsw/*\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "ignore runtime"], cwd=repo, check=True)

    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "RESEARCH_ROOT", research)
    monkeypatch.setattr(publish, "WORK_ROOT", repo / "work_zsw")
    monkeypatch.setattr(
        publish,
        "load_config",
        lambda require_frozen=True: (
            {
                "publication": {
                    "remote": "origin",
                    "commit_prefix": "autoresearch: hinter round",
                }
            },
            "config",
        ),
    )
    pushed = {}
    monkeypatch.setattr(
        publish,
        "_push",
        lambda commit, remote, branch: pushed.update(
            commit=commit,
            remote=remote,
            branch=branch,
        ),
    )
    receipt = publish.publish_round(1)
    assert receipt["commit"] == pushed["commit"]
    assert pushed["branch"] == "research"
    names = subprocess.run(
        [
            "git",
            "show",
            "--pretty=",
            "--name-only",
            receipt["commit"],
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "work_zsw/research/rounds/001/metrics.json" in names
    assert (
        "work_zsw/research/rounds/001/training_outputs/hint_01.json"
        not in names
    )

    (research / "final_book.json").write_text("{}\n", encoding="utf-8")
    (research / "STOPPED.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        publish,
        "_validate_existing_receipt",
        lambda *args, **kwargs: None,
    )
    terminal = publish.publish_round(1)
    assert terminal["commit"] != receipt["commit"]
    terminal_names = subprocess.run(
        [
            "git",
            "show",
            "--pretty=",
            "--name-only",
            terminal["commit"],
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "work_zsw/research/final_book.json" in terminal_names
    assert "work_zsw/research/STOPPED.json" in terminal_names
    assert (repo / "work_zsw/pool/control/STOP").is_file()
    assert publish.publish_round(1)["reused"] is True


def test_round_zero_human_stop_is_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "research"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
    )
    (repo / ".gitignore").write_text("work_zsw/*\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)
    research = repo / "work_zsw/research"
    research.mkdir(parents=True)
    core.atomic_write_json(research / "initial_book.json", {"generation": 0})
    core.atomic_write_json(research / "current_book.json", {"generation": 0})
    core.atomic_write_json(research / "final_book.json", {"generation": 0})
    core.atomic_write_json(
        research / "STOPPED.json",
        {"should_stop": True, "last_round": 0},
    )
    monkeypatch.setattr(publish, "REPO_ROOT", repo)
    monkeypatch.setattr(publish, "RESEARCH_ROOT", research)
    monkeypatch.setattr(publish, "WORK_ROOT", repo / "work_zsw")
    monkeypatch.setattr(
        publish,
        "load_config",
        lambda require_frozen=True: (
            {
                "publication": {
                    "remote": "origin",
                    "commit_prefix": "autoresearch: hinter round",
                }
            },
            "config",
        ),
    )
    monkeypatch.setattr(publish, "_push", lambda *args: None)
    receipt = publish.publish_round(0)
    assert receipt["round"] == 0
    names = subprocess.run(
        ["git", "show", "--pretty=", "--name-only", receipt["commit"]],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "work_zsw/research/final_book.json" in names
    assert "work_zsw/research/STOPPED.json" in names
    assert (repo / "work_zsw/pool/control/STOP").is_file()
