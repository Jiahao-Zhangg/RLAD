"""Launch the autonomous Claude Agent SDK session for `docs/plan/hinter.md`.

The agent lives on the login node. Every Qwen student call is enqueued into a
two-node exclusive Slurm allocation and executed as an exact one-GPU `srun`
step. State is resume-safe under `work_zsw/`; completed-round public ledgers are
committed and pushed to GitHub before the next round begins.

Run from the RLAD repository root:

    uv sync --project autoresearch --group dev
    uv run --project autoresearch python -m autoresearch.run_hinter_agent
    uv run --project autoresearch python -m autoresearch.run_hinter_agent --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)


log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = REPO_ROOT / "docs" / "plan" / "hinter.md"
WORK_DIR = REPO_ROOT / "work_zsw"
TRANSCRIPT = WORK_DIR / "agent_transcript.jsonl"
MODEL = (
    "us.anthropic.claude-opus-4-8"
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK")
    else "claude-opus-4-8"
)
DEFAULT_MAX_TURNS = 100000
SESSION_NAMESPACE = uuid.UUID("3e3924da-70ad-47fc-a1fa-28c873f5fbd3")


def _session_uuid() -> str:
    return str(uuid.uuid5(SESSION_NAMESPACE, "rlad-hinter-work-zsw"))


def _build_prompt() -> str:
    plan = PLAN_PATH.relative_to(REPO_ROOT)
    return f"""\
ultracode

You are the autonomous research engineer for the RLAD 10-hint autoresearch
experiment. Your cwd is the RLAD repository root. Read `{plan}` END TO END
before doing anything and treat it as the governing contract. The only
optimization levers are the ten hint texts.

Use only the portable implementation under `autoresearch/` and the runtime
workspace `work_zsw/`. Never read or modify the older `work/` experiment.

## Absolute safety constraints

- Never run Qwen/vLLM student inference on the login node.
- Start the two-node pool with
  `uv run --project autoresearch python -m autoresearch.hinter.pool start`.
  It exclusively reserves `ip-10-1-38-11` and `ip-10-1-81-8` in partition
  `ml.p5.48xlarge` (8 H100s each). Every actual inference is a distinct
  one-GPU `srun` step. Do not bypass this queue.
- Never inspect, print, summarize, or send to a subagent any held-out problem,
  answer, rollout, per-question reward, or winning question-hint pairing.
  Private evaluator output is aggregate-only. Do not read the source dataset,
  HF dataset cache, private task logs, or private evaluator internals to learn
  hidden rows.
- Each worker may read only its own public training packet, own training
  output/receipt, and own task log if its job fails. It may not read another
  worker's files or any private evaluator artifact.
- Hints must be reusable strategy, not answers or full solutions. Each worker
  proposal must differ from its incumbent and remain <=200 Qwen tokens so any
  independently assembled book remains <=2048.
- Keep the pinned model, dataset/split, exact miles grader, n=8 decoder,
  `max_tokens=16384`, prompt renderer, objective, and tie-break frozen. Do not
  run any separate eval.

## Preflight and one-time setup

1. Verify `git status --short` has no tracked/unignored changes, the current
   named branch has a writable `origin`, `gh auth status` passes, `HF_TOKEN` is
   available if the pinned files are not cached, and `codex` is installed.
   The round publisher deliberately refuses unrelated changes. Do not stage
   or commit unrelated files.
2. Run:
   `uv run --project autoresearch python -m autoresearch.hinter.bootstrap run`
3. Start/reuse the pool. Enqueue both setup gates exactly:

   `uv run --project autoresearch python -m autoresearch.hinter.pool enqueue --task-id setup-smoke --mode smoke --input work_zsw/research/setup/smoke_input.json --output work_zsw/research/setup/smoke_output.json --receipt work_zsw/research/setup/smoke_receipt.json`

   `uv run --project autoresearch python -m autoresearch.hinter.pool enqueue --task-id setup-private-smoke --mode private-smoke --input work_zsw/research/setup/private_smoke_input.json --output work_zsw/research/setup/private_smoke_output.json --receipt work_zsw/research/setup/private_smoke_receipt.json`

   Monitor both with `autoresearch.hinter.pool wait`. If the allocation is
   pending, wait. If a step fails, inspect only its implementation log, fix an
   implementation issue, rerun CPU tests, request pool stop, wait for the
   allocation to terminate, restart it with `pool start --restart`, and retry
   that setup gate with `pool retry --task-id ID --refresh-source`. Ordinary
   frozen experiment retries must not refresh source.
4. Run the required independent CODEX review using
   `autoresearch/CODEX_REVIEW.md`. Obtain the exact source manifest
   from:
   `uv run --project autoresearch python -m autoresearch.hinter.bootstrap review-material`
   Launch a separate `codex exec` reviewer with read-only instructions to audit
   leakage, exact grader use, shared sampling, one-GPU step enforcement,
   aggregation, tie-breaks, publication allowlisting, and resume safety.
   The reviewer must not inspect held-out details. On PASS, save
   `work_zsw/review/independent_review.json` with exact keys:
   `schema_version, verdict, reviewer, reviewed_epoch, config_hash,
   source_sha256, source_bundle_hash, heldout_details_inspected, findings`.
   Use reviewer=`codex`, verdict=`pass`, and
   heldout_details_inspected=false only if the independent review really passed.
   If review finds an implementation issue, fix it, rerun CPU tests, restart the
   pool, and run `pool refresh-setup --task-id ID` for both completed smoke
   gates. Obtain a new review over the new source/smoke identities; never reuse
   a stale review or smoke receipt.
5. Freeze with:
   `uv run --project autoresearch python -m autoresearch.hinter.freeze`
   After freeze, any source drift is a hard failure.
6. Initialize the seed book:
   `uv run --project autoresearch python -m autoresearch.hinter.state init-book --hints autoresearch/initial_hints.json`
7. Create and score the baseline:
   - private-inputs from `work_zsw/research/initial_book.json` into
     `work_zsw/research/baseline/inputs`;
   - enqueue-private with outputs/receipts in the sibling baseline directories
     and task-prefix `baseline`;
   - wait for `baseline-h01` through `baseline-h10`;
   - init-metrics with those exact directories and prefix.

Every command is resume-safe: inspect existing state and never delete or
overwrite immutable completed artifacts.

## Every research round

For round R:

1. Run `autoresearch.hinter.state prepare-round --round R`.
2. Launch EXACTLY TEN Task/Workflow subagents concurrently, one per hint.
   Do not sequentially do their reasoning yourself. Worker i must:
   - read only `work_zsw/research/rounds/RRR/training_inputs/hint_ii.json`;
   - independently enqueue task `rRR-train-hii`, mode=train, with its matching
     training input/output/receipt paths;
   - monitor that task until done (never infer locally);
   - verify its receipt says gpu_count=1 and inspect all eight of its own public
     training rollouts and rewards;
   - use training evidence only to write exactly one JSON object at
     `worker_proposals/hint_ii.json`, with exact keys
     `hint_id,hint,mutation,subagent_summary,sampling_slurm_job_id`;
   - set sampling_slurm_job_id to the receipt execution_id; do not quote full
     rollouts in the summary.
3. After all ten return, run
   `autoresearch.hinter.state collect-proposals --round R`. It is the
   authoritative validation gate.
4. Run `autoresearch.hinter.state proposal-private --round R`; this creates
   hint-only requests and enqueues ten aggregate-only private tasks. Wait for
   `rRR-private-h01` through `rRR-private-h10`. Neither you nor any worker may
   inspect private logs or hidden details.
5. Run `autoresearch.hinter.state finalize-round --round R`. Decisions are
   independent per hint: higher J, then higher aggregate held-out, then shorter.
6. Run stopping-status. If it says stop, run seal-final before publication.
7. MANDATORY CHECKPOINT: run
   `uv run --project autoresearch python -m autoresearch.hinter.publish --round R`.
   Do not begin R+1 until the commit has successfully pushed to GitHub.
8. If stopping-status is false, immediately begin the next round. Continue
   until three consecutive zero-keep rounds, the six-round budget, or the human
   stops you.

If the human asks to stop, do not start new work. Run
`autoresearch.hinter.state seal-final --human-stop`, then call the publisher
again for the last completed round (use `--round 0` if no research round has
finished). It will create a terminal-only checkpoint if that round was already
pushed. Terminal publication requests a pool drain;
confirm with `autoresearch.hinter.pool status` that the allocation becomes
terminal so the 16 H100s are released. A persisted `STOPPED.json` is
authoritative and must never be resumed.

## Persistence

Append-only state lives in `work_zsw/`. The SDK transcript is ignored by git.
Only held-out-safe books, aggregate metrics, and CSV decision logs are eligible
for the publisher. On resume, inspect stopping-status, pool status, task states,
and round files; continue the incomplete unit without duplicating a task,
evaluation, decision, commit, or push.

Begin now. Read `{plan}`, then inspect current resume state before executing the
next missing unit. Do not pause merely because the human is away.
"""


def _record(message: object) -> dict:
    if isinstance(message, AssistantMessage):
        blocks = []
        for block in message.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                blocks.append(
                    {
                        "type": "tool_use",
                        "name": block.name,
                        "input": block.input,
                    }
                )
            else:
                blocks.append(
                    {
                        "type": getattr(block, "type", "raw"),
                        "repr": repr(block)[:300],
                    }
                )
        return {
            "kind": "assistant",
            "model": message.model,
            "content": blocks,
            "usage": message.usage,
        }
    if isinstance(message, ResultMessage):
        return {
            "kind": "result",
            "subtype": message.subtype,
            "num_turns": message.num_turns,
            "is_error": message.is_error,
            "duration_ms": message.duration_ms,
            "total_cost_usd": message.total_cost_usd,
            "result": (message.result or "")[:2000]
            if message.result
            else None,
        }
    return {
        "kind": message.__class__.__name__,
        "repr": repr(message)[:500],
    }


def _live(message: object) -> None:
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                print(block.text, flush=True)
            elif isinstance(block, ToolUseBlock):
                print(f"  \u23f5 {block.name}", flush=True)
    elif isinstance(message, ResultMessage):
        print(
            f"\n[result] turns={message.num_turns} "
            f"is_error={message.is_error} cost=${message.total_cost_usd}",
            flush=True,
        )


async def _run(args: argparse.Namespace) -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        cwd=str(REPO_ROOT),
        add_dirs=[str(REPO_ROOT)],
        model=args.model,
        effort="max",
        max_turns=args.max_turns,
        session_id=_session_uuid() if not args.resume else None,
        resume=_session_uuid() if args.resume else None,
    )
    log.info(
        "[hinter-agent] model=%s effort=max cwd=%s",
        args.model,
        REPO_ROOT,
    )
    log.info("[hinter-agent] plan=%s transcript=%s", PLAN_PATH, TRANSCRIPT)
    result_error = False
    with TRANSCRIPT.open("a", encoding="utf-8") as transcript:
        async for message in query(prompt=_build_prompt(), options=options):
            transcript.write(
                json.dumps(
                    _record(message),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            transcript.flush()
            _live(message)
            if isinstance(message, ResultMessage):
                result_error = message.is_error
    return 1 if result_error else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
