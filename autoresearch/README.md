# RLAD two-node hint-book autoresearch

This directory is the portable implementation of
[`docs/plan/hinter.md`](../docs/plan/hinter.md). It is intended for the
cluster that exposes two exclusive `p5.48xlarge` nodes:

```text
ip-10-1-38-11
ip-10-1-81-8
```

The login-node Claude Agent SDK session drives the research loop. A persistent
Slurm allocation reserves both nodes (16 H100s total), while every Qwen
student evaluation runs as a distinct `srun` step with exactly one visible
GPU. The ten-hint protocol therefore runs all ten independent hint pipelines
concurrently without altering the fixed `n=8` sampling semantics. Six pool
slots remain available for overlap/recovery; manufacturing redundant samples
merely to occupy them would violate the frozen experiment.

## Install and launch

From the RLAD repository root on the target machine:

```bash
uv sync --project autoresearch --group dev
uv run --project autoresearch python -m pytest autoresearch/tests -q
uv run --project autoresearch python -m autoresearch.run_hinter_agent
```

Launch from a clean named branch whose implementation commit is already on a
writable GitHub `origin`. Runtime publication intentionally refuses to absorb
uncommitted source or unrelated user changes.

Resume the deterministic SDK session with:

```bash
uv run --project autoresearch python -m autoresearch.run_hinter_agent --resume
```

### Slurm coordinator wrapper

To keep the SDK coordinator itself under Slurm, submit the CPU-only wrapper
from the RLAD repository root:

```bash
sbatch --partition=<cpu-or-service-partition> \
  --export=ALL,RLAD_REPO_ROOT="$PWD" \
  autoresearch/jobs/hinter_agent.sbatch
```

Resume an interrupted SDK session with:

```bash
sbatch --partition=<cpu-or-service-partition> \
  --export=ALL,RLAD_REPO_ROOT="$PWD" \
  autoresearch/jobs/hinter_agent.sbatch --resume
```

The coordinator partition must not be `ml.p5.48xlarge` and must not place the
wrapper on either named H100 host. Otherwise the coordinator would occupy a
node while waiting for its own exclusive two-node pool, causing a nested-job
deadlock. The wrapper intentionally has no `#SBATCH --partition` line because
the repository does not know this cluster's CPU/service partition name. It
requests four CPUs and 16 GiB, installs the frozen environment, runs the CPU
tests, and launches the SDK session. On interruption or an incomplete agent
exit, it asks the GPU pool to drain so the H100 allocation is not abandoned.

The runner mirrors
`/home/jiahao/shared/AutoTeacher/curation/run_refine_codebook_agent.py`:
`ClaudeAgentOptions` uses the repository root as `cwd`, maximum effort,
`bypassPermissions`, a stable session UUID, an append-only JSONL transcript,
and a long turn backstop.

If an implementation fix is made after a smoke gate completed but before
freeze, `autoresearch.hinter.pool refresh-setup --task-id setup-smoke` (and the
corresponding private gate) archives the old attempt and recreates it with the
new source identity. Source refresh is refused after freeze.

## Layout

```text
autoresearch/
  config.json                 frozen model/data/grader/decoder/cluster pins
  initial_hints.json          answer-free seed book
  run_hinter_agent.py         Claude Agent SDK entry point
  hinter/
    bootstrap.py              pinned setup and review material
    pool.py                   two-node allocation and one-GPU task queue
    job.py                    train/smoke/private GPU task entry point
    state.py                  book, metrics, decisions, and stopping
    freeze.py                 setup/review/smoke freeze gate
    publish.py                allowlisted per-round Git commit/push
  jobs/hinter_agent.sbatch     CPU/service-node SDK coordinator
  jobs/hinter_pool.sbatch      exclusive two-node H100 worker pool
  tests/

work_zsw/
  research/                   durable experiment state
  pool/                       ignored task queue
  logs/                       ignored Slurm logs
  review/                     ignored independent-review scratch/receipt
  publication/                ignored push receipts
  agent_transcript.jsonl      ignored SDK transcript
```

## GPU scheduling

`python -m autoresearch.hinter.pool start` submits one allocation with:

```text
partition:       ml.p5.48xlarge
nodes:           2
nodelist:        ip-10-1-38-11,ip-10-1-81-8
exclusive:       yes
GPUs per node:   8
pool slots:      16
```

Tasks are immutable JSON requests. The parent requests all eight GPUs on each
exclusive node. The dispatcher owns sixteen fixed `(node, slot)` positions and
launches each task with `srun --exclusive --exact --nodelist=<node>
--gpus-per-task=1 --gpu-bind=single:1`. The GPU entry point refuses to run
without a numeric Slurm step, `SLURM_GPUS_PER_TASK=1`, and exactly one
`CUDA_VISIBLE_DEVICES` entry. Each receipt binds the input hash, allocation ID,
unique step ID, named node, pool slot, Slurm GPU grant, visible device,
source/config hashes, timing, and output hash.

Baseline and proposal held-out tasks are accepted only at their registered
paths and only when their hint identity exactly matches the initial book or
that round's proposal book. Round task IDs are limited to the frozen six-round
budget. Once `STOPPED.json` exists, enqueue, retry, restart, and task claiming
all refuse further work; active steps drain and the allocation exits.

## Privacy and GitHub publication

Public workers receive only one assigned training question/answer, their
incumbent hint and aggregate incumbent metrics, and their own eight training
rollouts/rewards. The private evaluator loads held-out rows internally and
writes only per-hint aggregate counts and objective values.

After every finalized round, `autoresearch.hinter.publish` stages an explicit
allowlist:

- initial/current/best hint books;
- round and per-hint aggregate CSV ledgers;
- `book_before.json`, `book_proposals.json`, `book_after.json`, and
  `metrics.json` for that round;
- final book/stop receipt on the terminal round.

It refuses unrelated or pre-staged changes, requires authenticated GitHub
access, force-adds only that explicit allowlist from the otherwise ignored
workspace, commits one round, and pushes the current named branch. A durable,
content-hashed Git transaction permits exact recovery after interruption
during staging, commit, or push; recovery never searches another branch. Every
effective Git push URL must resolve to the exact `github.com` host. On a human
stop after a round was already pushed, it creates a separate idempotent terminal
checkpoint containing `final_book.json` and `STOPPED.json`.
Successful terminal publication also tells the pool to stop claiming work and
release the two exclusive nodes after active steps finish. `.gitignore` keeps
every other worker packet/rollout, private evaluator artifact, queue, log,
cache, dependency clone, review file, and transcript out of Git.

A human stop after baseline but before round 1 is published with
`autoresearch.hinter.publish --round 0`; this checkpoints the frozen seed/final
book and releases the allocation without requiring nonexistent round files.

The same-account filesystem is an honest-worker boundary, not a hostile
sandbox. The SDK prompt and validators prevent accidental leakage; they cannot
make a malicious process unable to bypass Unix permissions.
