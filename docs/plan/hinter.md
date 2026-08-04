# Autoresearch: Parallel 10-Hint Book Pilot

## High-level idea

Start with 10 training questions and maintain one evolving hint for each question, forming a 10-entry hint book. In every round, 10 subagents revise the hints in parallel using only their assigned training question and its complete held-out-safe history from all previous rounds. The held-out questions are strictly invisible to all subagents: they are used only by a private evaluator, which returns aggregate transfer scores. Each revision is kept or discarded independently, so the hint book gradually becomes a collection of reusable mathematical strategies rather than problem-specific solutions.

You are the autonomous autoresearch teacher. Maintain exactly 10 hints for a fixed Qwen3-1.7B student.

## Data

Use:

```text
zjhhhh/DeepScaleR-Qwen3-1.7B-2k-strategy-error-200
```

Fixed zero-indexed rows:

```text
train:
[163, 28, 6, 189, 70, 62, 57, 35, 188, 26]

heldout:
[173, 139, 22, 151, 108, 8, 7, 23, 55, 59]
```

Never resample. Pin the dataset revision and fingerprint.

## Hint book

The book always contains exactly 10 hints.

```text
train question 1 -> hint 1
...
train question 10 -> hint 10
```

Each hint is permanently assigned to its training question. A hint may be revised, but not reassigned.

Limits:

```text
10 hints
256 tokens maximum per hint
2048 tokens maximum total
```

Hints should describe reusable strategies, not answers or full problem-specific solutions.

## Student sampling implementation

For sampling **without a hint**, reuse the existing response-sampling code in
this repository. Keep **thinking mode enabled** and use the repository
evaluator's default `max_tokens = 16384`.

For sampling **with a hint**, use the same sampling code and identical settings; only add the hint to the student prompt. Do not create a separate inference implementation or change decoding settings between no-hint and hint-assisted sampling.

All such sampling must follow the one-GPU Slurm rule below.

## Student and grader

Use the project Qwen3-1.7B checkpoint, or `Qwen/Qwen3-1.7B` if none is configured. Freeze all inference settings before research. Generate 8 rollouts for every evaluated `(question, hint)` pair. Hint-assisted sampling must differ from the repository's no-hint sampling only by inserting the hint into the prompt.

Use the exact grader from `radixark/miles` commit `9437366e0`:

```python
from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
```

Do not modify it.

Define:

```text
accuracy(q, h) = mean grader reward over 8 rollouts
```

## Held-out invisibility rule

The 10 held-out questions are evaluator-private.

No subagent may see or receive:

- held-out problem text;
- held-out answers or reference solutions;
- held-out student rollouts;
- per-question or per-rollout held-out rewards;
- which held-out questions a hint helped or failed on.

A subagent may receive only its own aggregate `heldout_i` and `J_i` after proposing a revision. It must revise its hint using training evidence only. The main autoresearch process must enforce this separation and must not copy hidden evaluator details into prompts, logs, summaries, or subagent context.

## Per-hint objective

Each hint has its own score.

For hint `h_i` assigned to training question `q_i`:

```text
train_i = accuracy(q_i, h_i)

heldout_i = mean over all 10 held-out questions q of accuracy(q, h_i)

J_i = train_i + lambda * heldout_i
```

Freeze `lambda` before setup and never change it within a run. The default
`work_zsw` run uses `lambda = 1`; the isolated comparison runs use
`lambda = 2`, `5`, and `10`.

The main process and all subagents may inspect the assigned training question, answer, rollouts, and grader results. They must not inspect held-out question text, answers, rollouts, or per-question results. The private evaluator reveals only `heldout_i` and `J_i` for each hint.

## Slurm GPU rule

Every subagent must obtain student samples through **Slurm**. Never run student inference directly on a login node.

For each round, subagent `i` must:

1. submit a Slurm job for its assigned training question and current hint;
2. request exactly **1 GPU**;
3. generate the 8 student rollouts inside that job;
4. monitor the job until completion and inspect its saved training artifacts;
5. use those training rollouts and grader results to propose one revision of `hint_i`.

Launch the 10 one-GPU jobs independently so they may run in parallel. Reuse the repository's existing `sbatch` template or Slurm script rather than inventing cluster-specific directives. Record the Slurm job ID, output path, exit status, and runtime.

If a job is pending, wait for it rather than replacing it with local inference. If it crashes, inspect the Slurm logs, fix only implementation issues, and resubmit the same experiment. A subagent's Slurm job may access only its assigned training question; it must never load held-out data.

All later student sampling used to score proposed hints must also run through Slurm with exactly one GPU per job. Held-out scoring remains inside the private evaluator and exposes only aggregate `heldout_i` and `J_i`.

## Parallel autoresearch round

In every round, launch **10 subagents in parallel**, one per hint.

Subagent `i` receives:

- its assigned training question and answer;
- the current `hint_i`;
- its previous `train_i`, aggregate `heldout_i`, and `J_i`;
- every held-out-safe artifact from all previous rounds for that same assigned
  question, exposed through its per-round `worker_history/hint_ii.json`
  manifest (including its own prior training rollouts, proposals, decisions,
  and aggregate-only private scores);
- the global hint token limits.

It must obtain fresh training rollouts and grader results by submitting its own one-GPU Slurm job as described above.

Each subagent proposes exactly one revised version of its own hint. It must not edit any other hint, read another hint's history, or access any held-out question or hidden held-out evaluation detail while forming the proposal.

After all 10 proposals return:

1. evaluate every proposed hint on its assigned training question;
2. privately evaluate every proposed hint on all 10 held-out questions;
3. compute the proposed `train_i`, `heldout_i`, and `J_i`;
4. compare each proposed hint only against its own incumbent;
5. decide **keep or discard independently for every hint**.

Keep proposed hint `i` when:

1. `J_i` is higher; or
2. `J_i` ties and `heldout_i` is higher; or
3. both tie and the proposed hint is shorter.

Otherwise discard it and keep the incumbent hint.

Some hints may advance while others are discarded in the same round. Assemble the next hint book from the 10 independent decisions, then start another parallel round.

Do not compare one hint against another and do not replace a hint with another hint's text.

## Setup

Before research:

1. implement the fixed split;
2. pin the student and grader revisions;
3. implement batched 8-rollout inference;
4. implement private per-hint held-out evaluation;
5. verify the pipeline with a Slurm GPU run;
6. run an independent Codex review for leakage and grader correctness.

After verification, freeze everything except the 10 hint texts.

The autoresearch agent is responsible only for optimizing the 10 hints on the train and held-out objective. Do not mention or access any separate evaluation set. Run all 20 rounds even if the search appears converged or several consecutive rounds keep no hints. After round 20, return the frozen hint book; evaluation will be performed later by a separate process. Only an explicit human stop may end the run before round 20.

## Logging

Because every round proposes 10 hint revisions, maintain both a **round-level summary** and a **per-hint decision log**.

### 1. Round summary

Append one row per round to:

```text
research/round_summary.csv
```

Columns:

```text
round
book_hash_before
book_hash_after
mean_train_before
mean_train_after
mean_heldout_before
mean_heldout_after
mean_J_before
mean_J_after
num_kept
num_discarded
total_tokens_before
total_tokens_after
elapsed_seconds
notes
```

Here:

```text
mean_train = mean_i train_i
mean_heldout = mean_i heldout_i
mean_J = mean_i J_i
```

### 2. Per-hint proposal log

Append exactly 10 rows per round to:

```text
research/hint_history.csv
```

Columns:

```text
round
hint_id
train_qid
sampling_slurm_job_id
proposal_eval_slurm_job_id
incumbent_hash
proposal_hash
final_hash
old_train
new_train
delta_train
old_heldout
new_heldout
delta_heldout
old_J
new_J
delta_J
old_tokens
new_tokens
decision
mutation
subagent_summary
```

`final_hash` is the proposal hash when kept and the incumbent hash when discarded.

### 3. Immutable artifacts

For every round, save:

```text
research/rounds/<round>/book_before.json
research/rounds/<round>/book_proposals.json
research/rounds/<round>/book_after.json
research/rounds/<round>/metrics.json
research/rounds/<round>/worker_history/hint_<id>.json
```

`metrics.json` should contain all 10 old/proposed/final metric records and the round aggregates.

Also maintain:

```text
research/current_book.json
research/best_per_hint.json
```

`best_per_hint.json` records the current incumbent and full metric history for each of the 10 independently optimized hints.

Never log held-out problem contents, answers, rollouts, per-question held-out scores, or winning held-out question–hint pairs.

## No-hack rules

Never inspect hidden held-out details. Never change the model, split, grader, rollout count, configured lambda, objective, or scoring procedure after research starts. Never put answers in hints or silently drop questions.

Run exactly 20 parallel rounds. A zero-keep round, any number of consecutive
zero-keep rounds, or apparent convergence is not a stopping condition. The
only permitted early termination is an explicit human stop.

Then save and return the final frozen 10-hint book. Do not run any separate evaluation.
