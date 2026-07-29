# `work_zsw` runtime workspace

This directory is populated by `autoresearch/run_hinter_agent.py` on the
two-node H100 cluster.

Only held-out-safe aggregate research ledgers are eligible for Git tracking:
the current/best hint books, round summaries, per-hint decision history, and
the four public artifacts for each completed round. Dataset caches, dependency
clones, worker packets and rollouts, private evaluator artifacts, Slurm queues
and logs, review scratch space, and the SDK transcript are ignored.

Do not manually place held-out questions, answers, rollouts, or per-question
scores here.
