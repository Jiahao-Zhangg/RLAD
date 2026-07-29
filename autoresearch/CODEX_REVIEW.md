# Independent Codex freeze review

This review is a mandatory gate from `docs/plan/hinter.md`. Run it in a
separate Codex process after both Slurm smoke tasks pass and before invoking
`autoresearch.hinter.freeze`.

The reviewer may inspect:

- all source under `autoresearch/`;
- `docs/plan/hinter.md`;
- `train/rl/eval/vllm_eval.py`;
- `train/rl/rlad_plugin/templates.py`;
- held-out-safe setup manifests, smoke aggregates, and one-GPU receipts under
  `work_zsw/research/setup/`.

The reviewer must not inspect:

- the source dataset or Hugging Face dataset cache;
- held-out row contents or answers;
- private evaluator internals beyond source review;
- private rollouts, per-question scores, or private task logs;
- any prior experiment’s hidden artifacts.

Audit all of the following:

1. Fixed dataset revision/fingerprint/hash and fixed train/held-out indices.
2. Qwen3-1.7B model revision, thinking-on prompt rendering, `n=8` decoder, and
   no application output cap.
3. Both prompt conditions call the repository’s shared sampling primitive and
   differ only by inserting the hint.
4. Exact clean `radixark/miles` commit and DeepScaleR source hashes.
5. Every student call requires a numeric Slurm step and exactly one visible
   GPU.
6. The parent allocation is exactly the two named exclusive 8×H100 nodes and
   allows at most sixteen one-GPU steps.
7. Training workers receive one public packet only. Private output is
   aggregate-only and never contains held-out text, answers, rollouts, or
   per-question rewards.
8. Hint assignment, token/leak gates, exact count-derived metrics, independent
   keep/discard comparisons, and tie-break ordering.
9. Three-zero-keep/six-round stopping behavior and immutable round artifacts.
10. Git publication's explicit allowlist, transaction-authorized staged-index
    recovery, exact GitHub host check, and ignored private/runtime paths.
11. Crash recovery and duplicate-task/duplicate-push prevention.

Obtain the exact source/config identity with:

```bash
uv run --project autoresearch \
  python -m autoresearch.hinter.bootstrap review-material
```

On a genuine PASS, produce
`work_zsw/review/independent_review.json` with exactly these keys:

```text
schema_version
verdict
reviewer
reviewed_epoch
config_hash
source_sha256
source_bundle_hash
heldout_details_inspected
findings
```

Required constants are `schema_version=1`, `verdict="pass"`,
`reviewer="codex"`, and `heldout_details_inspected=false`. Copy the exact
config/source values from `review-material`; do not reconstruct them.
