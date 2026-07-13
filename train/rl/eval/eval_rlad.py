"""RLAD dual evaluation — the PRIMARY metric (method_spec §5.1).

Reports three numbers per benchmark for a (pi_abs, pi_sol) pair:
  - w/o abs       : pi_sol(y|x), n samples/problem
  - w/ abs (avg)  : per problem, average pass@1 over K=4 abstractions z~pi_abs(.|x),
                    each with n samples/problem conditioned via render_prompt_with_abstraction
  - w/ abs (best) : per problem, the best of the K abstractions
PRIMARY = AIME2025 w/abs(avg) (aime2025_wabs_avg_acc). Grading reuses the validated
eval/vllm_eval deepscaler grader. Decoding temp 0.6 / top_p 0.95, n>=32, 32K budget.

Two models can't share one GPU cleanly, so three RESUMABLE stages (separate processes):
  gen-abs   : load pi_abs -> OUT/abstractions.jsonl {id, abs_idx, abstraction}
  solve     : load pi_sol -> OUT/solve_samples.jsonl {id, cond(woabs|abs<k>), sample_idx, correct}
              (cond set = {woabs} + {abs0..abs{K-1}} per problem; resumable by (id,cond))
  summarize : strictly validate all expected samples, then write OUT/summary.json
  validate  : recompute and verify an existing summary without loading either model
  compare   : combine untrained/RFT summaries into five metrics and log them to W&B

Usage (host vllm env), e.g. RLAD model:
  python eval_rlad.py gen-abs   --absgen-hf <pi_abs_hf> --benchmark aime25 --out OUT --k 4
  python eval_rlad.py solve     --solgen-hf <pi_sol_hf> --benchmark aime25 --out OUT --n 32 --max-tokens 32768
  python eval_rlad.py summarize --benchmark aime25 --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent          # .../train/rl
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
if str(_CODE / "eval") not in sys.path:
    sys.path.insert(0, str(_CODE / "eval"))

BENCH_DIR = Path(os.environ.get("RLAD_DATA", _CODE / "data")) / "benchmarks"


def _read(path, *, repair_trailing=False):
    """Read JSONL, optionally removing one truncated final record after a job kill."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [(number, line) for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(keepends=True), start=1
    ) if line.strip()]
    rows = []
    for offset, (number, line) in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            final_unterminated = offset == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if repair_trailing and final_unterminated:
                _atomic_jsonl(path, rows)
                print(f"resume: removed a truncated final JSONL record from {path}")
                break
            raise SystemExit(f"invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def _read_solve(out):
    """All solve rows across the single-file and N-way-sharded layouts (interchangeable)."""
    out = Path(out)
    single = sorted(out.glob("solve_samples.jsonl"))
    shards = sorted(out.glob("solve_samples.shard*.jsonl"))
    if single and shards:
        raise SystemExit(f"mixed sharded and unsharded solve outputs under {out}")
    rows = []
    for p in single + shards:
        rows += _read(p)
    return rows


def _solve_path(out, shard, num_shards):
    out = Path(out)
    return out / ("solve_samples.jsonl" if num_shards <= 1 else f"solve_samples.shard{shard}.jsonl")


def _load_problems(bench):
    p = BENCH_DIR / f"{bench}.jsonl"
    if not p.exists():
        sys.exit(f"benchmark not found: {p}")
    return _read(p)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _abstractions(out, benchmark, k, *, repair=False, require_complete=False):
    """Return id -> abs_idx -> text, optionally dropping interrupted per-id groups."""
    path = Path(out) / "abstractions.jsonl"
    problems = _load_problems(benchmark)
    valid_ids = {problem["id"] for problem in problems}
    rows_in_file = _read(path, repair_trailing=repair)
    grouped = defaultdict(list)
    bad_ids = set()
    for row in rows_in_file:
        pid, idx, text = row.get("id"), row.get("abs_idx"), row.get("abstraction")
        if pid not in valid_ids or not isinstance(idx, int) or not 0 <= idx < k or not isinstance(text, str) or not text.strip():
            raise SystemExit(f"malformed abstraction row under {path}: {row}")
        if any(existing["abs_idx"] == idx for existing in grouped[pid]):
            bad_ids.add(pid)
        grouped[pid].append(row)

    complete = {}
    incomplete = set(valid_ids)
    expected_indices = set(range(k))
    for pid, rows in grouped.items():
        indices = {row["abs_idx"] for row in rows}
        if pid not in bad_ids and len(rows) == k and indices == expected_indices:
            complete[pid] = {row["abs_idx"]: row["abstraction"] for row in rows}
            incomplete.discard(pid)
        else:
            bad_ids.add(pid)

    if repair and bad_ids:
        kept = [row for row in rows_in_file if row["id"] in complete]
        _atomic_jsonl(path, kept)
        print(f"gen-abs: removed interrupted abstraction groups for {len(bad_ids)} problem(s)")
    elif bad_ids:
        raise SystemExit(f"incomplete or duplicate abstraction groups for {len(bad_ids)} problem(s)")
    if require_complete and incomplete:
        raise SystemExit(f"missing complete abstraction groups for {len(incomplete)} problem(s)")
    return complete


def _repair_solve_file(path, allowed_keys, n):
    """Drop only incomplete condition groups; reject rows outside this evaluation plan."""
    path = Path(path)
    rows = _read(path, repair_trailing=True)
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("id"), row.get("cond"))
        idx, correct = row.get("sample_idx"), row.get("correct")
        if key not in allowed_keys or not isinstance(idx, int) or not 0 <= idx < n or correct not in (0, 1):
            raise SystemExit(f"malformed or stale solve row under {path}: {row}")
        grouped[key].append(row)
    expected_indices = set(range(n))
    complete = {
        key for key, values in grouped.items()
        if len(values) == n and {row["sample_idx"] for row in values} == expected_indices
    }
    partial = set(grouped) - complete
    if partial:
        _atomic_jsonl(path, [row for row in rows if (row["id"], row["cond"]) in complete])
        print(f"solve: removed {len(partial)} interrupted condition group(s) from {path}")
    return complete


def _strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def stage_gen_abs(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_absgen_prompt

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    abs_path = out / "abstractions.jsonl"
    problems = _load_problems(args.benchmark)
    done = set(_abstractions(args.out, args.benchmark, args.k, repair=True))
    todo = [p for p in problems if p["id"] not in done]
    print(f"gen-abs: {len(problems)} problems, {len(done)} done, {len(todo)} to do")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.absgen_hf, trust_remote_code=True)
    llm = LLM(model=args.absgen_hf, tensor_parallel_size=args.tp, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=0.7, top_p=0.95, max_tokens=args.abs_max_tokens, seed=args.seed)
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        outs = llm.generate([render_absgen_prompt(tok, p["problem"]) for p in chunk], sp)
        if len(outs) != len(chunk):
            raise RuntimeError(f"vLLM returned {len(outs)} prompt outputs; expected {len(chunk)}")
        generated = []
        for p, output in zip(chunk, outs):
            if len(output.outputs) != args.k:
                raise RuntimeError(
                    f"vLLM returned {len(output.outputs)} abstractions for {p['id']}; expected {args.k}"
                )
            texts = [_strip_think(candidate.text) for candidate in output.outputs]
            if any(not text for text in texts):
                raise RuntimeError(f"vLLM returned an empty abstraction for {p['id']}")
            generated.extend(
                {"id": p["id"], "abs_idx": index, "abstraction": text}
                for index, text in enumerate(texts)
            )
        with abs_path.open("a", encoding="utf-8") as handle:
            for row in generated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        print(f"  gen-abs: +{len(chunk)} problems [{i + len(chunk)}/{len(todo)}]")
    _abstractions(args.out, args.benchmark, args.k, require_complete=True)
    print(f"gen-abs -> {abs_path}")


def stage_solve(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_prompt, render_prompt_with_abstraction
    from vllm_eval import grade_response

    out = Path(args.out)
    problems = {p["id"]: p for p in _load_problems(args.benchmark)}
    abs_by_id = _abstractions(out, args.benchmark, args.k, require_complete=True)
    tok = AutoTokenizer.from_pretrained(args.solgen_hf, trust_remote_code=True)

    # build the full (id,cond) work list, then take this shard's deterministic slice (data-parallel
    # across GPUs); skip items already done in THIS shard's file (resume across cancels/timeouts)
    allwork = []   # (id, cond, prompt, gold)
    for pid, p in problems.items():
        if not args.skip_woabs:
            allwork.append((pid, "woabs", render_prompt(tok, p["problem"]), p["answer"]))
        for k, z in abs_by_id.get(pid, {}).items():
            allwork.append((pid, f"abs{k}", render_prompt_with_abstraction(tok, p["problem"], z), p["answer"]))
    allwork.sort(key=lambda w: (w[0], w[1]))   # stable order so the shard slice is deterministic
    shard_work = [w for i, w in enumerate(allwork) if i % args.num_shards == args.shard]
    samples_path = _solve_path(out, args.shard, args.num_shards)
    allowed = {(w[0], w[1]) for w in shard_work}
    done = _repair_solve_file(samples_path, allowed, args.n)
    work = [w for w in shard_work if (w[0], w[1]) not in done]
    print(f"solve[shard {args.shard}/{args.num_shards}]: {len(work)} (problem,condition) items to generate "
          f"({len(done)} done, {len(allwork)} total)")
    if not work:
        return
    llm = LLM(model=args.solgen_hf, tensor_parallel_size=args.tp, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    # chunk so a 4h timeout resumes at the (id,cond) level instead of losing the whole solve
    for i in range(0, len(work), args.chunk):
        ch = work[i:i + args.chunk]
        outs = llm.generate([w[2] for w in ch], sp)
        if len(outs) != len(ch):
            raise RuntimeError(f"vLLM returned {len(outs)} prompt outputs; expected {len(ch)}")
        with samples_path.open("a", encoding="utf-8") as fh:
            for (pid, cond, _, gold), o in zip(ch, outs):
                if len(o.outputs) != args.n:
                    raise RuntimeError(f"vLLM returned {len(o.outputs)} samples for {(pid, cond)}; expected {args.n}")
                for si, c in enumerate(o.outputs):
                    correct, *_ = grade_response(c.text, gold)
                    fh.write(json.dumps({"id": pid, "cond": cond, "sample_idx": si, "correct": int(correct)}) + "\n")
            fh.flush()
        print(f"  solve: +{len(ch)} (problem,cond) items [{i + len(ch)}/{len(work)}]")
    print(f"solve -> {samples_path}")


def stage_joint(args):
    """RLAD-Joint NATIVE eval: single model, ONE combined-instruction generation per
    sample (render_joint_prompt = write a <cheatsheet> then solve), graded on the boxed answer.
    This is RLAD-Joint's TRAINING distribution (unlike the separate hint/solve prompts of the dual eval).
    Writes cond='joint' rows to the same solve_samples file; sharded + resumable like stage_solve."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_joint_prompt
    from vllm_eval import grade_response

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    problems = {p["id"]: p for p in _load_problems(args.benchmark)}
    tok = AutoTokenizer.from_pretrained(args.solgen_hf, trust_remote_code=True)

    allwork = [(pid, "joint", render_joint_prompt(tok, p["problem"]), p["answer"]) for pid, p in problems.items()]
    allwork.sort(key=lambda w: (w[0], w[1]))
    shard_work = [w for i, w in enumerate(allwork) if i % args.num_shards == args.shard]
    samples_path = _solve_path(out, args.shard, args.num_shards)
    allowed = {(w[0], w[1]) for w in shard_work}
    done = _repair_solve_file(samples_path, allowed, args.n)
    work = [w for w in shard_work if (w[0], w[1]) not in done]
    print(f"joint[shard {args.shard}/{args.num_shards}]: {len(work)} problems to generate "
          f"({len(done)} done, {len(allwork)} total)")
    if not work:
        return
    llm = LLM(model=args.solgen_hf, tensor_parallel_size=args.tp, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    for i in range(0, len(work), args.chunk):
        ch = work[i:i + args.chunk]
        outs = llm.generate([w[2] for w in ch], sp)
        if len(outs) != len(ch):
            raise RuntimeError(f"vLLM returned {len(outs)} prompt outputs; expected {len(ch)}")
        with samples_path.open("a", encoding="utf-8") as fh:
            for (pid, cond, _, gold), o in zip(ch, outs):
                if len(o.outputs) != args.n:
                    raise RuntimeError(f"vLLM returned {len(o.outputs)} samples for {(pid, cond)}; expected {args.n}")
                for si, c in enumerate(o.outputs):
                    correct, *_ = grade_response(c.text, gold)
                    fh.write(json.dumps({"id": pid, "cond": cond, "sample_idx": si, "correct": int(correct)}) + "\n")
            fh.flush()
        print(f"  joint: +{len(ch)} problems [{i + len(ch)}/{len(work)}]")
    print(f"joint -> {samples_path}")


def _build_summary(args):
    out = Path(args.out)
    rows = _read_solve(out)
    if not rows:
        sys.exit("run solve first")
    problems = [p["id"] for p in _load_problems(args.benchmark)]
    if args.mode == "dual":
        _abstractions(out, args.benchmark, args.k, require_complete=True)
        expected = {
            (pid, condition)
            for pid in problems
            for condition in ([f"abs{i}" for i in range(args.k)] + ([] if args.skip_woabs else ["woabs"]))
        }
    else:
        expected = {(pid, "joint") for pid in problems}

    acc = defaultdict(dict)   # id -> cond -> {sample_idx: correct}
    for r in rows:
        key = (r.get("id"), r.get("cond"))
        idx, correct = r.get("sample_idx"), r.get("correct")
        if key not in expected or not isinstance(idx, int) or not 0 <= idx < args.n or correct not in (0, 1):
            raise SystemExit(f"malformed or stale solve row: {r}")
        if idx in acc[key[0]].setdefault(key[1], {}):
            raise SystemExit(f"duplicate solve sample for {(key[0], key[1], idx)}")
        acc[key[0]][key[1]][idx] = correct
    missing = []
    for pid, condition in sorted(expected):
        indices = set(acc.get(pid, {}).get(condition, {}))
        if indices != set(range(args.n)):
            missing.append((pid, condition, len(indices)))
    if missing:
        raise SystemExit(f"evaluation is incomplete for {len(missing)} condition group(s); first={missing[0]}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    woabs, wabs_avg, wabs_best, joint = [], [], [], []
    per_problem = {}
    for pid in problems:
        conds = acc.get(pid, {})
        wo_values = list(conds.get("woabs", {}).values())
        abs = [mean(list(conds[f"abs{i}"].values())) for i in range(args.k)] if args.mode == "dual" else []
        joint_values = list(conds.get("joint", {}).values())
        wo = mean(wo_values)
        jt = mean(joint_values)
        if wo_values:
            woabs.append(wo)
        if abs:
            wabs_avg.append(sum(abs) / len(abs))
            wabs_best.append(max(abs))
        if joint_values:
            joint.append(jt)
        per_problem[pid] = {"woabs": round(wo, 4), "wabs_avg": round(sum(abs) / len(abs), 4) if abs else None,
                            "wabs_best": round(max(abs), 4) if abs else None, "n_abs": len(abs),
                            "joint": round(jt, 4) if joint_values else None}
    summary = {
        "benchmark": args.benchmark, "n_problems": len(problems), "mode": args.mode,
        "n_samples": args.n, "n_hints": args.k if args.mode == "dual" else 0,
        "skip_woabs": bool(args.skip_woabs),
        "woabs_pass1": round(100 * mean(woabs), 2),
        "wabs_avg_pass1": round(100 * mean(wabs_avg), 2),
        "wabs_best_pass1": round(100 * mean(wabs_best), 2),
        "joint_pass1": round(100 * mean(joint), 2),   # RLAD-Joint native combined-prompt (None-arm -> 0.0)
        "n_problems_woabs": len(woabs), "n_problems_wabs": len(wabs_avg), "n_problems_joint": len(joint),
        "per_problem": per_problem,
    }
    return summary


def stage_summarize(args):
    summary = _build_summary(args)
    _atomic_json(Path(args.out) / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_problem"}, indent=2))


def stage_validate(args):
    expected = _build_summary(args)
    path = Path(args.out) / "summary.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid evaluation summary {path}: {exc}") from exc
    if saved != expected:
        raise SystemExit(f"evaluation summary does not match raw samples: {path}")
    print(f"valid evaluation summary: {path}")


def build_comparison(untrained, rft, benchmark, input_key):
    for name, summary in (("untrained", untrained), ("rft", rft)):
        if summary.get("benchmark") != benchmark or summary.get("mode") != "dual":
            raise SystemExit(f"{name} summary does not describe dual {benchmark} evaluation")
    n_problems = untrained.get("n_problems")
    if not isinstance(n_problems, int) or n_problems <= 0 or rft.get("n_problems") != n_problems:
        raise SystemExit("untrained and RFT summaries do not cover the same nonempty benchmark")
    if untrained.get("n_problems_woabs") != n_problems or untrained.get("n_problems_wabs") != n_problems:
        raise SystemExit("untrained evaluation is missing no-hint or hint-conditioned problems")
    if rft.get("n_problems_woabs") != 0 or rft.get("n_problems_wabs") != n_problems:
        raise SystemExit("RFT evaluation must contain every hint-conditioned problem and no duplicate baseline")
    if untrained.get("n_samples") != rft.get("n_samples") or untrained.get("n_hints") != rft.get("n_hints"):
        raise SystemExit("untrained and RFT evaluation sampling configurations differ")

    metric_sources = {
        "base_without_hint_pass1": (untrained, "woabs_pass1"),
        "untrained_hint_avg_pass1": (untrained, "wabs_avg_pass1"),
        "untrained_hint_best_pass1": (untrained, "wabs_best_pass1"),
        "rft_hint_avg_pass1": (rft, "wabs_avg_pass1"),
        "rft_hint_best_pass1": (rft, "wabs_best_pass1"),
    }
    metrics = {}
    for name, (summary, source) in metric_sources.items():
        value = summary.get(source)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 100.0:
            raise SystemExit(f"invalid metric {source} in comparison input")
        metrics[name] = float(value)
    return {
        "schema": 1,
        "benchmark": benchmark,
        "n_problems": n_problems,
        "n_samples_per_condition": untrained["n_samples"],
        "n_hints": untrained["n_hints"],
        "unit": "percent",
        "input_key": input_key,
        **metrics,
    }


def stage_compare(args):
    def load_summary(directory, label):
        path = Path(directory) / "summary.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid {label} summary {path}: {exc}") from exc

    untrained = load_summary(args.untrained_out, "untrained")
    rft = load_summary(args.rft_out, "RFT")
    comparison = build_comparison(untrained, rft, args.benchmark, args.input_key)
    out = Path(args.out)
    pending = out / "summary.pending.json"
    _atomic_json(pending, comparison)

    import wandb

    metrics = {
        f"eval/{name}": comparison[name]
        for name in (
            "base_without_hint_pass1",
            "untrained_hint_avg_pass1",
            "untrained_hint_best_pass1",
            "rft_hint_avg_pass1",
            "rft_hint_best_pass1",
        )
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        mode="online",
        group=args.wandb_group,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="allow",
        config={
            "benchmark": args.benchmark,
            "n_problems": comparison["n_problems"],
            "n_samples_per_condition": comparison["n_samples_per_condition"],
            "n_hints": comparison["n_hints"],
            "untrained_absgen": args.untrained_absgen,
            "rft_absgen": args.rft_absgen,
            "solver_model": args.solver_model,
            "source_commit": args.source_commit,
            "input_key": args.input_key,
        },
    )
    run.log(metrics, step=0)
    for name, value in metrics.items():
        run.summary[name] = value
    resolved_entity = args.wandb_entity or getattr(run, "entity", None)
    run_url = run.url or f"https://wandb.ai/{resolved_entity}/{run.project}/runs/{run.id}"
    run.finish()

    comparison["wandb"] = {
        "project": args.wandb_project,
        "entity": resolved_entity,
        "group": args.wandb_group,
        "run_id": args.wandb_run_id,
        "url": run_url,
    }
    _atomic_json(out / "summary.json", comparison)
    pending.unlink(missing_ok=True)
    print(json.dumps(comparison, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = lambda q: (q.add_argument("--benchmark", default="aime25"), q.add_argument("--out", required=True),
                        q.add_argument("--tp", type=int, default=1), q.add_argument("--seed", type=int, default=1234))
    g = sub.add_parser("gen-abs"); common(g); g.add_argument("--absgen-hf", required=True)
    g.add_argument("--k", type=int, default=4); g.add_argument("--abs-max-tokens", type=int, default=1024)
    g.add_argument("--chunk", type=int, default=8)
    def solve_args(q):
        q.add_argument("--solgen-hf", required=True)
        q.add_argument("--n", type=int, default=32); q.add_argument("--k", type=int, default=4)
        q.add_argument("--max-tokens", type=int, default=32768)
        q.add_argument("--temperature", type=float, default=0.6); q.add_argument("--top-p", type=float, default=0.95)
        q.add_argument("--chunk", type=int, default=8)   # small => frequent flush => resume loses <1 chunk on a kill
        q.add_argument("--shard", type=int, default=0); q.add_argument("--num-shards", type=int, default=1)
        q.add_argument("--skip-woabs", action="store_true")
    s = sub.add_parser("solve"); common(s); solve_args(s)
    j = sub.add_parser("joint"); common(j); solve_args(j)   # RLAD-Joint native combined-prompt eval
    def summary_args(q):
        q.add_argument("--n", type=int, default=32); q.add_argument("--k", type=int, default=4)
        q.add_argument("--mode", choices=("dual", "joint"), default="dual")
        q.add_argument("--skip-woabs", action="store_true")
    z = sub.add_parser("summarize"); common(z); summary_args(z)
    v = sub.add_parser("validate"); common(v); summary_args(v)
    c = sub.add_parser("compare")
    c.add_argument("--benchmark", default="dsr_hard")
    c.add_argument("--untrained-out", required=True); c.add_argument("--rft-out", required=True)
    c.add_argument("--out", required=True); c.add_argument("--input-key", required=True)
    c.add_argument("--untrained-absgen", required=True); c.add_argument("--rft-absgen", required=True)
    c.add_argument("--solver-model", required=True); c.add_argument("--source-commit", required=True)
    c.add_argument("--wandb-project", default="repro-paper003-rlad")
    c.add_argument("--wandb-entity", default=""); c.add_argument("--wandb-group", default="absgen-rft-eval")
    c.add_argument("--wandb-run-name", required=True); c.add_argument("--wandb-run-id", required=True)
    args = ap.parse_args()
    {"gen-abs": stage_gen_abs, "solve": stage_solve, "joint": stage_joint,
     "summarize": stage_summarize, "validate": stage_validate,
     "compare": stage_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
