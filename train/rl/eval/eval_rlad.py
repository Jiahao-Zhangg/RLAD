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
  summarize : OUT/summary.json {woabs, wabs_avg, wabs_best, per-problem}

Usage (host vllm env), e.g. RLAD model:
  python eval_rlad.py gen-abs   --absgen-hf <pi_abs_hf> --benchmark aime25 --out OUT --k 4
  python eval_rlad.py solve     --solgen-hf <pi_sol_hf> --benchmark aime25 --out OUT --n 32 --max-tokens 32768
  python eval_rlad.py summarize --benchmark aime25 --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent          # .../train/rl
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
if str(_CODE / "eval") not in sys.path:
    sys.path.insert(0, str(_CODE / "eval"))

SANDBOX = _CODE.parent
BENCH_DIR = SANDBOX / "data" / "benchmarks"


def _read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def _read_solve(out):
    """All solve rows across the single-file and N-way-sharded layouts (interchangeable)."""
    out = Path(out)
    rows = []
    for p in sorted(out.glob("solve_samples.jsonl")) + sorted(out.glob("solve_samples.shard*.jsonl")):
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


def _strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def stage_gen_abs(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_absgen_prompt

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    abs_path = out / "abstractions.jsonl"
    problems = _load_problems(args.benchmark)
    done = {r["id"] for r in _read(abs_path)}
    todo = [p for p in problems if p["id"] not in done]
    print(f"gen-abs: {len(problems)} problems, {len(done)} done, {len(todo)} to do")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.absgen_hf, trust_remote_code=True)
    llm = LLM(model=args.absgen_hf, tensor_parallel_size=args.tp, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=0.7, top_p=0.95, max_tokens=args.abs_max_tokens, seed=args.seed)
    outs = llm.generate([render_absgen_prompt(tok, p["problem"]) for p in todo], sp)
    with abs_path.open("a", encoding="utf-8") as fh:
        for p, o in zip(todo, outs):
            for j, c in enumerate(o.outputs):
                fh.write(json.dumps({"id": p["id"], "abs_idx": j, "abstraction": _strip_think(c.text)}, ensure_ascii=False) + "\n")
    print(f"gen-abs -> {abs_path}")


def stage_solve(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_prompt, render_prompt_with_abstraction
    from vllm_eval import grade_response

    out = Path(args.out)
    problems = {p["id"]: p for p in _load_problems(args.benchmark)}
    absrows = _read(out / "abstractions.jsonl")
    if not absrows:
        sys.exit("run gen-abs first")
    abs_by_id = defaultdict(dict)
    for r in absrows:
        abs_by_id[r["id"]][r["abs_idx"]] = r["abstraction"]

    samples_path = _solve_path(out, args.shard, args.num_shards)
    done = {(r["id"], r["cond"]) for r in _read(samples_path)}
    tok = AutoTokenizer.from_pretrained(args.solgen_hf, trust_remote_code=True)

    # build the full (id,cond) work list, then take this shard's deterministic slice (data-parallel
    # across GPUs); skip items already done in THIS shard's file (resume across cancels/timeouts)
    allwork = []   # (id, cond, prompt, gold)
    for pid, p in problems.items():
        allwork.append((pid, "woabs", render_prompt(tok, p["problem"]), p["answer"]))
        for k, z in abs_by_id.get(pid, {}).items():
            allwork.append((pid, f"abs{k}", render_prompt_with_abstraction(tok, p["problem"], z), p["answer"]))
    allwork.sort(key=lambda w: (w[0], w[1]))   # stable order so the shard slice is deterministic
    work = [w for i, w in enumerate(allwork)
            if i % args.num_shards == args.shard and (w[0], w[1]) not in done]
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
        with samples_path.open("a", encoding="utf-8") as fh:
            for (pid, cond, _, gold), o in zip(ch, outs):
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
    samples_path = _solve_path(out, args.shard, args.num_shards)
    done = {(r["id"], r["cond"]) for r in _read(samples_path)}
    tok = AutoTokenizer.from_pretrained(args.solgen_hf, trust_remote_code=True)

    allwork = [(pid, "joint", render_joint_prompt(tok, p["problem"]), p["answer"]) for pid, p in problems.items()]
    allwork.sort(key=lambda w: (w[0], w[1]))
    work = [w for i, w in enumerate(allwork)
            if i % args.num_shards == args.shard and (w[0], w[1]) not in done]
    print(f"joint[shard {args.shard}/{args.num_shards}]: {len(work)} problems to generate "
          f"({len(done)} done, {len(allwork)} total)")
    if not work:
        return
    llm = LLM(model=args.solgen_hf, tensor_parallel_size=args.tp, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    for i in range(0, len(work), args.chunk):
        ch = work[i:i + args.chunk]
        outs = llm.generate([w[2] for w in ch], sp)
        with samples_path.open("a", encoding="utf-8") as fh:
            for (pid, cond, _, gold), o in zip(ch, outs):
                for si, c in enumerate(o.outputs):
                    correct, *_ = grade_response(c.text, gold)
                    fh.write(json.dumps({"id": pid, "cond": cond, "sample_idx": si, "correct": int(correct)}) + "\n")
            fh.flush()
        print(f"  joint: +{len(ch)} problems [{i + len(ch)}/{len(work)}]")
    print(f"joint -> {samples_path}")


def stage_summarize(args):
    out = Path(args.out)
    rows = _read_solve(out)
    if not rows:
        sys.exit("run solve first")
    problems = [p["id"] for p in _load_problems(args.benchmark)]
    # per (id,cond) accuracy
    acc = defaultdict(lambda: defaultdict(list))   # id -> cond -> [correct...]
    for r in rows:
        acc[r["id"]][r["cond"]].append(r["correct"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    woabs, wabs_avg, wabs_best, joint = [], [], [], []
    per_problem = {}
    for pid in problems:
        conds = acc.get(pid, {})
        wo = mean(conds.get("woabs", []))
        abs = [mean(v) for c, v in conds.items() if c.startswith("abs")]
        jt = mean(conds.get("joint", []))
        if conds.get("woabs"):
            woabs.append(wo)
        if abs:
            wabs_avg.append(sum(abs) / len(abs))
            wabs_best.append(max(abs))
        if conds.get("joint"):
            joint.append(jt)
        per_problem[pid] = {"woabs": round(wo, 4), "wabs_avg": round(sum(abs) / len(abs), 4) if abs else None,
                            "wabs_best": round(max(abs), 4) if abs else None, "n_abs": len(abs),
                            "joint": round(jt, 4) if conds.get("joint") else None}
    summary = {
        "benchmark": args.benchmark, "n_problems": len(problems),
        "woabs_pass1": round(100 * mean(woabs), 2),
        "wabs_avg_pass1": round(100 * mean(wabs_avg), 2),
        "wabs_best_pass1": round(100 * mean(wabs_best), 2),
        "joint_pass1": round(100 * mean(joint), 2),   # RLAD-Joint native combined-prompt (None-arm -> 0.0)
        "n_problems_woabs": len(woabs), "n_problems_wabs": len(wabs_avg), "n_problems_joint": len(joint),
    }
    (out / "summary.json").write_text(json.dumps({**summary, "per_problem": per_problem}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = lambda q: (q.add_argument("--benchmark", default="aime25"), q.add_argument("--out", required=True),
                        q.add_argument("--tp", type=int, default=1), q.add_argument("--seed", type=int, default=1234))
    g = sub.add_parser("gen-abs"); common(g); g.add_argument("--absgen-hf", required=True)
    g.add_argument("--k", type=int, default=4); g.add_argument("--abs-max-tokens", type=int, default=1024)
    def solve_args(q):
        q.add_argument("--solgen-hf", required=True)
        q.add_argument("--n", type=int, default=32); q.add_argument("--max-tokens", type=int, default=32768)
        q.add_argument("--temperature", type=float, default=0.6); q.add_argument("--top-p", type=float, default=0.95)
        q.add_argument("--chunk", type=int, default=8)   # small => frequent flush => resume loses <1 chunk on a kill
        q.add_argument("--shard", type=int, default=0); q.add_argument("--num-shards", type=int, default=1)
    s = sub.add_parser("solve"); common(s); solve_args(s)
    j = sub.add_parser("joint"); common(j); solve_args(j)   # RLAD-Joint native combined-prompt eval
    z = sub.add_parser("summarize"); common(z)
    args = ap.parse_args()
    {"gen-abs": stage_gen_abs, "solve": stage_solve, "joint": stage_joint,
     "summarize": stage_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
