"""Offline abstraction-generator RFT scoring for RLAD (method_spec §4.2).

Implements the paper's r_sol(x,z) = E_{y~pi_sol(.|x,z)}[Acc] and rejection-fine-tuning corpus
construction. Three RESUMABLE stages (chunked, append-as-you-go + skip-completed, so a 4h
timeout just resumes — the score stage is the dominant cost):

  gen-abs  : load pi_abs (HF), sample K abstractions per D_abs problem -> abs_cache.jsonl
             {qid, answer, abs_idx, abstraction}.  (model = pi_abs; runs/sft_absgen_rft prior)
  score    : load pi_sol (base Qwen3-1.7B), for each (qid, abs_idx) sample M solutions
             conditioned via render_prompt_with_abstraction, grade (deepscaler) -> r_sol ->
             scored.jsonl {qid, answer, abs_idx, abstraction, r_sol}.  (model = pi_sol)
  build-rft: KEEP abstractions whose r_sol > the problem's base w/o-abs success + margin
             (Eq. 1) AND that don't leak the answer -> RFT corpus (SFT messages) ->
             train_absgen_rft.jsonl.

D_abs defaults to a 1.5k-problem subset of the curriculum (A-scoresubset). K=4, M=8 (M
reduced from the paper's 16 to fit the 240 GPU-h cap; A-rft).

Usage (host vllm env):
  python -m rlad_plugin.absgen_score gen-abs   --absgen-hf runs/sft_absgen/hf --k 4 --n-problems 1500
  python -m rlad_plugin.absgen_score score     --m 8 --max-tokens 16384
  python -m rlad_plugin.absgen_score build-rft
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RLAD_HOME = Path(os.environ.get("RLAD_HOME", Path(__file__).resolve().parents[1]))
DATA = RLAD_HOME / "data"
RUNS = RLAD_HOME / "runs"
DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
SOLVER = "Qwen/Qwen3-1.7B"
SCORE_SAMPLES = RUNS / "eval" / "dsr_pool_score" / "dsr_pool"
ABS_CACHE = DATA / "rft_abs_cache.jsonl"
SCORED = DATA / "rft_scored.jsonl"
RFT_OUT = DATA / "train_absgen_rft.jsonl"
CHUNK = 256   # problems (gen-abs) / abstraction-rows (score) per incremental write


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _read_glob(stem):
    """Read+concat all shard files for a stem (e.g. 'rft_scored' -> rft_scored.jsonl +
    rft_scored.shard*.jsonl), so single-process and N-way-sharded runs are interchangeable."""
    rows = []
    for p in sorted(DATA.glob(stem + ".jsonl")) + sorted(DATA.glob(stem + ".shard*.jsonl")):
        rows += _read_jsonl(p)
    return rows


def _shard_path(base, shard, num_shards):
    """Per-shard output path (num_shards==1 keeps the original single-file name)."""
    return base if num_shards <= 1 else base.with_name(f"{base.stem}.shard{shard}.jsonl")


def _done_keys(path, keyfn):
    return {keyfn(r) for r in _read_jsonl(path)}


def _curriculum_qids():
    qids = []
    for f in ("train_easy.jsonl", "train_medium.jsonl"):
        p = DATA / f
        if p.exists():
            qids += [json.loads(l)["metadata"]["qid"] for l in p.read_text().splitlines() if l.strip()]
    return qids


def _base_success():
    by = defaultdict(list)
    for sp in sorted(SCORE_SAMPLES.glob("samples*.jsonl")):
        for line in sp.read_text().splitlines():
            if line.strip():
                r = json.loads(line); by[r["id"]].append(int(r["correct"]))
    return {q: sum(v) / len(v) for q, v in by.items() if v}


def _strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def _leaks(abstraction, answer):
    a = re.sub(r"\s+", "", str(answer))
    if len(a) < 3:
        return False
    if re.fullmatch(r"[A-Za-z0-9]+", a):
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])",
                         re.sub(r"\s+", " ", abstraction)) is not None
    return a in re.sub(r"\s+", "", abstraction)


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def stage_gen_abs(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_absgen_prompt

    qids = _curriculum_qids()[: args.n_problems]
    qids = [q for i, q in enumerate(qids) if i % args.num_shards == args.shard]   # data-parallel slice
    out_path = _shard_path(ABS_CACHE, args.shard, args.num_shards)
    done = _done_keys(out_path, lambda r: r["qid"])
    todo = [q for q in qids if q not in done]
    print(f"gen-abs[shard {args.shard}/{args.num_shards}]: {len(qids)} problems, {len(done)} done, {len(todo)} to do")
    if not todo:
        print("gen-abs: nothing to do"); return
    ds = load_dataset(DATASET, split="train")
    tok = AutoTokenizer.from_pretrained(args.absgen_hf, trust_remote_code=True)
    llm = LLM(model=args.absgen_hf, tensor_parallel_size=1, seed=42)
    sp = SamplingParams(n=args.k, temperature=0.7, top_p=0.95, max_tokens=args.abs_max_tokens, seed=42)
    with out_path.open("a", encoding="utf-8") as fh:
        for ch in _chunks(todo, CHUNK):
            meta = [(q, str(ds[int(q.split("-")[1])]["answer"]), ds[int(q.split("-")[1])]["problem"]) for q in ch]
            outs = llm.generate([render_absgen_prompt(tok, p) for _, _, p in meta], sp)
            for (q, ans, _), o in zip(meta, outs):
                for j, c in enumerate(o.outputs):
                    z = _strip_think(c.text)
                    if len(z) >= 40:
                        fh.write(json.dumps({"qid": q, "answer": ans, "abs_idx": j, "abstraction": z}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  gen-abs: +{len(ch)} problems written")
    print(f"gen-abs: done -> {ABS_CACHE}")


def stage_score(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import render_prompt_with_abstraction
    sys.path.insert(0, str(RLAD_HOME / "miles"))
    from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
    from datasets import load_dataset

    rows = _read_glob("rft_abs_cache")          # all gen-abs shards
    if not rows:
        sys.exit("run gen-abs first")
    rows.sort(key=lambda r: (r["qid"], r["abs_idx"]))   # stable order so the shard slice is deterministic
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]   # data-parallel slice
    key = lambda r: (r["qid"], r["abs_idx"])
    out_path = _shard_path(SCORED, args.shard, args.num_shards)
    done = _done_keys(out_path, key)
    todo = [r for r in rows if key(r) not in done]
    print(f"score[shard {args.shard}/{args.num_shards}]: {len(rows)} abstractions, {len(done)} scored, {len(todo)} to do")
    if not todo:
        print("score: nothing to do"); return
    ds = load_dataset(DATASET, split="train")
    tok = AutoTokenizer.from_pretrained(SOLVER, trust_remote_code=True)
    llm = LLM(model=SOLVER, tensor_parallel_size=1, seed=42)
    sp = SamplingParams(n=args.m, temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, seed=42)
    with out_path.open("a", encoding="utf-8") as fh:
        for ch in _chunks(todo, CHUNK):
            prompts = [render_prompt_with_abstraction(tok, ds[int(r["qid"].split("-")[1])]["problem"], r["abstraction"]) for r in ch]
            outs = llm.generate(prompts, sp)
            for r, o in zip(ch, outs):
                accs = [int(get_deepscaler_rule_based_reward(c.text, r["answer"])) for c in o.outputs]
                r_sol = sum(accs) / len(accs) if accs else 0.0
                fh.write(json.dumps({**{k: r[k] for k in ("qid", "answer", "abs_idx", "abstraction")},
                                     "r_sol": r_sol}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  score: +{len(ch)} abstractions scored")
    print(f"score: done -> {SCORED}")


def stage_build_rft(args):
    from datasets import load_dataset
    from rlad_plugin.templates import ABSGEN_INSTRUCTION

    base = _base_success()
    rows = _read_glob("rft_scored")          # all score shards
    if not rows:
        sys.exit("run score first")
    ds = load_dataset(DATASET, split="train")
    kept = dropped_leak = dropped_ineff = 0
    with RFT_OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            b = base.get(r["qid"], 0.0)
            if r["r_sol"] <= b + args.margin:      # Eq.1: must increase solver success
                dropped_ineff += 1; continue
            if _leaks(r["abstraction"], r["answer"]):
                dropped_leak += 1; continue
            prob = ds[int(r["qid"].split("-")[1])]["problem"]
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": ABSGEN_INSTRUCTION + "\n\nProblem:\n" + prob},
                {"role": "assistant", "content": r["abstraction"]}],
                "metadata": {"qid": r["qid"], "r_sol": round(r["r_sol"], 3), "base": round(b, 3)}},
                ensure_ascii=False) + "\n")
            kept += 1
    meta = {"kept": kept, "dropped_ineffective": dropped_ineff, "dropped_leak": dropped_leak,
            "margin": args.margin, "out": str(RFT_OUT)}
    (DATA / "absgen_rft_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"build-rft: kept {kept} (dropped {dropped_ineff} ineffective, {dropped_leak} leak) -> {RFT_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    def _shardargs(p):
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    g = sub.add_parser("gen-abs"); g.add_argument("--absgen-hf", required=True); g.add_argument("--k", type=int, default=4)
    g.add_argument("--n-problems", type=int, default=1500); g.add_argument("--abs-max-tokens", type=int, default=1024); _shardargs(g)
    s = sub.add_parser("score"); s.add_argument("--m", type=int, default=8); s.add_argument("--max-tokens", type=int, default=16384); _shardargs(s)
    b = sub.add_parser("build-rft"); b.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()
    {"gen-abs": stage_gen_abs, "score": stage_score, "build-rft": stage_build_rft}[args.cmd](args)


if __name__ == "__main__":
    main()
