"""Build the RLAD solution-generator DAPO training set (method_spec §4.2).

The sol-gen DAPO trains on a MIXTURE of abstraction-conditioned prompts (x,z) and bare
prompts (x); the bare ones are reward-masked (reward_post.rlad_reward_post). We pre-generate
one abstraction z per curriculum problem from the (RFT'd) pi_abs offline and inject it
(A-absinject: 1/problem, reused across the rollout group), rather than serving pi_abs online.

Output train_solgen_rlad.jsonl (stock-rollout format {input, label, metadata}), in curriculum
order (easy then medium; shuffle OFF in the config). For each problem index i:
  - i % NOABS_EVERY == 0  -> BARE entry: render_prompt(problem),  metadata.has_abstraction=False
  - else                  -> ABSTRACTION entry: render_prompt_with_abstraction(problem, z),
                             metadata.has_abstraction=True
NOABS_EVERY=8 → ~1/8 bare (A-noabs). label = boxed-free answer (deepscaler grader handles it).

Usage (host vllm env):
  python -m rlad_plugin.solgen_data --absgen-hf runs/sft_absgen_rft/hf [--noabs-every 8]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RLAD_HOME = Path(os.environ.get("RLAD_HOME", Path(__file__).resolve().parents[1]))
DATA = RLAD_HOME / "data"
DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
SOLVER = "Qwen/Qwen3-1.7B"


def _curriculum(path=None):
    """Ordered list of {qid, answer, problem?} curriculum items.
    - path=None (legacy): DeepScaleR easy then medium; 'problem' absent -> resolved by qid index.
    - path=<file>: a self-contained curriculum (e.g. POPE) whose metadata embeds the raw 'problem'."""
    items = []
    files = [path] if path else [DATA / "train_easy.jsonl", DATA / "train_medium.jsonl"]
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            it = {"qid": r["metadata"]["qid"], "answer": r["label"]}
            if r["metadata"].get("problem"):
                it["problem"] = r["metadata"]["problem"]
            items.append(it)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--absgen-hf", required=True, help="RFT'd pi_abs HF dir")
    ap.add_argument("--noabs-every", type=int, default=8)
    ap.add_argument("--abs-max-tokens", type=int, default=1024)
    ap.add_argument("--curriculum", default=None,
                    help="self-contained curriculum jsonl (metadata.problem embedded); default=DeepScaleR easy+medium")
    ap.add_argument("--out", default=str(DATA / "train_solgen_rlad.jsonl"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlad_plugin.templates import (render_absgen_prompt, render_prompt,
                                       render_prompt_with_abstraction)

    items = _curriculum(args.curriculum)
    # only resolve raw problems via DeepScaleR when the curriculum did NOT embed them
    if all("problem" in it for it in items):
        problems = [it["problem"] for it in items]
    else:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="train")
        problems = [it.get("problem") or ds[int(it["qid"].split("-")[1])]["problem"] for it in items]
    abs_tok = AutoTokenizer.from_pretrained(args.absgen_hf, trust_remote_code=True)
    sol_tok = AutoTokenizer.from_pretrained(SOLVER, trust_remote_code=True)

    # one abstraction per problem from pi_abs
    llm = LLM(model=args.absgen_hf, tensor_parallel_size=1, seed=42)
    sp = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=args.abs_max_tokens, seed=42)
    outs = llm.generate([render_absgen_prompt(abs_tok, p) for p in problems], sp)

    def strip_think(t):
        return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()

    n_abs = n_bare = 0
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for i, (it, prob, o) in enumerate(zip(items, problems, outs)):
            qid, ans = it["qid"], it["answer"]
            if i % args.noabs_every == 0:
                fh.write(json.dumps({"input": render_prompt(sol_tok, prob), "label": ans,
                                     "metadata": {"qid": qid, "has_abstraction": False}}, ensure_ascii=False) + "\n")
                n_bare += 1
            else:
                z = strip_think(o.outputs[0].text)
                fh.write(json.dumps({"input": render_prompt_with_abstraction(sol_tok, prob, z), "label": ans,
                                     "metadata": {"qid": qid, "has_abstraction": True}}, ensure_ascii=False) + "\n")
                n_abs += 1
    (DATA / "solgen_data_meta.json").write_text(json.dumps(
        {"n_abstraction": n_abs, "n_bare": n_bare, "noabs_every": args.noabs_every,
         "absgen_hf": args.absgen_hf, "out": args.out}, indent=2) + "\n")
    print(f"wrote {n_abs} abstraction + {n_bare} bare prompts -> {args.out}")


if __name__ == "__main__":
    main()
