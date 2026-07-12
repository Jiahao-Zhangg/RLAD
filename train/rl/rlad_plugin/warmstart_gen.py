"""Warmstart-data generation for pi_abs SFT (method_spec §3.1-§4.2).

Builds the seed corpus {(problem, abstraction)} the abstraction generator is SFT'd on.
The paper summarizes solution traces with a STRONG model (o4-mini); the NIM key is empty
on this cluster, so we substitute a strong LOCAL model (default Qwen3-4B-Thinking-2507 —
stronger than the 1.7B solver, a thinking model good at summarizing) and summarize the
GOLD DeepScaleR solution into a non-leaking "cheatsheet". (assumptions A-nimmodel→local,
A-warmstart). Leakage handling: a cheap answer-string filter here; the paper's 16-sample
base-model leakage check is deferred (G2 efficacy gate is the real quality check).

Pipeline (resumable; one GPU vLLM pass):
  - load curriculum (train_easy + train_medium), map qid "dsr-<idx>" -> DeepScaleR row
    -> (problem, solution, answer)
  - generate K abstractions/problem from (problem, solution) with the generator model
  - drop abstractions whose text contains the answer (cheap leak filter)
  - write train_absgen_sft.jsonl: {input: render_absgen_prompt(problem), label: abstraction}

Usage (in eval/host env with vllm):
  python -m rlad_plugin.warmstart_gen --k 2 [--generator Qwen/Qwen3-4B-Thinking-2507] \
      [--limit N] [--max-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

RLAD_HOME = Path(os.environ.get("RLAD_HOME", Path(__file__).resolve().parents[1]))
DATA = RLAD_HOME / "data"
DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
GENERATOR = "Qwen/Qwen3-4B-Instruct-2507"  # NON-thinking: summarizes the GIVEN gold solution
# into a direct cheatsheet (no <think> to strip). The Thinking variant re-solved the problem
# and its un-closed <think> leaked into the label (smoke 4873645) — instruct is the right tool.
SOLVER = "Qwen/Qwen3-1.7B"  # tokenizer for the SFT prompt (pi_abs base)

GEN_INSTRUCTION = (
    "You are given a competition math problem and a correct worked solution. Distill the "
    "solution into a concise cheatsheet of transferable insights — key ideas, lemmas, "
    "strategies, reformulations, or common pitfalls — that would help someone solve this "
    "problem from scratch. Write a few focused bullet points. CRITICAL: do NOT state the "
    "final numerical answer and do NOT reveal the last computational result; give guidance, "
    "not the answer."
)


def _load_curriculum_qids() -> list[str]:
    qids = []
    for f in ("train_easy.jsonl", "train_medium.jsonl"):
        p = DATA / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    qids.append(json.loads(line)["metadata"]["qid"])
    return qids


def _strip_think(text: str) -> str:
    """Return the post-</think> content (thinking-model output); else the whole text."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def _leaks(abstraction: str, answer: str) -> bool:
    """Cheap leak check: the normalized answer appears in the abstraction. Plain
    alphanumeric answers use word boundaries (so 'pi'/'2'/'100' don't match
    'period'/'1000'); LaTeX/symbolic answers fall back to substring. len<3 skipped
    (too short to be a reliable signal; the paper's 16-sample base check is deferred)."""
    a = re.sub(r"\s+", "", str(answer))
    if len(a) < 3:
        return False
    if re.fullmatch(r"[A-Za-z0-9]+", a):
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])",
                         re.sub(r"\s+", " ", abstraction)) is not None
    return a in re.sub(r"\s+", "", abstraction)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=2, help="abstractions per problem")
    ap.add_argument("--generator", default=GENERATOR)
    ap.add_argument("--limit", type=int, default=0, help="cap #problems (0=all; for smoke)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=str(DATA / "train_absgen_sft.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from rlad_plugin.templates import ABSGEN_INSTRUCTION  # used in the messages-format SFT label

    qids = _load_curriculum_qids()
    if args.limit:
        qids = qids[: args.limit]
    print(f"curriculum problems: {len(qids)}")

    ds = load_dataset(DATASET, split="train")
    idx_of = lambda q: int(q.split("-")[1])
    probs = [{"qid": q, "problem": ds[idx_of(q)]["problem"],
              "solution": ds[idx_of(q)]["solution"], "answer": str(ds[idx_of(q)]["answer"])}
             for q in qids]

    gen_tok = AutoTokenizer.from_pretrained(args.generator, trust_remote_code=True)
    sft_tok = AutoTokenizer.from_pretrained(SOLVER, trust_remote_code=True)

    # generator prompt = problem + gold solution + distill instruction
    gen_prompts = []
    for p in probs:
        user = (GEN_INSTRUCTION + "\n\nProblem:\n" + p["problem"]
                + "\n\nCorrect solution:\n" + (p["solution"] or "")[:8000])
        gen_prompts.append(gen_tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.generator, tensor_parallel_size=1, seed=42)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=42)
    outs = llm.generate(gen_prompts, sp)

    out_path = Path(args.out)
    kept = leaked = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for p, o in zip(probs, outs):
            for c in o.outputs:
                abstraction = _strip_think(c.text)
                if len(abstraction) < 40:
                    continue
                if _leaks(abstraction, p["answer"]):
                    leaked += 1
                    continue
                # SFT data = chat messages (miles sft_rollout reads sample.prompt as a
                # message list; loss masked to the assistant turn). user = instruction+problem.
                fh.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": ABSGEN_INSTRUCTION + "\n\nProblem:\n" + p["problem"]},
                        {"role": "assistant", "content": abstraction},
                    ],
                    "metadata": {"qid": p["qid"], "generator": args.generator},
                }, ensure_ascii=False) + "\n")
                kept += 1
    meta = {"n_problems": len(probs), "k": args.k, "generator": args.generator,
            "kept": kept, "leaked_dropped": leaked, "out": str(out_path)}
    (DATA / "absgen_sft_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {kept} SFT rows ({leaked} dropped as leaking) -> {out_path}")


if __name__ == "__main__":
    main()
