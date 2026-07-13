"""Warmstart-data generation for pi_abs SFT (method_spec §3.1-§4.2).

Builds the seed corpus {(problem, abstraction)} the abstraction generator is SFT'd on.
The paper summarizes solution traces with a STRONG model (o4-mini); the NIM key is empty
on this cluster, so we substitute a strong LOCAL model (default Qwen3-4B-Thinking-2507 —
stronger than the 1.7B solver, a thinking model good at summarizing) and summarize the
GOLD DeepScaleR solution into a non-leaking "cheatsheet". (assumptions A-nimmodel→local,
A-warmstart). Leakage handling: a cheap answer-string filter here; the paper's 16-sample
base-model leakage check is deferred (G2 efficacy gate is the real quality check).

Pipeline (multi-node data parallel; one independent output per GPU shard):
  - load curriculum (train_easy + train_medium), map qid "dsr-<idx>" -> DeepScaleR row
    -> (problem, solution, answer)
  - generate K abstractions/problem from (problem, solution) with the generator model
  - drop abstractions whose text contains the answer (cheap leak filter)
  - merge shard files into train_absgen_sft.jsonl in chat-message format

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


def _shard_path(path: Path, shard: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard}{path.suffix}")


def _metadata_path(out_path: Path) -> Path:
    if out_path.name == "train_absgen_sft.jsonl":
        return out_path.with_name("absgen_sft_meta.json")
    return out_path.with_name(f"{out_path.stem}_meta.json")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _merge_shards(args: argparse.Namespace, qids: list[str]) -> None:
    out_path = Path(args.out)
    meta_path = _metadata_path(out_path)
    qid_rank = {qid: index for index, qid in enumerate(qids)}
    rows = []
    leaked = 0
    for shard in range(args.num_shards):
        shard_path = _shard_path(out_path, shard, args.num_shards)
        shard_meta_path = _shard_path(meta_path, shard, args.num_shards)
        try:
            shard_rows = [
                json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            shard_meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid warm-start shard {shard}: {exc}") from exc
        expected_qids = set(qids[shard::args.num_shards])
        if (
            shard_meta.get("shard") != shard
            or shard_meta.get("num_shards") != args.num_shards
            or shard_meta.get("n_problems") != len(expected_qids)
            or shard_meta.get("total_problems") != len(qids)
            or shard_meta.get("k") != args.k
            or shard_meta.get("generator") != args.generator
            or shard_meta.get("kept") != len(shard_rows)
        ):
            raise SystemExit(f"warm-start metadata mismatch for shard {shard}")
        per_qid = {}
        for row in shard_rows:
            qid = row.get("metadata", {}).get("qid")
            messages = row.get("messages")
            if (
                qid not in expected_qids
                or not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
                or not messages[1].get("content")
            ):
                raise SystemExit(f"malformed warm-start row in shard {shard}: {row}")
            per_qid[qid] = per_qid.get(qid, 0) + 1
            if per_qid[qid] > args.k:
                raise SystemExit(f"too many warm-start rows for {qid} in shard {shard}")
        leaked += int(shard_meta.get("leaked_dropped", 0))
        rows.extend(shard_rows)

    rows.sort(key=lambda row: qid_rank[row["metadata"]["qid"]])
    _atomic_jsonl(out_path, rows)
    _atomic_json(meta_path, {
        "n_problems": len(qids),
        "k": args.k,
        "generator": args.generator,
        "kept": len(rows),
        "leaked_dropped": leaked,
        "num_shards": args.num_shards,
        "out": str(out_path),
    })
    print(f"merged {args.num_shards} warm-start shards -> {out_path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=2, help="abstractions per problem")
    ap.add_argument("--generator", default=GENERATOR)
    ap.add_argument("--limit", type=int, default=0, help="cap #problems (0=all; for smoke)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=str(DATA / "train_absgen_sft.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", action="store_true")
    args = ap.parse_args()

    if args.k < 1 or args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("--k/--num-shards must be positive and --shard must be in range")

    qids = _load_curriculum_qids()
    if args.limit:
        qids = qids[: args.limit]
    if args.merge_shards:
        _merge_shards(args, qids)
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from rlad_plugin.templates import ABSGEN_INSTRUCTION  # used in the messages-format SFT label

    total_problems = len(qids)
    qids = qids[args.shard::args.num_shards]
    print(f"warm-start shard {args.shard}/{args.num_shards}: {len(qids)}/{total_problems} problems")

    ds = load_dataset(DATASET, split="train")
    idx_of = lambda q: int(q.split("-")[1])
    probs = [{"qid": q, "problem": ds[idx_of(q)]["problem"],
              "solution": ds[idx_of(q)]["solution"], "answer": str(ds[idx_of(q)]["answer"])}
             for q in qids]

    gen_tok = AutoTokenizer.from_pretrained(args.generator, trust_remote_code=True)

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
    if len(outs) != len(probs):
        raise RuntimeError(f"vLLM returned {len(outs)} prompt outputs; expected {len(probs)}")

    base_out_path = Path(args.out)
    out_path = _shard_path(base_out_path, args.shard, args.num_shards)
    meta_path = _shard_path(_metadata_path(base_out_path), args.shard, args.num_shards)
    kept = leaked = 0
    rows = []
    for p, o in zip(probs, outs):
        if len(o.outputs) != args.k:
            raise RuntimeError(
                f"vLLM returned {len(o.outputs)} abstractions for {p['qid']}; expected {args.k}"
            )
        for c in o.outputs:
            abstraction = _strip_think(c.text)
            if len(abstraction) < 40:
                continue
            if _leaks(abstraction, p["answer"]):
                leaked += 1
                continue
            # SFT data = chat messages (miles sft_rollout reads sample.prompt as a
            # message list; loss masked to the assistant turn). user = instruction+problem.
            rows.append({
                "messages": [
                    {"role": "user", "content": ABSGEN_INSTRUCTION + "\n\nProblem:\n" + p["problem"]},
                    {"role": "assistant", "content": abstraction},
                ],
                "metadata": {"qid": p["qid"], "generator": args.generator},
            })
            kept += 1
    _atomic_jsonl(out_path, rows)
    meta = {"n_problems": len(probs), "total_problems": total_problems,
            "shard": args.shard, "num_shards": args.num_shards,
            "k": args.k, "generator": args.generator,
            "kept": kept, "leaked_dropped": leaked, "out": str(out_path)}
    _atomic_json(meta_path, meta)
    print(f"wrote {kept} SFT rows ({leaked} dropped as leaking) -> {out_path}")


if __name__ == "__main__":
    main()
