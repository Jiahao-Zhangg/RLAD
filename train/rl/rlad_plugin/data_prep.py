"""DeepScaleR data prep + base-success curriculum for the RLAD reproduction.

Two resumable steps (method_spec §4 curriculum; assumptions A-curriculum/A-scoresubset):

  build-pool : load agentica-org/DeepScaleR-Preview-Dataset (cols problem/answer/solution),
               filter to non-empty problem + non-empty short answer, take a deterministic
               seed-42 subset of --n-pool, and write it as an eval-harness benchmark
               data/benchmarks/dsr_pool.jsonl  ({"id","problem","answer"}) so the VALIDATED
               eval/vllm_eval.py can score base-model success on it (no separate scorer).

  partition  : read the base-model scoring samples produced by running
               jobs/eval.sbatch on dsr_pool (runs/eval/dsr_pool_score/dsr_pool/samples*.jsonl),
               compute per-problem success rate, bin into easy/medium/hard, and write:
                 data/train_easy.jsonl, data/train_medium.jsonl   (templated {input,label,metadata})
                 data/benchmarks/dsr_hard.jsonl                   (held-out eval {id,problem,answer})
               Cutoffs (A-curriculum): hard succ<=HARD_MAX, medium (HARD_MAX,EASY_MIN], easy >EASY_MIN.

Usage:
  python -m rlad_plugin.data_prep build-pool --n-pool 6000 [--seed 42]
  # then: rlad_inference_sbatch jobs/eval.sbatch  (BENCHMARKS=dsr_pool N_SAMPLES=8 MAX_TOKENS=8192 OUT_DIR=runs/eval/dsr_pool_score MODEL_PATH=Qwen/Qwen3-1.7B)
  python -m rlad_plugin.data_prep partition [--hard-max 0.125 --easy-min 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
RLAD_HOME = Path(os.environ.get("RLAD_HOME", Path(__file__).resolve().parents[1]))
DATA = RLAD_HOME / "data"
BENCH = DATA / "benchmarks"
SCORE_SAMPLES_DIR = RLAD_HOME / "runs" / "eval" / "dsr_pool_score" / "dsr_pool"
MODEL = "Qwen/Qwen3-1.7B"
MAX_ANSWER_CHARS = 64


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_pool(n_pool: int, seed: int) -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    print(f"loaded {DATASET}: {len(ds)} rows")
    eligible = []
    for i in range(len(ds)):
        prob = (ds[i]["problem"] or "").strip()
        ans = str(ds[i]["answer"] or "").strip()
        if not prob or not ans or len(ans) > MAX_ANSWER_CHARS:
            continue
        eligible.append((i, prob, ans))
    print(f"eligible after filter: {len(eligible)}")
    if len(eligible) < n_pool:
        raise SystemExit(f"only {len(eligible)} eligible, need {n_pool}")
    random.Random(seed).shuffle(eligible)
    pool = eligible[:n_pool]
    rows = [{"id": f"dsr-{idx}", "problem": prob, "answer": ans} for idx, prob, ans in pool]
    out = BENCH / "dsr_pool.jsonl"
    _write_jsonl(out, rows)
    meta = {"dataset": DATASET, "seed": seed, "n_pool": n_pool, "n_eligible": len(eligible),
            "output": str(out)}
    (DATA / "dsr_pool_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(rows)} pool problems -> {out}")


def _load_scores() -> dict[str, float]:
    """Per-problem base success rate from the eval-harness samples.jsonl shards."""
    by_id: dict[str, list[int]] = defaultdict(list)
    shards = sorted(SCORE_SAMPLES_DIR.glob("samples*.jsonl"))
    if not shards:
        raise SystemExit(f"no scoring samples at {SCORE_SAMPLES_DIR} — run eval.sbatch on dsr_pool first")
    for sp in shards:
        with sp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_id[row["id"]].append(int(row["correct"]))
    return {pid: sum(v) / len(v) for pid, v in by_id.items() if v}


def partition(hard_max: float, easy_min: float) -> None:
    from transformers import AutoTokenizer

    from .templates import render_prompt

    scores = _load_scores()
    pool = {json.loads(l)["id"]: json.loads(l)
            for l in (BENCH / "dsr_pool.jsonl").read_text().splitlines() if l.strip()}
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    bins = {"easy": [], "medium": [], "hard": []}
    for pid, sr in scores.items():
        if pid not in pool:
            continue
        b = "hard" if sr <= hard_max else ("easy" if sr > easy_min else "medium")
        bins[b].append((pid, sr))
    print("bin counts:", {k: len(v) for k, v in bins.items()})

    def train_rows(items):
        rows = []
        for pid, sr in items:
            p = pool[pid]
            rows.append({"input": render_prompt(tok, p["problem"]), "label": p["answer"],
                         "metadata": {"qid": pid, "success_rate": round(sr, 4), "problem": p["problem"]}})
        return rows

    _write_jsonl(DATA / "train_easy.jsonl", train_rows(bins["easy"]))
    _write_jsonl(DATA / "train_medium.jsonl", train_rows(bins["medium"]))
    _write_jsonl(BENCH / "dsr_hard.jsonl",
                 [{"id": pid, "problem": pool[pid]["problem"], "answer": pool[pid]["answer"]}
                  for pid, _ in bins["hard"]])
    meta = {"hard_max": hard_max, "easy_min": easy_min, "n_scored": len(scores),
            "counts": {k: len(v) for k, v in bins.items()}}
    (DATA / "curriculum_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote train_easy ({len(bins['easy'])}), train_medium ({len(bins['medium'])}), "
          f"dsr_hard ({len(bins['hard'])})")


POPE_DATASET = "CMU-AIRe/POPE-more-64x32k"


def _pope_cols():
    """Read only {problem, answer, mean_reward, sources} from the cached POPE parquet shards,
    skipping the huge responses/tokenized_prompt columns (load_dataset would materialize them)."""
    import glob
    import os

    import pyarrow.parquet as pq

    hub = os.path.join(os.path.expanduser(os.environ["HF_HOME"]), "hub")
    shards = sorted(glob.glob(hub + "/datasets--CMU-AIRe--POPE-more-64x32k/**/*.parquet", recursive=True))
    if not shards:
        from datasets import load_dataset
        load_dataset(POPE_DATASET, split="train")  # one-time fetch -> parquet cache
        shards = sorted(glob.glob(hub + "/datasets--CMU-AIRe--POPE-more-64x32k/**/*.parquet", recursive=True))
    cols = ["problem", "answer", "mean_reward", "sources"]
    rows = []
    for s in shards:
        t = pq.read_table(s, columns=cols)
        rows.extend(t.to_pylist())
    return rows


def build_pope(mr_lo: float, mr_hi: float, n_cap: int, seed: int, holdout: int) -> None:
    """Build a HARD self-contained curriculum from POPE-more-64x32k, filtered by mean_reward
    (the generator's empirical pass rate over 64 samples; LOW = hard). Output rows EMBED the raw
    problem (metadata.problem) so the downstream solver-DAPO data build needs no source lookup.
    A held-out slice is written as an eval benchmark (pope_hard.jsonl)."""
    from transformers import AutoTokenizer

    from .templates import render_prompt

    rows = _pope_cols()
    print(f"loaded {POPE_DATASET}: {len(rows)} rows")
    elig = []
    for r in rows:
        prob = (r.get("problem") or "").strip()
        ans = str(r.get("answer") or "").strip()
        mr = r.get("mean_reward")
        if not prob or not ans or len(ans) > MAX_ANSWER_CHARS or mr is None:
            continue
        if mr_lo < mr <= mr_hi:                       # headroom band (hard but not impossible)
            elig.append((prob, ans, float(mr), r.get("sources")))
    print(f"in band ({mr_lo},{mr_hi}]: {len(elig)} problems (of {len(rows)})")
    random.Random(seed).shuffle(elig)
    if n_cap and len(elig) > n_cap + holdout:
        elig = elig[: n_cap + holdout]
    train, held = elig[holdout:], elig[:holdout]

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    train_rows = [{"input": render_prompt(tok, p), "label": a,
                   "metadata": {"qid": f"pope-{i}", "mean_reward": round(mr, 4),
                                "problem": p, "sources": src}}
                  for i, (p, a, mr, src) in enumerate(train)]
    _write_jsonl(DATA / "train_pope_hard.jsonl", train_rows)
    _write_jsonl(BENCH / "pope_hard.jsonl",
                 [{"id": f"popeh-{i}", "problem": p, "answer": a} for i, (p, a, mr, src) in enumerate(held)])
    meta = {"dataset": POPE_DATASET, "mr_lo": mr_lo, "mr_hi": mr_hi, "seed": seed,
            "n_in_band": len(elig) - holdout if n_cap else len(elig), "n_train": len(train_rows),
            "n_holdout_bench": len(held), "out": str(DATA / "train_pope_hard.jsonl")}
    (DATA / "pope_curriculum_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote train_pope_hard ({len(train_rows)}) + pope_hard bench ({len(held)})  meta={meta}")


def build_joint(src: str) -> None:
    """INT-002 RLAD-Joint training data: re-render the SAME curriculum problems
    (default train_curriculum.jsonl) with the combined 'write a cheatsheet then solve' prompt
    (templates.render_joint_prompt). Same problems/labels/qids as the faithful arm — only
    the prompt differs — so RLAD-Joint is comparable. The raw problem is read from metadata.problem."""
    from transformers import AutoTokenizer

    from .templates import render_joint_prompt

    src_path = DATA / src
    rows = [json.loads(line) for line in src_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    out = []
    for r in rows:
        md = r.get("metadata") or {}
        problem = md.get("problem")
        if not problem:
            raise KeyError(f"{src} row {md.get('qid')} lacks metadata.problem (needed to re-render RLAD-Joint prompt)")
        out.append({"input": render_joint_prompt(tok, problem), "label": r["label"],
                    "metadata": {"qid": md.get("qid"), "problem": problem}})
    _write_jsonl(DATA / "train_joint.jsonl", out)
    print(f"wrote train_joint ({len(out)}) from {src} -> {DATA / 'train_joint.jsonl'}")


def build_probe(per_bin: int, seed: int) -> None:
    """Stratified Qwen3-headroom probe: sample per-bin across POPE mean_reward bands into a
    benchmark (pope_probe.jsonl) + a sidecar id->mean_reward map, so scoring Qwen3-1.7B on it
    reveals which POPE band gives OUR 1.7B solver trainable difficulty (pass ~0.1-0.7)."""
    edges = [(0.0, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
             (0.5, 0.625), (0.625, 0.75), (0.75, 0.875)]
    rows = _pope_cols()
    rng = random.Random(seed)
    bench, idmap = [], {}
    for lo, hi in edges:
        cand = [(p, a, float(mr)) for r in rows
                if (p := (r.get("problem") or "").strip()) and (a := str(r.get("answer") or "").strip())
                and len(a) <= MAX_ANSWER_CHARS and (mr := r.get("mean_reward")) is not None and lo < mr <= hi]
        rng.shuffle(cand)
        for j, (p, a, mr) in enumerate(cand[:per_bin]):
            pid = f"popeprobe-{int(lo*1000):03d}-{j:03d}"
            bench.append({"id": pid, "problem": p, "answer": a}); idmap[pid] = mr
    _write_jsonl(BENCH / "pope_probe.jsonl", bench)
    (DATA / "pope_probe_meta.json").write_text(json.dumps(
        {"per_bin": per_bin, "edges": edges, "n": len(bench), "id_mean_reward": idmap}, indent=2) + "\n")
    print(f"wrote pope_probe ({len(bench)} problems across {len(edges)} bands) -> {BENCH/'pope_probe.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build-pool"); bp.add_argument("--n-pool", type=int, default=6000); bp.add_argument("--seed", type=int, default=42)
    pa = sub.add_parser("partition"); pa.add_argument("--hard-max", type=float, default=0.125); pa.add_argument("--easy-min", type=float, default=0.5)
    pp = sub.add_parser("build-pope")
    pp.add_argument("--mr-lo", type=float, default=0.0); pp.add_argument("--mr-hi", type=float, default=0.5)
    pp.add_argument("--n-cap", type=int, default=3254); pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--holdout", type=int, default=300)
    pr = sub.add_parser("build-probe")
    pr.add_argument("--per-bin", type=int, default=50); pr.add_argument("--seed", type=int, default=42)
    bj = sub.add_parser("build-joint")
    bj.add_argument("--src", type=str, default="train_curriculum.jsonl", help="source bare-problem JSONL under data/")
    args = ap.parse_args()
    if args.cmd == "build-pool":
        build_pool(args.n_pool, args.seed)
    elif args.cmd == "build-pope":
        build_pope(args.mr_lo, args.mr_hi, args.n_cap, args.seed, args.holdout)
    elif args.cmd == "build-probe":
        build_probe(args.per_bin, args.seed)
    elif args.cmd == "build-joint":
        build_joint(args.src)
    else:
        partition(args.hard_max, args.easy_min)


if __name__ == "__main__":
    main()
