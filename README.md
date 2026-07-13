# RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems

Official implementation of **RLAD** ([arXiv:2510.02263](https://arxiv.org/abs/2510.02263)).

RLAD teaches a language model to **discover reusable abstractions** — concise "cheatsheets"
of insights, lemmas, strategies, and pitfalls — and to **solve reasoning problems conditioned
on them**. It trains two roles:

- **Abstraction generator** `π_abs(z | x)` — proposes an abstraction `z` for a problem `x`
  (a short natural-language cheatsheet that must *not* reveal the answer).
- **Solution generator** `π_sol(y | x, z)` — solves the problem conditioned on the abstraction
  (and is also evaluated without one, as `π_sol(y | x)`).

At test time the model proposes several abstractions per problem and solves under each; the
abstraction-conditioned setting is the paper's headline.

> **About this repository.** The original RLAD experiments were built on an internal research
> codebase that is no longer actively maintained. **This repository reimplements RLAD from
> scratch as a plugin on top of [`miles`](https://github.com/radixark/miles)** (Megatron-LM +
> SGLang), an actively-maintained RL post-training framework, so the method is easy to run and
> extend today. It provides the **original RLAD method** *and* **two new fully-online variants**
> (below). `miles` is referenced as an external dependency (a pinned commit + a one-line patch),
> not vendored — see [`train/rl/REPRODUCTION.md`](train/rl/REPRODUCTION.md).

## ✨ Two online extensions

The original RLAD recipe is an **offline two-stage** pipeline: it warm-starts and then
*rejection-finetunes* `π_abs` **offline** (scoring each abstraction by how much it helps a
separately-trained solver), and only the solver is trained online. This repository adds two
**single-model, fully on-policy** variants that put the abstraction generator *inside* the RL
reward loop — one model learns to both abstract and solve, with no offline scoring stage:

- **RLAD-Joint** — the model writes a `<cheatsheet>` and then solves the problem in a **single
  trajectory**; the whole trajectory is trained end-to-end by the final-answer reward (plain
  GRPO over the samples per problem). *Learns to self-abstract and immediately use it, inline.*
- **RLAD-Hierarchical** — the model proposes **`n` abstractions** per problem and then **`m`
  solutions per abstraction**; each solution is rewarded by correctness and each abstraction is
  credited with the **mean reward of its own `m` solutions** (the paper's abstraction reward,
  Eq. 3, realized fully on-policy in one model). *Learns which abstractions help by how much
  they improve downstream solving, with hierarchical credit assignment.*

Both extensions and the faithful original share the same data pipeline, evaluation harness, and
`miles` setup — see the reproduction guide for a one-command path to each.

## Repository layout

| path | description |
|---|---|
| `train/rl/` | RL training + evaluation — the RLAD layer on top of `miles` |
| `train/rl/REPRODUCTION.md` | **full recipe**: environment, data, configs, training commands, evaluation |
| `train/rl/rlad_plugin/` | RLAD method code: data prep, on-policy rollouts, reward shaping, prompt templates |
| `train/rl/rlad_plugin/configs/` | per-arm training configs (original RLAD, the `+DAPO` baseline, and the two online extensions) |
| `train/rl/eval/` | dual evaluation (w/o-abs, w/abs-avg over K, w/abs-best) + rule-based graders |
| `train/rl/jobs/` | Slurm launch scripts: singleton-chained resumable training, checkpoint conversion, sharded eval |
| `train/rl/patches/` | the one-line `miles` sampling-temperature patch applied on top of the pinned commit |

## Reproducing

Everything needed is under **`train/rl/`** — see **[`train/rl/REPRODUCTION.md`](train/rl/REPRODUCTION.md)**
for the full recipe (environment setup, dataset preparation, configs, run commands, and
evaluation). For the original abstraction-generator SFT/RFT stages on an FSx-backed P5 Slurm
cluster—including two-node inference/data generation, HF publication, and the DeepScaleR-hard
five-metric W&B evaluation—the setup command also imports the pinned Miles training container—run
**[`RFT_pipeline.sh`](RFT_pipeline.sh)** or see
**[`train/rl/P5_FSX.md`](train/rl/P5_FSX.md)**. Each version has its own training config:

| version | config | one-line description |
|---|---|---|
| **RLAD** (original, offline two-stage) | `sft_absgen.sh` → `rft_absgen.sh` → `dapo_solgen_rlad.sh` | SFT `π_abs` → offline RFT `π_abs` → online DAPO `π_sol` with abstraction reward-masking |
| `+DAPO` baseline | `dapo_baseline.sh` | the paper's RL baseline (same setup, no abstractions) |
| **RLAD-Joint** (online) | `rlad_joint.sh` | single-trajectory self-abstract-then-solve, whole-trajectory reward |
| **RLAD-Hierarchical** (online) | `rlad_hierarchical.sh` | `n` abstractions × `m` solutions, per-abstraction credit from its solutions |

## Citation

If you use this code, please cite RLAD:

```bibtex
@misc{qu2025rladtrainingllmsdiscover,
      title={RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems},
      author={Yuxiao Qu and Anikait Singh and Yoonho Lee and Amrith Setlur and Ruslan Salakhutdinov and Chelsea Finn and Aviral Kumar},
      year={2025},
      eprint={2510.02263},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.02263},
}
```

## Acknowledgements & licensing

This repository is released under the **MIT License** (see [`LICENSE`](LICENSE)). The RL training
code builds on [`miles`](https://github.com/radixark/miles) (Megatron-LM + SGLang), which is
licensed under **Apache-2.0**; see [`NOTICE`](NOTICE). Thanks to the `miles` authors and to the
broader open RL post-training community.
