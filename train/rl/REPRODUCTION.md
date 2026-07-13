# RLAD — Reproduction Guide

This guide reproduces **RLAD** and the two online extensions from scratch on the
[`miles`](https://github.com/radixark/miles) RL post-training framework (Megatron-LM + SGLang).
It covers environment setup, dataset preparation, configs, training, checkpoint conversion, and
evaluation. All paths are relative to this directory (`train/rl/`).

The pipeline trains two roles: an **abstraction generator** `π_abs(z|x)` and an
**abstraction-conditioned solution generator** `π_sol(y|x,z)`. The repository provides four arms:

| arm | what it is | config(s) |
|---|---|---|
| **RLAD** (original) | SFT `π_abs` → offline RFT `π_abs` → online DAPO `π_sol` (reward-masked) | `sft_absgen.sh`, `rft_absgen.sh`, `dapo_solgen_rlad.sh` |
| `+DAPO` baseline | the paper's RL baseline (no abstractions) | `dapo_baseline.sh` |
| **RLAD-Joint** (online) | single model: write a `<cheatsheet>` then solve, in one trajectory; whole-trajectory reward | `rlad_joint.sh` |
| **RLAD-Hierarchical** (online) | single model: `n` abstractions × `m` solutions; per-abstraction credit = mean of its solutions | `rlad_hierarchical.sh` |

---

## 1. Environment setup

### 1a. The `miles` training framework (required for all training)

Training runs inside `miles` (Megatron-LM + SGLang). Clone it next to this directory, pin the
tested commit, and apply the one-line sampling-temperature patch shipped here:

```bash
cd train/rl
git clone https://github.com/radixark/miles.git miles
cd miles
git checkout 9437366e0                                   # pinned base commit
git apply ../patches/miles_div_temperature.patch         # in-place temperature div (avoids a [T,V] fp32 copy OOM)
cd ..
```

`miles` runs in a container image (pyxis/enroot `.sqsh`, or your own) that provides Megatron-LM +
SGLang and CUDA. Build/obtain that image per the `miles` README and point `RLAD_CONTAINER` at it
(next step). The training `.sbatch` scripts launch `python train.py` inside this container.

### 1b. Host environment (for data preparation + evaluation)

Data prep and evaluation run on the host (vLLM + HuggingFace), not in the training container:

```bash
conda create -n rlad python=3.12 -y && conda activate rlad
pip install -r ../../requirements.txt
```

Alternatively, source a site profile and run `scripts/bootstrap_host.sh`; it creates the host
environment when needed, installs the requirements, and prepares the pinned `miles` checkout.

### 1c. Cluster settings

All Slurm scripts source [`jobs/cluster_env.sh`](jobs/cluster_env.sh), which auto-detects the repo
layout (`RLAD_HOME`, `MILES_DIR`, `RLAD_RUNS`, `RLAD_DATA`) and exposes cluster knobs. Copy
`.env.cluster.example` to the ignored `.env.cluster`, customize it, and source it before submitting:

```bash
cp .env.cluster.example .env.cluster
source .env.cluster

export RLAD_ACCOUNT=<your_slurm_account>       # omit when the site does not require an account
export RLAD_PARTITION=<your_gpu_partition>
export RLAD_CONTAINER=/path/to/miles.sqsh
export RLAD_CONTAINER_MOUNTS=/shared:/shared   # e.g. /fsx:/fsx
```

`#SBATCH` directives do not expand shell variables. Single-node jobs therefore use
`jobs/sbatch.sh`; sharded host-side GPU jobs use `rlad_inference_sbatch`, which adds the configured
node count, nodelist, and GPUs per node. Both apply site options before allocation, while
`jobs/chain.sh` uses the single-node wrapper internally. Training uses one 8×H100 node with TP=2,
CP=2 and chains `≤4h` segments as a resumable singleton chain (see §5). Host-side generation and
evaluation use one process per allocated GPU. Submit all jobs from `train/rl/`; the wrappers create
`logs/`, route output there, and set `RLAD_HOME` as the job working directory.
See [`P5_FSX.md`](P5_FSX.md) for the concrete FSx/P5 profile.

---

## 2. Dataset preparation (DeepScaleR)

RLAD trains on [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
with a base-model-success **easy→medium curriculum** (hard split held out for eval).

```bash
cd train/rl
export PYTHONPATH=$PWD

# (1) sample a problem pool
python -m rlad_plugin.data_prep build-pool --n-pool 6000        # -> data/benchmarks/dsr_pool.jsonl

# (2) score the base model on the pool (per-problem success rate drives the curriculum)
rlad_inference_sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-1.7B,BENCHMARKS=dsr_pool,N_SAMPLES=8,MAX_TOKENS=8192,OUT_DIR=$RLAD_RUNS/eval/dsr_pool_score jobs/eval.sbatch

# (3) partition into easy / medium (+ held-out hard); rows carry the raw problem in metadata
python -m rlad_plugin.data_prep partition --hard-max 0.125 --easy-min 0.5
#   -> data/train_easy.jsonl, data/train_medium.jsonl, data/benchmarks/dsr_hard.jsonl

# (4) build the easy->medium curriculum file (shuffle OFF preserves the ordering)
cat data/train_easy.jsonl data/train_medium.jsonl > data/train_curriculum.jsonl
```

**Abstraction warm-start corpus** (seed abstractions for `π_abs` SFT). `warmstart_gen.py` distills a
non-leaking cheatsheet from each gold solution with a stronger instruct model (default
`Qwen3-4B-Instruct-2507`; configure your own via the script's flags) and applies a leakage filter:

```bash
rlad_inference_sbatch jobs/warmstart.sbatch # -> data/train_absgen_sft.jsonl  (chat "messages" format)
```

**Online-variant data.** RLAD-Hierarchical trains directly on `train_curriculum.jsonl` (bare
problems; it generates abstractions on-policy). RLAD-Joint uses a combined "write-cheatsheet-then-
solve" prompt built from the same problems:

```bash
python -m rlad_plugin.data_prep build-joint --src train_curriculum.jsonl   # -> data/train_joint.jsonl
```

> **Alternative dataset.** `data_prep.py build-pope` builds a curriculum from
> [`CMU-AIRe/POPE-more-64x32k`](https://huggingface.co/datasets/CMU-AIRe/POPE-more-64x32k) filtered
> to a difficulty band (`--mr-lo/--mr-hi`), useful if the base model saturates on DeepScaleR. Point
> any config's `PROMPT_DATA` at the resulting files to use it.

---

## 3. Configuration files

All arms source [`rlad_plugin/configs/common.sh`](rlad_plugin/configs/common.sh) (shared
Qwen3-1.7B model args, DAPO/GRPO knobs, TP2/CP2 perf args). Each per-arm config sets `ARM`, the
prompt data, batch sizes, and any `CUSTOM_ARGS` (custom rollout / reward hooks). Key knobs:

| config | role | notable settings |
|---|---|---|
| `sft_absgen.sh` | SFT `π_abs` | `--loss-type sft_loss`, 5 epochs, TP1, messages corpus |
| `rft_absgen.sh` | offline RFT `π_abs` | init from SFT `π_abs`, SFT on kept (high-reward) abstractions |
| `dapo_baseline.sh` | `+DAPO` baseline | stock GRPO, no abstractions |
| `dapo_solgen_rlad.sh` | online DAPO `π_sol` | abstraction-conditioned + bare mixture; Eq. 3 reward-masking (zero advantage on bare prompts, KL retained) |
| `rlad_joint.sh` | **RLAD-Joint** | stock GRPO on the combined prompt (`n_samples=16`) |
| `rlad_hierarchical.sh` | **RLAD-Hierarchical** | custom generate + reward hooks; `N_HINTS`×`(1+M_SOLS)` samples/prompt |

RLAD-Hierarchical exposes two extra knobs (defaults `4`/`4`, reach the rollout via env):
`N_HINTS` (`n` abstractions/problem = `--n-samples-per-prompt`) and `M_SOLS` (`m` solutions per
abstraction). The global batch is set to `ROLLOUT_BATCH × N_HINTS × (1+M_SOLS)` so a prompt's
samples stay grouped; `HINT_MAX_TOKENS` caps the (short) abstraction generation.

---

## 4. One-time base checkpoint

Convert the base model to Megatron `torch_dist` (used as `--ref-load` and init for all RL arms):

```bash
jobs/sbatch.sh jobs/prep_megatron_ckpt.sbatch      # Qwen/Qwen3-1.7B -> runs/qwen3_1p7b_torch_dist
```

---

## 5. Training

All training is launched with `jobs/chain.sh <config> <N_segments> [job_name]`, which submits `N`
resumable `≤4h` segments as a Slurm **singleton chain** (each segment resumes from the last
checkpoint; abort with `scancel --name=<job_name>`). Checkpoints land in `runs/<ARM>/ckpts`.

### RLAD (original, offline two-stage)

```bash
cd train/rl

# (1) SFT pi_abs (uses the SFT launcher)
LAUNCHER=$PWD/jobs/sft_launch.sh jobs/chain.sh rlad_plugin/configs/sft_absgen.sh 2 rlad-sft-absgen
#   convert the final SFT checkpoint to HF (see §6) -> runs/sft_absgen/hf/iter_<N>

# (2) offline RFT pi_abs: score sampled abstractions by downstream solver success, keep the best
rlad_inference_sbatch --export=ALL,ABSGEN_HF=$RLAD_RUNS/sft_absgen/hf/iter_<N> jobs/rft_data.sbatch   # -> data/train_absgen_rft.jsonl
LAUNCHER=$PWD/jobs/sft_launch.sh jobs/chain.sh rlad_plugin/configs/rft_absgen.sh 1 rlad-rft-absgen
#   convert -> runs/sft_absgen_rft/hf/iter_<N>   (this is the shared pi_abs used at eval)

# (3) build the abstraction-conditioned solver data from the RFT'd pi_abs, then online DAPO pi_sol
jobs/sbatch.sh --export=ALL,ABSGEN_HF=$RLAD_RUNS/sft_absgen_rft/hf/iter_<N> jobs/solgen_data.sbatch   # -> data/train_solgen_rlad.jsonl
jobs/chain.sh rlad_plugin/configs/dapo_solgen_rlad.sh 8 rlad-dapo-solgen
```

### `+DAPO` baseline

```bash
jobs/chain.sh rlad_plugin/configs/dapo_baseline.sh 8 rlad-dapo-baseline
```

### RLAD-Joint (online)

```bash
jobs/chain.sh rlad_plugin/configs/rlad_joint.sh 4 rlad-joint
```

### RLAD-Hierarchical (online)

```bash
N_HINTS=4 M_SOLS=4 jobs/chain.sh rlad_plugin/configs/rlad_hierarchical.sh 4 rlad-hierarchical
```

> **Resumable-training note.** Because training is chained across short segments, verify the LR
> schedule spans the full horizon on resume (a mis-restored `consumed_samples` can collapse a
> cosine LR early). Plot the per-step LR from the training logs before trusting long runs.

---

## 6. Checkpoint conversion

`miles` in-training HF export under TP+CP is unreliable; always convert offline before evaluating:

```bash
jobs/sbatch.sh --export=ALL,CKPT_DIR=$RLAD_RUNS/<ARM>/ckpts,OUT_DIR=$RLAD_RUNS/<ARM>/hf_iter<N>,ITER=<N> jobs/convert_hf.sbatch
```

The output `hf_iter<N>/` is a standard HuggingFace model directory (load with
`AutoModelForCausalLM.from_pretrained(...)` or serve with vLLM).

---

## 7. Evaluation

`jobs/eval_rlad.sbatch` runs the **dual evaluation** on AIME 2025 (default): it proposes `K=4`
abstractions from `π_abs` and solves under each, reporting **w/o-abs**, **w/abs-avg** (average over
the K), and **w/abs-best**. It is resumable and shards the solve across GPUs. Provide the two roles
as HuggingFace checkpoint dirs (from §6):

```bash
# original RLAD / +DAPO / base: pi_abs = the RFT'd generator, pi_sol = the arm's solver checkpoint
rlad_inference_sbatch --export=ALL,ABSGEN_HF=$RLAD_RUNS/sft_absgen_rft/hf/iter_<N>,SOLGEN_HF=$RLAD_RUNS/<ARM>/hf_iter<N>,OUT=$RLAD_RUNS/eval/<ARM>,BENCHMARK=aime25 jobs/eval_rlad.sbatch
```

For the online variants the single model is both roles:

```bash
# RLAD-Hierarchical (and RLAD-Joint) self-hint dual eval: point both roles at the variant checkpoint
rlad_inference_sbatch --export=ALL,ABSGEN_HF=$RLAD_RUNS/rlad_hierarchical/hf_iter<N>,SOLGEN_HF=$RLAD_RUNS/rlad_hierarchical/hf_iter<N>,OUT=$RLAD_RUNS/eval/rlad_hierarchical,BENCHMARK=aime25 jobs/eval_rlad.sbatch

# RLAD-Joint native combined-prompt eval (its training distribution): MODE=joint (no separate pi_abs)
rlad_inference_sbatch --export=ALL,MODE=joint,SOLGEN_HF=$RLAD_RUNS/rlad_joint/hf_iter<N>,OUT=$RLAD_RUNS/eval/rlad_joint_joint,BENCHMARK=aime25 jobs/eval_rlad.sbatch
```

Results are written to `<OUT>/summary.json` (`woabs_pass1`, `wabs_avg_pass1`, `wabs_best_pass1`,
and `joint_pass1` for `MODE=joint`). Other benchmarks: pass `BENCHMARK=` a JSONL built with
`eval/prep_benchmarks.py` (e.g. `amc23`, `aime24`, `hmmt2025`), or the held-out `dsr_hard`.

For the original abstraction-generator pipeline, the root `RFT_pipeline.sh` automates a stricter
five-number `dsr_hard` comparison after RFT. It evaluates the base Qwen3-1.7B solver without a
hint and with `K=4` hints from (a) the untrained Qwen3-1.7B abstraction generator and (b) the
RFT-trained generator. The no-hint condition is shared, `N=32` samples are required for every
condition, and incomplete or duplicate groups are rejected before summary. The combined
percentages and run URL are saved under `runs/eval/absgen_compare/` and logged to W&B. The same
pipeline also publishes both corpora and both converted checkpoints to private-by-default HF
repositories; see `train/rl/P5_FSX.md` for destination overrides.

---

## 8. Tests

Pure-Python unit tests for the on-policy rollout and reward shaping (no GPU) live in
`rlad_plugin/tests/`:

```bash
cd train/rl
PYTHONPATH=$PWD:$PWD/miles python -m pytest rlad_plugin/tests/ -q
```
