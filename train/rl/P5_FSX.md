# Original RLAD SFT/RFT on P5 + FSx

This runbook prepares the original RLAD abstraction generator: warm-start SFT, offline RFT,
Hugging Face publication, and the final DeepScaleR-hard comparison. It assumes the repository is
cloned at `/fsx/gstevenw/testing_alignment_algos/RLAD`, each P5 node exposes eight GPUs, and
Pyxis/enroot accepts `--container-image`. GPU data generation and evaluation reserve
`ip-10-1-81-8` and `ip-10-1-38-11` together; training and conversion remain single-node.

## Automated controller

From the repository root, the stateful controller performs the stages below, validates every
handoff, and resumes recorded Slurm jobs without submitting duplicates:

```bash
cp train/rl/.env.cluster.example train/rl/.env.cluster
chmod 600 train/rl/.env.cluster
${EDITOR:-vi} train/rl/.env.cluster

./RFT_pipeline.sh setup
tmux new -s rlad-rft
# Then, inside tmux:
./RFT_pipeline.sh run
```

The controller remains in the foreground while GPU jobs run. If it is interrupted, the Slurm job
is deliberately left alone; reconnect and run `./RFT_pipeline.sh resume`. Use
`./RFT_pipeline.sh status` from another shell for a read-only summary. RFT caches are tied to the
exact SFT checkpoint and sampling settings. For stale or corrupt RFT artifacts under the same
settings, `./RFT_pipeline.sh archive-rft` moves RFT data, checkpoints, and local comparison outputs
aside instead of deleting them, and refuses to run while a related Slurm job is active. It does
not delete already-published HF repositories or W&B runs. To change pipeline parameters or
the repository commit, move both `train/rl/data` and `train/rl/runs` aside and start a fresh run.

The checked-in profile sets `RLAD_INFERENCE_NODES=2`, the two-node nodelist above,
`RLAD_GPUS_PER_NODE=8`, and therefore `NSHARDS=16`. The controller passes these settings only to
base scoring, warm-start generation, RFT generation/scoring, and final evaluation. Each allocation
starts one launcher task per node and one vLLM process per local GPU, producing disjoint global
shards `0` through `15`. A barrier completes before merge, validation, or summarization. Do not
change only `--nodes`: the launcher and global shard count must remain consistent.

After training, the same `run` command publishes four private repositories under the username
returned by the existing HF login:

- `rlad-original-absgen-sft-data` and `rlad-original-absgen-rft-data`
- `rlad-original-absgen-sft-model` and `rlad-original-absgen-rft-model`

Set `HF_REPO_PREFIX`, `HF_NAMESPACE`, or the four exact `HF_*_REPO` values in `.env.cluster` to
rename them. Set `HF_REPO_PRIVATE=0` only when the artifacts should be public. Uploads are
idempotent and `runs/rft_pipeline/hf_publish.json` records the verified destinations.

The final evaluation uses the codebase defaults on `data/benchmarks/dsr_hard.jsonl`: base solver
`Qwen/Qwen3-1.7B`, `K=4` hints, `N=32` solutions per condition, temperature `0.6`, top-p `0.95`,
and 32,768 output tokens. It logs exactly these percentages to the W&B project configured by
`WANDB_PROJECT` (default `repro-paper003-rlad`):

1. base without a hint;
2. base with untrained-base hints, averaged across hints;
3. base with the best untrained-base hint;
4. base with RFT-generator hints, averaged across hints;
5. base with the best RFT-generator hint.

The no-hint samples are generated only once. Evaluation shards resume at complete
problem/condition groups and may require many `03:55:00` segments; the controller submits up to
`EVAL_SEGMENTS=12` per invocation, then a later `./RFT_pipeline.sh resume` continues. Use
`./RFT_pipeline.sh status` to print the five values, W&B URL, and HF repository URLs.

The manual commands below remain useful for inspection and recovery.

## One-time machine setup

From `train/rl/`, create the private site profile and inspect every path before sourcing it:

```bash
cp .env.cluster.example .env.cluster
chmod 600 .env.cluster
source .env.cluster

mkdir -p "$RLAD_DATA" "$RLAD_RUNS" "$RLAD_LOGS" "$HF_HOME" "$(dirname "$RLAD_CONTAINER")"
scripts/bootstrap_host.sh
rlad_activate_conda
hf auth whoami
wandb login --verify
```

The bootstrap creates/updates the `rlad` Conda environment and prepares `miles` at commit
`9437366e0` with the repository patch. It does not create the training container. Copy or build a
compatible `miles.sqsh`, then verify `test -f "$RLAD_CONTAINER"`. All container-visible paths are
under `/fsx`, which is mounted as `/fsx:/fsx`; home mounting remains disabled. Run the HF check
after sourcing the profile because cached HF credentials live under its configured `HF_HOME`.

Always source `.env.cluster` in a new login shell. Submit single-node jobs through
`jobs/sbatch.sh` and sharded GPU generation/evaluation through `rlad_inference_sbatch`; both apply
the P5 partition, 32 CPUs per task, 400 GiB RAM per node, job working directory, and optional
account before Slurm parses the request.

## 1. Curriculum and SFT corpus

```bash
cd "$RLAD_HOME"
source .env.cluster
rlad_activate_conda
export PYTHONPATH=$PWD

python -m rlad_plugin.data_prep build-pool --n-pool 6000
rlad_inference_sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-1.7B,BENCHMARKS=dsr_pool,N_SAMPLES=8,MAX_TOKENS=8192,OUT_DIR=$RLAD_RUNS/eval/dsr_pool_score jobs/eval.sbatch
```

The base evaluation is resumable. Under a short wall-time, resubmit the same command until the
`samples*.jsonl` files contain 48,000 rows (6,000 problems × 8 samples):

```bash
awk 'END {print NR}' "$RLAD_RUNS"/eval/dsr_pool_score/dsr_pool/samples*.jsonl
```

Only then build the curriculum and launch local-teacher distillation:

```bash
python -m rlad_plugin.data_prep partition --hard-max 0.125 --easy-min 0.5
rlad_inference_sbatch jobs/warmstart.sbatch
```

The required output is `data/train_absgen_sft.jsonl`. `warmstart.sbatch` distributes curriculum
problems over all 16 GPUs, writes one atomic file per shard, then merges them in original curriculum
order. Increase `WARMSTART_TIME` if a shard cannot finish within the default `03:55:00`, then
inspect the failed log before explicitly retrying.

## 2. Base checkpoint and SFT

The base conversion can run in parallel with corpus generation:

```bash
jobs/sbatch.sh jobs/prep_megatron_ckpt.sbatch
```

After both prerequisites finish successfully, submit the two resumable SFT segments:

```bash
LAUNCHER=$RLAD_HOME/jobs/sft_launch.sh \
  jobs/chain.sh rlad_plugin/configs/sft_absgen.sh 2 rlad-sft-absgen
```

Wait until both SFT segments leave the queue successfully. Then convert the final Megatron
checkpoint offline into the exact layout consumed by RFT:

```bash
SFT_ITER=$(cat "$RLAD_RUNS/sft_absgen/ckpts/latest_checkpointed_iteration.txt")
jobs/sbatch.sh --export=ALL,CKPT_DIR=$RLAD_RUNS/sft_absgen/ckpts,OUT_DIR=$RLAD_RUNS/sft_absgen/hf/iter_${SFT_ITER},ITER=${SFT_ITER} jobs/convert_hf.sbatch
```

Wait for conversion and verify `runs/sft_absgen/hf/iter_${SFT_ITER}/config.json` exists.

## 3. RFT corpus and training

RFT scoring is resumable and is the expensive stage. Queue six singleton segments for the
reference four-hour limit:

```bash
SFT_HF=$RLAD_RUNS/sft_absgen/hf/iter_${SFT_ITER}
for _ in $(seq 6); do
  rlad_inference_sbatch --job-name=rlad-rft-data --dependency=singleton \
    --export=ALL,ABSGEN_HF=${SFT_HF} jobs/rft_data.sbatch
done
```

Do not start RFT training until all cached abstractions are scored and the corpus is nonempty:

```bash
N_ABS=$(awk 'END {print NR}' data/rft_abs_cache*.jsonl)
N_SCORED=$(awk 'END {print NR}' data/rft_scored*.jsonl)
test "$N_ABS" -gt 0
test "$N_ABS" -eq "$N_SCORED"
test -s data/train_absgen_rft.jsonl
python -m json.tool data/absgen_rft_meta.json
```

Then submit rejection fine-tuning:

```bash
LAUNCHER=$RLAD_HOME/jobs/sft_launch.sh \
  jobs/chain.sh rlad_plugin/configs/rft_absgen.sh 1 rlad-rft-absgen
```

Wait for the RFT training job to finish successfully before reading its checkpoint tracker and
submitting the final conversion:

```bash
RFT_ITER=$(cat "$RLAD_RUNS/sft_absgen_rft/ckpts/latest_checkpointed_iteration.txt")
jobs/sbatch.sh --export=ALL,CKPT_DIR=$RLAD_RUNS/sft_absgen_rft/ckpts,OUT_DIR=$RLAD_RUNS/sft_absgen_rft/hf/iter_${RFT_ITER},ITER=${RFT_ITER} jobs/convert_hf.sbatch
```

The final model is `runs/sft_absgen_rft/hf/iter_${RFT_ITER}`. Continue resuming an interrupted run
with the same artifacts, but do not reuse `data/rft_*` caches after changing the SFT checkpoint,
`K`, `M`, or problem subset; archive them or use a fresh checkout first.
