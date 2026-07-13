# Original RLAD SFT/RFT on P5 + FSx

This runbook prepares only the original RLAD abstraction generator: warm-start SFT followed by
offline RFT. It assumes the repository is cloned at
`/fsx/gstevenw/testing_alignment_algos/RLAD`, Slurm exposes eight GPUs with
`--gpus-per-node=8`, and Pyxis/enroot accepts `--container-image`.

## One-time machine setup

From `train/rl/`, create the private site profile and inspect every path before sourcing it:

```bash
cp .env.cluster.example .env.cluster
chmod 600 .env.cluster
source .env.cluster

mkdir -p "$RLAD_DATA" "$RLAD_RUNS" "$RLAD_LOGS" "$HF_HOME" "$(dirname "$RLAD_CONTAINER")"
scripts/bootstrap_host.sh
```

The bootstrap creates/updates the `rlad` Conda environment and prepares `miles` at commit
`9437366e0` with the repository patch. It does not create the training container. Copy or build a
compatible `miles.sqsh`, then verify `test -f "$RLAD_CONTAINER"`. All container-visible paths are
under `/fsx`, which is mounted as `/fsx:/fsx`; home mounting remains disabled.

Always source `.env.cluster` in a new login shell. Submit direct jobs through `jobs/sbatch.sh` so
the P5 partition, 32 CPUs, 400 GiB RAM, job working directory, and optional account are applied
before Slurm parses the request.

## 1. Curriculum and SFT corpus

```bash
cd "$RLAD_HOME"
source .env.cluster
rlad_activate_conda
export PYTHONPATH=$PWD

python -m rlad_plugin.data_prep build-pool --n-pool 6000
jobs/sbatch.sh --export=ALL,MODEL_PATH=Qwen/Qwen3-1.7B,BENCHMARKS=dsr_pool,N_SAMPLES=8,MAX_TOKENS=8192,OUT_DIR=$RLAD_RUNS/eval/dsr_pool_score jobs/eval.sbatch
```

The base evaluation is resumable. Under a short wall-time, resubmit the same command until the
`samples*.jsonl` files contain 48,000 rows (6,000 problems × 8 samples):

```bash
awk 'END {print NR}' "$RLAD_RUNS"/eval/dsr_pool_score/dsr_pool/samples*.jsonl
```

Only then build the curriculum and launch local-teacher distillation:

```bash
python -m rlad_plugin.data_prep partition --hard-max 0.125 --easy-min 0.5
jobs/sbatch.sh jobs/warmstart.sbatch
```

The required output is `data/train_absgen_sft.jsonl`. `warmstart.sbatch` uses one GPU for the
Qwen3-4B-Instruct teacher even though the full P5 node is allocated.

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
  jobs/sbatch.sh --job-name=rlad-rft-data --dependency=singleton \
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
