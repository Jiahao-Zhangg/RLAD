# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `train/rl/rlad_plugin/`: data preparation, prompt templates, rollout logic, and reward shaping. Per-training-arm shell configurations are under `rlad_plugin/configs/`, while focused pytest coverage is in `rlad_plugin/tests/`. Evaluation utilities belong in `train/rl/eval/`; Slurm launchers and cluster setup live in `train/rl/jobs/`; the compatibility patch for the external `miles` framework is in `train/rl/patches/`. Treat `train/rl/REPRODUCTION.md` as the authoritative workflow. The cloned `train/rl/miles/` tree and generated `data/`, `runs/`, and `logs/` directories are intentionally ignored.

## Setup, Test, and Development Commands

Host-side data preparation and evaluation use Python 3.12:

```bash
pip install -r requirements.txt
cd train/rl
export PYTHONPATH=$PWD
python -m rlad_plugin.data_prep build-pool --n-pool 6000
PYTHONPATH=$PWD:$PWD/miles python -m pytest rlad_plugin/tests/ -q
```

The first module command builds a training-problem pool; the pytest command runs the CPU-only unit suite, but requires the pinned `miles` checkout described in the reproduction guide. Training is not a local build: submit resumable Slurm segments with, for example, `jobs/chain.sh rlad_plugin/configs/dapo_baseline.sh 8 rlad-dapo-baseline`.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for functions and variables, `CapWords` for classes, `_leading_underscores` for internal helpers, and `UPPER_CASE` for constants. Prefer `pathlib.Path`, type hints where practical, and import groups ordered standard-library, third-party, then local. No formatter or linter is configured, so match surrounding code and keep diffs focused. In Bash, preserve uppercase environment knobs, quoted expansions, argument arrays, and `${VAR:-default}` overrides.

## Testing Guidelines

Name files `test_*.py` and functions `test_<behavior>`. Add focused regression tests beside `test_rollout_rlad.py`; stub tokenizers, network generation, and external services so tests remain deterministic and GPU-free. There is no configured coverage threshold, but new rollout and reward branches should be exercised. Run the full pytest command above before opening a PR.

## Commit & Pull Request Guidelines

History is currently minimal. Follow the existing scoped-subject style (`RLAD: ...`; component scopes such as `eval:` are also clear) and explain rationale or operational impact in the body. PRs should summarize the change, identify affected arms/configs/jobs, list validation, and note data, checkpoint, cluster, or pinned-`miles` compatibility. Update `README.md` or `train/rl/REPRODUCTION.md` when workflows change; link issues and include logs only when relevant.

## Configuration & Secrets

Set site-specific values in `jobs/cluster_env.sh` or export `RLAD_ACCOUNT`, `RLAD_PARTITION`, and `RLAD_CONTAINER`. Keep `HF_TOKEN`, `WANDB_API_KEY`, checkpoints, and `.env` files out of commits.
