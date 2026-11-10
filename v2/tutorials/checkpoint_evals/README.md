# Evaluate every checkpoint

A trainer fine-tunes SmolLM2-360M on math word problems and publishes every checkpoint as a new
version of the `math-sft-checkpoint` artifact, without pausing training. A separately deployed
evaluator has an artifact trigger on that artifact, so each new checkpoint starts its own eval
run: GSM8K with vLLM, and ARC-Easy and MMLU by log-likelihood, in parallel on their own GPUs.
Every benchmark result is cached per (checkpoint, benchmark, benchmark version). The trainer's
report draws learning curves as the scores arrive, applies a ship rule (best GSM8K whose MMLU
stays within two points of the base model), and promotes the winner as
`math-sft-release-candidate`.

The tutorial that walks through this code is at
https://www.union.ai/docs/v2/union/tutorials/frontier-ai/checkpoint-evals/.

## Files

| File | What it does |
| --- | --- |
| `config.py` | Artifact names, the benchmark registry, the ship rule, and the images |
| `train.py` | The fine-tuning task that publishes checkpoints, and the driver with the live leaderboard |
| `evals.py` | The artifact trigger, the per-checkpoint eval run, the benchmark tasks, and backfill |
| `benchmarks.py` | Benchmark loading, prompts, answer extraction, and scoring |
| `reports.py` | The scorecard, training report, leaderboard, and release card |
| `viz.py` | Shared HTML and SVG building blocks for reports and cards |

## Requirements

A Union.ai cluster with one NVIDIA L40S for training and L4s for evals (any GPU with 24 GB or more
works; change `train_env` in `train.py` and the environments in `evals.py`). Artifacts and
artifact triggers are Union features.

## Run it

```bash
uv run train.py                                    # deploy the evaluator, then train and watch
uv run train.py --smoke                            # 2 checkpoints, 40 problems per benchmark
uv run evals.py                                    # deploy only the evaluator and its trigger
uv run evals.py --backfill <train_run> arc_challenge   # after adding arc_challenge to BENCHMARKS
uv run train.py --leaderboard <train_run>          # redraw the leaderboard after a backfill
```
