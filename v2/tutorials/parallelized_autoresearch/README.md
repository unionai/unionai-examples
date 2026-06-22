# Parallelized autoresearch agent

An ML-engineer **agent** runs [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)-style
experiments on Flyte: it edits `train.py`, batches parallel sandbox runs on the
climbmix corpus, reads back **val_bpb** (validation bits per byte — lower is
better), keeps or discards, and iterates.

Built from the `flyte.ai.agents` toolkit with **code mode** (`code_mode=True`):
each turn the LLM writes a Python plan that calls batch tools and `flyte.map`.

## Files

| File | Role |
|------|------|
| `parallelized_autoresearch.py` | Entry point — agent task, `run_experiment`, and CLI. |
| `bundle.py` | Flyte task environments and climbmix bundle tasks. |
| `tools.py` | Agent tools, sandbox execution, memory, and batch fan-out helpers. |
| `ui.py` | Directives, leaderboard parsing, and Flyte report HTML. |
| `autoresearch_types.py` | Shared dataclasses (`ExperimentConfig`, leaderboard entries, …). |
| `prepare.py` | Data prep (climbmix shards + BPE tokenizer). |
| `train.py` | TinyGPT training loop — baseline for agent code edits. |

## Run it

```bash
cd v2/tutorials/parallelized_autoresearch
uv run --script parallelized_autoresearch.py -- --n-experiments 6 --batch-size 3 --num-shards 1
```

Or invoke the agent task directly:

```bash
flyte run parallelized_autoresearch.py mle_autoresearch_code_fanout_agent \
  --n_experiments 6 --batch_size 3 --num_shards 1 --max_turns 12
```

Re-run with the same `--memory-key` to resume prior experiments and saved code edits.
