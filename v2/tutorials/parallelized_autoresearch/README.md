# Parallelized autoresearch agent

An ML-engineer **agent** runs [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)-style
experiments on Flyte: it edits `train.py`, batches parallel sandbox runs on the
climbmix corpus, reads back **val_bpb** (validation bits per byte — lower is
better), keeps or discards, and iterates.

Built from the `flyte.ai.agents` toolkit with **code mode** (`code_mode=True`):
each turn the LLM writes a Python plan that calls batch tools and `flyte.map`.

## Entry point

| File | Role |
|------|------|
| `parallelized_autoresearch.py` | Code-mode fan-out agent — parallel batch experiments. |

## Supporting modules

| File | Role |
|------|------|
| `prepare.py` | Data prep (climbmix shards + BPE tokenizer) + eval utilities. |
| `train.py` | TinyGPT model + optimizer + training loop. Baseline for code edits. |
| `autoresearch_types.py` | Shared dataclasses (`ExperimentConfig`, leaderboard entries, …). |
| `bundle.py` | Flyte task environments + cached bundle build/profile tasks. |
| `research_tools.py` | Shared agent tools: arXiv search, dataset inspect, hypotheses, leaderboard. |
| `call_handlers.py` | Resource floor/ceiling helpers for OOM retries on `run_experiment`. |
| `sandbox_runner.py` | Stages files and runs edited `train.py` inside `unionai-sandbox`. |
| `code_edit_tools.py` | Code-edit tools: `edit_train_code`, `edit_train_code_batch`, promising-code memory. |
| `fanout_tools.py` | Batch tools: `record_batch_plan`, `run_experiment_batch`, `evaluate_batch_results`. |
| `reports.py` / `reports_code_edit_fanout.py` | Parse experiment results from the transcript; build the agent directive. |
| `report.py` | HTML for live Flyte reports (activity, leaderboard, memory, code edits). |

## The four constructs

1. **`flyte.ai.agents.Agent`** — a **code mode** loop. Tools are `@env.task` Flyte tasks;
   the agent orchestrates literature search, dataset inspection, code edits, and training.
2. **Self-healing on `run_experiment`** — inline memory bump + retry on Flyte or sandbox OOM
   (works with `flyte_map` and `run_experiment_batch`).
3. **Memory** — a keyed `MemoryStore` carries the transcript, leaderboard, saved
   `train.py` edits, and batch plans across runs.
4. **Reports** — live **Activity**, **Leaderboard**, **Code edits**, and **Memory** tabs
   (`report=True` on the parent task).

## Run it

Configure Flyte for the demo cluster:

```yaml
admin:
  endpoint: dns:///demo.hosted.unionai.cloud
image:
  builder: remote
task:
  org: demo
  project: flytesnacks
  domain: development
```

Launch the agent (images build remotely; no local Docker needed):

```bash
cd v2/tutorials/parallelized_autoresearch
uv run --script parallelized_autoresearch.py -- --n-experiments 6 --batch-size 3 --num-shards 1
```

Or invoke the agent task directly:

```bash
flyte run parallelized_autoresearch.py mle_autoresearch_code_fanout_agent \
  --n_experiments 6 --batch_size 3 --num_shards 1 --max_turns 12
```

## Run larger experiments

```bash
uv run --script parallelized_autoresearch.py -- \
  --n-experiments 250 --batch-size 10 --max-turns 1000 \
  --memory-key mle-autoresearch-code-fanout-001 --num-shards 4
```

> **Tips:** Code mode needs more turns than JSON tool mode (default `--max-turns 50`).
> Each turn is one Python plan, not one tool call. If the agent hits the turn limit,
> check the **Activity** tab for sandbox errors (Monty restrictions: no `+=`, no dict
> mutation). Successful batch runs are persisted to `memory/leaderboard.json` even when
> the agent does not finish with a plain-text summary.

> Scripts load `~/.flyte/config.yaml` and force the remote image builder. Override
> with `--config` or `FLYTE_CONFIG`.

Open the printed run URL to watch report tabs update live. Re-run with the same
`--memory-key` and the agent continues from prior experiments and saved code edits.

## Batch workflow

The agent typically runs a loop like:

1. **`record_batch_plan`** — persist 2–4 hypotheses for the next sweep.
2. **`edit_train_code_batch(edits=[...])`** — save a distinct `train.py` per title (one memory transaction).
3. **`record_batch_hypotheses`** — write expected effects before spending compute.
4. **`run_experiment_batch(titles, concurrency=3)`** — parallel sandbox runs via `flyte.map`.
5. **`evaluate_batch_results`** — rank by `val_bpb`, fork winners into the next batch.

In code mode the agent can express the same plan inline:

```python
titles = ["deeper-6L", "wider-384", "higher-lr"]
budgets = [45, 45, 45]
keys = ["mle-autoresearch-code-fanout"] * 3
results = list(flyte_map("run_experiment", titles, budgets, keys, concurrency=3))
evaluate_batch_results(results, batch_id="batch-1")
```

Several leaderboard rows may come from the **same batch** (fan-out group in the Flyte UI).
Check the **Memory** tab for `memory/batches.json`, `memory/promising_code.json`, and use
`get_promising_code` to inspect winning edits.
