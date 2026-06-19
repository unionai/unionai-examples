# Autoresearch with an MLE agent

The running examples behind the workshop. An ML-engineer **agent** runs
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)-style
experiments on Flyte: it trains a TinyGPT variant for a short budget on the
climbmix corpus, reads back **val_bpb** (validation bits per byte — lower is
better), keeps or discards, and iterates.

All examples are built from the `flyte.ai.agents` toolkit and demonstrate the
constructs that make agents *durable* and *self-healing* on Flyte.

## Three agent variants

| | **`mle_agent.py`** | **`mle_agent_code_edit.py`** | **`mle_agent_code_edit_fanout.py`** |
|---|---|---|---|
| **How the agent proposes experiments** | Structured tool args (`n_layer`, `n_embd`, …) | Edits full `train.py` source (upstream karpathy style) | Same code edits, orchestrated in **batches** |
| **Agent loop** | JSON tool calls | JSON tool calls | **`code_mode=True`** — LLM writes Python plans each turn |
| **`run_experiment`** | Calls `train.run_training(config)` directly | Runs edited code in [`unionai-sandbox`](https://www.union.ai/docs/v2/union/user-guide/sandboxing/interactive-sandboxes/) | Same sandbox run, fan-out via **`flyte.map`** / `run_experiment_batch` |
| **OOM self-healing** | `call_handlers.heal_oom` (`flyte.errors.OOMError`) | `call_handlers_sandbox.heal_sandbox_oom` (sandbox stderr) | Inline retry loop on `run_experiment` (works with `flyte_map`) |
| **Memory key default** | `mle-autoresearch` | `mle-autoresearch-code` | `mle-autoresearch-code-fanout` |
| **Extra memory artifacts** | Leaderboard + hypotheses | + `memory/code/*.py`, promising-code ledger | + `memory/batches.json` batch plans |

Use **`mle_agent.py`** for structured tools and Flyte-native OOM recovery (slides default).

Use **`mle_agent_code_edit.py`** for the upstream-faithful code-editing loop and sandbox isolation.

Use **`mle_agent_code_edit_fanout.py`** for the **scaled-up** story: plan multiple
hypotheses, edit several `train.py` variants, run them **in parallel**, rank the
batch, and iterate — all from generated Python that calls `flyte_map("run_experiment", ...)`.

## Files

| File | Role |
|------|------|
| `prepare.py` | Data prep (climbmix shards + BPE tokenizer) + eval utilities. **Fixed infrastructure.** |
| `train.py` | TinyGPT model + optimizer + training loop. Baseline for all agents. |
| `autoresearch_types.py` | Shared dataclasses (`ExperimentConfig`, leaderboard entries, …). |
| `bundle.py` | Flyte task environments + cached bundle build/profile tasks. |
| `research_tools.py` | Shared agent tools: arXiv search, dataset inspect, hypotheses, leaderboard. |
| `call_handlers.py` | OOM-healing `call_handler` for structured `run_experiment`. |
| `call_handlers_sandbox.py` | OOM-healing `call_handler` for sandbox-backed `run_experiment`. |
| `sandbox_runner.py` | Stages files and runs edited `train.py` inside `unionai-sandbox`. |
| `code_edit_tools.py` | Code-edit tools: `edit_train_code`, `edit_train_code_batch`, `get_baseline_train_code`, promising-code memory. |
| `fanout_tools.py` | Batch tools: `record_batch_plan`, `run_experiment_batch`, `evaluate_batch_results`. |
| `reports.py` / `reports_code_edit.py` / `reports_code_edit_fanout.py` | Agent directives; parse results from the transcript. |
| `report.py` | HTML for live Flyte reports (activity, leaderboard, memory). |
| `mle_agent.py` | Structured-hyperparameter agent. |
| `mle_agent_code_edit.py` | Code-editing agent with sandbox execution. |
| `mle_agent_code_edit_fanout.py` | Code-mode fan-out agent — parallel batch experiments. |

## The four constructs (all examples)

1. **`flyte.ai.agents.Agent`** — a tool-use loop (or **code mode** loop). Tools are
   `@env.task` Flyte tasks; the agent orchestrates literature search, dataset
   inspection, and training.
2. **Self-healing on `run_experiment`** — right-size compute and **double memory +
   retry** on OOM (`call_handler` for JSON mode; inline retry for fan-out / `flyte_map`).
3. **Memory** — a keyed `MemoryStore` carries the transcript, leaderboard, saved
   `train.py` edits, and (fan-out) batch plans across runs.
4. **Reports** — live **Activity**, **Leaderboard**, and **Memory** tabs
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

Install deps and launch any agent (images build remotely; no local Docker needed):

```bash
uv sync

# Structured hyperparameters (workshop default)
uv run python mle_agent.py --n-experiments 4 --num-shards 1

# Code-editing + unionai-sandbox (one experiment at a time)
uv run python mle_agent_code_edit.py --n-experiments 4 --num-shards 1

# Scaled-up: code mode + parallel batches
uv run python mle_agent_code_edit_fanout.py --n-experiments 6 --batch-size 3 --max-turns 50 --num-shards 1
```

## Run larger experiments

```bash
uv run python mle_agent.py --n-experiments 10 --max-turns 100 --num-shards 1 --memory-key mle-autoresearch-001

uv run python mle_agent_code_edit.py --n-experiments 10 --max-turns 100 --num-shards 1 --memory-key mle-autoresearch-code-001

uv run python mle_agent_code_edit_fanout.py --n-experiments 250 --batch-size 10 --max-turns 1000 --memory-key mle-autoresearch-code-fanout-001 --num-shards 4
```

> **Fan-out tips:** Code mode needs more turns than JSON tool mode (default
> `--max-turns 50`). Each turn is one Python plan, not one tool call. If the
> agent hits the turn limit, check the **Activity** tab for sandbox errors
> (Monty restrictions: no `+=`, no dict mutation). Successful batch runs are
> persisted to `memory/leaderboard.json` even when the agent does not finish
> with a plain-text summary.

> Scripts load `~/.flyte/config.yaml` and force the remote image builder. Override
> with `--config` or `FLYTE_CONFIG`.

Open the printed run URL to watch report tabs update live. Re-run with the same
`--memory-key` and the agent continues from prior experiments (and saved code edits).

## Fan-out batch workflow

The scaled-up agent (`code_mode=True`) typically runs a loop like:

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

## What a successful structured run looks like

A 4-experiment run with `mle_agent.py` improves `val_bpb` and heals an OOM along
the way (experiment 2 retried once at higher memory):

| # | Experiment | val_bpb | Resources | OOM retries |
|---|------------|---------|-----------|-------------|
| 1 | Baseline | 3.130 | cpu=2, mem=4Gi | — |
| 2 | Deeper (6 layers) | 2.983 | cpu=4, mem=8Gi | 1× |
| 3 | Wider (embd=384) | 2.906 | cpu=4, mem=8Gi | — |
| 4 | Higher LR (1e-3) | **2.880** 🏆 | cpu=2, mem=4Gi | — |

With `mle_agent_code_edit.py`, each row corresponds to a **saved `train.py` edit**.
With `mle_agent_code_edit_fanout.py`, several rows may come from the **same batch**
(fan-out group in the Flyte UI). Check the **Memory** tab for `memory/batches.json`,
`memory/promising_code.json`, and use `get_promising_code` to inspect winning edits.
