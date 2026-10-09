# What it costs to not validate

A benchmark that measures what happens to training when corrupt batches reach the model,
compared with a loop that validates every batch with [pandera](https://pandera.readthedocs.io)
and skips the ones that fail. It's the other side of
[`pandera_validation_overhead`](../pandera_validation_overhead/), whose baseline skips corrupt
batches with a precomputed mask that a real pipeline doesn't have.

## Requirements / spec

- **Inputs** (all entrypoint parameters of `main`)
  - `rates: list[float] = [0.01, 0.05, 0.2]`: share of batches corrupted.
  - `seeds: int = 30`: seeds per cell.
- **Pipeline**
  1. `run_cell`: for one (corruption, rate) cell, trains a REINFORCE policy (a 4-64-2 MLP,
     Adam at lr 1e-2) on CartPole-v1 for every seed, in both modes. Each update collects
     8 episodes into a `TensorDict`; a share of batches gets one `NaN` reward
     (`nan_reward`) or observations 100x too large (`obs_scale`). *Without pandera* trains on
     every batch; *with pandera* calls `Transition.validate(td, inplace=True)` and skips the
     batch on `SchemaError`.
  2. `main`: fans out one `run_cell` per cell with `asyncio.gather` (a clean cell plus each
     corruption at each rate), summarizes medians and quartiles, and calls `plot`.
  3. `plot`: renders the chart in light and dark themes into the report tab.
- **Outputs**
  - `main` returns a JSON string with the config, a per-cell summary, and every run. The
    `__main__` driver writes it to `results/results.json` and renders `cost.png` and
    `cost-dark.png` locally.

The cost is environment steps to reach a 475 return (CartPole-v1's "solved" threshold,
averaged over the last 20 episodes), counting steps lost to rollbacks and skipped batches.
Steps are deterministic for a given seed. A run that hasn't solved it by 1M steps stops and
counts at 1M. The committed `results/` come from 30 seeds per cell (420 runs) on x86_64,
torch 2.14.1+cpu, pandera 0.34.0, tensordict 0.14.3, gymnasium 1.4.0.

## Run it

```bash
# On the cluster in ~/.flyte/config.yaml; builds the image remotely, then writes results/
uv run main.py

# Locally, with a short configuration
uv run main.py --local --seeds 3 --rates 0.05 --out /tmp/cost

# Through the examples test harness
make test-local FILE=v2/tutorials/pandera_cost_of_not_validating/main.py
```

Driver flags: `--rates`, `--seeds`, `--out`.

## What to look at

- `build_schema`: dtypes and shapes on every key, observations within ±10, actions in
  `{0, 1}`, and rewards in `[0, 1]`, which also rejects `NaN`.
- The `corrupt-and-validate` fragment in `train`: corruption uses its own random stream, so
  both modes see the same schedule.
- The `rollback` fragment in `train`: a `NaN` update crashes the next rollout, and the loop
  rolls back to the newest checkpoint whose weights and optimizer state are finite. It loads
  copies, because `Optimizer.load_state_dict` keeps references to the tensors it's given.
