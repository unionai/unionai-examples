# Pandera TensorDict validation overhead

A benchmark that measures how much wall-clock time it adds to a training loop when
[pandera](https://pandera.readthedocs.io) validates every `TensorDict` batch and the loop
skips the ones that fail. Two tasks run on one CPU pod each, and a `report=True` task
renders the charts and tables into the run's report tab.

## Requirements / spec

- **Inputs** (all entrypoint parameters of `main`)
  - `lengths: list[int] = [100, 300, 1_000, 3_000, 10_000]`: training lengths, in steps.
  - `repeats: int = 5`: runs per length and mode.
  - `batch_size: int = 256`, `obs_dim: int = 64`, `hidden: int = 256`: batch and MLP shape.
  - `corrupt_rate: float = 0.05`: share of batches corrupted.
  - `pool_size: int = 200`: pre-generated batches the loop cycles through.
  - `widths: list[int] = [64, 256, 1_024, 2_048, 4_096]`: hidden widths for the step-size sweep.
  - `sweep_repeats: int = 5`, `sweep_target_seconds: float = 15.0`: runs per width, and
    roughly how long each run takes.
- **Pipeline**
  1. `run_benchmark`: trains a 3-layer MLP policy (REINFORCE-style loss, Adam) at every
     length in two modes. *Without pandera* skips corrupt batches with a precomputed mask;
     *with pandera* calls `Transition.validate(td)` and skips on `SchemaError`. Both modes
     train on the same batches, and the task asserts they skip the same steps.
  2. `run_step_size_sweep`: keeps the batch and schema fixed and grows the MLP's hidden
     width, so the training step gets more expensive while validation work stays the same.
  3. `plot`: renders both charts in light and dark themes into the report tab.
- **Outputs**
  - `main` returns two JSON strings (length benchmark and sweep). The `__main__` driver
    writes them to `results/results.json` and `results/step-size.json` and renders
    `overhead.png`, `overhead-dark.png`, `step-size.png`, and `step-size-dark.png` locally.

5% of batches are corrupted in one of three ways: `float64` observations, one action outside
the action space, or one `NaN` reward. The schema checks dtype, shape, and values on all four
keys. The committed `results/` come from a run on 4 CPU cores (x86_64, torch 2.14.1+cpu,
pandera 0.34.0, tensordict 0.14.2).

## Run it

```bash
# On the cluster in ~/.flyte/config.yaml; builds the image remotely, then writes results/
uv run main.py

# Locally, with a short configuration
uv run main.py --local --lengths 100 300 --repeats 2 \
    --widths 64 256 --sweep-repeats 2 --sweep-target-seconds 1 --out /tmp/overhead

# Through the examples test harness
make test-local FILE=v2/tutorials/pandera_validation_overhead/main.py
```

Driver flags: `--lengths`, `--repeats`, `--batch-size`, `--obs-dim`, `--hidden`,
`--corrupt-rate`, `--widths`, `--sweep-repeats`, `--sweep-target-seconds`, `--out`.

Results come back from the run as inline JSON strings rather than `File`s, so the driver
doesn't need credentials for the cluster's object store.

## What to look at

- `build_schema`: the `Transition` `TensorDictModel`.
- `train`: the loop, and where validation sits in it.
- `run_benchmark`: alternating mode order across repeats, and the assertion that pandera
  skips exactly the batches the mask does.
- `env`: the CPU limit is set above the torch thread count on purpose. With the limit equal
  to the thread count, CFS throttling skewed the comparison for larger models.
