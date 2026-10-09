# Pandera whole-dataset validation

A benchmark that measures how [pandera](https://pandera.readthedocs.io) validation time grows
with dataset size when you validate a whole `TensorDict` dataset once, where it's produced,
instead of every batch inside a training loop. It's the companion to
[`pandera_validation_overhead`](../pandera_validation_overhead/), which measures the
per-batch cost.

## Requirements / spec

- **Inputs** (all entrypoint parameters of `main`)
  - `sizes_mb: list[int] = [16, 64, 256, 1024, 4096]`: dataset sizes in memory.
  - `repeats: int = 3`: runs per point.
  - `chunk_mb: int = 64`: chunk size for streaming validation.
  - `sweep_size_mb: int = 1024`: dataset size for the chunk-size sweep.
  - `sweep_chunk_rows: list[int] = [256, 4_096, 65_536, 1_048_576]`: chunk sizes to sweep
    (the whole dataset at once is always added).
- **Pipeline**
  1. `run_dataset_benchmark`: for each of four synthetic datasets (continuous control, Atari
     frames, tabular classification, token sequences) and each size, builds one in-memory
     `TensorDict` and streams it through `validate(inplace=True)` in chunks. Before timing a
     dataset, it corrupts one value in the last row of a small copy and asserts validation
     raises. Then it validates the same 1 GB continuous-control dataset at each chunk size.
  2. `plot`: renders the chart in light and dark themes, plus tables, into the report tab.
- **Outputs**
  - `main` returns a JSON string. The `__main__` driver writes it to `results/results.json`
    and renders `dataset-validation.png` and `dataset-validation-dark.png` locally.

The schemas use the object-based `TensorDictSchema` API because several keys need custom
`Check`s, which `TensorDictModel` fields don't accept in pandera 0.34. The task requests
4 CPUs and 12 GiB. The committed `results/` come from a run on x86_64, torch 2.14.1+cpu,
pandera 0.34.0, tensordict 0.14.2.

## Run it

```bash
# On the cluster in ~/.flyte/config.yaml; builds the image remotely, then writes results/
uv run main.py

# Locally, with small sizes
uv run main.py --local --sizes-mb 8 32 128 --repeats 2 \
    --sweep-size-mb 64 --sweep-chunk-rows 256 4096 65536 --out /tmp/dataset-validation

# Through the examples test harness
make test-local FILE=v2/tutorials/pandera_dataset_validation/main.py
```

Driver flags: `--sizes-mb`, `--repeats`, `--chunk-mb`, `--sweep-size-mb`,
`--sweep-chunk-rows`, `--out`.

## What to look at

- `build_schema`: one `TensorDictSchema` per dataset, with custom checks for one-hot rows,
  finite values, blank frames, and the `-100` ignore index.
- `assert_checks_scan_everything`: the guard that refuses to time a schema that could pass
  without scanning every row.
- `validate_streamed`: slicing a `TensorDict` returns views, so chunking adds no copies, and
  `inplace=True` skips the defensive clone `validate()` makes by default.
