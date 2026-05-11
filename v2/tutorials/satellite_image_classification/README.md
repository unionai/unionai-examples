# Satellite Image Classification

A Flyte v2 pipeline that fine-tunes EfficientNet-B0 on the EuroSAT satellite
imagery dataset, logs metrics and t-SNE visualizations to Weights & Biases,
and renders training curves in the Union UI.

The pipeline has three tasks:

1. `load_dataset` — downloads EuroSAT JPEGs via `torchvision` and caches them
   as a `flyte.io.Dir` (CPU).
2. `train_model` — runs two-phase EfficientNet-B0 training with Lightning and
   W&B logging (GPU, `T4:1`).
3. `create_report` — renders loss/accuracy curves in the Union UI (CPU).

## Install dependencies

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

These match the packages baked into the task image in `config.py`, so the
local environment used to submit the run mirrors the remote execution
environment.

## Configure Flyte / Union

The driver calls `flyte.init_from_config()`, which loads your Flyte/Union
connection from `~/.flyte/config.yaml` (or the `FLYTE_CONFIG` environment
variable). Make sure you have a working config that points at the cluster
you want to submit to.

## Configure Weights & Biases

The GPU task expects a Flyte secret named `wandb_api_key` that is mounted as
the `WANDB_API_KEY` environment variable (see `training_env` in `config.py`).
Create it once in your project, e.g.:

```bash
flyte create secret wandb_api_key
```

By default the run logs to the `satellite-classification` W&B project with no
entity override. To change this, edit `TRAINING_CONFIG` in `run.py` and set
`wandb_project` / `wandb_entity`, or update the `TrainingConfig` defaults in
`config.py`.

## Run the pipeline

```bash
python run.py
```

The script submits the pipeline to your configured Flyte/Union cluster and
prints a run URL you can open in the UI to follow progress, view the W&B
links, and see the rendered report once training finishes.
