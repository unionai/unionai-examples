# # Forecasting with GluonTS & PyTorch on GPUs
#
# In this tutorial, we learn how to train and evaluate a time series forecasting model
# with [GluonTS](https://ts.gluon.ai/stable/) on GPUs. We'll train a
# [DeepAR modal](https://arxiv.org/abs/1704.04110), an auto-regressive deep network that
# produces probabilistic forecasts. On Union, we show how to easily request for an A100,
# specify dependencies for this workflow, and visualize our results with Flyte Deck.

# {{run-on-union}}

# ## Managing Dependencies
#
# First, let's import the workflow dependencies:

import os
from pathlib import Path

from flytekit import task, Resources, current_context, workflow, Deck, ImageSpec
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekit.extras.accelerators import A100, L4

# We use Flytekit's `ImageSpec` to specify our Python dependencies as a list of `packages`.
# We'll use `gluonts` which uses PyTorch Lightning for training and `matplotlib` for
# visualizations.

gluon_image = ImageSpec(
    "gluon-time-series",
    builder="union",
    packages=[
        "torch==2.3.1",
        "gluonts[torch]==0.15.1",
        "matplotlib==3.9.1",
        "orjson==3.10.6",
        "union",
        "pandas==2.2.2",
    ],
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)

# ## Download our time series dataset
#
# First, we download the `m4_hourly` dataset and cache it, so we can reuse the data
# in sequential task without re downloading it every time:


@task(container_image=gluon_image, requests=Resources(cpu="2", mem="6Gi"))
def download_m4_hourly_dataset() -> FlyteFile:
    from gluonts.dataset.repository import get_dataset

    working_dir = Path(current_context().working_directory)
    dataset_path = working_dir / "dataset_path"
    dataset_path.mkdir(exist_ok=True)

    get_dataset("m4_hourly", path=dataset_path)

    dataset_compressed = working_dir / "dataset_path.tar.gz"
    _compress(dataset_path, dataset_compressed)
    return dataset_compressed


# ## Training DeepAR on A100
#
# For our model training task, we use `flytekit's` `@task` decorator to configure the
# hardware and image used to train our model. With `requests=Resources(gpu="1")` and
# `accelerator=A00`, we declare the hardware require to train out deep learning
# forecasting model. The body of the function is training code from DeepAR's
# [quick start tutorial](https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html).
# The task returns a `FlyteDirectory` containing the serialized predictor model.


@task(
    container_image=gluon_image,
    requests=Resources(gpu="1", cpu="2", mem="6Gi"),
    accelerator=A100,
)
def train_predictor(dataset: FlyteFile) -> FlyteDirectory:
    from gluonts.dataset.repository import get_dataset
    from gluonts.torch import DeepAREstimator

    dataset.download()
    working_dir = Path(current_context().working_directory)
    dataset_path = working_dir / "dataset"
    _decompress(dataset.path, dataset_path)
    dataset = get_dataset("m4_hourly", path=dataset_path)

    model = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq="h",
        trainer_kwargs={"max_epochs": 5},
    )

    predictor = model.train(dataset.train)

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    predictor_path = working_dir / "predictor"
    predictor_path.mkdir(exist_ok=True)

    predictor.serialize(predictor_path)
    return predictor_path


# ## Forecasting and Evaluating Model
#
# For computing forecast, we declare a `L4` GPU as our accelerator and a specific amount
# of CPU and memory. We also pass in `enable_deck=True` to visualize our evaluation
# in a Flyte Deck.


@task(
    container_image=gluon_image,
    requests=Resources(gpu="1", cpu="4", mem="4Gi"),
    accelerator=L4,
    enable_deck=True,
)
def compute_forecasts(dataset: FlyteFile, predictor_directory: FlyteDirectory):
    from gluonts.dataset.repository import get_dataset
    from gluonts.torch.model.predictor import PyTorchPredictor
    from gluonts.evaluation import make_evaluation_predictions
    from gluonts.evaluation import Evaluator
    import pandas as pd

    # Download input data
    dataset.download()
    working_dir = Path(current_context().working_directory)
    dataset_path = working_dir / "dataset"
    _decompress(dataset.path, dataset_path)
    dataset = get_dataset("m4_hourly", path=dataset_path)

    predictor_directory.download()

    predictor = PyTorchPredictor.deserialize(Path(predictor_directory))

    # Create forecast
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=100,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    ts_entry = tss[0]
    forecast_entry = forecasts[0]

    # Plot forecast
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ts_entry[-150:].to_timestamp())
    ax.tick_params(axis="x", rotation=30)
    forecast_entry.plot(show_label=True, ax=ax)
    fig.legend()

    ctx = current_context()
    forecast_deck = Deck("Forecast", _fig_to_html(fig))
    ctx.decks.insert(0, forecast_deck)

    # Evaluate
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(tss, forecasts)

    agg_df = pd.DataFrame.from_dict(agg_metrics, orient="index")
    agg_df.columns = ["metric values"]

    metrics_deck = Deck("Metrics", agg_df.to_html())
    ctx.decks.insert(1, metrics_deck)


# ## Complete Workflow
#
# Finally, we define the workflow that calls `train_predictor` and passes it's output
# to `compute_forecasts`. We run the workflow by:
#
# ```shell
# $ union run --remote gluonts_time_series.py gluonts_wf
# ```


@workflow
def gluonts_wf() -> FlyteDirectory:
    dataset = download_m4_hourly_dataset()
    predictor_directory = train_predictor(dataset=dataset)
    compute_forecasts(dataset=dataset, predictor_directory=predictor_directory)
    return predictor_directory


# ## Appendix
#
# The following are helper functions used by our Flyte tasks. We include functions to
# convert a matplotlib figure to HTML, decompress and compress a directory.


def _fig_to_html(fig) -> str:
    """Convert matplotlib figure to HTML."""
    import io
    import base64

    fig_bytes = io.BytesIO()
    fig.savefig(fig_bytes, format="jpg")
    fig_bytes.seek(0)
    image_base64 = base64.b64encode(fig_bytes.read()).decode()
    return f'<img src="data:image/png;base64,{image_base64}" alt="Rendered Image" />'


def _decompress(src: Path, dest: Path):
    """Decompress src into dest."""
    import tarfile

    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(path=dest)


def _compress(src: Path, dest: Path):
    """Compress src into dest."""
    import tarfile

    with tarfile.open(dest, "w:gz") as tar:
        for file in src.rglob("*"):
            tar.add(file, arcname=file.relative_to(src))
