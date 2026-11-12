# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.0",
# ]
# main = "forecast_pipeline"
# params = "n_series=32"
# ///
"""Batch forecasting with Timestral.

Backtests every Timestral checkpoint on 321 electricity customers, picks the most
accurate one, and uses it to forecast the next week for every customer.

The work fans out over (model, shard) tasks. Each task is cached, retried, and
checkpointed, so a re-run skips finished shards and a preempted shard resumes from
its last completed backtest window.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from common import (
    QUANTILE_LEVELS,
    TIMESTRAL_MODELS,
    fan_chart_svg,
    load_forecaster,
    mase,
    timestral_image,
    weighted_quantile_loss,
)

# {{docs-fragment config}}
# Hourly electricity consumption (kW) for 321 customers, 2012-2014, from the
# Monash forecasting archive.
DATA_URL = (
    "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
    "monash_electricity_hourly/train-00000-of-00001.parquet"
)
SEASON = 24  # daily seasonality, for the seasonal-naive MASE baseline
# {{/docs-fragment config}}

# {{docs-fragment envs}}
# Forecasting tasks run on CPU. `interruptible=True` lets them use spot capacity;
# retries absorb preemptions, and checkpoints keep the work already done.
forecast_env = flyte.TaskEnvironment(
    name="timestral-batch",
    image=timestral_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    interruptible=True,
)

# The driver only orchestrates and renders the report, so it stays small and
# on on-demand capacity.
driver_env = flyte.TaskEnvironment(
    name="timestral-batch-driver",
    image=timestral_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[forecast_env],
)
# {{/docs-fragment envs}}


@dataclass
class ShardScore:
    model: str
    n_forecasts: int
    mase: float
    wql: float
    seconds: float


@dataclass
class ModelScore:
    model: str
    n_forecasts: int
    mase: float
    wql: float
    ms_per_series: float


# {{docs-fragment load_dataset}}
@forecast_env.task(cache="auto", retries=3)
async def load_dataset(url: str = DATA_URL) -> flyte.io.File:
    """Copy the dataset into object storage once. Later runs hit the cache."""
    local = Path(tempfile.mkdtemp()) / "electricity_hourly.parquet"
    urllib.request.urlretrieve(url, local)
    return await flyte.io.File.from_local(str(local))


# {{/docs-fragment load_dataset}}


async def read_series(data: flyte.io.File, series_ids: list[str]):
    """Return (ids, timestamps, values) for the requested series as numpy arrays."""
    import numpy as np
    import pandas as pd

    path = await data.download()
    df = pd.read_parquet(path, filters=[("id", "in", series_ids)])
    df = df.set_index("id").loc[series_ids]
    values = np.stack(df["target"].to_numpy()).astype("float32")
    timestamps = pd.to_datetime(df["timestamp"].iloc[0])
    return series_ids, timestamps, values


# {{docs-fragment backtest_shard}}
@forecast_env.task(cache="auto", retries=3)
async def backtest_shard(
    data: flyte.io.File,
    series_ids: list[str],
    model: str,
    cutoffs: list[int],
    context_length: int,
    horizon: int,
) -> ShardScore:
    """Score one model on one shard of series over several backtest windows."""
    import numpy as np

    _, _, values = await read_series(data, series_ids)
    forecaster = load_forecaster(TIMESTRAL_MODELS[model])

    # Resume from the windows a previous attempt already scored.
    checkpoint = flyte.ctx().checkpoint
    done: dict[str, dict] = {}
    if checkpoint is not None and (prev := await checkpoint.load()) is not None:
        done = json.loads(prev.read_bytes())

    for cutoff in cutoffs:
        if str(cutoff) in done:
            continue
        context = values[:, cutoff - context_length : cutoff]
        actual = values[:, cutoff : cutoff + horizon]

        start = time.perf_counter()
        forecast = forecaster.forecast(context, prediction_length=horizon)
        seconds = time.perf_counter() - start

        quantiles = forecast.quantiles[:, 0].numpy()  # (series, horizon, 21)
        median = forecast.median[:, 0].numpy()  # (series, horizon)
        done[str(cutoff)] = {
            "mase": [
                mase(actual[i], median[i], context[i], SEASON) for i in range(len(series_ids))
            ],
            "wql": [
                weighted_quantile_loss(actual[i], quantiles[i]) for i in range(len(series_ids))
            ],
            "seconds": seconds,
        }
        if checkpoint is not None:
            await checkpoint.save(json.dumps(done).encode())

    return ShardScore(
        model=model,
        n_forecasts=sum(len(w["mase"]) for w in done.values()),
        mase=float(np.nanmean([m for w in done.values() for m in w["mase"]])),
        wql=float(np.nanmean([q for w in done.values() for q in w["wql"]])),
        seconds=sum(w["seconds"] for w in done.values()),
    )


# {{/docs-fragment backtest_shard}}


# {{docs-fragment forecast_shard}}
@forecast_env.task(cache="auto", retries=3)
async def forecast_shard(
    data: flyte.io.File,
    series_ids: list[str],
    model: str,
    context_length: int,
    horizon: int,
) -> flyte.io.File:
    """Forecast the next `horizon` steps for a shard and write them as Parquet."""
    import numpy as np
    import pandas as pd

    ids, timestamps, values = await read_series(data, series_ids)
    forecaster = load_forecaster(TIMESTRAL_MODELS[model])
    forecast = forecaster.forecast(values[:, -context_length:], prediction_length=horizon)
    quantiles = forecast.quantiles[:, 0].numpy()  # (series, horizon, 21)

    future = pd.date_range(timestamps[-1], periods=horizon + 1, freq="h")[1:]
    frame = pd.DataFrame(
        {
            "id": np.repeat(ids, horizon),
            "timestamp": np.tile(future, len(ids)),
            **{
                f"q{level:g}": quantiles[:, :, k].ravel()
                for k, level in enumerate(QUANTILE_LEVELS)
            },
        }
    )
    out = Path(tempfile.mkdtemp()) / f"forecast_{model}_{ids[0]}.parquet"
    frame.to_parquet(out)
    return await flyte.io.File.from_local(str(out))


# {{/docs-fragment forecast_shard}}


def shard(items: list[str], n_shards: int) -> list[list[str]]:
    size = -(-len(items) // n_shards)
    return [items[i : i + size] for i in range(0, len(items), size)]


# {{docs-fragment backtest}}
@driver_env.task
async def backtest(
    data: flyte.io.File,
    series_ids: list[str],
    models: list[str],
    n_shards: int,
    n_windows: int,
    context_length: int,
    horizon: int,
    series_length: int,
) -> list[ModelScore]:
    """Fan out every (model, shard) pair and aggregate a leaderboard."""
    # Rolling-origin backtest: the last `n_windows` horizons of each series.
    cutoffs = [series_length - horizon * k for k in range(n_windows, 0, -1)]
    shards = shard(series_ids, n_shards)

    async def score_model(model: str) -> ModelScore:
        with flyte.group(f"backtest-{model}"):
            results = await asyncio.gather(
                *(
                    backtest_shard(data, ids, model, cutoffs, context_length, horizon)
                    for ids in shards
                )
            )
        n = sum(r.n_forecasts for r in results)
        return ModelScore(
            model=model,
            n_forecasts=n,
            mase=sum(r.mase * r.n_forecasts for r in results) / n,
            wql=sum(r.wql * r.n_forecasts for r in results) / n,
            ms_per_series=1000 * sum(r.seconds for r in results) / n,
        )

    return list(await asyncio.gather(*(score_model(m) for m in models)))


# {{/docs-fragment backtest}}


# {{docs-fragment pipeline}}
@driver_env.task(report=True)
async def forecast_pipeline(
    models: list[str] = list(TIMESTRAL_MODELS),
    n_series: int = 0,
    n_shards: int = 8,
    n_windows: int = 4,
    context_length: int = 2048,
    horizon: int = 168,
) -> tuple[str, flyte.io.File]:
    """Backtest all models, pick the champion by WQL, forecast every series with it.

    Returns the champion's Hugging Face repo id and a Parquet file of forecasts.
    """
    import pandas as pd

    data = await load_dataset()
    path = await data.download()
    meta = pd.read_parquet(path, columns=["id"])
    series_ids = meta["id"].tolist()[: n_series or None]
    series_length = len(
        pd.read_parquet(path, filters=[("id", "==", series_ids[0])])["target"].iloc[0]
    )

    scores = await backtest(
        data, series_ids, models, n_shards, n_windows, context_length, horizon, series_length
    )
    champion = min(scores, key=lambda s: s.wql)

    with flyte.group(f"forecast-{champion.model}"):
        shard_files = await asyncio.gather(
            *(
                forecast_shard(data, ids, champion.model, context_length, horizon)
                for ids in shard(series_ids, n_shards)
            )
        )
    frames = [pd.read_parquet(await f.download()) for f in shard_files]
    forecasts = pd.concat(frames, ignore_index=True)
    out = Path(tempfile.mkdtemp()) / "forecasts.parquet"
    forecasts.to_parquet(out)
    forecasts_file = await flyte.io.File.from_local(str(out))

    await render_report(path, scores, champion, forecasts, horizon)
    return TIMESTRAL_MODELS[champion.model], forecasts_file


# {{/docs-fragment pipeline}}


async def render_report(
    path, scores: list[ModelScore], champion: ModelScore, forecasts, horizon: int
):
    import pandas as pd

    rows = "".join(
        f"<tr{' style=\"font-weight:bold\"' if s is champion else ''}>"
        f"<td>Timestral-{s.model}</td><td>{s.wql:.4f}</td><td>{s.mase:.3f}</td>"
        f"<td>{s.ms_per_series:.1f}</td><td>{s.n_forecasts}</td></tr>"
        for s in sorted(scores, key=lambda s: s.wql)
    )
    charts = []
    for series_id in forecasts["id"].unique()[:4]:
        history = pd.read_parquet(path, filters=[("id", "==", series_id)])["target"].iloc[0][
            -2 * horizon :
        ]
        fc = forecasts[forecasts["id"] == series_id]
        quantiles = fc[[f"q{level:g}" for level in QUANTILE_LEVELS]].to_numpy()
        charts.append(
            fan_chart_svg(history, quantiles, title=f"{series_id}: next {horizon} h")
        )
    await flyte.report.replace.aio(
        "<h2>Timestral backtest</h2>"
        "<p>Lower is better. WQL is the weighted quantile loss over all 21 quantiles; "
        "MASE is relative to a seasonal-naive forecast.</p>"
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        "<tr><th>Model</th><th>WQL</th><th>MASE</th><th>ms / series</th><th>Forecasts</th></tr>"
        f"{rows}</table>"
        f"<h2>Forecasts from Timestral-{champion.model}</h2>" + "".join(charts)
    )


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    run = flyte.run(forecast_pipeline, n_series=32)
    print(run.url)
# {{/docs-fragment main}}
