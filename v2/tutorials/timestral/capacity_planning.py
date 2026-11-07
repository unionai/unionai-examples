# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.0",
# ]
# main = "capacity_plan"
# params = ""
# ///
"""Covariate-aware load forecasting and capacity planning with Timestral.

Forecasts hourly electricity demand for the eight ERCOT (Texas) weather zones.

1. Backtest: measure how much calendar and temperature covariates improve
   Timestral's forecasts over the target-only model.
2. Plan: forecast the next week under weather scenarios drawn from past years,
   and size each zone's capacity to the worst-case scenario.
3. Approve: pause for a planner to set the reserve margin before the plan is
   published.
"""

from __future__ import annotations

import asyncio
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import flyte
import flyte.errors
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

# {{docs-fragment data}}
# Hourly load (MW) for the eight ERCOT weather zones, 2004 to August 2021.
ERCOT_URL = (
    "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
    "ercot/train-00000-of-00001.parquet"
)

# One representative city per weather zone, for hourly temperatures from the
# Open-Meteo historical weather API.
ZONES = {
    "COAST": (29.76, -95.37),  # Houston
    "EAST": (32.35, -95.30),  # Tyler
    "FAR_WEST": (31.99, -102.08),  # Midland
    "NORTH": (33.91, -98.49),  # Wichita Falls
    "NORTH_C": (32.78, -96.80),  # Dallas
    "SOUTHERN": (27.80, -97.40),  # Corpus Christi
    "SOUTH_C": (30.27, -97.74),  # Austin
    "WEST": (32.45, -99.73),  # Abilene
}
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
# {{/docs-fragment data}}

# Covariate configurations compared in the backtest.
CONFIGS = ("target only", "+ calendar", "+ calendar + temperature")

# {{docs-fragment env}}
env = flyte.TaskEnvironment(
    name="timestral-capacity",
    image=timestral_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
)
# {{/docs-fragment env}}


# {{docs-fragment fetch}}
@env.task(cache="auto", retries=3)
async def load_ercot(url: str = ERCOT_URL) -> flyte.io.File:
    local = Path(tempfile.mkdtemp()) / "ercot.parquet"
    urllib.request.urlretrieve(url, local)
    return await flyte.io.File.from_local(str(local))


# The weather API is rate-limited. Exponential backoff rides out 429s and
# transient errors; the cache means each zone is fetched once, ever.
@env.task(
    cache="auto",
    retries=flyte.RetryStrategy(
        count=5,
        backoff=flyte.Backoff(
            base=timedelta(seconds=10), factor=2.0, cap=timedelta(minutes=2)
        ),
    ),
)
async def fetch_temperature(zone: str, start: str, end: str) -> flyte.io.File:
    """Hourly 2 m temperature (°C) in US Central local time, as Parquet."""
    import httpx
    import pandas as pd

    lat, lon = ZONES[zone]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "America/Chicago",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(WEATHER_URL, params=params)
        response.raise_for_status()
    hourly = response.json()["hourly"]
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(hourly["time"]), "temp_c": hourly["temperature_2m"]}
    )
    out = Path(tempfile.mkdtemp()) / f"temperature_{zone}.parquet"
    frame.to_parquet(out)
    return await flyte.io.File.from_local(str(out))


# {{/docs-fragment fetch}}


async def zone_frame(ercot: flyte.io.File, temperature: flyte.io.File, zone: str):
    """Load, temperature, and calendar features for one zone on one hourly index."""
    import pandas as pd

    df = pd.read_parquet(await ercot.download(), filters=[("id", "==", zone)])
    load = pd.Series(
        df["target"].iloc[0], index=pd.to_datetime(df["timestamp"].iloc[0]), name="load_mw"
    )
    temps = pd.read_parquet(await temperature.download()).set_index("timestamp")["temp_c"]
    frame = load.to_frame()
    frame["temp_c"] = temps.reindex(frame.index).interpolate(limit_direction="both")
    frame["load_mw"] = frame["load_mw"].interpolate(limit_direction="both")
    return frame


# {{docs-fragment covariates}}
def calendar_codes(index):
    """Categorical known-future covariates: hour of day, day of week, holiday."""
    import numpy as np
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar

    holidays = USFederalHolidayCalendar().holidays(index.min(), index.max())
    return np.stack(
        [
            index.hour.to_numpy(),
            index.dayofweek.to_numpy(),
            pd.DatetimeIndex(index.normalize()).isin(holidays).astype(int),
        ]
    ).astype(
        "float32"
    )  # (3, len(index))


def forecast_with(forecaster, load, temps, codes, horizon: int, config: str):
    """Forecast one or more series under a covariate configuration.

    `load` is (B, T); `temps` is (B, T + horizon); `codes` is (3, T + horizon).
    """
    import numpy as np

    batch = load.shape[0]
    kwargs = {}
    if config != "target only":
        kwargs["categorical_known_covariates"] = np.repeat(codes[None], batch, axis=0)
    if config == "+ calendar + temperature":
        kwargs["known_covariates"] = temps[:, None, :]
    return (
        forecaster.forecast(load, prediction_length=horizon, **kwargs).quantiles[:, 0].numpy()
    )


# {{/docs-fragment covariates}}


@dataclass
class ConfigScore:
    zone: str
    config: str
    wql: float
    mase: float


# {{docs-fragment backtest_zone}}
@env.task(cache="auto", retries=3)
async def backtest_zone(
    ercot: flyte.io.File,
    temperature: flyte.io.File,
    zone: str,
    model: str,
    n_windows: int,
    context_length: int,
    horizon: int,
) -> list[ConfigScore]:
    """Score every covariate configuration on the last `n_windows` weeks of data.

    The backtest uses observed temperatures over the horizon, as a stand-in for a
    perfect weather forecast. That is an upper bound on what temperature adds.
    """
    import numpy as np

    frame = await zone_frame(ercot, temperature, zone)
    load = frame["load_mw"].to_numpy("float32")
    temps = frame["temp_c"].to_numpy("float32")
    codes = calendar_codes(frame.index)
    forecaster = load_forecaster(TIMESTRAL_MODELS[model])

    scores = []
    for config in CONFIGS:
        wql, mase_ = [], []
        for k in range(n_windows, 0, -1):
            cutoff = len(load) - k * horizon
            window = slice(cutoff - context_length, cutoff + horizon)
            quantiles = forecast_with(
                forecaster,
                load[None, cutoff - context_length : cutoff],
                temps[None, window],
                codes[:, window],
                horizon,
                config,
            )[0]
            actual = load[cutoff : cutoff + horizon]
            wql.append(weighted_quantile_loss(actual, quantiles))
            mase_.append(
                mase(actual, quantiles[:, QUANTILE_LEVELS.index(0.5)], load[:cutoff], 168)
            )
        scores.append(ConfigScore(zone, config, float(np.mean(wql)), float(np.mean(mase_))))
    return scores


# {{/docs-fragment backtest_zone}}


@dataclass
class ZonePlan:
    zone: str
    median_peak_mw: float
    p99_peak_mw: float
    worst_year: int
    chart: str


# {{docs-fragment plan_zone}}
@env.task(cache="auto", retries=3)
async def plan_zone(
    ercot: flyte.io.File,
    temperature: flyte.io.File,
    zone: str,
    model: str,
    context_length: int,
    horizon: int,
    heat_shift_c: float,
) -> ZonePlan:
    """Forecast the week after the data ends under one weather scenario per past
    year, in a single batched forward pass."""
    import numpy as np
    import pandas as pd

    frame = await zone_frame(ercot, temperature, zone)
    load = frame["load_mw"].to_numpy("float32")[-context_length:]
    future = pd.date_range(frame.index[-1], periods=horizon + 1, freq="h")[1:]
    index = frame.index[-context_length:].append(future)
    codes = calendar_codes(index)

    # Scenario y: the same calendar week's temperatures from year y, plus an
    # optional heat-stress shift. History is the same in every scenario.
    years = sorted(set(frame.index.year) - {future[0].year})
    scenarios = []
    for year in years:
        analog = frame["temp_c"].reindex(future - pd.DateOffset(years=future[0].year - year))
        if analog.notna().all():
            scenarios.append((year, analog.to_numpy("float32") + heat_shift_c))
    history_temps = frame["temp_c"].to_numpy("float32")[-context_length:]
    temps = np.stack([np.concatenate([history_temps, s]) for _, s in scenarios])

    forecaster = load_forecaster(TIMESTRAL_MODELS[model])
    batch_load = np.repeat(load[None], len(scenarios), axis=0)
    quantiles = forecast_with(
        forecaster, batch_load, temps, codes, horizon, "+ calendar + temperature"
    )

    median_peaks = quantiles[:, :, QUANTILE_LEVELS.index(0.5)].max(axis=1)
    p99_peaks = quantiles[:, :, -1].max(axis=1)
    worst = int(np.argmax(p99_peaks))
    return ZonePlan(
        zone=zone,
        median_peak_mw=float(np.median(median_peaks)),
        p99_peak_mw=float(p99_peaks[worst]),
        worst_year=scenarios[worst][0],
        chart=fan_chart_svg(
            load[-2 * horizon :],
            quantiles[worst],
            title=f"{zone}: worst scenario ({scenarios[worst][0]} weather)",
        ),
    )


# {{/docs-fragment plan_zone}}


# {{docs-fragment capacity_plan}}
@env.task(report=True)
async def capacity_plan(
    zones: list[str] = list(ZONES),
    model: str = "209M",
    n_windows: int = 8,
    context_length: int = 4096,
    horizon: int = 168,
    heat_shift_c: float = 0.0,
    approval_timeout_hours: float = 24.0,
) -> flyte.io.File:
    ercot = await load_ercot()
    with flyte.group("weather"):
        temperatures = await asyncio.gather(
            *(fetch_temperature(zone, "2004-01-01", "2021-09-01") for zone in zones)
        )
    with flyte.group("backtest"):
        backtests = await asyncio.gather(
            *(
                backtest_zone(ercot, temps, zone, model, n_windows, context_length, horizon)
                for zone, temps in zip(zones, temperatures)
            )
        )
    with flyte.group("plan"):
        plans = await asyncio.gather(
            *(
                plan_zone(ercot, temps, zone, model, context_length, horizon, heat_shift_c)
                for zone, temps in zip(zones, temperatures)
            )
        )
    await render_report([s for scores in backtests for s in scores], plans)

    margin = await approve_reserve_margin(plans, approval_timeout_hours)
    return await publish_plan(plans, margin)


# {{/docs-fragment capacity_plan}}


# {{docs-fragment approval}}
async def approve_reserve_margin(plans: list[ZonePlan], timeout_hours: float) -> float:
    """Pause until a planner enters the reserve margin. A margin of 0 or less
    rejects the plan; no answer before the timeout fails the run."""
    total = sum(p.p99_peak_mw for p in plans)
    condition = await flyte.new_condition.aio(
        "reserve-margin",
        prompt=(
            f"Worst-case peak across zones: {total:,.0f} MW (see the report). "
            "Enter the reserve margin in percent to publish the plan, or 0 to reject it."
        ),
        data_type=float,
        timeout=timedelta(hours=timeout_hours),
    )
    margin = await condition.wait.aio()
    if margin <= 0:
        raise flyte.errors.RuntimeUserError(
            "PlanRejected", "The planner rejected the capacity plan."
        )
    return margin


# {{/docs-fragment approval}}


async def publish_plan(plans: list[ZonePlan], margin_pct: float) -> flyte.io.File:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "zone": [p.zone for p in plans],
            "median_peak_mw": [round(p.median_peak_mw) for p in plans],
            "p99_peak_mw": [round(p.p99_peak_mw) for p in plans],
            "worst_weather_year": [p.worst_year for p in plans],
            "planned_capacity_mw": [
                round(p.p99_peak_mw * (1 + margin_pct / 100)) for p in plans
            ],
        }
    )
    out = Path(tempfile.mkdtemp()) / "capacity_plan.csv"
    frame.to_csv(out, index=False)
    return await flyte.io.File.from_local(str(out))


async def render_report(scores: list[ConfigScore], plans: list[ZonePlan]) -> None:
    import pandas as pd

    table = pd.DataFrame([s.__dict__ for s in scores]).pivot(
        index="zone", columns="config", values="wql"
    )
    table = table[list(CONFIGS)]
    table.loc["mean"] = table.mean()
    plan_rows = "".join(
        f"<tr><td>{p.zone}</td><td>{p.median_peak_mw:,.0f}</td><td>{p.p99_peak_mw:,.0f}</td>"
        f"<td>{p.worst_year}</td></tr>"
        for p in plans
    )
    await flyte.report.replace.aio(
        "<h2>Backtest: weighted quantile loss by covariates (lower is better)</h2>"
        + table.to_html(float_format="{:.4f}".format)
        + "<h2>Next-week peak load under past-year weather</h2>"
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        "<tr><th>Zone</th><th>Median peak (MW)</th><th>Worst-case P99 peak (MW)</th><th>Worst weather year</th></tr>"
        + plan_rows
        + "</table>"
        + "".join(p.chart for p in plans)
    )


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    run = flyte.run(capacity_plan)
    print(run.url)
# {{/docs-fragment main}}
