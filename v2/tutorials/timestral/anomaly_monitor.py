# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.0",
# ]
# main = "monitor_fleet"
# params = "cycles=8"
# ///
"""Forecast-driven anomaly detection with Timestral.

A fleet of pumps reports vibration, bearing temperature, and discharge pressure
every minute. For each pump, a long-running monitor loops: forecast the next window
with Timestral, wait for the actual readings to arrive, and flag the window if the
readings leave the forecast's 1-99% band. A sustained anomaly pauses the monitor on
a human decision: dispatch a maintenance crew or not.

The loop is durable. Every sensor read, forecast, and alert is a trace, and the
wait is a durable sleep, so a monitor that crashes mid-loop resumes where it was
instead of re-reading, re-forecasting, or re-alerting.

The sensor feed is simulated and time-compressed: one 30-minute window of plant
time passes in `cycle_seconds` of wall-clock time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import flyte
import flyte.durable
import flyte.errors
import flyte.report

from common import TIMESTRAL_MODELS, fan_chart_svg, load_forecaster, timestral_image

VARIATES = ("vibration_mm_s", "bearing_temp_c", "pressure_bar")

# {{docs-fragment env}}
monitor_env = flyte.TaskEnvironment(
    name="timestral-monitor",
    image=timestral_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)
# {{/docs-fragment env}}


# {{docs-fragment simulator}}
def operating_mode(minutes):
    """Planned duty cycle: 0 = idle (00-06h), 2 = high load (10-14h, 18-20h), 1 = normal."""
    import numpy as np

    hour = (np.asarray(minutes) // 60) % 24
    return np.where(
        hour < 6,
        0,
        np.where(((hour >= 10) & (hour < 14)) | ((hour >= 18) & (hour < 20)), 2, 1),
    )


def simulate(asset: str, start: int, end: int, fault_asset: str, fault_start: int):
    """Deterministic readings for minutes [start, end): a pure function of the asset
    and the minute, so re-reading a window always returns the same values."""
    import numpy as np

    minutes = np.arange(start, end)
    seed = sum(map(ord, asset))

    def noise(k):  # hash-based Gaussian noise, stable per (asset, minute, variate)
        u1 = np.modf(
            np.abs(np.sin(minutes * 12.9898 + seed * 78.233 + k * 37.719)) * 43758.5453
        )[0]
        u2 = np.modf(
            np.abs(np.sin(minutes * 93.9898 + seed * 47.233 + k * 11.131)) * 24634.6345
        )[0]
        return np.sqrt(-2 * np.log(np.clip(u1, 1e-9, 1))) * np.cos(2 * np.pi * u2)

    mode = operating_mode(minutes)
    ambient = np.sin(2 * np.pi * ((minutes / 60) % 24 - 9) / 24)
    vibration = 0.5 + 1.5 * mode + 0.15 * noise(0)
    temperature = 45 + 8 * mode + 3 * ambient + 0.4 * noise(1)
    pressure = 2 + 1.5 * mode + 0.05 * noise(2)

    if asset == fault_asset:  # a failing bearing: vibration and heat ramp up
        since = np.clip(minutes - fault_start, 0, None)
        vibration = vibration + 0.06 * since**0.9
        temperature = temperature + 0.05 * since

    return np.stack([vibration, temperature, pressure]), mode


# {{/docs-fragment simulator}}


# {{docs-fragment traced_steps}}
@flyte.trace
async def read_sensors(
    asset: str, start: int, end: int, fault_asset: str, fault_start: int
) -> dict:
    """Stand-in for a query against your plant historian."""
    values, mode = simulate(asset, start, end, fault_asset, fault_start)
    return {"values": values.round(4).tolist(), "mode": mode.tolist()}


@flyte.trace
async def forecast_window(
    model: str, history: dict, observed: list[bool], future_mode: list[int], horizon: int
) -> dict:
    """Forecast all three variates jointly, conditioned on the planned operating mode.

    `observed` is False for readings in windows already flagged as anomalous. The
    model treats them as missing, so it keeps forecasting healthy behavior instead
    of learning the fault as the new normal.
    """
    import numpy as np

    forecaster = load_forecaster(TIMESTRAL_MODELS[model])
    context = np.asarray(history["values"], dtype="float32")[None]  # (1, 3, T)
    mask = np.repeat(np.asarray(observed)[None, None], context.shape[1], axis=1)
    modes = np.asarray(history["mode"] + future_mode, dtype="float32")[
        None, None
    ]  # (1, 1, T + H)
    forecast = forecaster.forecast(
        context,
        prediction_length=horizon,
        observation_mask=mask,
        categorical_known_covariates=modes,
    )
    quantiles = forecast.quantiles[0].numpy()  # (3, H, 21)
    return {"quantiles": quantiles.round(4).tolist()}


@flyte.trace
async def send_alert(asset: str, window_start: int, message: str) -> str:
    """Stand-in for a pager or ticketing call. As a trace, it runs once per window,
    even if the monitor crashes and replays after sending it."""
    print(f"ALERT {asset} @ minute {window_start}: {message}", flush=True)
    return f"alert-{asset}-{window_start}"


# {{/docs-fragment traced_steps}}


@dataclass
class WindowResult:
    start: int
    outside_band: dict[str, float]
    anomaly: bool
    chart: str = ""


@dataclass
class AssetReport:
    asset: str
    windows: list[WindowResult] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    dispatched: bool = False
    attempt: int = 0


def score(actual: list[list[float]], quantiles: list) -> dict[str, float]:
    """Fraction of readings outside the forecast's 1-99% band, per variate."""
    import numpy as np

    a = np.asarray(actual)
    q = np.asarray(quantiles)
    outside = (a < q[:, :, 0]) | (a > q[:, :, -1])
    return {name: float(outside[i].mean()) for i, name in enumerate(VARIATES)}


# {{docs-fragment monitor_asset}}
@monitor_env.task(retries=2)
async def monitor_asset(
    asset: str,
    start_minute: int,
    cycles: int,
    horizon: int,
    context_minutes: int,
    cycle_seconds: int,
    model: str,
    fault_asset: str,
    fault_cycle: int,
    alert_threshold: float,
    crash_at_cycle: int,
) -> AssetReport:
    import numpy as np

    attempt = flyte.ctx().attempt_number
    fault_start = start_minute + fault_cycle * horizon
    report = AssetReport(asset=asset, attempt=attempt)
    consecutive = 0
    flagged: list[int] = []  # start minutes of anomalous windows

    for cycle in range(cycles):
        t0 = start_minute + cycle * horizon
        history = await read_sensors(asset, t0 - context_minutes, t0, fault_asset, fault_start)
        minutes = np.arange(t0 - context_minutes, t0)
        observed = ~np.isin(minutes - (minutes - start_minute) % horizon, flagged)
        future_mode = operating_mode(np.arange(t0, t0 + horizon)).tolist()
        forecast = await forecast_window(
            model, history, observed.tolist(), future_mode, horizon
        )

        # Wait for the window's readings to arrive. On a retry, the sleep only
        # waits out whatever time is left since it first started.
        await flyte.durable.sleep.aio(cycle_seconds)

        if attempt == 0 and cycle == crash_at_cycle:
            raise flyte.errors.RuntimeSystemError(
                "SimulatedCrash", f"{asset} monitor crashed in cycle {cycle}"
            )

        actual = await read_sensors(asset, t0, t0 + horizon, fault_asset, fault_start)
        outside = score(actual["values"], forecast["quantiles"])
        anomaly = max(outside.values()) >= alert_threshold
        consecutive = consecutive + 1 if anomaly else 0
        if anomaly:
            flagged.append(t0)

        chart = ""
        if anomaly:
            worst = int(np.argmax(list(outside.values())))
            chart = fan_chart_svg(
                history["values"][worst][-4 * horizon :],
                forecast["quantiles"][worst],
                actual["values"][worst],
                title=f"{asset} {VARIATES[worst]}, window starting minute {t0}",
            )
            report.alerts.append(
                await send_alert(
                    asset,
                    t0,
                    f"{VARIATES[worst]} outside forecast band {outside[VARIATES[worst]]:.0%}",
                )
            )
        report.windows.append(
            WindowResult(start=t0, outside_band=outside, anomaly=anomaly, chart=chart)
        )

        # Two anomalous windows in a row: ask a human whether to dispatch a crew.
        if consecutive == 2 and not report.dispatched:
            report.dispatched = await request_dispatch(asset, t0, outside)

    return report


# {{/docs-fragment monitor_asset}}


# {{docs-fragment approval}}
async def request_dispatch(asset: str, window_start: int, outside: dict[str, float]) -> bool:
    """Pause the monitor until an operator answers. No answer in 15 minutes
    means dispatch: for a failing bearing, a false alarm is the cheaper mistake."""
    summary = ", ".join(f"{k} {v:.0%}" for k, v in outside.items())
    condition = await flyte.new_condition.aio(
        f"dispatch-{asset}-{window_start}",
        prompt=f"{asset} has left its forecast band for two windows ({summary}). Dispatch a crew?",
        data_type=bool,
        timeout=timedelta(minutes=15),
    )
    try:
        return await condition.wait.aio()
    except flyte.errors.ConditionTimedoutError:
        return True


# {{/docs-fragment approval}}


# {{docs-fragment fleet}}
@monitor_env.task(
    report=True,
    # Once deployed with `flyte deploy`, the trigger starts a run at the top of every hour.
    triggers=flyte.Trigger(
        "hourly-fleet-monitor",
        flyte.Cron("0 * * * *"),
        inputs={"trigger_time": flyte.TriggerTime},
        auto_activate=False,
    ),
)
async def monitor_fleet(
    assets: list[str] = ["pump-01", "pump-02", "pump-03"],
    cycles: int = 8,
    horizon: int = 30,
    context_minutes: int = 2880,
    cycle_seconds: int = 20,
    model: str = "28M",
    fault_asset: str = "pump-02",
    fault_cycle: int = 3,
    alert_threshold: float = 0.2,
    crash_at_cycle: int = -1,
    trigger_time: datetime | None = None,
) -> list[AssetReport]:
    # Plant time starts at 08:00 so the run crosses the 10:00 shift to high load.
    start_minute = 14 * 24 * 60 + 8 * 60
    reports = await asyncio.gather(
        *(
            monitor_asset(
                asset,
                start_minute,
                cycles,
                horizon,
                context_minutes,
                cycle_seconds,
                model,
                fault_asset,
                fault_cycle,
                alert_threshold,
                crash_at_cycle,
            )
            for asset in assets
        )
    )
    await render_report(reports, trigger_time or await flyte.durable.now.aio())
    return list(reports)


# {{/docs-fragment fleet}}


async def render_report(reports: list[AssetReport], started: datetime) -> None:
    rows = []
    for r in reports:
        flags = "".join("🔴" if w.anomaly else "🟢" for w in r.windows)
        rows.append(
            f"<tr><td>{r.asset}</td><td>{flags}</td><td>{len(r.alerts)}</td>"
            f"<td>{'yes' if r.dispatched else 'no'}</td><td>{r.attempt + 1}</td></tr>"
        )
    charts = [w.chart for r in reports for w in r.windows if w.chart][:6]
    await flyte.report.replace.aio(
        f"<h2>Fleet monitor, started {started:%Y-%m-%d %H:%M} UTC</h2>"
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        "<tr><th>Asset</th><th>Windows</th><th>Alerts</th><th>Crew dispatched</th><th>Attempts</th></tr>"
        + "".join(rows)
        + "</table><h3>Anomalous windows</h3>"
        "<p>Grey: history. Blue: forecast median with 10-90% and 1-99% bands. Red: actual readings.</p>"
        + "".join(charts)
    )


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    # Crash every monitor once in cycle 4 to watch the retry replay cycles 0-3.
    run = flyte.run(monitor_fleet, crash_at_cycle=4)
    print(run.url)
# {{/docs-fragment main}}
