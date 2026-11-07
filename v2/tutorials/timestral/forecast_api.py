# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.0",
#    "fastapi",
#    "uvicorn",
# ]
# ///
"""Online forecasting with Timestral.

A FastAPI app that serves Timestral forecasts over HTTP. Concurrent requests are
coalesced into batched forward passes by `DynamicBatcher`, and the app scales its
replicas with request concurrency.

Deploy with a fixed checkpoint:

    python forecast_api.py --model mistralai/Timestral-28M

or with the champion picked by the latest run of `batch_forecasting.py`:

    python forecast_api.py --champion
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import flyte
import flyte.app
from fastapi import FastAPI, HTTPException
from flyte.app.extras import FastAPIAppEnvironment
from flyte.extras import DynamicBatcher
from pydantic import BaseModel, Field

from common import QUANTILE_LEVELS, load_forecaster, timestral_image

MAX_CONTEXT = 8192  # cap per-request history so one request can't stall a batch


# {{docs-fragment schema}}
class ForecastRequest(BaseModel):
    values: list[float | None] = Field(
        ..., min_length=2, description="History, oldest first. null marks a missing value."
    )
    horizon: int = Field(24, ge=1, le=2048)
    quantiles: list[float] = Field(default=[0.1, 0.5, 0.9])
    known_covariates: list[list[float | None]] | None = Field(
        None,
        description="Covariates known over history and horizon: (n_covariates, len(values) + horizon).",
    )
    past_covariates: list[list[float | None]] | None = Field(
        None, description="Covariates known over history only: (n_covariates, len(values))."
    )


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    median: list[float]
    quantiles: dict[str, list[float]]


# {{/docs-fragment schema}}


# {{docs-fragment batching}}
class TimestralService:
    """Owns the model and the batcher for one app replica."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.forecaster = load_forecaster(model_id)
        # One forward pass at a time: torch already uses every core of the replica.
        self.lock = asyncio.Lock()
        self.batcher = DynamicBatcher(
            process_fn=self._process,
            max_batch_size=64,
            batch_timeout_s=0.01,  # wait at most 10 ms for more requests to join
        )

    async def _run(self, **kwargs):
        async with self.lock:
            return await asyncio.to_thread(self.forecaster.forecast, **kwargs)

    async def _process(self, batch: list[ForecastRequest]) -> list:
        """Forecast a batch of plain (no-covariate) requests, one pass per horizon."""
        import numpy as np

        results: list = [None] * len(batch)
        by_horizon: dict[int, list[int]] = {}
        for i, request in enumerate(batch):
            by_horizon.setdefault(request.horizon, []).append(i)

        for horizon, indices in by_horizon.items():
            histories = [batch[i].values[-MAX_CONTEXT:] for i in indices]
            length = max(len(h) for h in histories)
            # Left-pad shorter histories with NaN; Timestral treats NaN as missing.
            context = np.full((len(indices), length), np.nan, dtype="float32")
            for row, history in enumerate(histories):
                context[row, length - len(history) :] = [
                    np.nan if v is None else v for v in history
                ]
            forecast = await self._run(context=context, prediction_length=horizon)
            quantiles = forecast.quantiles[:, 0].numpy()  # (batch, horizon, 21)
            for row, i in enumerate(indices):
                results[i] = quantiles[row]
        return results

    async def forecast(self, request: ForecastRequest):
        """Return a (horizon, 21) quantile array for one request."""
        import numpy as np

        if request.known_covariates is None and request.past_covariates is None:
            future = await self.batcher.submit(request)
            return await future

        # Requests with covariates have their own shapes, so they skip the batcher.
        def as_array(rows):
            return None if rows is None else np.array(rows, dtype="float32")[None]

        values = np.array(
            [np.nan if v is None else v for v in request.values], dtype="float32"
        )
        forecast = await self._run(
            context=values[None],
            prediction_length=request.horizon,
            known_covariates=as_array(request.known_covariates),
            past_covariates=as_array(request.past_covariates),
        )
        return forecast.quantiles[0, 0].numpy()


# {{/docs-fragment batching}}


# {{docs-fragment fastapi_app}}
service: TimestralService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the checkpoint once per replica, before the replica reports ready.
    global service
    service = TimestralService(os.environ.get("TIMESTRAL_MODEL", "mistralai/Timestral-28M"))
    await service.batcher.start()
    yield
    await service.batcher.stop()


app = FastAPI(title="Timestral forecasting API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": service.model_id if service else None}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    unknown = sorted(set(request.quantiles) - set(QUANTILE_LEVELS))
    if unknown:
        raise HTTPException(
            422, f"Unsupported quantile levels {unknown}; choose from {QUANTILE_LEVELS}"
        )
    try:
        quantiles = await service.forecast(request)
    except (AssertionError, ValueError) as e:
        raise HTTPException(422, str(e)) from e

    def pick(level: float) -> list[float]:
        return quantiles[:, QUANTILE_LEVELS.index(level)].astype(float).round(4).tolist()

    return ForecastResponse(
        model=service.model_id,
        horizon=request.horizon,
        median=pick(0.5),
        quantiles={f"{level:g}": pick(level) for level in request.quantiles},
    )


# {{/docs-fragment fastapi_app}}


# {{docs-fragment app_env}}
app_env = FastAPIAppEnvironment(
    name="timestral-forecast-api",
    app=app,
    image=timestral_image.with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    parameters=[
        flyte.app.Parameter(
            name="model", value="mistralai/Timestral-28M", env_var="TIMESTRAL_MODEL"
        ),
    ],
    # Keep one replica warm, add a replica for every 32 requests in flight (up to
    # four), and scale back down after five idle minutes.
    scaling=flyte.app.Scaling(
        replicas=(1, 4),
        metric=flyte.app.Scaling.Concurrency(val=32),
        scaledown_after=300,
    ),
    requires_auth=False,
)
# {{/docs-fragment app_env}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Timestral-28M")
    parser.add_argument(
        "--champion",
        action="store_true",
        help="Serve the model picked by the latest batch_forecasting.py run.",
    )
    args = parser.parse_args()

    if args.champion:
        # Resolved at deploy time to the first output of the latest pipeline run.
        model = flyte.app.RunOutput(
            type="string", task_name="timestral-batch-driver.forecast_pipeline"
        )
    else:
        model = args.model
    app_env.parameters = [
        flyte.app.Parameter(name="model", value=model, env_var="TIMESTRAL_MODEL")
    ]

    flyte.init_from_config(root_dir=Path(__file__).parent)
    deployed = flyte.serve(app_env)
    print(f"Forecast API: {deployed.url}")
    print(f"OpenAPI docs: {deployed.url}/docs")
# {{/docs-fragment deploy}}
