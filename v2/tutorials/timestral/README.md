# Time-series forecasting with Timestral

Four Flyte 2 examples built on [Timestral](https://github.com/mistralai/timestral),
Mistral AI's time-series foundation models. Each one pairs a Timestral capability
with the Flyte features that keep the workload durable.

| Script | What it does | Timestral features | Flyte features |
| --- | --- | --- | --- |
| `batch_forecasting.py` | Backtests all four checkpoints on 321 electricity customers, picks the best, forecasts every series | Batched zero-shot forecasting, 21 quantiles | Fan-out, caching, retries on spot, intra-task checkpoints, reports |
| `forecast_api.py` | Serves forecasts over HTTP | Per-request histories, covariates | FastAPI app, `DynamicBatcher`, autoscaling, `RunOutput` |
| `anomaly_monitor.py` | Flags pump sensors that leave their forecast band | Multivariate targets, categorical known covariates, observation mask | Traces, durable sleep, external conditions, cron trigger |
| `capacity_planning.py` | Forecasts ERCOT demand under weather scenarios and gates the plan on approval | Numeric and categorical known covariates, scenario batches | Cached API calls with backoff, fan-out, typed approval |

`common.py` holds the container image, model loading, metrics, and the report chart.
`forecast_client.py` load-tests the API.

## Run

```bash
pip install "flyte>=2.12.0" fastapi uvicorn
flyte create config --endpoint <your-endpoint> --project <project> --domain development --builder remote

python batch_forecasting.py               # backtest + forecast, 32 customers
python forecast_api.py --champion         # serve the checkpoint the batch run picked
python forecast_client.py <app-url>       # load-test the API
python anomaly_monitor.py                 # durable monitor with a simulated crash
python capacity_planning.py               # pauses on the `reserve-margin` condition
```

Everything runs on CPU. Data comes from the
[autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)
collection on Hugging Face and the [Open-Meteo](https://open-meteo.com/) historical
weather API; neither needs credentials.
