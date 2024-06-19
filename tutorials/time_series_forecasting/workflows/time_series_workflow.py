from datetime import datetime, timedelta
from typing import Annotated, List, Tuple

import numpy as np
import pandas as pd
from flytekit import task, workflow, ImageSpec, dynamic, Deck
from flytekit.core.artifact import Artifact, Granularity
from flytekit.deck import MarkdownRenderer
from unionai.artifacts import ModelCard

TimeSeriesData = Artifact(
    name="time-series-data",
    time_partitioned=True,
    time_partition_granularity=Granularity.DAY,
)

TimeSeriesForecast = Artifact(
    name="time-series-forecast",
    partition_keys=["model"]
)


def generate_card(df: pd.DataFrame) -> str:
    contents = "# Dataset Card\n"
    contents = contents + df.to_markdown()
    return contents


@task(container_image=ImageSpec(builder="unionai", packages=["pandas==2.2.2"]))
def get_data() -> Tuple[datetime.date, Annotated[List[float], TimeSeriesData]]:
    #  Dummy task that will theoretically return the last 25 days of data
    start_date = datetime(2024, 1, 3).date()
    days_of_data = 25
    data = np.sin(np.linspace(0, 20, days_of_data))

    return start_date, TimeSeriesData.create_from(
        [float(x) for x in data],
        time_partition=datetime.combine(start_date, datetime.min.time())
    )


@task(container_image=ImageSpec(builder="unionai", packages=["pandas==2.2.2", "statsmodels==0.14.2", "tabulate==0.9.0"]))
def sarima_forecast(start_date: datetime.date, steps: int, data: List[float] = TimeSeriesData.query()) -> Annotated[
    pd.DataFrame, TimeSeriesForecast]:
    from forecasters.sarima_forecaster import SARIMAForecaster
    sarima_forecaster = SARIMAForecaster()
    forecast = sarima_forecaster.forecast(data, steps, start_date)
    return TimeSeriesForecast.create_from(forecast, ModelCard(generate_card(forecast)), model='sarima')


@task(container_image=ImageSpec(builder="unionai", packages=["pandas==2.2.2", "prophet==1.1.5", "tabulate==0.9.0"]))
def prophet_forecast(start_date: datetime.date, steps: int, data: List[float] = TimeSeriesData.query()) -> Annotated[
    pd.DataFrame, TimeSeriesForecast]:
    from forecasters.prophet_forecaster import ProphetForecaster
    prophet_forecaster = ProphetForecaster()
    forecast = prophet_forecaster.forecast(data, steps, start_date)
    return TimeSeriesForecast.create_from(forecast, ModelCard(generate_card(forecast)), model='prophet')


@task(container_image=ImageSpec(builder="unionai", packages=["pandas==2.2.2", "numpy==1.26.4", "torch==2.3.1", "tabulate==0.9.0"]))
def lstm_forecast(start_date: datetime.date, steps: int, data: List[float] = TimeSeriesData.query()) -> Annotated[
    pd.DataFrame, TimeSeriesForecast]:
    from forecasters.lstm_forecaster import LSTMForecaster
    lstm_forecaster = LSTMForecaster()
    forecast = lstm_forecaster.forecast(data, steps, start_date)
    return TimeSeriesForecast.create_from(forecast, ModelCard(generate_card(forecast)), model='lstm')


@dynamic(enable_deck=True, container_image=ImageSpec(builder="unionai",
                                                     packages=["pandas==2.2.2", "flytekitplugins-deck-standard==1.12.3",
                                                               "plotly==5.22.0"]))
def show_results(start_date: datetime.date, preds: List[pd.DataFrame],
                 historical_data: List[float] = TimeSeriesData.query()):
    import plotly

    # Create the historical dataframe
    hist_dates = pd.date_range(start=start_date - timedelta(days=len(historical_data)), periods=len(historical_data))
    hist_df = pd.DataFrame({'datetime': hist_dates, 'Historical': historical_data})

    # Append the last historical data point to each forecast dataframe for clean plotting
    last_hist_point = hist_df.iloc[-1]
    for df in preds:
        last_point = {'datetime': last_hist_point['datetime'], df.columns[1]: last_hist_point['Historical']}
        df.loc[-1] = last_point
        df.index = df.index + 1
        df.sort_index(inplace=True)

    # Combine the dataframes on the 'datetime' column
    forecast_df = preds[0]
    for df in preds[1:]:
        forecast_df = forecast_df.merge(df, on='datetime')

    # Create traces for historical data and forecasts
    traces = [plotly.graph_objs.Scatter(x=hist_df['datetime'], y=hist_df['Historical'], mode='lines+markers',
                                        name='Historical Data')]
    for column in forecast_df.columns:
        if column != 'datetime':
            traces.append(
                plotly.graph_objs.Scatter(x=forecast_df['datetime'], y=forecast_df[column], mode='lines+markers',
                                          name=f'{column} Forecast'))

    # Create the figure
    layout = plotly.graph_objs.Layout(
        title='Model Forecasts',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Values'),
        legend=dict(x=0, y=1)
    )
    fig = plotly.graph_objs.Figure(data=traces, layout=layout)

    main_deck = Deck("Forecasts", MarkdownRenderer().to_html("### Plot of Forecasts"))
    main_deck.append(plotly.io.to_html(fig))


@workflow
def time_series_workflow(steps: int = 5):
    start_date, data = get_data()

    sarima_pred = sarima_forecast(start_date=start_date, steps=steps, data=data)
    prophet_pred = prophet_forecast(start_date=start_date, steps=steps, data=data)
    lstm_pred = lstm_forecast(start_date=start_date, steps=steps, data=data)

    show_results(start_date=start_date, historical_data=data, preds=[sarima_pred, prophet_pred, lstm_pred])

