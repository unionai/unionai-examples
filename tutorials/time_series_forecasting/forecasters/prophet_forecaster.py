from datetime import timedelta

from prophet import Prophet
import pandas as pd

from .forecaster import Forecaster


class ProphetForecaster(Forecaster):
    def forecast(self, data, steps, start_date):
        # Generate dates for the data
        dates = pd.date_range(start=start_date - timedelta(days=len(data)), periods=len(data))
        # Create DataFrame from dates and data
        df = pd.DataFrame({'ds': dates, 'y': data})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].tail(steps)
        return forecast_df.rename(columns={'ds': 'datetime', 'yhat': 'Prophet'}).reset_index(drop=True)