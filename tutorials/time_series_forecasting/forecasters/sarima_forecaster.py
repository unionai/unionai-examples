import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAForecaster:
    def forecast(self, data, steps, start_date) -> pd.DataFrame:
        data_series = pd.Series(data)
        # Define forecaster and make prediction
        model = SARIMAX(data_series, order=(2, 0, 0), seasonal_order=(2, 0, 0, len(data_series) // 2))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
        # Create a dataframe for the forecast
        index = pd.date_range(start=start_date, periods=steps, freq='D')
        forecast_df = pd.DataFrame({'datetime': index, 'SARIMA': forecast})
        return forecast_df
