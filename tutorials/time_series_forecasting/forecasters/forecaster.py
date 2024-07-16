from typing import Protocol
import pandas as pd


class Forecaster(Protocol):
    def forecast(self, data, steps, start_date) -> pd.DataFrame:
        pass
