from abc import ABC, abstractmethod


class Forecaster(ABC):
    @abstractmethod
    def forecast(self, data, steps, start_date):
        pass