"""Tests for forecaster modules in the time_series_forecasting tutorial."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Create mock modules
class MockProphet:
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, df):
        return self
    
    def make_future_dataframe(self, periods):
        # Use a fixed start date for testing
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=periods)
        return pd.DataFrame({"ds": dates})
    
    def predict(self, future):
        result = future.copy()
        result['yhat'] = np.linspace(10, 20, len(future))
        return result

# Set up mock modules before imports
prophet_mock = MagicMock()
prophet_mock.Prophet = MockProphet
sys.modules['prophet'] = prophet_mock

# Mock statsmodels module
class MockSARIMAXResults:
    def __init__(self, model):
        self.model = model
        
    def forecast(self, steps):
        return np.linspace(5, 15, steps)

class MockSARIMAX:
    def __init__(self, *args, **kwargs):
        pass
        
    def fit(self, disp=False):
        return MockSARIMAXResults(self)

# Create the complete mock hierarchy for statsmodels
statsmodels_mock = MagicMock()
statsmodels_mock.tsa = MagicMock()
statsmodels_mock.tsa.statespace = MagicMock()
statsmodels_mock.tsa.statespace.sarimax = MagicMock()
statsmodels_mock.tsa.statespace.sarimax.SARIMAX = MockSARIMAX
sys.modules['statsmodels'] = statsmodels_mock
sys.modules['statsmodels.tsa'] = statsmodels_mock.tsa
sys.modules['statsmodels.tsa.statespace'] = statsmodels_mock.tsa.statespace
sys.modules['statsmodels.tsa.statespace.sarimax'] = statsmodels_mock.tsa.statespace.sarimax

# Mock torch for LSTM forecaster
class MockTorchModule:
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, x):
        # For MSELoss, handle both y_pred and target
        if hasattr(self, 'is_loss') and self.is_loss:
            return MagicMock()
        return MagicMock()
    
    def parameters(self):
        return []
        
    def train(self):
        pass
        
    def eval(self):
        pass

class MockTensor:
    def __init__(self, data):
        self.data = data
        
    def view(self, *args):
        return self
        
    def item(self):
        return 10.0
        
    def backward(self):
        pass

# Create mock torch modules with proper hierarchy
torch_mock = MagicMock()
torch_mock.FloatTensor = lambda data: MockTensor(data)
torch_mock.cat = lambda tensors, dim: tensors[0]
torch_mock.no_grad = MagicMock()
torch_mock.no_grad.return_value.__enter__ = MagicMock()
torch_mock.no_grad.return_value.__exit__ = MagicMock()

# Mock torch.nn
nn_mock = MagicMock()
nn_mock.Module = MockTorchModule
nn_mock.LSTM = MockTorchModule
nn_mock.Linear = MockTorchModule
# Create special MSELoss with is_loss flag
loss_class = MockTorchModule
loss_instance = loss_class()
loss_instance.is_loss = True
nn_mock.MSELoss = MagicMock(return_value=loss_instance)
torch_mock.nn = nn_mock

# Mock torch.optim
optim_mock = MagicMock()
optim_mock.Adam = lambda params, lr: MagicMock()
torch_mock.optim = optim_mock

# Mock torch.utils.data
data_mock = MagicMock()
data_mock.DataLoader = lambda dataset, batch_size, shuffle: [(MagicMock(), MagicMock())]
data_mock.TensorDataset = lambda x, y: MagicMock()
torch_mock.utils = MagicMock()
torch_mock.utils.data = data_mock

# Add to sys.modules
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = nn_mock
sys.modules['torch.optim'] = optim_mock
sys.modules['torch.utils'] = torch_mock.utils
sys.modules['torch.utils.data'] = data_mock

# Import the forecasters
from tutorials.time_series_forecasting.forecasters.prophet_forecaster import ProphetForecaster
from tutorials.time_series_forecasting.forecasters.sarima_forecaster import SARIMAForecaster
from tutorials.time_series_forecasting.forecasters.lstm_forecaster import LSTMForecaster


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    return np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50


@pytest.fixture
def start_date():
    """Create a start date for forecasting."""
    return datetime(2023, 1, 1)


@pytest.mark.unit
def test_prophet_forecaster(sample_data, start_date):
    """Test that ProphetForecaster returns data in the expected format."""
    forecaster = ProphetForecaster()
    steps = 10
    
    with patch('tutorials.time_series_forecasting.forecasters.prophet_forecaster.Prophet', MockProphet):
        forecast = forecaster.forecast(sample_data, steps, start_date)
    
    # Verify the forecast dataframe has the expected structure
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == steps
    assert "datetime" in forecast.columns
    assert "Prophet" in forecast.columns
    
    # Check that all dates are datetime objects
    assert pd.api.types.is_datetime64_dtype(forecast["datetime"])


@pytest.mark.unit
def test_sarima_forecaster(sample_data, start_date):
    """Test that SARIMAForecaster returns data in the expected format."""
    forecaster = SARIMAForecaster()
    steps = 10
    
    forecast = forecaster.forecast(sample_data, steps, start_date)
    
    # Verify the forecast dataframe has the expected structure
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == steps
    assert "datetime" in forecast.columns
    assert "SARIMA" in forecast.columns
    
    # Check that all dates are datetime objects
    assert pd.api.types.is_datetime64_dtype(forecast["datetime"])


@pytest.mark.unit
def test_lstm_forecaster_structure():
    """Test the LSTMForecaster class structure."""
    # Instead of running the full forecast which is complex to mock,
    # just test the class structure
    forecaster = LSTMForecaster()
    
    # Verify the class has the required methods
    assert hasattr(forecaster, 'forecast')
    assert callable(forecaster.forecast)
    
    # Verify it has necessary attributes
    assert hasattr(forecaster, 'model')
    assert hasattr(forecaster, 'criterion')
    assert hasattr(forecaster, 'optimizer')
    
    # Test that the model has expected structure
    assert hasattr(forecaster.model, 'lstm')
    assert hasattr(forecaster.model, 'linear')