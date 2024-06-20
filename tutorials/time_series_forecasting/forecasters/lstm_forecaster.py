import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster:
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        self.model = LSTMModel(input_size, hidden_size, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forecast(self, data, steps, start_date, sequence_length=5) -> pd.DataFrame:
        # Convert data to numpy array
        data = np.array(data)

        # Prepare the dataset
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X = torch.FloatTensor(X).view(-1, sequence_length, 1)
        y = torch.FloatTensor(y).view(-1, 1)

        # Train the model
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            for seq, target in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(seq)
                loss = self.criterion(y_pred, target)
                loss.backward()
                self.optimizer.step()

        # Forecasting
        self.model.eval()
        predictions = []
        current_seq = torch.FloatTensor(data[-sequence_length:]).view(1, sequence_length, 1)
        for _ in range(steps):
            with torch.no_grad():
                next_step = self.model(current_seq)
                predictions.append(next_step.item())
                current_seq = torch.cat((current_seq[:, 1:, :], next_step.view(1, 1, 1)), dim=1)

        # Create a date range for the forecast
        index = pd.date_range(start=start_date, periods=steps, freq='D')
        forecast_df = pd.DataFrame({'datetime': index, 'LSTM': predictions})
        return forecast_df


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions