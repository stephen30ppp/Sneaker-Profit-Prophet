"""
GRU-based Price Forecasting Model
"""
import torch
import torch.nn as nn

from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FORECAST_HORIZON


class SneakerGRU(nn.Module):
    """
    Multi-layer GRU for multivariate time series forecasting.
    Fewer parameters than LSTM — faster training, often comparable accuracy.
    """

    def __init__(self, input_size: int):
        """
        Args:
            input_size: Number of input features per time step
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, FORECAST_HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            predictions: (batch, FORECAST_HORIZON)
        """
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)
