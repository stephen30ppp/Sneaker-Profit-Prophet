"""
PyTorch Dataset for Sneaker Price Time Series

Handles windowed sequence creation from multivariate time series
(price + sentiment features).
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from config import SEQUENCE_LENGTH, FORECAST_HORIZON


class SneakerPriceDataset(Dataset):
    """
    Sliding-window dataset for time series forecasting.

    Each sample is a (input_sequence, target) pair where:
    - input_sequence: shape (SEQUENCE_LENGTH, num_features)
    - target: shape (FORECAST_HORIZON,) — future prices
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], target_col: str = "price"):
        """
        Args:
            df: DataFrame sorted by date with feature columns
            feature_cols: List of column names to use as input features
            target_col: Column name for prediction target
        """
        self.features = torch.FloatTensor(df[feature_cols].values)
        self.targets = torch.FloatTensor(df[target_col].values)
        self.seq_len = SEQUENCE_LENGTH
        self.horizon = FORECAST_HORIZON

    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y
