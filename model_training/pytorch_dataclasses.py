import pandas as pd
from torch.utils.data import Dataset
from project_configuration import DataCFG, TrainCFG
import torch


class TimeSeriesWithFeatures(Dataset):
    def __init__(self, data: pd.DataFrame, data_cfg: DataCFG, train_cfg: TrainCFG) -> None:
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.lookback_length = self.train_cfg.lookback_length
        self.validation_time_steps = self.train_cfg.validation_time_steps
        self.data = data.pivot(columns=data_cfg.group_column, index=data_cfg.time_column)
        self.y = data.pop(data_cfg.target_column)

    def __len__(self) -> int:
        return len(self.data) - self.lookback_length - self.validation_time_steps

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx:idx + self.lookback_length, :], self.y[idx + self.lookback_length:idx + self.lookback_length + self.validation_time_steps,:]
