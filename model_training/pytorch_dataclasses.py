import pandas as pd
from torch.utils.data import Dataset
from project_configuration import DataCFG, TrainCFG
import torch


class TimeSeriesWithFeatures(Dataset):
    def __init__(self, data: pd.DataFrame, data_cfg: DataCFG, train_cfg: TrainCFG) -> None:
        torch.set_default_device(train_cfg.device)
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.lookback_length = self.train_cfg.lookback_length
        self.validation_time_steps = self.train_cfg.validation_time_steps
        self.data = data.pivot(columns=data_cfg.group_column, index=data_cfg.time_column)
        self.y = self.data.pop(data_cfg.target_column)
        self.data = torch.from_numpy(self.data.values).to(train_cfg.device).float()
        self.y = torch.from_numpy(self.y.values).to(train_cfg.device).float()

    def __len__(self) -> int:
        return len(self.data) - self.lookback_length - self.validation_time_steps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_start = idx + self.lookback_length
        target_end = target_start + self.validation_time_steps
        target = self.y[target_start:target_end, :]
        target_mask = (target != -1).float()
        return self.data[idx:idx + self.lookback_length, :], target, target_mask
