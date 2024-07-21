import numpy as np
import pandas as pd
from project_configuration.DataCFG import DataCFG


def engineer_basic_date_features(_data: pd.DataFrame, data_cfg: DataCFG) -> pd.DataFrame:
    _data[data_cfg.time_column] = pd.to_datetime(_data[data_cfg.time_column])
    _data['year'] = _data[data_cfg.time_column].dt.year
    _data['day'] = _data[data_cfg.time_column].dt.day
    _data['month'] = _data[data_cfg.time_column].dt.month
    _data['month_name'] = _data[data_cfg.time_column].dt.month_name()
    _data['day_of_week'] = _data[data_cfg.time_column].dt.day_name()
    _data['week'] = _data[data_cfg.time_column].dt.isocalendar().week
    _data['year_sin'] = np.sin(2 * np.pi * _data['year'])
    _data['year_cos'] = np.cos(2 * np.pi * _data['year'])
    _data['month_sin'] = np.sin(2 * np.pi * _data['month'] / 12)
    _data['month_cos'] = np.cos(2 * np.pi * _data['month'] / 12)
    _data['day_sin'] = np.sin(2 * np.pi * _data['day'] / 31)
    _data['day_cos'] = np.cos(2 * np.pi * _data['day'] / 31)
    return _data


def engineer_rohlik_specific_features(_data: pd.DataFrame, data_cfg: DataCFG) -> pd.DataFrame:
    raise NotImplementedError
