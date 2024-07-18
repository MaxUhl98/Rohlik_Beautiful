from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from project_configuration.DataCFG import DataCFG


def get_time_splits(_data: pd.DataFrame, data_cfg: DataCFG) -> None:
    time_series = _data[data_cfg.time_column]
    splitter = TimeSeriesSplit
