from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from project_configuration.DataCFG import DataCFG
from project_configuration.TrainCFG import TrainCFG
import numpy as np
from datetime import datetime
from typing import *
import os


def get_sorted_timeseries_array(series: pd.Series) -> np.ndarray:
    series = series.unique().tolist()
    series.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return np.array(series)


def create_new_save_directory(train_cfg: TrainCFG) -> str:
    n = 1
    save_dir_path = train_cfg.run_save_directory + train_cfg.run_name
    while 1:

        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
            break
        else:
            n += 1
            save_dir_path = f'{train_cfg.run_save_directory + train_cfg.run_name}_{n}'
    return save_dir_path


def execute_cv_loop(_data: pd.DataFrame, train_function: Callable, train_cfg: TrainCFG, data_cfg: DataCFG,
                    **train_kwargs) -> None:
    save_directory_path = create_new_save_directory(train_cfg)
    time_series = get_sorted_timeseries_array(_data[data_cfg.time_column])
    splitter = TimeSeriesSplit(test_size=train_cfg.validation_time_steps, n_splits=train_cfg.num_folds)

    oof_predictions = []
    for train_times, test_times in splitter.split(time_series):
        X_train = _data.loc[_data[data_cfg.time_column] <= time_series[train_times[-1]]]
        X_test = _data.loc[time_series[test_times[-1]] <= _data[data_cfg.time_column] <= time_series[test_times[-1]]]
        y_train, y_test = X_train.pop(data_cfg.target_column), X_test.pop(data_cfg.target_column)
        train_data = train_function(X_train, y_train, X_test, y_test, train_cfg, **train_kwargs)
        oof_predictions.append(train_data['oof_predictions'])
    all_oof_predictions = np.concatenate(oof_predictions)
    np.save(save_directory_path + '/all_oof_predictions.npy', all_oof_predictions)
