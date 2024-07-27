import logging
import os
from datetime import datetime
from typing import Any, Callable, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from project_configuration import DataCFG, TrainCFG, PreprocessingCFG
import pickle
from preprocessing import CVPreprocessor
from torch import nn


def get_sorted_timeseries_array(series: pd.Series) -> np.ndarray:
    """Convert series to a sorted numpy array of unique dates.

    :param series: Series to create the sorted array from.
    :return: Sorted numpy array of unique dates.
    """
    return np.array(sorted(series.unique()))


def create_new_save_directory(train_cfg: TrainCFG) -> str:
    """Create a new save directory, incrementing suffix if it already exists.

    :param train_cfg: TrainCFG object used to determine the directory name
    :return: Path of the new save directory
    """
    base_dir = os.path.join(train_cfg.run_save_directory, train_cfg.run_name)
    save_dir_path = base_dir
    n = 1
    while os.path.exists(save_dir_path):
        save_dir_path = f'{base_dir}_{n}'
        n += 1
    os.mkdir(save_dir_path)
    return save_dir_path


def get_fold_data(
        _data: pd.DataFrame, data_cfg: DataCFG, time_series: np.ndarray,
        train_times: np.ndarray, test_times: np.ndarray, lookback_window: int = 0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the data into training and testing sets based on time series.

    :param _data: Pandas dataframe containing train and test data
    :param data_cfg: Configuration for data usage
    :param time_series: Array containing all unique sorted time steps of data
    :param train_times: Indices of train dates in time series
    :param test_times: Indices of test dates in time series
    :param lookback_window: Number of days from the past to include in test data
    :return: Tuple containing (X_train, y_train, X_test, y_test)
    """
    X_train = _data[_data[data_cfg.time_column] <= time_series[train_times[-1]]]
    X_test = _data[
        (_data[data_cfg.time_column] >= time_series[test_times[0] - lookback_window]) &
        (_data[data_cfg.time_column] <= time_series[test_times[-1]])
        ]
    y_train, y_test = X_train.pop(data_cfg.target_column), X_test.pop(data_cfg.target_column)
    return X_train, y_train, X_test, y_test


def log_and_save_results(
        train_data: dict[str, Any], additional_metrics: Iterable[Callable],
        additional_metric_losses: dict[str, Any], oof_predictions: list[Any],
        cv_train_losses: list[Any], cv_test_losses: list[Any], cv_logger: logging.Logger,
        y_test: pd.Series, num: int
) -> None:
    """Log and save cross-validation results.

    :param train_data: Data returned by the train function.
    :param additional_metrics: List of metrics to use outside the train loop
    :param additional_metric_losses: Dictionary containing lists of additional metric values from previous folds
    :param oof_predictions: List of oof predictions from previous folds
    :param cv_train_losses: List of train losses from previous folds
    :param cv_test_losses: List of test losses from previous folds
    :param cv_logger: Logger to use for cross-validation results
    :param y_test: Test targets from current fold
    :param num: Number of the current fold
    :return: None
    """
    for metric in additional_metrics:
        additional_metric_losses[metric.__name__].append(metric(y_test, train_data['oof_predictions']))
    oof_predictions.append(train_data['oof_predictions'])
    cv_train_losses.append(train_data['train_loss'])
    cv_test_losses.append(train_data['test_loss'])
    cv_logger.info(
        f'Fold {num} Train Loss: {train_data["train_loss"].item()}, Test Loss: {train_data["test_loss"].item()}')


def save_cv_results_and_settings(
        train_cfg: TrainCFG, data_cfg: DataCFG, save_directory_path: str, oof_predictions: list[Any],
        cv_train_losses: list[Any], cv_test_losses: list[Any], additional_metric_losses: dict[str, Any]
) -> None:
    """Save the cross-validation results and training configuration.

    :param train_cfg: Training configuration
    :param save_directory_path: Path to run directory
    :param oof_predictions: List containing all oof predictions of the current run
    :param cv_train_losses: List containing all training losses of the current run
    :param cv_test_losses: List containing all test losses of the current run
    :param additional_metric_losses: Dict containing all lists of additional losses of the current run
    :return: None (saves everything inside the save directory)
    """
    train_cfg.save(os.path.join(save_directory_path, 'train_cfg.txt'))
    data_cfg.save(os.path.join(save_directory_path, 'data_cfg.txt'))
    for k, v in additional_metric_losses.items():
        additional_metric_losses[k] = np.concatenate(v)
    df_cv = pd.DataFrame({'train_loss': np.array(cv_train_losses), 'test_loss': np.array(cv_test_losses),
                          **additional_metric_losses})
    df_cv.to_csv(os.path.join(save_directory_path, 'losses.csv'), index=False)
    np.save(os.path.join(save_directory_path, 'all_oof_predictions.npy'), np.concatenate(oof_predictions))


def save_models_and_run_data(save_directory_path: str, train_cfg: TrainCFG, data_cfg: DataCFG,
                             oof_predictions: list[Any],
                             cv_train_losses: list[Any], cv_test_losses: list[Any],
                             additional_metric_losses: dict[str, Any], models: list[Any]) -> None:
    """Saved models and CV results to save_directory_path. Also saves CV results in aggregated view.

    :param save_directory_path: Path to the current run directory
    :param train_cfg: Training configuration
    :param oof_predictions: List of all oof predictions of the current run
    :param cv_train_losses: List of all training losses of the current run
    :param cv_test_losses: List of all test losses of the current run
    :param additional_metric_losses: Dict containing all lists of additional losses of the current run
    :param models: List of all models trained models from the current run
    :return: None (saves everything inside the save directory)
    """
    save_cv_results_and_settings(train_cfg, data_cfg, save_directory_path, oof_predictions, cv_train_losses,
                                 cv_test_losses,
                                 additional_metric_losses)

    with open(save_directory_path + '/models.pickle', 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    aggregated_runs = pd.read_csv(train_cfg.aggregate_runs_path, index_col=0)
    aggregated_runs = pd.concat([aggregated_runs, pd.DataFrame(
        {'run_name': [save_directory_path.rsplit('/', 1)[1]], 'avg_cv_test_loss': [np.mean(cv_test_losses)],
         'num_folds': [train_cfg.num_folds], 'model_name': [models[0].__class__.__name__]}, index=None)],
                                ignore_index=True, axis=0).reset_index(drop=True)
    aggregated_runs.to_csv(train_cfg.aggregate_runs_path)


def execute_cv_loop(
        _data: pd.DataFrame, train_function: Callable, models: list[Any], train_cfg: TrainCFG, data_cfg: DataCFG,
        preprocess_cfg: PreprocessingCFG, cv_logger: logging.Logger, additional_metrics: Iterable[Callable] = (),
        **train_kwargs
) -> list[float]:
    """Execute the cross-validation loop. Saves Out-of-fold predictions (OOF), train and test loss and trained models.

    :param _data: Pandas dataframe containing training and testing data.
    :param train_function: Function used to train the models.
    :param models: List of models to train.
    :param train_cfg: Configuration used to train the models.
    :param data_cfg: Data configuration used to train the models.
    :param cv_logger: Logger used to log the Cross-Validation results.
    :param additional_metrics: Additional metrics used to evaluate the models.
    :param train_kwargs: Additional arguments used by the train function.
    :return: None
    """
    save_directory_path = create_new_save_directory(train_cfg)
    time_series = get_sorted_timeseries_array(_data[data_cfg.time_column])
    splitter = TimeSeriesSplit(test_size=train_cfg.validation_time_steps, n_splits=train_cfg.num_folds)

    oof_predictions, cv_train_losses, cv_test_losses = [], [], []
    additional_metric_losses = {metric.__name__: [] for metric in additional_metrics}

    for num, (train_times, test_times) in enumerate(splitter.split(time_series), start=1):
        if issubclass(models[num - 1], nn.Module):
            X_train, y_train, X_test, y_test = get_fold_data(_data, data_cfg, time_series, train_times, test_times,
                                                             train_cfg.lookback_length)
        else:
            X_train, y_train, X_test, y_test = get_fold_data(_data, data_cfg, time_series, train_times, test_times, 0)
        if data_cfg.target_encoding_cols and preprocess_cfg.use_target_encoding:
            processor = CVPreprocessor(preprocess_cfg, data_cfg)
            X_train = processor.run(X_train, y_train)
            X_test = processor.transform_features(X_test)
        if not issubclass(models[num - 1], nn.Module):
            X_train, X_test = X_train.drop(columns=data_cfg.time_column), X_test.drop(columns=data_cfg.time_column)
        train_data, models[num - 1] = train_function(X_train, y_train, X_test, y_test, models[num - 1], train_cfg,
                                                     **train_kwargs)
        log_and_save_results(train_data, additional_metrics, additional_metric_losses,
                             oof_predictions, cv_train_losses, cv_test_losses, cv_logger, y_test, num)

    save_models_and_run_data(str(save_directory_path), train_cfg, data_cfg, oof_predictions, cv_train_losses,
                             cv_test_losses,
                             additional_metric_losses, models)
    return cv_test_losses
