import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import CVPreprocessor, InitialPreprocessor
from project_configuration import TrainCFG, DataCFG, PreprocessingCFG
from model_training.time_series_cv import execute_cv_loop
from utils.log_helpers import get_logger
from typing import *
from model_training.model_behaviour import ModelBehaviour


def get_data_and_logger(train_cfg: TrainCFG, data_cfg: DataCFG, preprocessing_cfg: PreprocessingCFG) -> tuple[
    pd.DataFrame, logging.Logger]:
    train_logger = get_logger(train_cfg.run_name, train_cfg.log_dir)
    train_logger.info(f'Starting training with\nTrainCFG:{train_cfg}\nPreprocessingCFG:{preprocessing_cfg}')
    preprocessor = InitialPreprocessor(preprocessing_cfg, data_cfg, logger=train_logger)
    data = pd.read_csv(data_cfg.train_path)
    data = preprocessor.run(data)
    return data, train_logger


def cross_validate_model(model: Callable, train_function: Callable, model_init_kwargs, model_fit_kwargs,
                         train_cfg: TrainCFG,
                         data_cfg: DataCFG, preprocessing_cfg: PreprocessingCFG,
                         additional_metrics: Iterable[Callable] = ()) -> \
        List[float]:
    data, train_logger = get_data_and_logger(train_cfg, data_cfg, preprocessing_cfg)

    models = [model(**model_init_kwargs) for _ in range(train_cfg.num_folds)]
    test_losses = execute_cv_loop(data, train_function, models, train_cfg, data_cfg, preprocessing_cfg, train_logger,
                                  additional_metrics,
                                  **model_fit_kwargs)
    return test_losses


def train_model_on_full_data(model: Callable, train_function: Callable, model_init_kwargs: dict[str, Any],
                             model_fit_kwargs: dict[str, Any], train_cfg: TrainCFG, data_cfg: DataCFG,
                             preprocessing_cfg: PreprocessingCFG, ) -> tuple[dict[str, Any], Callable]:
    model = model(**model_init_kwargs)
    data, train_logger = get_data_and_logger(train_cfg, data_cfg, preprocessing_cfg)
    X = data
    y = X.pop(data_cfg.target_column)
    train_data, model = train_function(X, y, X.head(train_cfg.validation_time_steps),
                                       y.head(train_cfg.validation_time_steps), model, train_cfg, **model_fit_kwargs)

    return train_data, model


def get_training_setup_from_model_name(model_name: str) -> tuple[Callable, Callable]:
    model_name = model_name.lower()
    return ModelBehaviour.name_to_model_map[model_name], ModelBehaviour.get_train_function(model_name)


def execute_cv_model_training(model_name: str, model_init_kwargs, model_fit_kwargs, train_cfg: TrainCFG,
                              data_cfg: DataCFG,
                              preprocessing_cfg: PreprocessingCFG) -> List[float]:
    model, train_function = get_training_setup_from_model_name(model_name)
    data_from_training = cross_validate_model(model, train_function, model_init_kwargs, model_fit_kwargs, train_cfg,
                                              data_cfg, preprocessing_cfg)
    return data_from_training
