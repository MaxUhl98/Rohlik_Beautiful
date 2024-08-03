"""Keep in mind, that during this project all training functions HAVE to take the following arguments:
X_train: train features,
y_train: train targets,
X_test: test features,
y_test: test targets,
model: model to train,
train_cfg: TrainCFG object,
everything else can be passed to the CV function through **train_kwargs
and HAVE to return only the trained model and a single dictionary containing:
{'oof_predictions':array containing test(also called out-of-fold or oof during cross validation) predictions,
 'train_loss': Loss on training dataset, 'test_loss': Loss on test dataset}
"""
import pandas as pd
from typing import *
from project_configuration import TrainCFG
from model_training.pytorch_dataclasses import TimeSeriesWithFeatures
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


def train_sklearn_api_like(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                           model: Callable, train_cfg: TrainCFG, **train_kwargs) -> tuple[dict[str, Any], Callable]:
    """Train model via classical Scikit-learn commands

    :param X_train: Train Features
    :param y_train: Train Target
    :param X_test: Test features
    :param y_test: Test targets
    :param model: Model to train
    :param train_cfg: Training Configuration
    :param train_kwargs: Training Keyword-Arguments
    :return: ({'train_loss': Train Loss, 'test_loss': Test Loss,'oof_predictions': Out-of-fold predictions}, model)
    """
    fit_kwargs = train_kwargs.get('fit_kwargs')
    model = model.fit(X_train, y_train, **fit_kwargs) if fit_kwargs else model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_preds = model.predict(X_train)
    loss_fn = train_cfg.loss_fn
    return {'train_loss': loss_fn(y_train, train_preds), 'test_loss': loss_fn(y_test, y_pred),
            'oof_predictions': y_pred}, model


def train_pytorch_like_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.DataFrame,
                             model: Callable, train_cfg: TrainCFG, **train_kwargs) -> tuple[dict[str, Any], Callable]:
    """Pytorch train loop with gradient clipping, optional lr scheduling and patience

    :param X_train: Train Features
    :param y_train: Train Target
    :param X_test: Test features
    :param y_test: Test targets
    :param model: Model to train
    :param train_cfg: Training Configuration
    :param train_kwargs: Training Keyword-Arguments
    :return: ({'train_loss': Train Loss, 'test_loss': Test Loss,'oof_predictions': Out-of-fold predictions}, model)
    """
    fit_kwargs = train_kwargs.get('fit_kwargs')
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.apex)
    optimizer_kwargs = train_kwargs.get('optimizer_kwargs')
    train_dataset = TimeSeriesWithFeatures(pd.concat([X_train, y_train], axis=1), train_cfg=train_cfg,
                                           data_cfg=train_kwargs['data_config'])
    test_dataset = TimeSeriesWithFeatures(pd.concat([X_test, y_test], axis=1), train_cfg=train_cfg,
                                          data_cfg=train_kwargs['data_config'])
    train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=train_cfg.shuffle)
    test_loader = DataLoader(test_dataset, train_cfg.batch_size, shuffle=train_cfg.shuffle)
    optimizer = train_kwargs['optimizer'](model.parameters(), **optimizer_kwargs)
    loss_fn = train_kwargs['loss_function']
    oof_predictions = None
    best_test_loss = 10 ** 10
    best_train_loss = 10 ** 10
    patience_count = 0

    for epoch in range(train_cfg.num_epochs):
        model.train()
        for (X, y, mask) in train_loader:
            if train_cfg.lr_scheduling:
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=train_cfg.apex):
                y = y[mask]
                y_pred = model(X[mask, :], **fit_kwargs)

            with torch.cuda.amp.autocast(enabled=train_cfg.apex):
                loss = loss_fn(y_pred, y)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            if train_cfg.lr_scheduling:
                scaler.step(optimizer.optimizer)
                optimizer.step()
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        with torch.inference_mode():
            for (X, y, mask) in test_loader:
                y_pred = model(X[mask, :])
                test_loss = loss_fn(y_pred, y[mask])
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_count = 0
                    best_model = model.state_dict()
                    oof_predictions = y_pred
                    best_train_loss = loss
                else:
                    patience_count += 1
                if patience_count >= train_cfg.patience:
                    break
        return {'train_loss': best_train_loss,
                'test_loss': best_test_loss,
                'oof_predictions': oof_predictions}, model.load_state_dict(best_model)
