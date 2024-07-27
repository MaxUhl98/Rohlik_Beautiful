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


def train_sklearn_api_like(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                           model: Callable, train_cfg: TrainCFG, **train_kwargs) -> tuple[dict[str, Any], Callable]:
    fit_kwargs = train_kwargs.get('fit_kwargs')
    model = model.fit(X_train, y_train, **fit_kwargs) if fit_kwargs else model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_preds = model.predict(X_train)
    loss_fn = train_cfg.loss_fn
    return {'train_loss': loss_fn(y_train, train_preds), 'test_loss': loss_fn(y_test, y_pred),
            'oof_predictions': y_pred}, model


def train_pytorch_like_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.DataFrame,
                             model: Callable,train_cfg: TrainCFG, **train_kwargs) -> tuple[dict[str, Any], Callable]:
    fit_kwargs = train_kwargs.get('fit_kwargs')

