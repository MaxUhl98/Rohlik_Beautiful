from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, StackingRegressor, \
    VotingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from transformers.models.patchtst import PatchTSTModel
from typing import *
from model_training.train_functions import train_sklearn_api_like, train_pytorch_like_model


class ModelBehaviour:
    sklearn_like_models = ['lgbm', 'xgb', 'cat', 'gbt', 'adaboost', 'lr', 'ridge', 'lasso', 'stacking', 'voting',
                           'elasticnet', 'rf']
    pytorch_like_models = ['patchtst']

    name_to_model_map = {'lgbm': LGBMRegressor, 'xgb': XGBRegressor, 'cat': CatBoostRegressor,
                         'gbt': GradientBoostingRegressor, 'adaboost': AdaBoostRegressor, 'lr': LinearRegression,
                         'ridge': Ridge, 'lasso': Lasso, 'stacking': StackingRegressor, 'voting': VotingRegressor,
                         'elasticnet': ElasticNet, 'patchtst': PatchTSTModel, 'rf': RandomForestRegressor}

    @staticmethod
    def get_train_function(model_name: str) -> Callable:
        if model_name in ModelBehaviour.sklearn_like_models:
            return train_sklearn_api_like
        elif model_name in ModelBehaviour.pytorch_like_models:
            return train_pytorch_like_model
        else:
            raise ValueError(f'Model {model_name} not recognized')
