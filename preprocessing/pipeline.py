import os.path
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder, StandardScaler
from project_configuration.DataCFG import DataCFG
from project_configuration.PreprocessingCFG import PreprocessingCFG
from preprocessing.feature_engineering import *
from openfe import OpenFE, transform
from typing import *
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from xgboost import XGBRegressor


class InitialPreprocessor:
    def __init__(self, preprocess_cfg: PreprocessingCFG, data_cfg: DataCFG):
        self.preprocess_cfg = preprocess_cfg
        self.data_cfg = data_cfg
        self.openfe_filename: str = f'{preprocess_cfg.openfe_feature_save_directory}/{preprocess_cfg.openfe_name}.pkl'
        self.data_save_path = f'{preprocess_cfg.data_save_directory}/{preprocess_cfg.name}.feather'

    def run(self, data: pd.DataFrame) -> pd.DataFrame:

        if os.path.exists(self.data_save_path):
            data = pd.read_feather(self.data_save_path)
        else:
            data = self.transform_data(data)
            data.to_feather(self.data_save_path)
        return data

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        X = data[self.data_cfg.usable_columns]
        X = self.engineer_manual_features(X)
        y = X.pop(self.data_cfg.target_column)

        if self.preprocess_cfg.use_openfe:
            X, y = self.engineer_openfe_features(X, y)

        X = self.encode_categorical_data(X)
        X = self.standardize(X)
        X = self.select_features(X)
        return pd.concat([X, y], axis=1)

    def engineer_openfe_features(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:

        if not os.path.exists(
                self.preprocess_cfg.openfe_feature_save_directory + f'/{self.preprocess_cfg.openfe_name}'):
            features = OpenFE().fit(X, y, **self.preprocess_cfg.openfe_kwargs)
            self.save_openfe_features(features)

        else:
            features = self.load_openfe_features()

        X, _ = transform(X, X.iloc[0:0], features, self.preprocess_cfg.openfe_kwargs['n_jobs'])
        return X, y

    def encode_categorical_data(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_cfg.use_onehot:
            X = pd.get_dummies(X, columns=self.data_cfg.categorical_columns)
        if self.preprocess_cfg.use_ordinal_encoding:
            X[self.data_cfg.ordinal_columns] = OrdinalEncoder().fit_transform(X[self.data_cfg.ordinal_columns])
        return X

    def engineer_manual_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_cfg.use_basic_timeseries_preprocessing:
            X = engineer_basic_date_features(X, self.data_cfg)
        if callable(self.preprocess_cfg.specialized_feature_engineering_function):
            X = self.preprocess_cfg.specialized_feature_engineering_function(X)
        return X

    def load_openfe_features(self) -> Any:
        with open(self.openfe_filename, 'rb') as f:
            return pickle.load(f)

    def save_openfe_features(self, features: Any) -> None:
        with open(self.openfe_filename, 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

    def standardize(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_cfg.standardize:
            X[self.data_cfg.standardize_columns] = StandardScaler().fit_transform(X[self.data_cfg.standardize_columns])
        return X

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        match self.preprocess_cfg.feature_selection_method:
            case None:
                return X
            case 'SFS':
                select = SequentialFeatureSelector(XGBRegressor(), scoring=self.preprocess_cfg.scoring)
                X[X.columns] = select.fit_transform(X.values, y.values)
            case 'Model':
                select = SelectFromModel(XGBRegressor())
                X[X.columns] = select.fit_transform(X.values, y.values)
        return X


class CVPreprocessor:
    def __init__(self, preprocess_cfg: PreprocessingCFG, data_cfg: DataCFG):
        self.preprocess_cfg = preprocess_cfg
        self.data_cfg = data_cfg
