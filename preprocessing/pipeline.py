import os.path
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder, StandardScaler
from project_configuration import DataCFG, PreprocessingCFG
from preprocessing.feature_engineering import *
from openfe import OpenFE, transform
from typing import *
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import TargetEncoder
from preprocessing.BasePipeline import BasePipeline


class InitialPreprocessor(BasePipeline):
    """A class to perform initial preprocessing of the dataset.

    :param preprocess_cfg: Configuration object for preprocessing settings.
    :param data_cfg: Configuration object for data settings.
    """

    def __init__(self, preprocess_cfg: PreprocessingCFG, data_cfg: DataCFG):
        super().__init__(preprocess_cfg, data_cfg)
        self.openfe_filename: str = f'{preprocess_cfg.openfe_feature_save_directory}/{preprocess_cfg.openfe_name}.pkl'
        self.data_save_path = f'{preprocess_cfg.data_save_directory}/{preprocess_cfg.name}.feather'

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the preprocessing pipeline on the provided data.

        :param data: Input data to preprocess.
        :return: Preprocessed data.
        """
        if os.path.exists(self.data_save_path):
            data = pd.read_feather(self.data_save_path)
        else:
            data = self.transform_data(data)
            data.to_feather(self.data_save_path)
        return data

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data through feature engineering, encoding, and selection.
         Performs these steps exactly as specified in the PreprocessingCFG object.

        :param data: Input data to transform.
        :return: Transformed data.
        """
        X = data[self.data_cfg.usable_columns + [self.data_cfg.target_column]]
        X = self.engineer_manual_features(X)
        y = X.pop(self.data_cfg.target_column)
        if self.preprocess_cfg.use_openfe:
            X, y = self.engineer_openfe_features(X, y)
            X = self.round_values(X)

        X = self.encode_categorical_data(X)
        X = self.drop_zero_variance_features(X)
        X = self.standardize(X)
        X = self.select_features(X, y)
        return pd.concat([X, y], axis=1)

    def engineer_openfe_features(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Engineer features using OpenFE if not already saved.

        :param X: Feature data.
        :param y: Target data.
        :return: Tuple of transformed feature data and target data.
        """
        if not os.path.exists(self.openfe_filename):
            if X.isna().any().any():
                raise AssertionError(f'Feature data has NaN values in {self.openfe_filename}. '
                                     f'NaN values are incompatible with OpenFE, remove NaN values or disable OpenFE')
            features = OpenFE().fit(X, y, **self.preprocess_cfg.openfe_kwargs)
            self.save_openfe_features(features)
        else:
            features = self.load_openfe_features()

        X, _ = transform(X, X.iloc[0:0], features, self.preprocess_cfg.openfe_kwargs['n_jobs'])
        return X, y

    def encode_categorical_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical data using specified encoders.

        :param X: Feature data to encode.
        :return: Encoded feature data.
        """
        if self.preprocess_cfg.use_onehot:
            X = pd.get_dummies(X, columns=self.data_cfg.categorical_columns)
        if self.preprocess_cfg.use_ordinal_encoding:
            X[self.data_cfg.ordinal_columns] = OrdinalEncoder().fit_transform(X[self.data_cfg.ordinal_columns])
        return X

    def engineer_manual_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer manual features based on the configuration.

        :param X: Feature data to engineer.
        :return: Feature data with engineered features.
        """
        X = self.round_values(X)
        if self.preprocess_cfg.use_basic_timeseries_preprocessing:
            X = engineer_basic_date_features(X, self.data_cfg)
            X = self.round_values(X)
        if callable(self.preprocess_cfg.specialized_feature_engineering_function):
            X = self.preprocess_cfg.specialized_feature_engineering_function(X)
            X = self.round_values(X)
        return X

    def load_openfe_features(self) -> Any:
        """Load OpenFE features from a file.

        :return: Loaded OpenFE features.
        """
        with open(self.openfe_filename, 'rb') as f:
            return pickle.load(f)

    def save_openfe_features(self, features: Any) -> None:
        """Save OpenFE features to a file.

        :param features: OpenFE features to save.
        """
        with open(self.openfe_filename, 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

    def standardize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize the feature data.

        :param X: Feature data to standardize.
        :return: Standardized feature data.
        """
        if self.preprocess_cfg.standardize:
            X[self.data_cfg.standardize_columns] = StandardScaler().fit_transform(X[self.data_cfg.standardize_columns])
        return X

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Select features based on the configuration method.

        :param X: Feature data.
        :param y: Target data.
        :return: Feature data with selected features.
        """
        match self.preprocess_cfg.feature_selection_method:
            case None:
                pass
            case 'SFS':
                select = SequentialFeatureSelector(XGBRegressor(), scoring=self.preprocess_cfg.scoring).fit(X, y)
                X = X[[feature for feature, is_supported in zip(X.columns, select.support_) if is_supported]]
            case 'Model':
                select = SelectFromModel(XGBRegressor()).fit(X.values, y.values)
                X = X[[feature for feature, is_supported in zip(X.columns, select.support_) if is_supported]]
        return X

    def round_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rounds float values in X to specified precision

        :param X: Feature data to round.
        :return: Rounded feature data.
        """
        X_categorical = X.select_dtypes(include=['object', 'category'])
        X = X.select_dtypes(exclude=['object', 'category']).round(decimals=self.preprocess_cfg.rounding_precision)
        return pd.concat([X, X_categorical], axis=1)

    def drop_zero_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop zero variance features.

        :param X: Feature data.
        :return: Feature data without zero variance features.
        """
        standard_deviations = X.select_dtypes(exclude=['object', 'category']).std()
        drop_cols = standard_deviations.loc[standard_deviations == 0].index
        return X.drop(columns=drop_cols)


class CVPreprocessor(BasePipeline):
    """A class to perform preprocessing inside the CV loop.

    :param preprocess_cfg: Configuration object for preprocessing settings.
    :param data_cfg: Configuration object for data settings.
    """

    def __init__(self, preprocess_cfg: PreprocessingCFG, data_cfg: DataCFG):
        super().__init__(preprocess_cfg, data_cfg)

    def run(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Execute the preprocessing pipeline on the provided data.

        :param X: Feature data to preprocess.
        :param y: Target data.
        :return: Encoded feature data.
        """
        X = self.encode_target(X, y)
        return X

    def encode_target(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Applies target encoding on the configured columns

        :param X: Feature Dataframe
        :param y: Target Series
        :return: Target encoded feature Dataframe
        """
        encoder = TargetEncoder(target_type='continuous', shuffle=False)
        X[self.data_cfg.target_encoding_cols] = encoder.fit_transform(X[self.data_cfg.target_encoding_cols], y)
        return X
