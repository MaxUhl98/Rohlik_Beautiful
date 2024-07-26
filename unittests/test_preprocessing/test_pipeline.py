import pickle

import numpy as np
import pytest

from preprocessing.pipeline import InitialPreprocessor
from unittests.mock_configurations import MockPreprocessingCFG, MockDataCFG
import pandas as pd
import os
from openfe.FeatureGenerator import Node


class TestInitialPreprocessor:

    def setup_method(self):
        if os.getcwd().rsplit('\\', 1)[1] == 'test_preprocessing':
            os.chdir('../..')
        self.data_cfg = MockDataCFG()
        self.preprocessing_cfg = MockPreprocessingCFG()
        self.preprocessor = InitialPreprocessor(self.preprocessing_cfg, self.data_cfg)
        self.X = pd.read_csv('unittests/test_preprocessing/files/unittest_dataframe.csv', index_col=0)

    def teardown_method(self):
        del self.preprocessing_cfg
        del self.data_cfg
        del self.preprocessor
        del self.X

    def test_drop_zero_variance_features(self):
        X_test = self.X.copy()
        X_test['test_0_variance_feature'] = 55
        X_test['test_another_0_variance_feature'] = 1
        X_test = self.preprocessor.drop_zero_variance_features(self.X)
        assert X_test.columns.tolist() == self.X.columns.tolist()

    def test_round_values(self):
        self.X['float_column'] = [1.99, 2.3454, 2.4534, 5.6765, 44.567, 4, 11.2323, 45.34543, 123421.4354, 23131.123,
                                  23432.324, 45.13, 123.12, 12.12, 42354.214]
        rounded_values = [2, 2.3, 2.5, 5.7, 44.6, 4, 11.2, 45.3, 123421.4, 23131.1, 23432.3, 45.1, 123.1, 12.1, 42354.2]
        self.X = self.preprocessor.round_values(self.X)
        assert self.X.float_column.tolist() == rounded_values

    def test_load_and_save_openfe_features(self):
        features = self.preprocessor.load_openfe_features()
        assert isinstance(features[0], Node)

        self.preprocessor.save_openfe_features(features)

        assert [f.name for f in features] == [f.name for f in self.preprocessor.load_openfe_features()]

    def test_standardize(self):
        self.data_cfg.standardize_columns = ['orders']
        self.preprocessing_cfg.standardize = True
        self.X = self.preprocessor.standardize(self.X)
        assert self.X[self.data_cfg.standardize_columns[0]].mean().round(0) == 0
        assert self.X[self.data_cfg.standardize_columns[0]].std().round(0) == 1

    def test_engineer_manual_features(self):
        correctly_engineered_data = self.preprocessor.round_values(
            pd.read_csv('unittests/test_preprocessing/files/engineered_unittest_dataframe.csv', index_col=0))
        processed_data = self.preprocessor.round_values(self.preprocessor.engineer_manual_features(self.X))
        assert (processed_data == correctly_engineered_data[processed_data.columns]).all().all()

    def test_encode_categorical_data(self):
        df_val = self.preprocessor.round_values(
            pd.read_csv('unittests/test_preprocessing/files/encoded_engineered_unittest_dataframe.csv',
                        index_col=0))
        df_test = self.preprocessor.round_values(
            self.preprocessor.encode_categorical_data(self.preprocessor.engineer_manual_features(self.X)))
        assert (df_test == df_val[df_test.columns]).all().all()

    def test_engineer_openfe_features_nan_assertion_error(self):
        self.preprocessor.openfe_filename = ''
        y = self.X.pop(self.data_cfg.target_column)
        try:
            X_nan = self.X.copy()
            X_nan['nan_col'] = np.nan
            self.preprocessor.engineer_openfe_features(X_nan, y)
        except AssertionError as ex:
            assert str(
                ex) == f'Feature data has NaN values in {self.preprocessor.openfe_filename}. NaN values are incompatible with OpenFE, remove NaN values or disable OpenFE'

    def test_engineer_loaded_openfe_features(self):
        y = self.X.pop(self.data_cfg.target_column)
        transformed_df = self.preprocessor.round_values(self.preprocessor.engineer_openfe_features(self.X, y)[0])
        assert (transformed_df == self.preprocessor.round_values(
            pd.read_csv('unittests/test_preprocessing/files/openfe_engineered_unittest_dataframe.csv', index_col=0)[
                transformed_df.columns])).all().all()

    def test_no_feature_selection(self):
        y = self.X.pop(self.data_cfg.target_column)
        new_df = self.preprocessor.select_features(self.X, y)
        assert (new_df == pd.concat([self.X, y], axis=1)[new_df.columns]).all().all()

    def test_transform_data(self):
        transformed_data = self.preprocessor.transform_data(self.X)
        df_val = pd.read_csv('unittests/test_preprocessing/files/unittest_full_run_dataframe.csv',
                             index_col=0)[transformed_data.columns]
        transformed_data = self.preprocessor.round_values(transformed_data)
        df_val = self.preprocessor.round_values(df_val)
        assert (transformed_data == df_val[transformed_data.columns]).all().all()

    def test_run(self):
        transformed_data = self.preprocessor.run(self.X)
        df_val = pd.read_csv('unittests/test_preprocessing/files/unittest_full_run_dataframe.csv',
                             index_col=0)[transformed_data.columns]
        transformed_data = self.preprocessor.round_values(transformed_data)
        df_val = self.preprocessor.round_values(df_val)
        assert (transformed_data == df_val[transformed_data.columns]).all().all()
