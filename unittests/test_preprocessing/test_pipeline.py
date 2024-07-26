import pickle

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
        self.X = self.preprocessor.standardize(self.X)
        assert self.X['orders'].mean().round(0) == 0
        assert self.X['orders'].std().round(0) == 1

    def test_engineer_manual_features(self):
        correctly_engineered_data = self.preprocessor.round_values(pd.read_csv('unittests/test_preprocessing/files/engineered_unittest_dataframe.csv', index_col=0))
        processed_data = self.preprocessor.round_values(self.preprocessor.engineer_manual_features(self.X))
        assert (processed_data == correctly_engineered_data[processed_data.columns]).all().all()


    def test_encode_categorical_data(self):
        raise NotImplementedError

    def test_engineer_openfe_features(self):
        raise NotImplementedError



