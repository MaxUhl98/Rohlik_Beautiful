from preprocessing.pipeline import InitialPreprocessor
from unittests.mock_configurations import MockPreprocessingCFG, MockDataCFG


class TestInitialPreprocessor:

    def setup_method(self):
        self.data_cfg = MockDataCFG()
        self.preprocessing_cfg = MockPreprocessingCFG()
        self.preprocessor = InitialPreprocessor(self.data_cfg, self.preprocessing_cfg)

    def teardown_method(self):
        del self.preprocessing_cfg
        del self.data_cfg
        del self.preprocessor

    def test_engineer_openfe_features(self):
        raise NotImplementedError



