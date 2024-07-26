from project_configuration import PreprocessingCFG
import os
def test_preprocessing_func(*args, ** kwargs):
    return args, kwargs

class TestPreprocessingCFG:
    def setup_method(self):
        if os.getcwd().rsplit('\\', 1)[1] == 'test_project_configuration':
            os.chdir('../..')
        self.preprocessCFG = PreprocessingCFG().load(
            'unittests/test_project_configuration/files/test_preprocessing_cfg.txt')
    def teardown_method(self):
        del self.preprocessCFG

    def test_get_feature_engineering_function_name(self):

        assert self.preprocessCFG.get_feature_engineering_function_name() == 'None'
        self.preprocessCFG.specialized_feature_engineering_function = test_preprocessing_func
        assert self.preprocessCFG.get_feature_engineering_function_name() == 'test_preprocessing_func'

    def test_get_openfe_name(self):
        assert self.preprocessCFG.get_openfe_name() == 'True_regression_2_True_None'
