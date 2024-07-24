import os
from project_configuration.TrainCFG import TrainCFG
from ast import literal_eval


class TestTrainCFG:

    def setUp(self):
        self.cfg = TrainCFG()
        if os.getcwd().rsplit('\\', 1)[1] == 'test_project_configuration':
            os.chdir('../..')
        self.save_directory = 'unittests/test_project_configuration/files'
        self.value_dict = {
            'run_name': 'test',
            'validation_time_steps': 60,
            'num_folds': 10,
            'run_save_directory': 'runs/',
            'log_dir': 'logs/training'}

    def tearDown(self):
        del self.cfg
        del self.save_directory
        del self.value_dict

    def test_save(self):
        self.setUp()
        self.cfg.save(self.save_directory + '/test.txt')
        assert os.path.isfile(self.save_directory + '/test.txt')
        with open(self.save_directory + '/test.txt', 'r') as f:
            assert literal_eval(f.read()) == self.value_dict

    def test_load(self):
        self.setUp()
        self.cfg.load(self.save_directory + '/load_test.txt')
        assert {k: v for k, v in vars(self.cfg).items() if not callable(v) and not k.startswith("__")} == {
            'run_name': 'test_new', 'validation_time_steps': 10, 'num_folds': 100, 'run_save_directory': 'runs/test/','log_dir': 'logs/training'}
