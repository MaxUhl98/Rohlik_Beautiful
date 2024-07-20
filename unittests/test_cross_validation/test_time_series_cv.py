import shutil
import pickle
from cross_validation.time_series_cv import *
import pandas as pd
from project_configuration.TrainCFG import TrainCFG
from project_configuration.DataCFG import DataCFG
from sklearn.linear_model import LinearRegression


def gen():
    yield {'train_loss': np.array([1.34]), 'test_loss': np.array([3.4]), 'oof_predictions': np.array([3])}
    yield {'train_loss': np.array([10.34]), 'test_loss': np.array([309.4]),
           'oof_predictions': np.array([1, 2])}
    yield {'train_loss': np.array([11.3]), 'test_loss': np.array([23.56]),
           'oof_predictions': np.array([1, 2, 3])}

g = gen()
def mock_train_func(X_train, y_train, X_test, y_test, model, train_cfg, **train_kwargs):
    assert model is not None
    assert not isinstance(model, Iterable)
    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(X_test) == np.ndarray
    assert type(y_test) == np.ndarray
    assert type(train_cfg) == TrainCFG
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    return next(g)


class TestTimeSeriesCV:
    def setup_method(self):
        if os.getcwd().rsplit('\\', 1)[1] == 'test_cross_validation':
            os.chdir('../..')
        self.save_directory = 'unittests/test_cross_validation/files'
        self.train_cfg = TrainCFG()
        self.train_cfg.num_folds = 3
        self.train_cfg.validation_time_steps = 1
        self.train_cfg.run_save_directory = self.save_directory
        self.data_cfg = DataCFG()
        self._data = pd.DataFrame(
            {'date': ['2022-01-01', '2022-05-01', '2022-04-10', '2022-04-09', '2022-01-01'], 'orders': [1, 2, 3, 4, 5],
             'warehouse': ['A', 'A', 'A', 'A', 'B']})

        self.test_oof_predictions = [np.array([1, 2, 3]), np.array([3, 4, 5])]
        self.test_cv_train_losses = [np.array([.01, .2, .3]), np.array([.03, .14, .0095])]
        self.test_cv_test_losses = [np.array([.11, .02, 1.93]), np.array([.13, .34, .0195])]
        self.additional_metric_losses = {
            'MAE': [np.array([5.345, 1.234, 67.89]), np.array([11234.4, 2342.23, 1221.452])]}
        self.models = [LinearRegression() for _ in range(self.train_cfg.num_folds)]

    def teardown_method(self):
        del self.train_cfg
        del self.data_cfg
        del self._data
        del self.test_oof_predictions
        del self.test_cv_train_losses
        del self.test_cv_test_losses
        del self.additional_metric_losses

    def test_create_new_save_directory(self):
        create_new_save_directory(self.train_cfg)
        assert os.path.exists(self.train_cfg.run_save_directory)
        os.rmdir(self.train_cfg.run_save_directory + f'/{self.train_cfg.run_name}')

    @staticmethod
    def test_get_sorted_timeseries_array():
        date_series = pd.Series(['2024-01-01', '2023-01-01', '2024-01-02', '2023-01-02', '2024-02-03', '2024-02-05'])
        to_test = get_sorted_timeseries_array(date_series)
        sorted_array = np.array(['2023-01-01', '2023-01-02', '2024-01-01', '2024-01-02', '2024-02-03', '2024-02-05'])
        assert (to_test == sorted_array).all()

    def test_get_fold_data(self):
        time_series = get_sorted_timeseries_array(self._data[self.data_cfg.time_column])
        train_idx = np.array([0, 1])
        test_idx = np.array([2])
        X_train, y_train, X_test, y_test = get_fold_data(self._data, self.data_cfg, time_series, train_idx, test_idx)
        assert (X_train == np.array([['2022-01-01', 'A'], ['2022-04-09', 'A'], ['2022-01-01', 'B'], ],
                                    dtype=np.object_)).all()
        assert (y_train == np.array([1, 4, 5])).all()
        assert (X_test == np.array([['2022-04-10', 'A']], dtype=np.object_)).all()
        assert (y_test == np.array([3])).all()

    def test_save_cv_results_and_settings(self):
        save_cv_results_and_settings(self.train_cfg, self.save_directory, self.test_oof_predictions,
                                     self.test_cv_train_losses, self.test_cv_test_losses, self.additional_metric_losses)
        assert (np.load(self.save_directory + '/all_oof_predictions.npy') == np.concatenate(
            self.test_oof_predictions)).all()
        assert (pd.read_csv(self.save_directory + '/losses.csv') == pd.DataFrame(
            {'train_loss': np.concatenate(self.test_cv_train_losses),
             'test_loss': np.concatenate(self.test_cv_test_losses),
             **self.additional_metric_losses})).all().all()
        assert TrainCFG().load(os.path.join(self.save_directory, 'train_cfg.txt')) == self.train_cfg

    def test_execute_cv_loop(self):
        execute_cv_loop(self._data, mock_train_func, self.models, self.train_cfg, self.data_cfg)
        with open(self.save_directory+'/test/models.pickle', 'rb') as f:
            models = pickle.load(f)
        assert [model.__dict__ for model in models] == [model.__dict__ for model in self.models]
        shutil.rmtree(self.train_cfg.run_save_directory + f'/{self.train_cfg.run_name}')
