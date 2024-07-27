from project_configuration.ConfigurationFunctionalities import BasicFunctionalities
from sklearn.metrics import mean_absolute_percentage_error
from typing import *

class TrainCFG(BasicFunctionalities):
    """Class to configure the training process"""
    run_name: str = 'test'
    validation_time_steps: int = 60
    num_folds: int = 10
    run_save_directory: str = 'runs/saved_runs'
    log_dir: str = 'logs/training'
    aggregate_runs_path: str = 'runs/aggregate_runs.csv'

    loss_fn: Callable = mean_absolute_percentage_error
    loss_fn = staticmethod(loss_fn)

    lookback_length:int = 420

    additional_metrics  = {}

    def __init__(self):
        """This is necessary for the saving of variables
        (__dict__ values do not get set for class attributes)"""
        super().__init__()
        self.run_name = self.run_name
        self.validation_time_steps = self.validation_time_steps
        self.num_folds = self.num_folds
        self.run_save_directory = self.run_save_directory
        self.log_dir = self.log_dir
        self.loss_fn_name = self.loss_fn.__name__


